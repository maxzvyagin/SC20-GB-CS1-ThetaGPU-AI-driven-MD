import os
import glob
import logging
from pathlib import Path
import shutil
import time
import random

import parmed as pmd
import simtk.unit as u
import simtk.openmm as omm
import simtk.openmm.app as app
from deepdrivemd.util import FileLock
from deepdrivemd.sim.utils import concat_h5, cleanup_h5

from deepdrivemd.sim.openmm_reporter import SparseContactMapReporter
from deepdrivemd.util import LocalCopySender, RemoteCopySender

logger = logging.getLogger(__name__)


def configure_amber_implicit(
    pdb_file, top_file, dt_ps, temperature_kelvin, platform, platform_properties
):

    # Configure system
    if top_file:
        pdb = pmd.load_file(top_file, xyz=pdb_file)
        system = pdb.createSystem(
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * u.nanometer,
            constraints=app.HBonds,
            implicitSolvent=app.OBC1,
        )
    else:
        pdb = pmd.load_file(pdb_file)
        forcefield = app.ForceField("amber99sbildn.xml", "amber99_obc.xml")
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * u.nanometer,
            constraints=app.HBonds,
        )

    # Congfigure integrator
    integrator = omm.LangevinIntegrator(temperature_kelvin, 91.0 / u.picosecond, dt_ps)
    integrator.setConstraintTolerance(0.00001)

    sim = app.Simulation(
        pdb.topology, system, integrator, platform, platform_properties
    )

    # Return simulation and handle to coordinates
    return sim, pdb


def configure_amber_explicit(
    pdb_file, top_file, dt_ps, temperature_kelvin, platform, platform_properties
):

    # Configure system
    top = pmd.load_file(top_file, xyz=pdb_file)
    system = top.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * u.nanometer,
        constraints=app.HBonds,
    )

    # Congfigure integrator
    integrator = omm.LangevinIntegrator(temperature_kelvin, 1 / u.picosecond, dt_ps)
    system.addForce(omm.MonteCarloBarostat(1 * u.bar, temperature_kelvin))

    sim = app.Simulation(
        top.topology, system, integrator, platform, platform_properties
    )

    # Return simulation and handle to coordinates
    return sim, top


def configure_simulation(
    ctx,
    sim_type="implicit",
    gpu_index=0,
    dt_ps=0.002 * u.picoseconds,
    temperature_kelvin=300 * u.kelvin,
):
    logger.info(f"configure_simulation: {sim_type} {ctx.pdb_file}")
    # Configure hardware
    try:
        platform = omm.Platform_getPlatformByName("CUDA")
        platform_properties = {"DeviceIndex": str(gpu_index), "CudaPrecision": "mixed"}
    except Exception:
        platform = omm.Platform_getPlatformByName("OpenCL")
        platform_properties = {"DeviceIndex": str(gpu_index)}

    # Select implicit or explicit solvent
    args = (
        ctx.pdb_file,
        ctx.top_file,
        dt_ps,
        temperature_kelvin,
        platform,
        platform_properties,
    )
    if sim_type == "implicit":
        sim, coords = configure_amber_implicit(*args)
    else:
        assert sim_type == "explicit"
        sim, coords = configure_amber_explicit(*args)

    # Set simulation positions
    if coords.get_coordinates().shape[0] == 1:
        sim.context.setPositions(coords.positions)
    else:
        positions = random.choice(coords.get_coordinates())
        sim.context.setPositions(positions / 10)

    # Minimize energy and equilibrate
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(300 * u.kelvin, random.randint(1, 10000))

    return sim


def configure_reporters(sim, ctx, report_interval_ps, dt_ps, frames_per_h5, wrap):
    # Configure reporters
    report_freq = int(report_interval_ps / dt_ps)

    # Overwriting sim.reporters: we're writing new report files but with same
    # simulation state
    sim.reporters = []

    # Configure DCD file reporter
    sim.reporters.append(app.DCDReporter(ctx.traj_file, report_freq))

    # Configure contact map reporter
    senders = [ctx.scp_sender] if ctx.scp_sender is not None else []
    sim.reporters.append(
        SparseContactMapReporter(
            ctx.h5_prefix,
            report_freq,
            wrap_pdb_file=ctx.pdb_file if wrap else None,
            reference_pdb_file=ctx.reference_pdb_file,
            senders=senders,
            batch_size=frames_per_h5,
        )
    )

    # Configure simulation output log
    sim.reporters.append(
        app.StateDataReporter(
            ctx.log_file,
            report_freq,
            step=True,
            time=True,
            speed=True,
            potentialEnergy=True,
            temperature=True,
            totalEnergy=True,
        )
    )


class SimulationContext:
    def __init__(
        self,
        reference_pdb_file,
        omm_dir_prefix,
        omm_parent_dir,
        input_dir,
        h5_scp_path,
        result_dir,
        initial_configs_dir,
    ):

        # Index for naming new omm_dirs
        self._file_id = 0
        self._omm_dir_prefix = omm_dir_prefix
        self._omm_parent_dir = omm_parent_dir

        self._initial_configs_dir = Path(initial_configs_dir)
        # Input dir for receiving new PDBs, topologies and halt signal
        self._input_dir = input_dir  # md_runs/input_run0004
        self._result_dir = result_dir

        # Copies /raid/scratch/run0004_0001 to /experiment_dir/md_runs
        self._cp_sender = LocalCopySender(result_dir)

        if h5_scp_path:
            self.scp_sender = RemoteCopySender(h5_scp_path)
        else:
            self.scp_sender = None

        self.pdb_file = None
        self.top_file = None
        self.reference_pdb_file = reference_pdb_file

        # Set fields in H5 file to write
        self._fields = ["contact_map"]
        if self.reference_pdb_file is not None:
            self._fields.append("rmsd")

    @property
    def sim_prefix(self):
        """
        run0004_0001
        """
        return f"{self._omm_dir_prefix}_{self._file_id:06d}"

    @property
    def workdir(self):
        """
        /raid/scratch/run0004_0001
        """
        return os.path.join(self._omm_parent_dir, self.sim_prefix)

    @property
    def h5_prefix(self):
        """
        /raid/scratch/run0004_0001/run0004_0001
        """
        return os.path.join(self.workdir, self.sim_prefix)

    @property
    def traj_file(self):
        return os.path.join(self.workdir, self.sim_prefix + ".dcd")

    @property
    def log_file(self):
        return os.path.join(self.workdir, self.sim_prefix + ".log")

    def _find_in_input_dir(self, pattern):
        match = list(Path(self._input_dir).glob(pattern))
        if match:
            return match[0]
        return None

    def is_new_pdb(self):
        pdb_file = self._find_in_input_dir("*.pdb")
        if pdb_file is None:
            logger.debug(f"No new PDB yet")
            return False

        self.new_context(copy=self.pdb_file is not None, have_new_pdb=True)

        with FileLock(pdb_file):
            self.pdb_file = shutil.move(pdb_file.as_posix(), self.workdir)
        logger.info(f"New PDB file detected; launching new sim: {self.pdb_file}")

        # TODO: this is brittle; be careful!
        system_dir = Path(self.pdb_file).with_suffix("").name.split("__")[1]
        top_file = list(self._initial_configs_dir.joinpath(system_dir).glob("*.top"))
        top_file = top_file[1] if top_file else None
        if top_file is not None:
            with FileLock(top_file):
                self.top_file = shutil.copy(top_file.as_posix(), self.workdir)
        return True

    def copy_context(self):
        """Copy data from node local storage to file system."""
        result_h5 = self.h5_prefix + ".h5"

        logger.debug("Performing H5 concat...")
        concat_h5(self.workdir, result_h5, self._fields)
        logger.debug("H5 concat finished")

        if self.scp_sender is not None:
            self.scp_sender.wait_all()

        cleanup_h5(self.workdir, keep=result_h5)
        self._cp_sender.send(self.workdir, touch_done_file=True)

    def new_context(self, copy=True, have_new_pdb=False):
        # Backup previous context
        if copy:
            self.copy_context()

        # Update run counter
        self._file_id += 1

        # Make new omm directory
        os.makedirs(self.workdir)

        # Copy previous pdb to current workdir if we didn't get a new one
        if not have_new_pdb:
            self.pdb_file = shutil.copy(str(self.pdb_file), self.workdir)

    def halt_signal(self):
        return "halt" in glob.glob(self._input_dir)


def run_simulation(
    reference_pdb_file,
    omm_dir_prefix,
    local_run_dir,
    gpu_index,
    sim_type,
    report_interval_ps,
    frames_per_h5,
    sim_time,
    dt_ps,
    temperature_kelvin,
    h5_scp_path,
    result_dir,
    input_dir,
    initial_configs_dir,
    wrap
):

    # Context to manage files and directory structure
    logger.debug("Creating simulation context")
    ctx = SimulationContext(
        reference_pdb_file=reference_pdb_file,
        omm_dir_prefix=omm_dir_prefix,
        omm_parent_dir=local_run_dir,
        input_dir=input_dir,
        h5_scp_path=h5_scp_path,
        result_dir=result_dir,
        initial_configs_dir=initial_configs_dir,
    )
    logger.debug(f"simulation context created: {ctx.sim_prefix}")

    # Number of steps to run each simulation
    nsteps = int(sim_time / dt_ps)
    logger.info(f"nsteps={nsteps}")

    logger.debug("Blocking until new PDB is received...")
    while not ctx.is_new_pdb():
        time.sleep(5)

    logger.info(f"Received initial PDB: {ctx.pdb_file}")
    logger.debug(f"Configuring new simulation")
    sim = configure_simulation(
        ctx=ctx,
        gpu_index=gpu_index,
        sim_type=sim_type,
        dt_ps=dt_ps,
        temperature_kelvin=temperature_kelvin,
    )

    while not ctx.halt_signal():
        configure_reporters(sim, ctx, report_interval_ps, dt_ps, frames_per_h5, wrap)
        logger.info(f"START sim.step(nsteps={nsteps})")
        sim.step(nsteps)
        logger.info("END sim.step")

        if not ctx.is_new_pdb():
            logger.debug(f"No new PDB: calling new_context() and keeping same traj")
            ctx.new_context()
        else:
            logger.debug(f"Configuring new simulation")
            sim = configure_simulation(
                ctx=ctx,
                gpu_index=gpu_index,
                sim_type=sim_type,
                dt_ps=dt_ps,
                temperature_kelvin=temperature_kelvin,
            )

    # Copy data generated by last simulation to file system
    ctx.copy_context()
