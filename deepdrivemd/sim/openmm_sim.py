import os
import glob
import logging
from pathlib import Path
import shutil
import random

import parmed as pmd
import simtk.unit as u
import simtk.openmm as omm
import simtk.openmm.app as app
from deepdrivemd.util import FileLock
from deepdrivemd.sim.utils import concat_h5, cleanup_h5

from deepdrivemd.sim.openmm_reporter import SparseContactMapReporter
from deepdrivemd.util import CopySender

logger = logging.getLogger(__name__)


def configure_amber_implicit(pdb_file, top_file, dt_ps, platform, platform_properties):

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
    integrator = omm.LangevinIntegrator(300 * u.kelvin, 91.0 / u.picosecond, dt_ps)
    integrator.setConstraintTolerance(0.00001)

    sim = app.Simulation(
        pdb.topology, system, integrator, platform, platform_properties
    )

    # Return simulation and handle to coordinates
    return sim, pdb


def configure_amber_explicit(pdb_file, top_file, dt_ps, platform, platform_properties):

    # Configure system
    top = pmd.load_file(top_file, xyz=pdb_file)
    system = top.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * u.nanometer,
        constraints=app.HBonds,
    )

    # Congfigure integrator
    integrator = omm.LangevinIntegrator(300 * u.kelvin, 1 / u.picosecond, dt_ps)
    system.addForce(omm.MonteCarloBarostat(1 * u.bar, 300 * u.kelvin))

    sim = app.Simulation(
        top.topology, system, integrator, platform, platform_properties
    )

    # Return simulation and handle to coordinates
    return sim, top


def configure_simulation(
    ctx,
    checkpoint_file=None,
    sim_type="implicit",
    gpu_index=0,
    dt_ps=0.002 * u.picoseconds,
    report_interval_ps=10 * u.picoseconds,
):

    # Configure hardware
    try:
        platform = omm.Platform_getPlatformByName("CUDA")
        platform_properties = {"DeviceIndex": str(gpu_index), "CudaPrecision": "mixed"}
    except Exception:
        platform = omm.Platform_getPlatformByName("OpenCL")
        platform_properties = {"DeviceIndex": str(gpu_index)}

    # Select implicit or explicit solvent
    args = ctx.pdb_file, ctx.top_file, dt_ps, platform, platform_properties
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
        # parmed \AA to OpenMM nm
        # TODO: remove this copy? Or do we need it?
        coords.write_pdb("start.pdb", coordinates=positions)

    # Minimize energy and equilibrate
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(300 * u.kelvin, random.randint(1, 10000))

    # Configure reporters
    report_freq = int(report_interval_ps / dt_ps)

    # Configure DCD file reporter
    sim.reporters.append(app.DCDReporter(ctx.traj_file, report_freq))

    # Configure contact map reporter
    sim.reporters.append(
        SparseContactMapReporter(ctx.h5_prefix, report_freq, senders=[ctx.scp_sender])
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

    # Configure simulation checkpoint reporter
    sim.reporters.append(app.CheckpointReporter(ctx.checkpoint_file, report_freq))

    # Optionally, load simulation checkpoint
    if checkpoint_file:
        sim.loadCheckpoint(checkpoint_file)

    return sim


class SimulationContext:
    def __init__(
        self,
        pdb_file,
        reference_pdb_file,
        top_file,
        omm_dir_prefix,
        omm_parent_dir,
        traj_file,
        log_file,
        checkpoint_file,
        input_dir,
        h5_scp_path,
        result_dir,
    ):

        # Index for naming new omm_dirs
        self._file_id = 0
        self._omm_dir_prefix = omm_dir_prefix
        self._omm_parent_dir = omm_parent_dir
        # Input dir for receiving new PDBs, topologies and halt signal
        self._input_dir = input_dir
        self._result_dir = result_dir
        self._cp_sender = CopySender(result_dir, method="cp -r")

        if h5_scp_path:
            self.scp_sender = CopySender(h5_scp_path, method="scp")
        else:
            self.scp_sender = None

        self.pdb_file = pdb_file
        self.top_file = top_file
        self.traj_file = os.path.join(self.workdir, traj_file)
        self.log_file = os.path.join(self.workdir, log_file)
        self.checkpoint_file = os.path.join(self.workdir, checkpoint_file)

        self._new_context(pdb_file, top_file, copy=False)

    @property
    def sim_prefix(self):
        """
        run0004_0001
        """
        return f"{self._omm_dir_prefix}_{self._file_id:04d}"

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

    def _find_in_workdir(self, pattern):
        match = list(Path(self._input_dir).glob(pattern))
        if match:
            return match[0]
        return None

    def is_new_pdb(self):
        pdb_file = self._find_in_workdir("*.pdb")
        if not pdb_file:
            return False

        logger.info(f"New PDB file detected; launching new sim: {self.pdb_file}")
        self._new_context(copy=True)

        with FileLock(pdb_file):
            self.pdb_file = shutil.copy(pdb_file, self.workdir)

        top_file = self._find_in_workdir("*.top")
        if top_file is not None:
            with FileLock(top_file):
                self.top_file = shutil.copy(top_file, self.workdir)
        return True

    def copy_context(self):
        """Copy data from node local storage to file system."""
        result_h5 = self.h5_prefix + ".h5"
        concat_h5(self.workdir, result_h5)
        self.scp_sender.wait_all()
        cleanup_h5(self.workdir, keep=result_h5)
        self._cp_sender.send(self.workdir)

    def _new_context(self, copy=True):
        # Backup previous context
        if copy:
            self.copy_context()

        # Update run counter
        self._file_id += 1

        # Make new omm directory
        os.makedirs(self.workdir)

    def halt_signal(self):
        return "halt" in glob.glob(self._input_dir)


def run_simulation(
    pdb_file,
    reference_pdb_file,
    top_file,
    checkpoint_file,
    omm_dir_prefix,
    local_run_dir,
    output_traj,
    gpu_index,
    sim_type,
    output_log,
    report_interval_ps,
    sim_time,
    reeval_time,
    dt_ps,
    h5_scp_path,
    result_dir,
    input_dir,
):

    # Context to manage files and directory structure
    ctx = SimulationContext(
        pdb_file=pdb_file,
        reference_pdb_file=reference_pdb_file,
        top_file=top_file,
        omm_dir_prefix=omm_dir_prefix,
        omm_parent_dir=local_run_dir,
        traj_file=output_traj,
        log_file=output_log,
        checkpoint_file=checkpoint_file,
        h5_scp_path=h5_scp_path,
        result_dir=result_dir,
        input_dir=input_dir,
    )

    # Number of steps to run each simulation
    nsteps = int(sim_time / dt_ps)
    # Number of times to run each simulation before
    # restarting with different initial conditions
    niter = int(sim_time / reeval_time)

    # If a new PDB arrives before the simulation has run niter
    # times, the new PDB is favored and simulated once the old
    # simulation finishes it's final run.

    # Loop until halt signal is sent
    while not ctx.halt_signal():

        sim = configure_simulation(
            ctx=ctx,
            checkpoint_file=checkpoint_file,
            gpu_index=gpu_index,
            sim_type=sim_type,
            dt_ps=dt_ps,
            report_interval_ps=report_interval_ps,
        )

        for _ in range(niter):
            sim.step(nsteps)

            # If new PDB is found, reconfigure simulation, otherwise
            # continue runing old simulation
            if ctx.is_new_pdb():
                break
            if ctx.halt_signal():
                logger.info("Detected halt signal: breaking")
                break

    # Copy data generated by last simulation to file system
    ctx.copy_context()
