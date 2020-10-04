import os
import glob
import shutil
import random

import parmed as pmd
import simtk.unit as u
import simtk.openmm as omm
import simtk.openmm.app as app

from MD_utils.utils import create_md_path
from MD_utils.openmm_reporter import SparseContactMapReporter

def configure_amber_implicit(pdb_file, top_file, dt, platform, platform_properties):

    # Configure system
    if top_file:
        pdb = pmd.load_file(top_file, xyz=pdb_file)
        system = pdb.createSystem(
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * u.nanometer,
            constraints=app.HBonds,
            implicitSolvent=app.OBC1
        )
    else:
        pdb = pmd.load_file(pdb_file)
        forcefield = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=1. * u.nanometer,
            constraints=app.HBonds
        )

    # Congfigure integrator
    integrator = omm.LangevinIntegrator(300 * u.kelvin, 91.0 / u.picosecond, dt)
    integrator.setConstraintTolerance(0.00001)

    sim = app.Simulation(
        pdb.topology,
        system,
        integrator,
        platform,
        platform_properties
    )

    # Return simulation and handle to coordinates
    return sim, pdb

    
def configure_amber_explicit(pdb_file, top_file, dt, platform, platform_properties):
    
    # Configure system
    top = pmd.load_file(top_file, xyz=pdb_file)
    system = top.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=1. * u.nanometer,
        constraints=app.HBonds
    )

    # Congfigure integrator
    integrator = omm.LangevinIntegrator(300 * u.kelvin, 1 / u.picosecond, dt)
    system.addForce(omm.MonteCarloBarostat(1 * u.bar, 300 * u.kelvin))

    sim = app.Simulation(
        top.topology,
        system,
        integrator,
        platform,
        platform_properties
    )

    # Return simulation and handle to coordinates
    return sim, top


def configure_simulation(
    ctx,
    check_point=None,
    sim_type='implicit',
    gpu_index=0,
    dt = 0.002 * u.picoseconds,
    report_time=10 * u.picoseconds,
    senders=[]
    ):

    # Configure hardware
    try:
        platform = omm.Platform_getPlatformByName("CUDA")
        platform_properties = {'DeviceIndex': str(gpu_index), 'CudaPrecision': 'mixed'}
    except Exception:
        platform = omm.Platform_getPlatformByName("OpenCL")
        platform_properties = {'DeviceIndex': str(gpu_index)}

    # Select implicit or explicit solvent
    args = ctx.pdb_file, ctx.top_file, dt, platform, platform_properties
    if sim_type == 'implicit':
        sim, coords = configure_amber_implicit(*args)
    elif sim_type == 'explicit':
        sim, coords = configure_amber_explicit(*args)

    # Set simulation positions
    if coords.get_coordinates().shape[0] == 1:
        sim.context.setPositions(coords.positions)
    else:
        positions = random.choice(coords.get_coordinates())
        sim.context.setPositions(positions / 10)
        # parmed \AA to OpenMM nm
        # TODO: remove this copy? Or do we need it?
        coords.write_pdb('start.pdb', coordinates=positions)

    # Minimize energy and equilibrate
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(
        300 * u.kelvin,
        random.randint(1, 10000)
    )

    # Configure reporters
    report_freq = int(report_time / dt)

    # Configure DCD file reporter
    sim.reporters.append(
        app.DCDReporter(
            ctx.traj_file,
            report_freq
        )
    )

    # Configure contact map reporter
    sim.reporters.append(
        SparseContactMapReporter(
            ctx.h5_prefix,
            report_freq,
            senders=senders
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
            totalEnergy=True
        )
    )

    # Configure simulation checkpoint reporter
    sim.reporters.append(
        app.CheckpointReporter(
            ctx.checkpoint_file,
            report_freq
        )
    )

    # Optionally, load simulation checkpoint
    if check_point:
        sim.loadCheckpoint(check_point)

    return sim


class SimulationContext:
    def __init__(
        self,
        pdb_file,
        top_file=None,
        omm_prefix='run001',
        omm_dir='/raid/scratch',
        traj_file='output.dcd',
        log_file='output.log',
        checkpoint_file='checkpnt.chk',
        h5_prefix = 'output_cm',
        input_dir='/raid/scratch/input',
        sender=None
    ):

        # Index for naming new omm_dirs
        self._file_id = 0
        self._omm_prefix = omm_prefix
        self._omm_dir = omm_dir
        # Input dir for receiving new PDBs, topologies and halt signal
        self._input_dir = input_dir
        self._sender = sender

        self.pdb_file = pdb_file
        self.top_file = top_file
        self.traj_file = os.path.join(self.omm_dir, traj_file)
        self.log_file = os.path.join(self.omm_dir, log_file)
        self.checkpoint_file = os.path.join(self.omm_dir, checkpoint_file)
        self.h5_prefix = os.path.join(self.omm_dir, h5_prefix)

        self.new_context(pdb_file, top_file, copy=False)

    @property
    def omm_dir(self):
        sim_dir = f'{self._omm_prefix}_{self._file_id}'
        return os.path.join(self._omm_dir, sim_dir)

    def copy_context(self):
        """Copy data from node local storage to file system."""
        if self._sender is not None:
            pass # TODO: implement, call concat_h5 in utils and copy all data to lustre

    def new_context(self, pdb_file, top_file, copy=True):
        # Backup previous context
        if copy:
            self.copy_context()
        
        # Update run counter
        self._file_id += 1

        # Make new omm directory
        os.makedirs(self.omm_dir)

        # Copy PDB and topology file to omm directory
        self.pdb_file = shutil.copy(pdb_file, self.omm_dir)
        if top_file is not None:
            self.top_file = shutil.copy(top_file, self.omm_dir)

    def halt_signal(self):
        return 'halt' in glob.glob(self._input_dir)

    def new_pdb(self):
        # TODO: not finished
        return glob.glob(self._input_dir)


def run_simulation(
    pdb_file,
    top_file,
    dt = 0.002 * u.picoseconds,
    sim_time=10 * u.nanoseconds,
    reeval_time=None,
    omm_path="/raid/scratch",
    **kwargs
):

    # Context to manage files and directory structure
    ctx = SimulationContext(pdb_file, top_file)

    # Number of steps to run each simulation
    nsteps = int(sim_time / dt)
    # Number of times to run each simulation before
    # restarting with different initial conditions
    niter = int(sim_time / reeval_time)

    # If a new PDB arrives before the simulation has run niter
    # times, the new PDB is favored and simulated once the old
    # simulation finishes it's final run.

    # Loop until halt signal is sent
    while not ctx.halt_signal():
    
        sim = configure_simulation(ctx, dt=dt, **kwargs)

        for _ in range(niter):
            sim.step(nsteps)

            # If new PDB is found, reconfigure simulation, otherwise 
            # continue runing old simulation
            if os.path.exists('new.pdb'):
                print("Found new.pdb, starting new sim...")
                pdb_file = glob.glob(os.path.join(omm_path, '*.pdb'))[0]
                # Initialize new directories and paths
                ctx.new_context(pdb_file)
                break
            elif ctx.halt_signal():
                break

    # Copy data generated by last simulation to file system
    ctx.copy_context()
    