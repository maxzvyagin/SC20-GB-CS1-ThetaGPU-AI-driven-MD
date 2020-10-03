import os 
import argparse 
import simtk.unit as u
from MD_utils.openmm_reporter import CopySender 
from MD_utils.openmm_simulation import openmm_simulate_amber_implicit

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='YAML config file')
args = parser.parse_args()

cfg = args.config['md_config']

# Handle optional parameters
if not cfg['top_file']:
    cfg['top_file'] = None

if not cfg['checkpoint_file']:
    cfg['checkpoint_file'] =  None

# 'medulla1.cels.anl.gov:/data/shared/vishal/new_dataV2/'

# Configure senders for copying data to local and remote
senders = []
if cfg['scp_path']:
    # Send HDF5 files to medulla
    # Requires user to execute "export MEDULLA_IDENTITY_FILE=~/.ssh/my-identity-file"
    # identity_file = os.environ['MEDULLA_IDENTITY_FILE']
    # method = f'scp -i {identity_file}'
    method = 'scp'
    senders.append(CopySender(cfg['scp_path'], method=method))

if cfg['cp_path']:
    # Send HDF5 files to any local path
    senders.append(CopySender(cfg['cp_path'], method='cp'))

# check_point = None
openmm_simulate_amber_implicit(
        cfg['pdb_file'],
        top_file=cfg['top_file'],
        check_point=cfg['check_point'],
        GPU_index=0,
        output_traj="output.dcd",
        output_log="output.log",
        output_cm="output_cm",
        report_time=float(cfg['report_interval']) * u.picoseconds,
        sim_time=float(cfg['simulation_length']) * u.nanoseconds, 
        reeval_time=float(cfg['reeval_time']) * u.nanoseconds,
        senders=senders)
