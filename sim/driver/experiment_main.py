import sys
from typing import Optional
from pathlib import Path
from mpi_launcher import (
    ComputeNodeManager,
    MPIRun,
)
from config import read_yaml_config, MDConfig

def start_md_run(workdir, md_config):

def main(config_filename):
    config = read_yaml_config(config_filename)
    manager = ComputeNodeManager()

    if config.outlier_detection.num_jobs is None:
        config.outlier_detection.num_jobs = 1

    if config.md_runner.num_jobs is None:
        config.md_runner.num_jobs = manager.num_nodes

if __name__ == "__main__":
    main(sys.argv[1])
