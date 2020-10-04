import sys
from typing import Optional
from pathlib import Path
from mpi_launcher import (
    ComputeNodeManager,
    MPIRun,
)
import time
from config import read_yaml_config, MDConfig
from cs1_manager import get_connection, write_configuration, launch_cs1_trainer, stop_cs1_trainer


def start_md_run(workdir, md_config):
    workdir.mkdir(exist_ok=True)
    with NamedTemporaryFile(
        dir=workdir,
        delete=False,
        mode="w",
        prefix="",
        suffix=".yml"
    ) as fp:
        md_config.dump_yaml(fp)


def dispatch_md_runs(manager, config):
    md_runs = []

    md_dir = config.experiment_directory.joinpath("md_runs")
    local_h5_dir = md_dir.joinpath("h5_contact_maps")
    md_dir.mkdir()
    local_h5_dir.mkdir()

    if config.cs1_training is not None:
        scp_dir

    for i in range(config.num_jobs):
        workdir = md_dir.joinpath(f"{i:03d}")
        checkpoint_fname = f"{config.experiment_directory.name}_{i:03d}.chk"
        md_config = MDConfig(
            pdb_file=config.md_runner.pdb_file,
            reference_pdb_file=config.md_runner.reference_pdb_file,
            top_file=config.md_runner.top_file,
            checkpoint_file=checkpoint_fname,
            simulation_length_ns=config.md_runner.simulation_length_ns,
            report_interval_ps=config.md_runner.report_interval_ps,
            h5_cp_path=local_h5_dir,
        )
    return md_runs


def main(config_filename):
    config = read_yaml_config(config_filename)

    config.experiment_directory.mkdir(
        exist_ok=False  # No duplicate experiment directories!
    )

    conn = get_connection("medulla1")
    write_configuration(conn, config.cs1_training, config.experiment_directory)
    train_thread = launch_cs1_trainer(conn, config.cs1_training)
    print("sleeping for a while")
    time.sleep(30)
    print("sending STOP_FILE and joining on thread...")
    stop_cs1_trainer(conn, config.cs1_training, train_thread)
    print("join done!")
    sys.exit(0)
    manager = ComputeNodeManager()
    #md_runs = dispatch_md_runs(manager, config)

    if config.cs1_training is not None:
        dispatch_cs1_trainer(config.cs1_training)
    elif config.gpu_training is not None:
        dispatch_gpu_trainer(config.gpu_training)

    if config.outlier_detection.num_jobs is None:
        config.outlier_detection.num_jobs = 1

    if config.md_runner.num_jobs is None:
        config.md_runner.num_jobs = manager.num_nodes


if __name__ == "__main__":
    config_filename = sys.argv[1]
    main(config_filename)
