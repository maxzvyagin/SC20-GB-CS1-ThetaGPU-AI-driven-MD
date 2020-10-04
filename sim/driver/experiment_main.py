from fabric import Connection
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
    md_dir.mkdir()
    MPIRun.set_preamble_commands(*config.md_environ_setup)

    if config.cs1_training is not None:
        remote_experiment_path = config.cs1_training.medulla_experiment_path
        h5_dest = remote_experiment_path.joinpath("h5_data")
        h5_scp_path = f"{config.medulla_ssh_hostname}:{h5_dest}"
    else:
        h5_scp_path = None

    for i in range(config.num_jobs):
        nodes, gpu_ids = manager.request(num_nodes=1, gpus_per_node=1)
        node_hostname = nodes[0].id
        run_base_id = (f"run{i:04d}")
        checkpoint_fname = f"{config.experiment_directory.name}_{i:05d}.chk"
        md_config = MDConfig(
            pdb_file=config.md_runner.pdb_file,
            reference_pdb_file=config.md_runner.reference_pdb_file,
            top_file=config.md_runner.top_file,
            checkpoint_file=checkpoint_fname,
            simulation_length_ns=config.md_runner.simulation_length_ns,
            report_interval_ps=config.md_runner.report_interval_ps,
            reeval_time_ns=config.md_runner.reeval_time_ns,
            run_base_id=run_base_id,  # like "run058",
            local_run_dir=config.md_runner.local_run_dir,
            h5_scp_path=h5_scp_path,
            result_dir=md_dir,
        )
        with NamedTemporaryFile(prefix=run_base_id) as fp:
            md_config.dump_yaml(fp)
            fp.flush()
            Connection(node_hostname).put(fp.name, md_config.local_run_dir)
            cfg_path = md_config.local_run_dir.joinpath(fp.name)
            MPIRun(
                config.md_run_command + f" -c {cfg_path}",
                node_list=nodes,
                ranks_per_node=1,
                gpu_ids=gpus,
                output_file=md_dir.joinpath(run_base_id + ".out"),
            )

    return md_runs


def main(config_filename):
    config = read_yaml_config(config_filename)

    config.experiment_directory.mkdir(
        exist_ok=False  # No duplicate experiment directories!
    )

    conn = get_connection(config.cs1_training.medulla_ssh_hostname)
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
