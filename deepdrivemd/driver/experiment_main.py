import sys
import time
from tempfile import NamedTemporaryFile
from typing import Set, List, Optional
from pathlib import Path

from fabric import Connection

from .mpi_launcher import ComputeNodeManager, MPIRun, ComputeNode
from .config import read_yaml_config, MDConfig, MDRunnerConfig, ExperimentConfig
from .cs1_manager import (
    get_connection,
    write_configuration,
    launch_cs1_trainer,
    stop_cs1_trainer,
)


def launch_md(
    nodes: List[ComputeNode],
    gpu_ids: Set[int],
    config: MDRunnerConfig,
    md_dir: Path,
    omm_dir_prefix: str,
    h5_scp_path: Optional[str],
) -> MPIRun:
    """
    Start one instance of OpenMM and return the MPIRun handle
    """
    hostname = nodes[0].id
    checkpoint_fname = md_dir.joinpath(omm_dir_prefix + ".chk")

    input_dir = md_dir.joinpath("input_" + omm_dir_prefix)  # input_run058
    input_dir.mkdir()

    run_config = MDConfig(
        pdb_file=config.pdb_file,
        reference_pdb_file=config.reference_pdb_file,
        top_file=config.top_file,
        sim_type=config.sim_type,
        checkpoint_file=checkpoint_fname,
        simulation_length_ns=config.simulation_length_ns,
        report_interval_ps=config.report_interval_ps,
        reeval_time_ns=config.reeval_time_ns,
        omm_dir_prefix=omm_dir_prefix,  # like "run058",
        local_run_dir=config.local_run_dir,
        h5_scp_path=h5_scp_path,
        result_dir=md_dir,
        input_dir=input_dir,
    )

    # Push the YAML over to node-local storage, then start run
    with NamedTemporaryFile(prefix=omm_dir_prefix) as fp:
        run_config.dump_yaml(fp)
        fp.flush()
        Connection(hostname).put(fp.name, run_config.local_run_dir)
        cfg_path = run_config.local_run_dir.joinpath(fp.name)
        run = MPIRun(
            config.md_run_command + f" -c {cfg_path}",
            node_list=nodes,
            ranks_per_node=1,
            gpu_ids=gpu_ids,
            output_file=md_dir.joinpath(omm_dir_prefix + ".out"),
        )
    return run


def dispatch_md_runs(
    manager: ComputeNodeManager, config: ExperimentConfig
) -> List[MPIRun]:
    """
    Launch the full set of MD Runs for this experiment
    """
    md_runs = []
    md_dir = config.experiment_directory.joinpath("md_runs")
    md_dir.mkdir()
    MPIRun.set_preamble_commands(*config.md_runner.md_environ_setup)

    if config.cs1_training is not None:
        remote_experiment_path = config.cs1_training.medulla_experiment_path
        h5_dest = remote_experiment_path.joinpath("h5_data")
        h5_scp_path = f"{config.cs1_training.hostname}:{h5_dest}"
    else:
        h5_scp_path = None

    for i in range(config.md_runner.num_jobs):
        nodes, gpu_ids = manager.request(num_nodes=1, gpus_per_node=1)
        omm_dir_prefix = f"run{i:04d}"
        run = launch_md(
            nodes,
            gpu_ids,
            config.md_runner,
            md_dir,
            omm_dir_prefix,
            h5_scp_path,
        )
        md_runs.append(run)
    return md_runs


def main(config_filename: str):
    config = read_yaml_config(config_filename)

    config.experiment_directory.mkdir(
        exist_ok=False  # No duplicate experiment directories!
    )

    if config.cs1_training is not None:
        conn = get_connection(config.cs1_training.hostname)
        write_configuration(conn, config.cs1_training, config.experiment_directory)
        train_thread = launch_cs1_trainer(conn, config.cs1_training)
        print("sleeping for a while")
        time.sleep(30)
        print("sending STOP_FILE and joining on thread...")
        stop_cs1_trainer(conn, config.cs1_training, train_thread)
        print("join done!")

    manager = ComputeNodeManager()
    md_runs = dispatch_md_runs(manager, config)
    while md_runs:
        time.sleep(5)
        md_runs = [run for run in md_runs if run.poll() is None]


if __name__ == "__main__":
    config_filename = sys.argv[1]
    main(config_filename)
