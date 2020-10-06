import itertools
import random
import os
import sys
import shutil
import time
from tempfile import NamedTemporaryFile
from typing import Set, List, Optional, Tuple
from pathlib import Path

from fabric import Connection

from .mpi_launcher import ComputeNodeManager, MPIRun, ComputeNode
from .config import (
    read_yaml_config,
    MDConfig,
    MDRunnerConfig,
    ExperimentConfig,
    OutlierDetectionConfig,
    LoggingConfig,
)
from .cs1_manager import (
    get_connection,
    write_configuration,
    launch_cs1_trainer,
    stop_cs1_trainer,
)

from deepdrivemd import config_logging
import logging

logger = logging.getLogger(__name__)


def launch_md(
    nodes: List[ComputeNode],
    gpu_ids: Set[int],
    config: MDRunnerConfig,
    md_dir: Path,
    omm_dir_prefix: str,
    h5_scp_path: Optional[str],
    pdb_file: Path,
    logging_config: LoggingConfig,
) -> MPIRun:
    """
    Start one instance of OpenMM and return the MPIRun handle
    """
    hostname = nodes[0].id

    input_dir = md_dir.joinpath("input_" + omm_dir_prefix)  # input_run058
    input_dir.mkdir()

    # pdb_file = /nsp10_16/comp_inputs/input_comp_088/comp.pdb
    system_name = pdb_file.parent.name  # system_name --> "input_comp_088"
    basename = pdb_file.with_suffix("").name  # basename --> "comp"
    if "_" in basename:
        raise ValueError("Cannot have underscore in PDB file names!")

    # initial_pdb --> /experiments/md_runs/input_run058/comp__input_comp_088.pdb
    # TODO: this is brittle; be careful!
    # Requires a directory structure with one pdb/top pair per subdirectory
    # And files must not contain a double underscore
    initial_pdb = input_dir.joinpath(basename + "__" + system_name + ".pdb")
    shutil.copy(pdb_file, initial_pdb)
    logger.info(f"Copied initial pdb {pdb_file} to  {initial_pdb}")

    run_config = MDConfig(
        initial_configs_dir=config.initial_configs_dir,
        reference_pdb_file=config.reference_pdb_file,
        sim_type=config.sim_type,
        simulation_length_ns=config.simulation_length_ns,
        report_interval_ps=config.report_interval_ps,
        reeval_time_ns=config.reeval_time_ns,
        frames_per_h5=config.frames_per_h5,
        omm_dir_prefix=omm_dir_prefix,  # like "run058",
        local_run_dir=config.local_run_dir,
        h5_scp_path=h5_scp_path,
        result_dir=md_dir,
        input_dir=input_dir,
        logging=logging_config,
    )

    # Push the YAML over to node-local storage, then start run
    cfg_path = input_dir.joinpath("omm.yaml")
    with open(cfg_path, mode="w") as fp:
        run_config.dump_yaml(fp)
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
) -> Tuple[List[MPIRun], Path]:
    """
    Launch the full set of MD Runs for this experiment
    """
    md_runs = []
    md_dir = config.experiment_directory.joinpath("md_runs")
    md_dir.mkdir()
    MPIRun.set_preamble_commands(*config.md_runner.md_environ_setup)

    pdb_files = list(config.md_runner.initial_configs_dir.glob("*/*.pdb"))
    random.shuffle(pdb_files)

    if config.cs1_training is not None:
        remote_experiment_path = config.cs1_training.medulla_experiment_path
        h5_dest = remote_experiment_path.joinpath("h5_data")
        h5_scp_path = f"{config.cs1_training.hostname}:{h5_dest}"
    else:
        h5_scp_path = None

    for pdb_file, i in zip(
        itertools.cycle(pdb_files), range(config.md_runner.num_jobs)
    ):
        pdb_dir = pdb_file.parent.name
        nodes, gpu_ids = manager.request(num_nodes=1, gpus_per_node=1)
        omm_dir_prefix = f"run{i:04d}"
        run = launch_md(
            nodes,
            gpu_ids,
            config.md_runner,
            md_dir,
            omm_dir_prefix,
            h5_scp_path,
            pdb_file=pdb_file,
            logging_config=config.logging,
        )
        md_runs.append(run)
    return md_runs, md_dir


class CS1Training:
    def __init__(self, config: ExperimentConfig):
        self.config = config.cs1_training
        assert self.config is not None
        conn = get_connection(self.config.hostname)
        write_configuration(conn, self.config, config.experiment_directory)

        num_h5s, rem = divmod(
            self.config.num_frames_per_training, config.md_runner.frames_per_h5
        )
        if rem != 0:
            raise ValueError(
                f"frames_per_h5 {config.md_runner.frames_per_h5} must evenly divide "
                f"num_frames_per_training {self.config.num_frames_per_training}"
            )
        self.train_thread = launch_cs1_trainer(conn, self.config, num_h5s)

    def stop(self):
        conn = get_connection(self.config.hostname)
        assert self.config is not None
        stop_cs1_trainer(conn, self.config, self.train_thread)


def dispatch_od_run(manager, config: ExperimentConfig, md_dir: Path):
    nodes, gpu_ids = manager.request(
        num_nodes=config.outlier_detection.num_nodes,
        gpus_per_node=config.outlier_detection.gpus_per_node,
    )
    outlier_cfg = config.outlier_detection
    outlier_cfg.md_dir = md_dir
    if config.cs1_training:
        outlier_cfg.cvae_dir = config.cs1_training.theta_gpu_path
    outlier_cfg.walltime_min = config.walltime_min

    cfg_path = config.experiment_directory.joinpath("lof.yaml")
    with open(cfg_path, "w") as fp:
        outlier_cfg.dump_yaml(fp)
    # od_run = MPIRun(
    #    config.outlier_detection.run_command + f" -c {cfg_path}",
    #    node_list=nodes,
    #    ranks_per_node=1,
    #    gpu_ids=gpu_ids,
    #    output_file=cfg_path.with_suffix(".out"),
    # )
    od_run = None
    return od_run


class LocalTraining:
    def __init__(self, config: ExperimentConfig):
        self.config = config


def main(config_filename: str):
    start = time.time()
    config = read_yaml_config(config_filename)

    config.experiment_directory.mkdir(
        exist_ok=False  # No duplicate experiment directories!
    )

    log_fname = config.experiment_directory.joinpath("experiment_main.log").as_posix()
    config_logging(filename=log_fname, **config.logging.dict())

    if config.cs1_training is not None:
        cs1_training = CS1Training(config)
        gpu_training = None
    elif config.gpu_training is not None:
        cs1_training = None
        gpu_training = LocalTraining(config)
    else:
        cs1_training = None
        gpu_training = None

    manager = ComputeNodeManager()
    md_runs, md_dir = dispatch_md_runs(manager, config)
    od_run = dispatch_od_run(manager, config, md_dir)

    elapsed_sec = time.time() - start
    remaining_sec = int(max(0, config.walltime_min * 60 - elapsed_sec))

    while remaining_sec > 0:
        min, sec = divmod(remaining_sec, 60)
        logger.info(f"experiment_main: f{min:02d}min:{sec:02d}sec remaining")
        time.sleep(60)
        elapsed_sec = time.time() - start
        remaining_sec = int(max(0, config.walltime_min * 60 - elapsed_sec))

    if cs1_training is not None:
        cs1_training.stop()


def test_log():
    from deepdrivemd.driver.config import LoggingConfig

    log_conf = LoggingConfig().dict()
    config_logging(filename="tester123.log", **log_conf)
    logger.info("This is only a test")


if __name__ == "__main__":
    config_filename = sys.argv[1]
    main(config_filename)
