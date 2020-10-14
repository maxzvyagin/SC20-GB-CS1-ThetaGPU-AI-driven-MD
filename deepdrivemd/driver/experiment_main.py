import itertools
import random
import sys
import shutil
import time
from typing import Set, List, Optional, Tuple
from pathlib import Path

from .mpi_launcher import ComputeNodeManager, MPIRun, ComputeNode
from .config import (
    MDConfig,
    MDRunnerConfig,
    ExperimentConfig,
    OutlierDetectionUserConfig,
    OutlierDetectionRunConfig,
    LoggingConfig,
    GPUTrainingUserConfig,
    GPUTrainingRunConfig,
    CVAEModelConfig,
)
from .cs1_manager import CS1Training

from deepdrivemd import config_logging
import logging

logger = None


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
    input_dir = md_dir.joinpath("input_" + omm_dir_prefix)  # input_run058
    input_dir.mkdir()

    # pdb_file = /nsp10_16/comp_inputs/input_comp_088/comp.pdb
    assert "__" not in pdb_file.as_posix()
    system_name = pdb_file.parent.name  # system_name --> "input_comp_088"
    basename = pdb_file.with_suffix("").name  # basename --> "comp"
    if "_" in basename:
        raise ValueError("Cannot have underscore in PDB file names!")

    # initial_pdb --> /experiments/md_runs/input_run058/comp__input_comp_088.pdb
    # TODO: this is brittle; be careful!
    # Requires a directory structure with one pdb/top pair per subdirectory
    # And files must not contain a double underscore

    initial_pdb = input_dir.joinpath(basename + "__" + system_name + ".pdb")
    dest = shutil.copy(pdb_file, initial_pdb)
    logger.info(f"Copied initial pdb: {dest}")

    run_config = MDConfig(
        initial_configs_dir=config.initial_configs_dir,
        reference_pdb_file=config.reference_pdb_file,
        sim_type=config.sim_type,
        temperature_kelvin=config.temperature_kelvin,
        simulation_length_ns=config.simulation_length_ns,
        report_interval_ps=config.report_interval_ps,
        frames_per_h5=config.frames_per_h5,
        omm_dir_prefix=omm_dir_prefix,  # like "run058",
        local_run_dir=config.local_run_dir,
        h5_scp_path=h5_scp_path,
        result_dir=md_dir,
        input_dir=input_dir,
        logging=logging_config,
        wrap=config.wrap,
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
    md_dir: Path,
    manager: ComputeNodeManager,
    config: ExperimentConfig,
    h5_scp_path: Optional[str],
) -> Tuple[List[MPIRun], Path]:
    """
    Launch the full set of MD Runs for this experiment
    """
    md_runs = []
    MPIRun.set_preamble_commands(*config.md_runner.md_environ_setup)

    pdb_files = list(config.md_runner.initial_configs_dir.glob("*/*.pdb"))
    random.shuffle(pdb_files)

    for pdb_file, i in zip(
        itertools.cycle(pdb_files), range(config.md_runner.num_jobs)
    ):
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


def dispatch_od_run(
    manager,
    user_config: OutlierDetectionUserConfig,
    md_dir: Path,
    outlier_predict_batch_size: int,
    logging_cfg,
    model_params,
    model_weights_dir,
    walltime_min,
    experiment_dir,
):
    nodes, gpu_ids = manager.request(
        num_nodes=user_config.num_nodes,
        gpus_per_node=user_config.gpus_per_node,
    )
    outlier_results_dir = experiment_dir.joinpath("outlier_runs")
    outlier_results_dir.mkdir()
    outlier_cfg = OutlierDetectionRunConfig(
        logging=logging_cfg,
        model_params=model_params,
        md_dir=md_dir,
        model_weights_dir=model_weights_dir,
        walltime_min=walltime_min,
        outlier_predict_batch_size=outlier_predict_batch_size,
        outlier_results_dir=outlier_results_dir,
        **user_config.dict(),
    )

    cfg_path = experiment_dir.joinpath("lof.yaml")
    with open(cfg_path, "w") as fp:
        outlier_cfg.dump_yaml(fp)
    MPIRun.set_preamble_commands(*user_config.environ_setup)
    od_run = MPIRun(
        outlier_cfg.run_command + f" -c {cfg_path}",
        node_list=nodes,
        ranks_per_node=1,
        gpu_ids=gpu_ids,
        output_file=cfg_path.with_suffix(".out"),
    )
    return od_run


def dispatch_gpu_training(
    manager,
    experiment_dir: Path,
    user_config: GPUTrainingUserConfig,
    model_config: CVAEModelConfig,
    model_weights_dir: Path,
    frames_per_h5: int,
    logging_cfg,
):
    top_dir = experiment_dir.joinpath("gpu_training")
    top_dir.mkdir(exist_ok=True)
    logger.info(f"Created gpu train dir: {top_dir}")

    num_h5s, rem = divmod(user_config.num_frames_per_training, frames_per_h5)
    if rem != 0:
        raise ValueError(
            f"frames_per_h5 {frames_per_h5} must evenly divide "
            f"num_frames_per_training {user_config.num_frames_per_training}"
        )

    config = GPUTrainingRunConfig(
        **user_config.dict(),
        **model_config.dict(),
        sim_data_dir=top_dir.joinpath("h5_data"),
        data_dir=top_dir.joinpath("records_loop"),
        eval_data_dir=top_dir.joinpath("eval_records_loop"),
        global_path=top_dir.joinpath("files_seen.txt"),
        model_dir=top_dir.joinpath("model_dir"),
        checkpoint_path=model_weights_dir,
        logging=logging_cfg,
        num_h5s_per_training=num_h5s,
    )
    cfg_path = top_dir.joinpath("config.yaml")
    with open(cfg_path, "w") as fp:
        config.dump_yaml(fp)

    nodes, gpu_ids = manager.request(
        num_nodes=user_config.num_nodes,
        gpus_per_node=user_config.gpus_per_node,
    )
    MPIRun.set_preamble_commands(*user_config.environ_setup)
    train_run = MPIRun(
        user_config.run_command + f" -c {fp.name}",
        node_list=nodes,
        ranks_per_node=1,
        gpu_ids=gpu_ids,
        output_file=cfg_path.with_suffix(".out"),
    )
    return train_run


def main(config_filename: str):
    global logger
    start = time.time()
    config = ExperimentConfig.from_yaml(config_filename)

    config.experiment_directory.mkdir(exist_ok=False)
    md_dir = config.experiment_directory.joinpath("md_runs")
    model_weights_dir = config.experiment_directory.joinpath("model_weights")
    model_weights_dir.mkdir()
    md_dir.mkdir()

    log_fname = config.experiment_directory.joinpath("experiment_main.log").as_posix()
    config_logging(filename=log_fname, **config.logging.dict())
    logger = logging.getLogger("deepdrivemd.driver.experiment_main")
    manager = ComputeNodeManager()

    if config.cs1_training is not None:
        logger.info("Dispatching CS1 Training")
        cs1_training = CS1Training(
            config.cs1_training,
            config.cvae_model,
            model_weights_dir,
            config.md_runner.frames_per_h5,
        )
        gpu_training = None
        remote_experiment_path = config.cs1_training.medulla_experiment_path
        h5_dest = remote_experiment_path.joinpath("h5_data")
        h5_scp_path = f"{config.cs1_training.hostname}:{h5_dest}"
    elif config.gpu_training is not None:
        logger.info("Dispatching GPU Training")
        cs1_training = None
        gpu_training = dispatch_gpu_training(
            manager=manager,
            user_config=config.gpu_training,
            experiment_dir=config.experiment_directory,
            model_config=config.cvae_model,
            model_weights_dir=model_weights_dir,
            frames_per_h5=config.md_runner.frames_per_h5,
            logging_cfg=config.logging,
        )
        h5_scp_path = (
            config.experiment_directory.joinpath("gpu_training")
            .joinpath("h5_data")
            .as_posix()
        )
    else:
        logger.info("Training is disabled for this run")
        cs1_training = None
        gpu_training = None
        h5_scp_path = None

    md_runs, md_dir = dispatch_md_runs(md_dir, manager, config, h5_scp_path)
    od_run = dispatch_od_run(
        manager=manager,
        user_config=config.outlier_detection,
        md_dir=md_dir,
        logging_cfg=config.logging,
        model_params=config.cvae_model,
        model_weights_dir=model_weights_dir,
        walltime_min=config.walltime_min,
        experiment_dir=config.experiment_directory,
        outlier_predict_batch_size=config.md_runner.frames_per_h5,
    )

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
    if gpu_training is not None:
        gpu_training.process.terminate()
        time.sleep(10)
        gpu_training.poll()
    od_run.process.terminate()


if __name__ == "__main__":
    config_filename = sys.argv[1]
    main(config_filename)
