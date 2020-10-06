from .config import CS1TrainingConfig
from fabric import Connection
from pathlib import Path
from tempfile import NamedTemporaryFile
import yaml
import threading
import json
import logging

STOP_FILENAME = "STOP_FILE"

logger = logging.getLogger(__name__)


def get_connection(host: str):
    logger.info(f"Creating fabric.Connection to: {host}")
    return Connection(host)


def write_configuration(
    conn: Connection, training_config: CS1TrainingConfig, theta_experiment_dir: Path
):
    """
    Sets up the directories and params.yaml file for the current experiment on Medulla
    Args:
        conn: fabric.Connection object
        training_config: CS1TrainingConfig object from parsed YAML
        theta_experiment_dir: ThetaGPU experiment directory
    """
    top_dir = training_config.medulla_experiment_path

    if not top_dir.is_absolute():
        raise ValueError("medulla_experiment_path must be absolute")

    training_config.sim_data_dir = top_dir.joinpath("h5_data")
    training_config.data_dir = top_dir.joinpath("records_loop")
    training_config.eval_data_dir = top_dir.joinpath("eval_records_loop")
    training_config.global_path = top_dir.joinpath("files_seen.txt")
    training_config.model_dir = top_dir.joinpath("model_dir")

    logger.info(f"Creating {top_dir} on {conn.host}")
    conn.run(f"mkdir -p {top_dir}")
    conn.run(f"mkdir -p {training_config.sim_data_dir}")
    conn.run(f"mkdir -p {training_config.data_dir}")
    conn.run(f"mkdir -p {training_config.eval_data_dir}")
    conn.run(f"mkdir -p {training_config.model_dir}")
    conn.run(f"touch  {training_config.global_path}")

    training_config.theta_gpu_path = theta_experiment_dir.joinpath("cvae_weights")
    training_config.theta_gpu_path.mkdir(exist_ok=True)

    with NamedTemporaryFile(mode="w", delete=False) as fp:
        yaml.dump(json.loads(training_config.json()), fp)
        fp.flush()
        conn.put(fp.name, top_dir.joinpath("params.yaml").as_posix())


def _launch_cs1_trainer(
    conn: Connection, training_config: CS1TrainingConfig, num_h5_per_run: int
):
    top_dir = training_config.medulla_experiment_path
    stop_path = top_dir.joinpath(STOP_FILENAME)
    result = conn.run(
        f"export stop_file={stop_path} && "
        f"export global_path={training_config.global_path} && "
        f"export sim_data_dir={training_config.sim_data_dir} && "
        f"export file_counter={num_h5_per_run} && "
        f"cd {top_dir} && nohup bash {training_config.run_script} >& run.log &",
        pty=False,  # no pseudo-terminal
    )


def launch_cs1_trainer(
    conn: Connection, training_config: CS1TrainingConfig, num_h5_per_run: int
) -> threading.Thread:
    """
    Returns a thread for the remotely-executed CS1 Training
    """
    t = threading.Thread(
        target=_launch_cs1_trainer,
        daemon=True,
        args=(conn, training_config, num_h5_per_run),
    )
    t.start()
    return t


def stop_cs1_trainer(
    conn: Connection, training_config: CS1TrainingConfig, train_thread: threading.Thread
):
    """
    Puts a STOP file in the CS1 training directory to signal end of training
    """
    top_dir = training_config.medulla_experiment_path
    stop_file = top_dir.joinpath(STOP_FILENAME)
    logger.info("Sending stop file to CS1")
    conn.run(f"touch {stop_file.as_posix()}")
    logger.info("Joining cs1 SSH-managing thread")
    train_thread.join()
    logger.info("Join done")
