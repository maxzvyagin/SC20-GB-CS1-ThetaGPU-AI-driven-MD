from .config import CS1TrainingRunConfig, CS1TrainingUserConfig, CVAEModelConfig
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


def write_configuration(conn: Connection, training_config: CS1TrainingRunConfig):
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

    logger.info(f"Creating {top_dir} on {conn.host}")
    conn.run(f"mkdir -p {top_dir}")
    conn.run(f"mkdir -p {training_config.sim_data_dir}")
    conn.run(f"mkdir -p {training_config.data_dir}")
    conn.run(f"mkdir -p {training_config.eval_data_dir}")
    conn.run(f"mkdir -p {training_config.model_dir}")
    conn.run(f"touch  {training_config.global_path}")

    with NamedTemporaryFile(mode="w", delete=False) as fp:
        yaml.dump(json.loads(training_config.json()), fp)
        fp.flush()
        conn.put(fp.name, top_dir.joinpath("params.yaml").as_posix())


def _launch_cs1_trainer(
    conn: Connection, training_config: CS1TrainingRunConfig, num_h5_per_run: int
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
    conn: Connection, training_config: CS1TrainingRunConfig, num_h5_per_run: int
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
    conn: Connection,
    training_config: CS1TrainingRunConfig,
    train_thread: threading.Thread,
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


class CS1Training:
    def __init__(
        self,
        user_config: CS1TrainingUserConfig,
        model_config: CVAEModelConfig,
        cvae_weights_dir: Path,
        frames_per_h5: int,
    ):
        top_dir = user_config.medulla_experiment_path

        self.config = CS1TrainingRunConfig(
            **user_config.dict(),
            **model_config.dict(),
            sim_data_dir=top_dir.joinpath("h5_data"),
            data_dir=top_dir.joinpath("records_loop"),
            eval_data_dir=top_dir.joinpath("eval_records_loop"),
            global_path=top_dir.joinpath("files_seen.txt"),
            model_dir=top_dir.joinpath("model_dir"),
            theta_gpu_path=cvae_weights_dir,
        )

        conn = get_connection(user_config.hostname)
        write_configuration(conn, self.config)

        num_h5s, rem = divmod(self.config.num_frames_per_training, frames_per_h5)
        if rem != 0:
            raise ValueError(
                f"frames_per_h5 {frames_per_h5} must evenly divide "
                f"num_frames_per_training {self.config.num_frames_per_training}"
            )
        self.train_thread = launch_cs1_trainer(conn, self.config, num_h5s)

    def stop(self):
        conn = get_connection(self.config.hostname)
        assert self.config is not None
        stop_cs1_trainer(conn, self.config, self.train_thread)