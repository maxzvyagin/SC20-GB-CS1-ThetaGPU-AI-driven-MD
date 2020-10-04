from fabric import Connection
from tempfile import NamedTemporaryFile
import yaml
import threading
import json

STOP_FILENAME="STOP_FILE"

def get_connection(host):
    return Connection(host)


def write_configuration(conn, training_config, theta_experiment_dir):
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


def _launch_cs1_trainer(conn, training_config):
    top_dir = training_config.medulla_experiment_path
    stop_path = top_dir.joinpath(STOP_FILENAME)
    result = conn.run(
        f"export stop_file={stop_path} && "
        f"export global_path={training_config.global_path} && "
        f"export sim_data_dir={training_config.sim_data_dir} && "
        f"cd {top_dir} && nohup bash {training_config.run_script} >& run.log &",
        pty=False,  # no pseudo-terminal
    )

def launch_cs1_trainer(conn, training_config):
    t = threading.Thread(
        target=_launch_cs1_trainer,
        daemon=True,
        args=(conn, training_config),
    )
    t.start()
    return t


def stop_cs1_trainer(conn, training_config, train_thread):
    top_dir = training_config.medulla_experiment_path
    stop_file = top_dir.joinpath(STOP_FILENAME)
    conn.run(f"touch {stop_file.as_posix()}")
    train_thread.join()
