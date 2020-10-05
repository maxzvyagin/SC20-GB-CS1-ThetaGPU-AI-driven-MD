# Schema of the YAML experiment file

import json
import yaml
from enum import Enum
from pydantic import BaseSettings
from pathlib import Path
from typing import Optional, List, Dict


class MDType(str, Enum):
    implicit = "implicit"
    explicit = "explicit"


class LoggingConfig(BaseSettings):
    level: str = "DEBUG"
    format: str = (
        "'%(asctime)s|%(process)d|%(levelname)8s|%(name)s:%(lineno)s] %(message)s'"
    )
    datefmt: str = "%d-%b-%Y %H:%M:%S"
    buffer_num_records: int = 10
    flush_period: int = 30


class MDConfig(BaseSettings):
    """
    Auto-generates configuration file for run_openmm.py
    """

    pdb_file: Path
    reference_pdb_file: Path
    top_file: Optional[Path]
    local_run_dir: Path
    sim_type: MDType
    simulation_length_ns: int = 10
    report_interval_ps: int = 50
    dt_ps: float = 0.002
    # Length of each simulation in nanoseconds if recursive mode is active
    reeval_time_ns: int = 10
    result_dir: Path
    h5_scp_path: Optional[str]
    omm_dir_prefix: str
    local_run_dir: Path = Path("/raid/scratch")
    input_dir: Path
    logging: LoggingConfig

    def dump_yaml(self, file):
        yaml.dump(json.loads(self.json()), file)


class MDRunnerConfig(BaseSettings):
    """
    Global MD configuration (written one per experiment)
    """

    num_jobs: int
    pdb_file: Path
    reference_pdb_file: Path
    top_file: Optional[Path]
    sim_type: MDType
    simulation_length_ns: int = 10
    report_interval_ps: int = 50
    # Length of each simulation in nanoseconds if recursive mode is active
    reeval_time_ns: int = 10
    local_run_dir: Path = Path("/raid/scratch")
    md_run_command: str = "python run_openmm.py"
    md_environ_setup: List[str] = [
        'eval "$(/lus/theta-fs0/projects/RL-fold/msalim/miniconda3/bin/conda shell.bash hook)"',
        "conda activate /lus/theta-fs0/projects/RL-fold/venkatv/software/conda_env/a100_rapids_openmm",
    ]


class CVAEModelConfig(BaseSettings):
    fraction: float = 0.2
    last_n_files: int = 1
    last_n_files_eval: int = 1
    batch_size: int = 1
    input_shape: List[int] = [1, 32, 32]
    itemsize: int = 1
    mixed_precision: bool = True
    h5_shape: List[int] = [1, 36, 36]
    tfrecord_shape: List[int] = [1, 36, 36]
    samples_per_file: int = 1

    # Model params
    enc_conv_kernels: List[int] = [5, 5, 5, 5]
    # Encoder filters define OUTPUT filters per layer
    enc_conv_filters: List[int] = [100, 100, 100, 100]  # 64, 64, 64, 32
    enc_conv_strides: List[int] = [1, 1, 2, 1]
    dec_conv_kernels: List[int] = [5, 5, 5, 5]
    # Decoder filters define INPUT filters per layer
    dec_conv_filters: List[int] = [100, 100, 100, 100]
    dec_conv_strides: List[int] = [1, 2, 1, 1]
    dense_units: int = 64  # 128
    latent_ndim: int = 10  # 3
    mixed_precision: bool = True
    activation: str = "relu"
    # Setting full_precision_loss to False as we do not support it yet.
    full_precision_loss: bool = False
    reconstruction_loss_reduction_type: str = "sum"
    kl_loss_reduction_type: str = "sum"
    model_random_seed: Optional[int] = None
    data_random_seed: Optional[int] = None

    # Optimizer params
    epsilon: float = 1.0e-8
    beta1: float = 0.2
    beta2: float = 0.9
    decay: float = 0.9
    momentum: float = 0.9
    optimizer_name: str = "rmsprop"
    allowed_optimizers: List[str] = ["sgd", "sgdm", "adam", "rmsprop"]
    learning_rate: float = 2.0e-5
    loss_scale: float = 1


class OutlierDetectionConfig(BaseSettings):
    md_dir: Path  # MD simulation direction
    cvae_dir: Path  # CVAE model directory
    pdb_file: Path
    reference_pdb_file: Path
    num_outliers: int = 500
    timeout_ns: float = 10.0
    model_params: CVAEModelConfig
    sklearn_num_cpus: int = 16
    logging: LoggingConfig


class GPUTrainingConfig(CVAEModelConfig):
    pass


class CS1TrainingConfig(CVAEModelConfig):
    hostname: str = "medulla1"
    medulla_experiment_path: Path
    run_script: Path = Path("/data/shared/vishal/ANL-shared/cvae_gb/run_mixed.sh")
    sim_data_dir: Optional[Path]
    data_dir: Optional[Path]
    eval_data_dir: Optional[Path]
    global_path: Optional[Path]  # files_seen36.txt
    theta_gpu_path: Optional[Path]
    model_dir: Optional[Path]

    # Logging params are not supported on CS-1 and are disabled in run.py.
    metrics: bool = True

    # Run params
    mode: str = "train"
    train_steps: int = 10
    eval_steps: int = 2
    runconfig_params: Dict[str, int] = {
        "save_checkpoints_steps": 10,
        "keep_checkpoint_max": 3,
        "save_summary_steps": 10,
        "log_step_count_steps": 10,
    }


class ExperimentConfig(BaseSettings):
    """
    Master configuration
    """

    experiment_directory: Path
    md_runner: MDRunnerConfig
    outlier_detection: OutlierDetectionConfig
    gpu_training: Optional[GPUTrainingConfig]
    cs1_training: Optional[CS1TrainingConfig]

    logging: LoggingConfig


def read_yaml_config(fname: str) -> ExperimentConfig:
    with open(fname) as fp:
        data = yaml.safe_load(fp)
    return ExperimentConfig(**data)
