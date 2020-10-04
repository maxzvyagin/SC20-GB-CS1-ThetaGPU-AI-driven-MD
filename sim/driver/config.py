# Schema of the YAML experiment file

import json
import yaml
from pydantic import BaseSettings
from pathlib import Path
from typing import Optional, List, Dict


class MDConfig(BaseSettings):
    """
    Auto-generates configuration file for run_openmm.py
    """
    pdb_file: Path
    reference_pdb_file: Path
    top_file: Optional[Path]
    checkpoint_file: Optional[Path]
    simulation_length_ns: int = 10
    report_interval_ps: int = 50
    h5_scp_path: Optional[str]
    h5_cp_path: Optional[str]

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
    simulation_length_ns: int = 10
    report_interval_ps: int = 50


class OutlierDetectionConfig(BaseSettings):
    num_jobs: int


class GPUTrainingConfig(BaseSettings):
    pass


class CS1TrainingConfig(BaseSettings):
    medulla_experiment_path: Path
    run_script: Path = "/data/shared/vishal/ANL-shared/cvae_gb/run_mixed.sh"
    sim_data_dir: Optional[Path]
    data_dir: Optional[Path]
    eval_data_dir: Optional[Path]
    global_path: Optional[Path]  # files_seen36.txt
    theta_gpu_path: Optional[Path]
    model_dir: Optional[Path]

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


def read_yaml_config(fname):
    with open(fname) as fp:
        data = yaml.safe_load(fp)
    return ExperimentConfig(**data)
