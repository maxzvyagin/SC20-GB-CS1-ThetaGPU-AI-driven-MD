# Schema of the YAML experiment file
import json
import yaml
from enum import Enum
from pydantic import BaseSettings as _BaseSettings
from pydantic import validator
from pathlib import Path
from typing import Optional, List, Dict


class BaseSettings(_BaseSettings):
    def dump_yaml(self, file):
        yaml.dump(json.loads(self.json()), file, indent=4)


class MDType(str, Enum):
    implicit = "implicit"
    explicit = "explicit"


class LoggingConfig(BaseSettings):
    level: str = "DEBUG"
    format: str = "%(asctime)s|%(process)d|%(thread)d|%(levelname)8s|%(name)s:%(lineno)s] %(message)s"
    datefmt: str = "%d-%b-%Y %H:%M:%S"
    buffer_num_records: int = 1024
    flush_period: int = 30


class MDConfig(BaseSettings):
    """
    Auto-generates configuration file for run_openmm.py
    """

    reference_pdb_file: Optional[Path]
    local_run_dir: Path
    sim_type: MDType
    simulation_length_ns: float = 10
    report_interval_ps: float = 50
    dt_ps: float = 0.002
    temperature_kelvin: float = 310.0
    # Length of each simulation in nanoseconds if recursive mode is active
    reeval_time_ns: float = 10
    frames_per_h5: int
    result_dir: Path
    h5_scp_path: Optional[str]
    omm_dir_prefix: str
    local_run_dir: Path = Path("/raid/scratch")
    input_dir: Path
    initial_configs_dir: Path
    logging: LoggingConfig


class MDRunnerConfig(BaseSettings):
    """
    Global MD configuration (written one per experiment)
    """

    num_jobs: int
    initial_configs_dir: Path
    reference_pdb_file: Optional[Path]
    sim_type: MDType
    frames_per_h5: int = 1024
    temperature_kelvin: float = 310.0
    simulation_length_ns: float = 10
    report_interval_ps: float = 50
    reeval_time_ns: float = 2
    local_run_dir: Path = Path("/raid/scratch")
    md_run_command: str = "python run_openmm.py"
    md_environ_setup: List[str] = [
        'eval "$(/lus/theta-fs0/projects/RL-fold/msalim/miniconda3/bin/conda shell.bash hook)"',
        "conda activate /lus/theta-fs0/projects/RL-fold/venkatv/software/conda_env/a100_rapids_openmm",
        "export PYTHONPATH=/lus/theta-fs0/projects/RL-fold/msalim/SC20-GB-CS1-ThetaGPU-AI-driven-MD:$PYTHONPATH",
    ]

    @validator("reeval_time_ns")
    def reeval_time_less_than_sim_time(cls, v, values):
        if values["simulation_length_ns"] <= v:
            raise ValueError("reeval_time_ns must be less than simulation_length_ns")
        return v


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

    # Logging params are not supported on CS-1 and are disabled in run.py.
    metrics: bool = True

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


class ExtrinsicScore(str, Enum):
    none = "none"
    rmsd_to_reference_state = "rmsd_to_reference_state"


class OutlierDetectionUserConfig(BaseSettings):
    num_outliers: int = 500
    extrinsic_outlier_score: ExtrinsicScore = ExtrinsicScore.none
    sklearn_num_cpus: int = 16
    local_scratch_dir: Path = Path("/raid/scratch")
    max_num_old_h5_files: int = 1000
    # Run parameters
    run_command: str = (
        "singularity run -B /lus:/lus:rw --nv "
        "/lus/theta-fs0/projects/RL-fold/msalim/tensorflow_20.09-tf1-py3.sif "
        "/lus/theta-fs0/projects/RL-fold/msalim/tf1-ngc-env/bin/python -m deepdrivemd.outlier.lof "
    )
    environ_setup: List[str] = []
    num_nodes: int = 1
    ranks_per_node: int = 1
    gpus_per_node: int = 8


class OutlierDetectionRunConfig(OutlierDetectionUserConfig):
    logging: LoggingConfig
    model_params: CVAEModelConfig
    md_dir: Path
    cvae_dir: Path
    walltime_min: int


class GPUTrainingUserConfig(BaseSettings):
    pass


class GPUTrainingRunConfig(CVAEModelConfig, GPUTrainingUserConfig):
    pass


class CS1TrainingUserConfig(BaseSettings):
    medulla_experiment_path: Path
    hostname: str = "medulla1"
    run_script: Path = Path("/data/shared/vishal/ANL-shared/cvae_gb/run_mixed.sh")
    num_frames_per_training: int = 16_000
    initial_h5_transfer_dir: Optional[Path] = None

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


class CS1TrainingRunConfig(CS1TrainingUserConfig, CVAEModelConfig):
    sim_data_dir: Path
    data_dir: Path
    eval_data_dir: Path
    global_path: Path  # files_seen36.txt
    theta_gpu_path: Path
    model_dir: Path


class ExperimentConfig(BaseSettings):
    """
    Master configuration
    """

    experiment_directory: Path
    walltime_min: int
    md_runner: MDRunnerConfig
    outlier_detection: OutlierDetectionUserConfig
    logging: LoggingConfig
    cvae_model: CVAEModelConfig
    gpu_training: Optional[GPUTrainingUserConfig] = None
    cs1_training: Optional[CS1TrainingUserConfig] = None


def read_yaml_config(fname: str) -> ExperimentConfig:
    with open(fname) as fp:
        data = yaml.safe_load(fp)
    return ExperimentConfig(**data)


def generate_sample_config():
    md_runner = MDRunnerConfig(
        num_jobs=10,
        initial_configs_dir="/path/to/initial_pdbs_and_tops",
        reference_pdb_file="/path/to/reference.pdb",
        sim_type="explicit",
        frames_per_h5=1024,
    )
    model = CVAEModelConfig()
    logging = LoggingConfig()
    outlier_detection = OutlierDetectionUserConfig()
    cs1_training = CS1TrainingUserConfig(
        medulla_experiment_path="/data/shared/experiment/on/medulla1"
    )

    return ExperimentConfig(
        experiment_directory="/path/to/experiment",
        walltime_min=120,
        md_runner=md_runner,
        outlier_detection=outlier_detection,
        logging=logging,
        cs1_training=cs1_training,
        cvae_model=model,
    )


if __name__ == "__main__":
    with open("deepdrivemd_template.yaml", "w") as fp:
        config = generate_sample_config()
        config.dump_yaml(fp)
