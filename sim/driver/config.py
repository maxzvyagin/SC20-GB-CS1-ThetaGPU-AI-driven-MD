# Schema of the YAML experiment file

import yaml
from pydantic import BaseSettings

class MDConfig(BaseSettings):
    pdb_file: Path
    reference_pdb_file: Path
    top_file: Optional[Path]
    checkpoint_file: Optional[Path]
    simulation_length_ns: int = 10
    report_interval_ps: int = 50
    h5_scp_path: Optional[str]
    h5_cp_path: Optional[str]

    def dump_yaml(self, file):
        yaml.dump(self.dict(), file)


class MDRunnerConfig(BaseSettings):
    num_jobs: int
    pdb_file: Path
    reference_pdb_file: Path
    top_file: Optional[Path]
    checkpoint_file: Optional[Path]
    simulation_length_ns: int = 10
    report_interval_ps: int = 50
    h5_scp_path: Optional[str]
    h5_cp_path: Optional[str]

class OutlierDetectionConfig(BaseSettings):
    num_jobs: int

class GPUTrainingConfig(BaseSettings):
    pass

class CS1TrainingConfig(BaseSettings):
    pass


class ExperimentConfig(BaseSettings):
    experiment_directory: Path
    md_runner: MDRunnerConfig
    outlier_detection: OutlierDetectionConfig
    gpu_training: Optional[GPUTrainingConfig]
    cs1_training: Optional[CS1TrainingConfig]

def read_yaml_config(fname):
    with open(fname) as fp:
        data = yaml.safe_load(fp)
    return ExperimentConfig(**data)
