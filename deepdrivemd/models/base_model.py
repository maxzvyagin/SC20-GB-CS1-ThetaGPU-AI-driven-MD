from typing import List, Optional, Any
from pathlib import Path


class ModelConfig:
    checkpoint_path: Path
    ssd_path: Path


class BaseModel:
    def __init__(self, model_config: ModelConfig) -> None:
        self.config = model_config
        self.config.weights_path

    def pre_train(self) -> None:
        return None

    def pre_predict(self) -> None:
        return None

    def train(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def preprocess(self, h5_files: List[Path]) -> None:
        raise NotImplementedError

    def predict(self) -> Any:
        raise NotImplementedError

    def get_weights_file(self) -> Optional[str]:
        """Returns a Path under self.config.weights_path"""
        return None
