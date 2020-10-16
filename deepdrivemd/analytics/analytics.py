from pydantic import BaseModel
import json
from typing import Union, Optional, List, Dict, Tuple
from pathlib import Path
import numpy as np

PathLike = Union[str, Path]


class Outlier(BaseModel):
    extrinsic_score: Optional[float]
    intrinsic_score: float
    rmsd: Optional[float]
    outlier_ind: int
    frame_index: int
    dcd_filename: Path
    pdb_filename: Path


class Analytics:
    def __init__(self, experiment_dir: PathLike):
        self.dir = Path(experiment_dir)

        self.json_files: List[Path] = self._find_outlier_files("outliers-*.json")
        self.embedding_files: List[Path] = self._find_outlier_files("embeddings-*.npy")
        self.rmsd_files: List[Path] = self._find_outlier_files("rmsds-*.npy")

        # All of these lists are indexed by the outlier detection iteration:
        self.json_data: List[Dict[int, dict]] = []
        for f in self.json_files:
            dat = json.loads(f.read_text())
            dat = {outlier.pop("outlier_ind"): outlier for outlier in dat}
            self.json_data.append(dat)

        self.embeddings: List[np.ndarray] = [np.load(f) for f in self.embedding_files]
        self.rmsds: List[np.ndarray] = [np.load(f) for f in self.rmsd_files]
        self._outlier_rmsds: Optional[List[np.ndarray]] = None

    def _find_outlier_files(self, pattern: str) -> List[Path]:
        return sorted(list(self.outlier_dir.glob(pattern)), key=self.extract_timestamp)

    @property
    def outlier_dir(self) -> Path:
        return self.dir.joinpath("outlier_runs")

    @staticmethod
    def extract_timestamp(fname: Path) -> str:
        return fname.with_suffix("").name.split("-")[1]

    def fetch_outlier(self, iteration_idx: int, outlier_idx: int) -> Outlier:
        outlier_iteration = self.json_data[iteration_idx]
        outlier_dict = outlier_iteration[outlier_idx]
        outlier = Outlier(**outlier_dict, outlier_ind=outlier_idx)
        if self.rmsds:
            outlier.rmsd = self.rmsds[iteration_idx][outlier_idx]
        return outlier

    @property
    def outlier_indices(self) -> List[Tuple[int, int]]:
        return list(
            (outlier_iter, outlier_idx)
            for outlier_iter in range(len(self.json_data))
            for outlier_idx in self.json_data[outlier_iter]
        )

    def extrinsic_scores(self) -> List[np.ndarray]:
        """
        Returns a list of extrinsic_score ndarrays; one array per iteration of outlier detection
        """
        scores = []
        for outliers in self.json_data:
            scores.append(
                np.array(
                    list(outlier["extrinsic_score"] for outlier in outliers.values())
                )
            )
        return scores

    def outlier_rmsds(self, use_cache=True) -> List[np.ndarray]:
        """
        Returns a list of extrinsic_score ndarrays; one array per iteration of outlier detection
        """
        if use_cache and self._outlier_rmsds:
            return self._outlier_rmsds
        outlier_rmsds = []
        for iter_idx, outliers in enumerate(self.json_data):
            rmsds = [
                self.rmsds[iter_idx][outlier_ind] for outlier_ind in outliers.keys()
            ]
            outlier_rmsds.append(np.array(rmsds))

        self._outlier_rmsds = outlier_rmsds
        return outlier_rmsds
