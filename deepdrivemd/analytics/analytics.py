from pydantic import BaseModel
import json
from typing import Union, Optional, List, Dict
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
        self.embeddings: List[np.ndarray] = []
        self.rmsds: List[np.ndarray] = []
        self._load_data()

    def _find_outlier_files(self, pattern: str) -> List[Path]:
        return sorted(list(self.outlier_dir.glob(pattern)), key=self.extract_timestamp)

    @property
    def outlier_dir(self) -> Path:
        return self.dir.joinpath("outlier_runs")

    @staticmethod
    def extract_timestamp(fname: Path) -> str:
        return fname.with_suffix("").name.split("-")[1]

    def _load_data(self):
        for json_f, embed_f, rmsd_f in zip(
            self.json_files, self.embedding_files, self.rmsd_files
        ):
            json_data = json.loads(json_f.read_text())
            json_data = {dat.pop("outlier_ind"): dat for dat in json_data}
            self.json_data.append(json_data)
            self.embeddings.append(np.load(embed_f))
            self.rmsds.append(np.load(rmsd_f))

    def fetch_outlier(self, iteration_idx: int, outlier_idx: int) -> Outlier:
        outlier_iteration = self.json_data[iteration_idx]
        outlier_dict = outlier_iteration[outlier_idx]
        outlier = Outlier(**outlier_dict)
        return outlier

    def extrinsic_scores(self) -> List[np.ndarray]:
        """
        Returns a list of extrinsic_score ndarrays; one array per iteration of outlier detection
        """
        scores = []
        for outliers in self.json_data:
            scores.append(
                np.array(
                    list(outlier["extrinisic_score"] for outlier in outliers.values())
                )
            )
        return scores

    def outlier_rmsds(
        self,
    ) -> List[np.ndarray]:
        """
        Returns a list of extrinsic_score ndarrays; one array per iteration of outlier detection
        """
        outlier_rmsds = []
        for iter_idx, outliers in enumerate(self.json_data):
            rmsds = [
                self.rmsds[iter_idx][outlier_ind] for outlier_ind in outliers.keys()
            ]
            outlier_rmsds.append(np.array(rmsds))
        return outlier_rmsds
