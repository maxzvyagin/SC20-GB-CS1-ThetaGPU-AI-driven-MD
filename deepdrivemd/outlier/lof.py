from concurrent.futures import ThreadPoolExecutor
from tempfile import NamedTemporaryFile
from typing import List, Optional
import json
from pathlib import Path
import time
import yaml
import shutil
import argparse
import queue
import numpy as np
import itertools

import h5py

import MDAnalysis as mda

from deepdrivemd import config_logging
from deepdrivemd.util import FileLock
from deepdrivemd.driver.config import OutlierDetectionRunConfig

from deepdrivemd.models.symmetric_cvae.predict_gpu import TFEstimatorModel

from deepdrivemd.outlier.utils import outlier_search_lof
from deepdrivemd.outlier.utils import find_frame

import logging

logger = None


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="YAML config file")
    config_filename = parser.parse_args().config
    with open(config_filename) as fp:
        dict_config = yaml.safe_load(fp)
    return OutlierDetectionRunConfig(**dict_config)


class OutlierDetectionContext:
    def __init__(
        self,
        local_scratch_dir: Path,
        outlier_results_dir: Path,
        md_dir: Path,
        model_weights_dir: Path,
        max_num_old_h5_files: int,
        model_params: dict,
        outlier_predict_batch_size: int,
        **kwargs,
    ):
        self.md_dir = Path(md_dir).resolve()
        self.model_weights_dir = Path(model_weights_dir).resolve()

        self._local_scratch_dir = local_scratch_dir
        self._outlier_results_dir = outlier_results_dir
        self._outlier_pdbs_dir = outlier_results_dir.joinpath("outlier_pdbs")
        self._outlier_pdbs_dir.mkdir()
        self._seen_h5_files_set = set()

        self._last_acquired_cvae_lock = None
        self._md_input_dirs = None
        self._h5_dcd_map = {}
        self.dcd_h5_map = {}
        self._pdb_outlier_queue = queue.PriorityQueue()
        self._h5_contact_map_length = None
        self.max_num_old_h5_files = max_num_old_h5_files
        self._seen_outliers = set()
        self.model = TFEstimatorModel(
            local_scratch_dir,
            model_params,
            outlier_predict_batch_size,
            self.model_weights_dir,
        )

    @property
    def h5_files(self):
        return list(self._h5_dcd_map.keys())

    @property
    def dcd_files(self):
        return list(self._h5_dcd_map.values())

    @property
    def md_input_dirs(self):
        if self._md_input_dirs is None:
            self._md_input_dirs = list(self.md_dir.glob("input_*"))
        return self._md_input_dirs

    def put_outlier(
        self, dcd_filename, frame_index, created_time, intrinsic_score, extrinsic_score
    ):
        assert created_time > 0
        assert intrinsic_score < 0
        if extrinsic_score is None:
            extrinsic_score = 0
        # TODO: Make combined score or have a function to compute the score
        #       which the user can specify score_func(dcd_file, frame_index) -> score
        #       where score is any object with a comparison operator (<).
        score = (-1 * created_time, extrinsic_score, intrinsic_score)
        outlier = (dcd_filename, frame_index)
        # Only keep new outliers
        if outlier in self._seen_outliers:
            logger.info(f"Outlier seen before: {outlier}")
            return
        self._seen_outliers.add(outlier)
        logger.debug(f"Enqueueing new (score, outlier) = {(score, outlier)}")
        self._pdb_outlier_queue.put_nowait((score, outlier))

    def _get_outlier(self):
        try:
            score, outlier = self._pdb_outlier_queue.get(block=False)
        except queue.Empty:
            return None
        else:
            return score, outlier

    def get_open_md_input_slots(self):
        return [d for d in self.md_input_dirs if not list(d.glob("*.pdb"))]

    def send_outliers(self) -> List[dict]:
        # Blocks until all PDBs are sent
        md_dirs = self.get_open_md_input_slots()
        logger.info(f"Sending new outlier pdbs to {len(md_dirs)} dirs")

        outliers = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            for data in ex.map(self.dispatch_next_outlier, md_dirs):
                if data is not None:
                    outliers.append(data)
        return outliers

    def backup_pdbs(self, outlier_results: List[dict]):
        count = len(outlier_results)
        logger.info(
            f"Moving {count} pdb files from local storage to {self._outlier_pdbs_dir}"
        )
        start = time.time()
        for outlier in outlier_results:
            dest = shutil.move(outlier["pdb_filename"], self._outlier_pdbs_dir)
            outlier["pdb_filename"] = dest
        elapsed = time.time() - start
        logger.info(f"Moved {count} pdb files in {elapsed:.2f} seconds")

    def generate_pdb_file(self, dcd_filename: Path, frame_index: int) -> Path:
        logger.debug(
            f"generate_pdb_file({dcd_filename}, {frame_index}): {list(dcd_filename.parent.glob('*'))}"
        )
        pdb_path = list(dcd_filename.parent.glob("*.pdb"))[0]
        system_name = pdb_path.with_suffix("").name.split("__")[1]
        # run_058/something.dcd -> something
        outlier_pdb_fname = (
            dcd_filename.with_suffix("").name + f"__{system_name}__{frame_index}.pdb"
        )
        outlier_pdb_file = self._local_scratch_dir.joinpath(outlier_pdb_fname)

        temp_pdb = NamedTemporaryFile(dir=self._local_scratch_dir, suffix=".pdb")
        temp_pdb.close()
        temp_dcd = NamedTemporaryFile(dir=self._local_scratch_dir, suffix=".dcd")
        temp_dcd.close()
        local_pdb = shutil.copy(
            pdb_path.as_posix(), Path(temp_pdb.name).resolve().as_posix()
        )
        local_dcd = shutil.copy(
            dcd_filename.as_posix(), Path(temp_dcd.name).resolve().as_posix()
        )

        mda_traj = mda.Universe(local_pdb, local_dcd)
        mda_traj.trajectory[frame_index]

        PDB = mda.Writer(outlier_pdb_file.as_posix())
        PDB.write(mda_traj.atoms)
        return outlier_pdb_file

    def dispatch_next_outlier(self, input_dir) -> Optional[dict]:
        item = self._get_outlier()
        if item is None:
            return None

        score, outlier = item
        _, extrinsic_score, intrinsic_score = score

        if extrinsic_score is not None:
            extrinsic_score = float(extrinsic_score)
        if intrinsic_score is not None:
            intrinsic_score = float(intrinsic_score)

        dcd_filename, frame_index = outlier

        logger.debug(f"Creating outlier .pdb from {dcd_filename} frame {frame_index}")
        outlier_pdb_file = self.generate_pdb_file(dcd_filename, frame_index)
        logger.debug("outlier .pdb write done")
        target = input_dir.joinpath(outlier_pdb_file.name)

        logger.debug(f"Acquiring FileLock to write {target}")
        with FileLock(target):
            logger.debug(f"FileLock acquired")
            start = time.perf_counter()
            shutil.copy(outlier_pdb_file.as_posix(), target)
            elapsed = time.perf_counter() - start
            logger.debug(f"shutil.copy wrote {target} in {elapsed:.2f} seconds")

        return {
            "extrinisic_score": extrinsic_score,
            "intrinsic_score": intrinsic_score,
            "dcd_filename": str(dcd_filename),
            "frame_index": int(frame_index),
            "pdb_filename": outlier_pdb_file.as_posix(),
        }

    def rescan_h5_dcd(self):
        # We *always* want to scan at least once
        # But on the first pass, block until at least some h5 files appear
        got_new_h5 = 0
        while got_new_h5 < 80:
            logger.debug("Outlier detection waiting for new H5 files")
            for done_path in self.md_dir.glob("*/DONE"):
                h5 = list(done_path.parent.glob("*.h5"))[0]
                dcd = list(done_path.parent.glob("*.dcd"))[0]
                if h5 not in self.h5_files:
                    self._h5_dcd_map[h5] = dcd
                    self.dcd_h5_map[dcd] = h5
                    got_new_h5 += 1
            if got_new_h5 < 80:
                time.sleep(10)

    def await_model_weights(self):
        """Blocks until model has weights"""
        while self.model.get_weights_file() is None:
            logger.debug("Outlier detection waiting for model checkpoint file")
            time.sleep(10)
        time.sleep(10)

    @property
    def h5_length(self):
        if self._h5_contact_map_length is not None:
            return self._h5_contact_map_length
        if not self.h5_files:
            raise Exception("Calling get_h5_length before data has arrived")
        h5_file = self.h5_files[0]
        with h5py.File(h5_file, "r") as f:
            self._h5_contact_map_length = len(f["contact_map"])
        return self._h5_contact_map_length

    def update_dataset(self):
        num_h5s = min(len(self._seen_h5_files_set), self.max_num_old_h5_files)
        if num_h5s > 0:
            stride = int(len(self._seen_h5_files_set) // num_h5s)
        else:
            stride = 1

        old_h5_indices = list(range(0, len(self._seen_h5_files_set), stride))
        new_h5_files = list(set(self.h5_files).difference(self._seen_h5_files_set))
        new_h5_indices = list(
            range(len(self.h5_files) - len(new_h5_files), len(self.h5_files))
        )

        indices = old_h5_indices + new_h5_indices
        all_dcd_files = (
            self.dcd_files
        )  # self.dcd_files is @property; don't put in listcomp!
        dcd_files = [all_dcd_files[i] for i in indices]

        model_input = self.model.preprocess(new_h5_files, dcd_files)
        self._seen_h5_files_set.update(new_h5_files)
        return (dcd_files, model_input)

    def backup_array(self, results, name, creation_time):
        if isinstance(results, list):
            results = np.array(results)
        result_file = self._outlier_results_dir.joinpath(f"{name}-{creation_time}.npy")
        np.save(result_file, results)

    def backup_json(self, results, name, creation_time):
        # Convert numeric arrays to string for JSON serialization
        result_file = self._outlier_results_dir.joinpath(f"{name}-{creation_time}.json")
        with open(result_file, "w") as f:
            json.dump(results, f, indent=4)

    def halt_simulations(self):
        for md_dir in self.md_input_dirs:
            md_dir.joinpath("halt").touch()


def main():
    global logger
    config = get_config()
    log_fname = config.md_dir.parent.joinpath("outlier_detection.log").as_posix()
    config_logging(filename=log_fname, **config.logging.dict())
    logger = logging.getLogger("deepdrivemd.outlier.lof")

    logger.info(f"Starting outlier detection main()")
    logger.info(f"{config.dict()}")

    ctx = OutlierDetectionContext(**config.dict())

    start_time = time.time()
    while True:
        # Will block until the first H5 files are received
        ctx.rescan_h5_dcd()
        dcd_files, model_input_data = ctx.update_dataset()

        # NOTE: It's up to the ML service to do model selection;
        # we're assuming the latest cvae weights file has the best model
        ctx.await_model_weights()

        embeddings = ctx.model.predict(model_input_data)
        logger.info(f"end model prediction: generated {len(embeddings)} embeddings")

        logger.info(
            f"Starting outlier searching with n_outliers="
            f"{config.num_outliers} and n_jobs={config.sklearn_num_cpus}"
        )

        outlier_inds, outlier_scores = outlier_search_lof(
            embeddings, n_outliers=config.num_outliers, n_jobs=config.sklearn_num_cpus
        )
        logger.info("Done with outlier searching")

        # A record of every trajectory length (they are all the same)
        logger.debug("Building traj_dict:")
        logger.debug(f"dcd_files = {dcd_files}")
        logger.debug(f"ctx.h5_length = {ctx.h5_length}")
        traj_dict = dict(zip(dcd_files, itertools.cycle([ctx.h5_length])))

        # Collect extrinsic scores for logging
        extrinsic_scores = []
        rmsds = []

        # Identify new outliers and add to queue
        creation_time = int(time.time())
        for outlier_ind, outlier_score in zip(outlier_inds, outlier_scores):
            # find the location of outlier in it's DCD file
            frame_index, dcd_filename = find_frame(traj_dict, outlier_ind)
            h5_file = ctx.dcd_h5_map[dcd_filename]
            extrinsic_score = None

            with h5py.File(h5_file, "r") as f:
                if "rmsd" in f.keys():
                    rmsd = f["rmsd"][...][frame_index]
                    rmsds.append(rmsd)
                    if config.extrinsic_outlier_score == "rmsd_to_reference_state":
                        extrinsic_score = rmsd
                        extrinsic_scores.append(extrinsic_score)

            ctx.put_outlier(
                dcd_filename, frame_index, creation_time, outlier_score, extrinsic_score
            )

        # Send outliers to MD simulation jobs
        outlier_results = ctx.send_outliers()
        logger.info("Finished sending new outliers")

        # outlier_results is a List of result_dicts:
        # With the keys: {extrinsic_score, intrinsic_score, dcd_filename, frame_index, pdb_filename}

        # Move outliers from scratch to persistent location
        ctx.backup_pdbs(outlier_results)
        ctx.backup_array(embeddings, "embeddings", creation_time)
        ctx.backup_array(rmsds, "rmsds", creation_time)
        ctx.backup_json(outlier_results, "outliers", creation_time)

        # Compute elapsed time
        mins, secs = divmod(int(time.time() - start_time), 60)
        logger.info(f"Outlier detection elapsed time {mins:02d}:{secs:02d} done")

        # If elapsed time is greater than specified walltime then stop
        # all MD simulations and end outlier detection process.
        if mins + secs / 60 >= config.walltime_min:
            logger.info("Walltime expired: halting simulations")
            ctx.halt_simulations()
            return


if __name__ == "__main__":
    main()
