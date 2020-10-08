from concurrent.futures import ThreadPoolExecutor
from tempfile import NamedTemporaryFile
import json
from pathlib import Path
import time
import yaml
import shutil
import argparse
import queue
import numpy as np
import itertools

import tensorflow as tf
import h5py

import MDAnalysis as mda

from deepdrivemd import config_logging
from deepdrivemd.util import FileLock
from deepdrivemd.driver.config import OutlierDetectionRunConfig, CVAEModelConfig

from deepdrivemd.models.symmetric_cvae.utils import (
    write_to_tfrecords,
    write_single_tfrecord,
)
from deepdrivemd.models.symmetric_cvae.data import parse_function_record_predict
from deepdrivemd.models.symmetric_cvae.model import model_fn


from deepdrivemd.outlier.utils import outlier_search_lof
from deepdrivemd.outlier.utils import find_frame

import logging

logger = None
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)


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
        cvae_dir: Path,
        max_num_old_h5_files: int,
        model_params: dict,
        outlier_predict_batch_size: int,
        **kwargs,
    ):
        self.md_dir = Path(md_dir).resolve()
        self.cvae_dir = Path(cvae_dir).resolve()

        self._local_scratch_dir = local_scratch_dir
        self._outlier_results_dir = outlier_results_dir
        self.tfrecords_dir = local_scratch_dir.joinpath("tfrecords")
        self.tfrecords_dir.mkdir(exist_ok=True)
        self._seen_h5_files_set = set()

        self.cvae_weights_file = None
        self._last_acquired_cvae_lock = None
        self._md_input_dirs = None
        self._h5_dcd_map = {}
        self.dcd_h5_map = {}
        self._pdb_outlier_queue = queue.PriorityQueue()
        self._model_params = model_params
        self._h5_contact_map_length = None
        self.max_num_old_h5_files = max_num_old_h5_files
        self._seen_outliers = set()
        self._outlier_predict_batch_size = outlier_predict_batch_size

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
        if extrinsic_score == None:
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
            _, outlier = self._pdb_outlier_queue.get(block=False)
        except queue.Empty:
            return None
        else:
            return outlier

    def get_open_md_input_slots(self):
        return [d for d in self.md_input_dirs if not list(d.glob("*.pdb"))]

    def send(self):
        # Blocks until all PDBs are sent
        md_dirs = self.get_open_md_input_slots()
        logger.info(f"Sending new outlier pdbs to {len(md_dirs)} dirs")
        with ThreadPoolExecutor(max_workers=8) as ex:
            for _ in ex.map(self.dispatch_next_outlier, md_dirs):
                pass

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
        local_pdb = shutil.copy(pdb_path.as_posix(), Path(temp_pdb.name).resolve().as_posix())
        local_dcd = shutil.copy(dcd_filename.as_posix(), Path(temp_dcd.name).resolve().as_posix())

        mda_traj = mda.Universe(local_pdb, local_dcd)
        mda_traj.trajectory[frame_index]

        PDB = mda.Writer(outlier_pdb_file.as_posix())
        PDB.write(mda_traj.atoms)
        return outlier_pdb_file

    def dispatch_next_outlier(self, input_dir):
        outlier = self._get_outlier()
        if outlier is None:
            return

        dcd_filename, frame_index = outlier
        logger.debug(f"Creating outlier .pdb from {dcd_filename} frame {frame_index}")
        outlier_pdb_file = self.generate_pdb_file(dcd_filename, frame_index)
        logger.debug("outlier .pdb write done")
        target = input_dir.joinpath(outlier_pdb_file.name)
        logger.debug(f"Acquiring FileLock to write {target}")
        with FileLock(target):
            logger.debug(f"FileLock acquired")
            start = time.perf_counter()
            shutil.move(outlier_pdb_file.as_posix(), target)
            elapsed = time.perf_counter() - start
            logger.debug(f"shutil.move wrote {target.name} in {elapsed:.2f} seconds")

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

    def update_model(self):
        """Gets most recent model weights."""
        while True:
            self.cvae_weights_file = tf.train.latest_checkpoint(self.cvae_dir)
            if self.cvae_weights_file is not None:
                break
            logger.debug("Outlier detection waiting for model checkpoint file")
            time.sleep(10)
        time.sleep(10)
        self.cvae_weights_file = tf.train.latest_checkpoint(self.cvae_dir)

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

        # tf.data.Dataset.list_files expects a list of strings, not pathlib.Path objects!
        # as_posix() converts a Path to a string
        tfrecord_files = [
            self.tfrecords_dir.joinpath(f.with_suffix(".tfrecords").name).as_posix()
            for f in dcd_files
        ]
        logger.debug(
            f"update_dataset: Will predict from tfrecord_files={tfrecord_files}"
        )

        self._seen_h5_files_set.update(new_h5_files)

        # Write to local node storage
        for h5_file in new_h5_files:
            write_single_tfrecord(
                h5_file=h5_file,
                initial_shape=self._model_params["h5_shape"][1:],
                final_shape=self._model_params["tfrecord_shape"][1:],
                tfrecord_dir=self.tfrecords_dir,
            )

        # Use files closure to get correct data sample
        def data_generator():
            dtype = tf.float16 if self._model_params["mixed_precision"] else tf.float32
            list_files = tf.data.Dataset.list_files(tfrecord_files)
            dataset = tf.data.TFRecordDataset(list_files)

            # TODO: We want drop_remainder=False but this needs to be rewritten:
            dataset = dataset.batch(
                self._outlier_predict_batch_size, drop_remainder=True
            )
            parse_sample = parse_function_record_predict(
                dtype,
                self._model_params["tfrecord_shape"],
                self._model_params["input_shape"],
            )
            return dataset.map(parse_sample)

        return (dcd_files, data_generator)

    def backup_array(self, results, name, creation_time):
        result_file = self._outlier_results_dir.joinpath(f"{name}-{creation_time}.npy")
        np.save(result_file, results)

    def backup_dict(self, results, name, creation_time):
        # Convert numeric arrays to string for JSON serialization
        json_result = {}
        for key, val in results.items():
            json_result[key] = list(map(str, val))

        result_file = self._outlier_results_dir.joinpath(f"{name}-{creation_time}.json")
        with open(result_file, "w") as f:
            json.dump(json_result, f)

    def halt_simulations(self):
        for md_dir in self.md_input_dirs:
            md_dir.joinpath("halt").touch()


def predict_from_cvae(
    workdir: Path,
    weights_file: str,
    config: CVAEModelConfig,
    data_generator,
    outlier_predict_batch_size: int,
):
    params = config.dict()
    params["sim_data_dir"] = workdir.as_posix()
    params["data_dir"] = workdir.as_posix()
    params["eval_data_dir"] = workdir.as_posix()
    params["global_path"] = workdir.joinpath("files_seen.txt").as_posix()
    params["fraction"] = 0.0
    params["batch_size"] = outlier_predict_batch_size

    tf_config = tf.estimator.RunConfig()
    est = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=tf_config,
    )
    gen = est.predict(
        input_fn=data_generator,
        checkpoint_path=weights_file,
        yield_single_examples=True,
    )
    return np.array([list(it.values())[0] for it in gen])


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
        dcd_files, data_generator = ctx.update_dataset()

        # NOTE: It's up to the ML service to do model selection;
        # we're assuming the latest cvae weights file has the best model
        ctx.update_model()

        assert ctx.cvae_weights_file is not None
        logger.info(f"start model prediction with weights: {ctx.cvae_weights_file}")
        embeddings = predict_from_cvae(
            ctx.tfrecords_dir,
            ctx.cvae_weights_file,
            config.model_params,
            data_generator,
            config.outlier_predict_batch_size,
        )
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
        # Identify new outliers and add to queue
        creation_time = int(time.time())
        for outlier_ind, outlier_score in zip(outlier_inds, outlier_scores):
            # find the location of outlier in it's DCD file
            frame_index, dcd_filename = find_frame(traj_dict, outlier_ind)

            # Rank the outlier PDBs according to their RMSD to reference state
            if config.extrinsic_outlier_score == "rmsd_to_reference_state":
                h5_file = ctx.dcd_h5_map[dcd_filename]
                with h5py.File(h5_file, "r") as f:
                    extrinsic_score = f["rmsd"][...][frame_index]
            else:
                extrinsic_score = None

            if extrinsic_score is not None:
                extrinsic_scores.append(extrinsic_score)

            ctx.put_outlier(
                dcd_filename, frame_index, creation_time, outlier_score, extrinsic_score
            )

        # Send outliers to MD simulation jobs
        ctx.send()
        logger.info("Finished sending new outliers")

        # Results dictionary storing outlier_inds, intrinsic & extrinsic scores
        results = {
            "outlier_inds": outlier_inds,
            "intrinsic_scores": outlier_scores,
            "extrinsic_scores": extrinsic_scores,
        }

        ctx.backup_array(embeddings, "embeddings", creation_time)
        ctx.backup_dict(results, "outliers", creation_time)

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
