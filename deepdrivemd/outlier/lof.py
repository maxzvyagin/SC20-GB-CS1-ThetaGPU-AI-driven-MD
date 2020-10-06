from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from functools import lru_cache
import time
import yaml
import shutil
import argparse
import queue
import random
import numpy as np
from glob import glob

import tensorflow as tf
import h5py

import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSD

from deepdrivemd.util import FileLock, config_logging
from deepdrivemd.driver.config import OutlierDetectionConfig, CVAEModelConfig

from deepdrivemd.models.symmetric_cvae.utils import write_to_tfrecords
from deepdrivemd.models.symmetric_cvae.data import parse_function_record
from deepdrivemd.models.symmetric_cvae.model import model_fn

from deepdrivemd.models.symmetric_cvae.prepare_dataset import update_dataset

from deepdrivemd.outlier.utils import predict_from_cvae, outliers_from_latent_loc
from deepdrivemd.outlier.utils import outliers_largeset
from deepdrivemd.outlier.utils import find_frame, write_pdb_frame

import logging

logger = logging.getLogger(__name__)


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="YAML config file")
    config_filename = parser.parse_args().config
    with open(config_filename) as fp:
        dict_config = yaml.safe_load(fp)
    return OutlierDetectionConfig(**dict_config)


class OutlierDetectionContext:
    def __init__(
        self,
        pdb_file,
        reference_pdb_file,
        local_scratch_dir,
        md_dir,
        cvae_dir,
        max_num_old_h5_files,
        model_params: CVAEModelConfig,
        **kwargs,
    ):
        self.pdb_file = Path(pdb_file).resolve()
        self.reference_pdb_file = Path(reference_pdb_file).resolve()
        self.md_dir = Path(md_dir).resolve()
        self.cvae_dir = Path(cvae_dir).resolve()

        self._local_scratch_dir = local_scratch_dir
        self._tfrecords_dir = local_scratch_dir.joinpath("tfrecords")
        self._seen_h5_files_set = set()

        self._cvae_weights_file = None
        self._last_acquired_cvae_lock = None
        self._md_input_dirs = None
        self._h5_dcd_map = {}
        self._pdb_outlier_queue = queue.PriorityQueue()
        self._model_params = model_params
        self._h5_contact_map_length = None
        self.max_num_old_h5_files = max_num_old_h5_files

    @property
    def h5_files(self):
        return list(self._h5_dcd_map.keys())

    @property
    def dcd_files(self):
        return list(self._h5_dcd_map.values())

    @property
    def h5_dcd_file_pairs(self):
        return list(self._h5_dcd_map.items())

    @property
    def md_input_dirs(self):
        if self._md_input_dirs is None:
            self._md_input_dirs = list(self.md_dir.glob("input_*"))
        return self._md_input_dirs

    def put_outlier(self, dcd_filename, frame_index, created_time, outlier_score):
        assert created_time > 0
        assert outlier_score < 0
        score = (-1 * created_time, outlier_score)
        outlier = (dcd_filename, frame_index)
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

    def generate_pdb_file(self, dcd_filename: Path, frame_index: int) -> str:
        outlier_pdb_fname = dcd_filename.with_suffix("").name + f"_{frame_index}.pdb"
        outlier_pdb_path = self._local_scratch_dir.joinpath(outlier_pdb_fname)

        pdb_path = dcd_filename.with_suffix(".pdb")
        mda_traj = mda.Universe(pdb_path.as_posix(), dcd_filename.as_posix())
        mda_traj.trajectory[frame_index]

        PDB = mda.Writer(outlier_pdb_path.as_posix())
        PDB.write(mda_traj.atoms)
        return outlier_pdb_path

    def dispatch_next_outlier(self, input_dir):
        outlier = self._get_outlier()
        if outlier is None:
            return False

        dcd_filename, frame_index = outlier
        logger.info(f"Creating outlier .pdb from {dcd_filename} frame {frame_index}")
        local_pdb_path = self.generate_pdb_file(dcd_filename, frame_index)
        logger.info("outlier .pdb write done")
        target = input_dir.joinpath(local_pdb_path)
        logger.debug(f"Acquiring FileLock to write {target}")
        with FileLock(target):
            logger.debug(f"FileLock acquired")
            start = time.perf_counter()
            shutil.move(local_pdb_path, target)
            elapsed = time.perf_counter() - start
            logger.info(f"shutil.move wrote {target.name} in {elapsed:.2f} seconds")

    def rescan_h5_dcd(self):
        for done_path in self.md_dir.glob("**/DONE"):
            h5 = list(done_path.parent.glob("*.h5"))[0]
            dcd = list(done_path.parent.glob("*.dcd"))[0]
            self._h5_dcd_map[h5] = dcd

    @property
    def cvae_weights_file(self):
        if self._last_acquired_cvae_lock == self._cvae_weights_file:
            return self._cvae_weights_file
        with FileLock(self._cvae_weights_file):
            self._last_acquired_cvae_lock = self._cvae_weights_file
            return self._cvae_weights_file

    def update_model(self):
        """Gets most recent model weights."""
        all_weights = self.cvae_dir.glob("*.h5")
        # Sort files by time of creation
        all_weights = sorted(all_weights, key=lambda file: os.path.getctime(file))
        self._cvae_weights_file = all_weights[-1]
        # TODO: Make sure to use FileLock before loading weights!

    def get_h5_length(self, h5_file):
        if self._h5_contact_map_length is None:
            with h5py.File(h5_file, "r") as f:
                self._h5_contact_map_length = len(f["contact_map"])
        return self._h5_contact_map_length

    def update_dataset(self):
        num_h5s = min(len(self.h5_files), self.max_num_old_h5_files)
        stride = int(len(self.h5_files) // num_h5s)
        h5_indices = list(range(0, len(self.h5_files), stride))
        old_h5_indices = random.choices(self.h5_files, k=min(len(self.h5_files, 100)))
        new_h5_files = list(set(self.h5_files).difference(self._seen_h5_files_set))
        self._seen_h5_files_set.update(new_h5_files)
        write_to_tfrecords(
            files=new_h5_files,
            initial_shape=self._model_params.h5_shape[1:],
            final_shape=self._model_params.tfrecord_shape[1:],
            num_samples=self.get_h5_length(new_h5_files[0]),
            train_dir_path=self._tfrecords_dir,
            eval_dir_path=self._tfrecords_dir,
            fraction=0.0,
        )


        def data_generator():
            dtype = tf.float16 if self._model_params.mixed_precision else tf.float32
            files = list(Path(self._tfrecords_dir).glob("*.tfrecords"))
            list_files = tf.data.Dataset.list_files(files)
            dataset = tf.data.TFRecordDataset(list_files)
            dataset = dataset.batch(self._model_params.batch_size, drop_remainder=False)
            parse_sample = parse_function_record(
                dtype, self._model_params.tfrecord_shape, self._model_params.input_shape
            )
            return dataset.map(parse_sample)


        return data_generator


def predict_from_cvae(workdir: Path, weights_file, config: CVAEModelConfig, h5_files):
    params = config.dict()
    params["sim_data_dir"] = workdir.as_posix()
    params["data_dir"] = workdir.as_posix()
    params["eval_data_dir"] = workdir.as_posix()
    params["global_path"] = workdir.joinpath("files_seen.txt").as_posix()
    params["fraction"] = 0.0

    config = tf.estimator.RunConfig()
    est = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=config,
    )
    gen = est.predict(
        input_fn=produce_dataset,
        checkpoint_path=weights_file,
        yield_single_examples=True,
    )
    return np.array([list(it.values())[0] for it in gen])


def main():
    config = get_config()
    log_fname = config.md_dir.joinpath("outlier_detection.log").as_posix()
    config_logging(filename=log_fname, **config.logging.dict())
    ctx = OutlierDetectionContext(**config.dict())

    timeout_ns = config.timeout_ns
    iteration = 0

    while not os.path.exists("halt"):
        ctx.rescan_h5_dcd()

        # NOTE: It's up to the ML service to do model selection;
        # we're assuming the latest cvae weights file has the best model
        ctx.update_model()
        data_generator = ctx.update_dataset()
        cm_predict = predict_from_cvae(
            ctx.cvae_weights_file,
            data_generator,
        )

        # A record of every trajectory length
        traj_dict = dict(zip(ctx.dcd_files, train_data_length))
        print(traj_dict)

        ## Unique outliers
        print("Starting outlier searching...")
        outlier_list_ranked, _ = outliers_from_latent_loc(
            cm_predict, n_outliers=config.num_outliers, n_jobs=config.sklearn_num_cpus
        )
        print("Done outlier searching...")
        # print(outlier_list_ranked)

        # Write the outliers using MDAnalysis
        outliers_pdb_path = os.path.abspath("./outlier_pdbs")
        os.makedirs(outliers_pdb_path, exist_ok=True)
        print("Writing outliers in %s" % outliers_pdb_path)

        # identify new outliers

        new_outliers_list = []
        for outlier in outlier_list_ranked:
            # find the location of outlier
            traj_dir, num_frame = find_frame(traj_dict, outlier)
            traj_file = os.path.join(traj_dir, "output.dcd")
            # get the outlier name - traj_label + frame number
            run_name = os.path.basename(traj_dir)
            pdb_name = f"{run_name}_{num_frame:06}.pdb"
            outlier_pdb_file = os.path.join(outliers_pdb_path, pdb_name)

            new_outliers_list.append(outlier_pdb_file)
            # Only write new pdbs to reduce I/O redundancy.
            if not os.path.exists(outlier_pdb_file):
                print(f"New outlier at frame {num_frame} of {run_name}")
                outlier_pdb = write_pdb_frame(
                    traj_file, pdb_file, num_frame, outlier_pdb_file
                )

        # Clean up outdated outliers (just for bookkeeping)
        outliers_list = glob(os.path.join(outliers_pdb_path, "omm_runs*.pdb"))
        for outlier in outliers_list:
            if outlier not in new_outliers_list:
                outlier_label = os.path.basename(outlier)
                print(
                    f"Old outlier {outlier_label} is now connected to \
                    a cluster and removing it from the outlier list "
                )
                # os.rename(outlier, os.path.join(outliers_pdb_path, '-'+outlier_label))

        # Set up input configurations for next batch of MD simulations
        ### Get the pdbs used once already
        used_pdbs = glob(os.path.join(md_path, "omm_runs_*/omm_runs_*.pdb"))
        used_pdbs_labels = [os.path.basename(used_pdb) for used_pdb in used_pdbs]
        print(used_pdbs_labels)
        ### Exclude the used pdbs
        # outliers_list = glob(os.path.join(outliers_pdb_path, 'omm_runs*.pdb'))
        restart_pdbs = [
            outlier
            for outlier in new_outliers_list
            if os.path.basename(outlier) not in used_pdbs_labels
        ]
        print("restart pdbs: ", restart_pdbs)

        # rank the restart_pdbs according to their RMSD to local state
        if ref_pdb_file:
            ref_traj = mda.Universe(ref_pdb_file)
            outlier_traj = mda.Universe(restart_pdbs[0], restart_pdbs)
            R = RMSD(outlier_traj, ref_traj, select="protein and name CA")
            R.run()
            # Make a dict contains outliers and their RMSD
            # outlier_pdb_RMSD = dict(zip(restart_pdbs, R.rmsd[:,2]))
            restart_pdbs = [pdb for _, pdb in sorted(zip(R.rmsd[:, 2], restart_pdbs))]
            if np.min(R.rmsd[:, 2]) < 0.1:
                with open("../halt", "w"):
                    pass
                break

        # identify currently running MDs
        running_MDs = [md for md in omm_runs if not os.path.exists(md + "/new_pdb")]
        # decide which MD to stop, (no outliers in past 10ns/50ps = 200 frames)
        n_timeout = timeout_ns / 0.05
        for md in running_MDs:
            md_label = os.path.basename(md)
            current_frames = traj_dict[md]
            # low bound for minimal MD runs, 2 * timeout
            if current_frames > n_timeout * 2:
                current_outliers = glob(outliers_pdb_path + f"{md_label}_*pdb")
                if current_outliers != []:
                    latest_outlier = current_outliers[-1]
                    latest_frame = int(latest_outlier.split(".")[0].split("_")[-1])
                    # last 10 ns had outliers
                    if current_frames - latest_frame < n_timeout:
                        continue
                restart_pdb = os.path.abspath(restart_pdbs.pop(0))
                with open(md + "/new_pdb", "w") as fp:
                    fp.write(restart_pdb)

        print(f"=======>Iteration {iteration} done<========")
        iteration += 1

    # ## Restarts from check point
    # used_checkpnts = glob(os.path.join(md_path, 'omm_runs_*/omm_runs_*.chk'))
    # restart_checkpnts = []
    # for checkpnt in checkpnt_list:
    #     checkpnt_filepath = os.path.join(outliers_pdb_path, os.path.basename(os.path.dirname(checkpnt) + '.chk'))
    #     if not os.path.exists(checkpnt_filepath):
    #         shutil.copy2(checkpnt, checkpnt_filepath)
    #         print([os.path.basename(os.path.dirname(checkpnt)) in outlier for outlier in outliers_list])
    #         # includes only checkpoint of trajectory that contains an outlier
    #         if any(os.path.basename(os.path.dirname(checkpnt)) in outlier for outlier in outliers_list):
    #             restart_checkpnts.append(checkpnt_filepath)

    # Write record for next step
    ## 1> restarting checkpoint; 2> unused outliers (ranked); 3> used outliers (shuffled)
    # random.shuffle(used_pdbs)
    # restart_points = restart_checkpnts + restart_pdbs + used_pdbs
    # print(restart_points)

    # restart_points_filepath = os.path.abspath('./restart_points.json')
    # with open(restart_points_filepath, 'w') as restart_file:
    #     json.dump(restart_points, restart_file)


if __name__ == "__main__":
    main()
