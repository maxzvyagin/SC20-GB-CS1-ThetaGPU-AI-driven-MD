from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time
import yaml
import random
import shutil
import argparse
import queue
import numpy as np
from glob import glob
import MDAnalysis as mda
from deepdrivemd.util import FileLock
from deepdrivemd.driver.config import OutlierDetectionConfig
from deepdrivemd.outlier.utils import predict_from_cvae, outliers_from_latent_loc
from deepdrivemd.outlier.utils import outliers_largeset
from deepdrivemd.outlier.utils import find_frame, write_pdb_frame
from MDAnalysis.analysis.rms import RMSD


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
        md_dir,
        cvae_dir,
        **kwargs,
    ):
        self.pdb_file = Path(pdb_file).resolve()
        self.reference_pdb_file = Path(reference_pdb_file).resolve()
        self.md_dir = Path(md_dir).resolve()
        self.cvae_dir = Path(cvae_dir).resolve()

        self._cvae_weights_file = None
        self._last_acquired_cvae_lock = None
        self._md_input_dirs = None
        self._h5_dcd_map = {}
        self._pdb_outlier_queue = queue.PriorityQueue()

    def push_new_outlier(self, pdb_filename, outlier_score):
        self._pdb_outlier_queue.put_nowait((outlier_score, pdb_filename))

    def get_outlier_pdb_filename(self):
        try:
            _, pdb_filename = self._pdb_outlier_queue.get(block=False)
        except queue.Empty:
            return None
        else:
            return pdb_filename

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

    def get_open_md_input_slot(self, pdb_file):
        for input_dir in self.md_input_dirs:
            if 

    def update_model(self):
        """Gets most recent model weights."""
        all_weights = self.cvae_dir.glob("*.h5")
        # Sort files by time of creation
        all_weights = sorted(all_weights, key=lambda file: os.path.getctime(file))
        self._cvae_weights_file = all_weights[-1]
        # TODO: Make sure to use FileLock before loading weights!


def main():
    config = get_config()
    ctx = OutlierDetectionContext(**config.dict())

    timeout_ns = config.timeout_ns
    iteration = 0

    while not os.path.exists("halt"):
        ctx.rescan_h5_dcd()

        # NOTE: It's up to the ML service to do model selection;
        # we're assuming the latest cvae weights file has the best model
        ctx.update_model()

        # Get the predicted embeddings
        cm_predict, train_data_length = predict_from_cvae(
            ctx.cvae_weights_file,
            config.model_params,  # This is a CVAEModelConfig object
            ctx.h5_files,
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
