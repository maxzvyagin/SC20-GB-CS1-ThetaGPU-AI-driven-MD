import argparse
import logging
from pathlib import Path
import shutil
import time

import tensorflow as tf
import horovod.tensorflow as hvd

from deepdrivemd.models.symmetric_cvae.model import model_fn
from deepdrivemd.models.symmetric_cvae.data import simulation_tf_record_input_fn
from deepdrivemd.models.symmetric_cvae.prepare_dataset import update_dataset
from deepdrivemd.driver.config import GPUTrainingRunConfig
from deepdrivemd import config_logging


logger = None


def main(params: GPUTrainingRunConfig):
    global logger
    hvd.init()
    log_fname = params.checkpoint_path.parent.joinpath("training.log").as_posix()
    if hvd.rank() == 0:
        config_logging(filename=log_fname, **params.logging.dict())
    logger = logging.getLogger("deepdrivemd.models.symmetric_cvae.train_gpu")

    if params.initial_weights_dir:
        warm_start_from = tf.train.latest_checkpoint(params.initial_weights_dir)
    else:
        warm_start_from = None

    params.fraction = 0
    logger.info("start create tf estimator")
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.gpu_options.visible_device_list = str(hvd.local_rank())
    tf_config = tf.estimator.RunConfig(
        session_config=session_config, **params.runconfig_params
    )
    if hvd.rank() == 0:
        local_checkpoint_path = params.scratch_dir.joinpath("train_checkpoints")
        local_checkpoint_path.mkdir(parents=True)
    else:
        local_checkpoint_path = None

    est = tf.estimator.Estimator(
        model_fn,
        model_dir=local_checkpoint_path.as_posix() if local_checkpoint_path else None,
        params=params.dict(),
        config=tf_config,
        warm_start_from=warm_start_from,
    )
    bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
    logger.info("end create tf estimator")

    seen_h5_files = set()
    while True:
        all_h5_files = set(params.sim_data_dir.glob("*.h5"))
        new_h5_files = all_h5_files.difference(seen_h5_files)
        if len(new_h5_files) < params.num_h5s_per_training:
            logger.info(
                f"Got {len(new_h5_files)} out of {params.num_h5s_per_training} new H5s "
                f"needed for training.  Sleeping..."
            )
            time.sleep(60)
            continue
        seen_h5_files.update(new_h5_files)
        logger.info("start update_dataset")
        update_dataset(params.dict())
        logger.info("end update_dataset")

        logger.info(f"start train (steps={params.train_steps})")
        est.train(
            input_fn=simulation_tf_record_input_fn,
            steps=params.train_steps,
            hooks=[bcast_hook],
        )
        logger.info(f"end train")

        if hvd.rank() == 0:
            for fname in local_checkpoint_path.glob("*"):
                dest = shutil.copy(fname, params.checkpoint_path)
                logger.info(f"Copied file: {dest}")


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", required=True)
    yml_file = parser.parse_args().config_file
    yml_file = Path(yml_file).resolve()
    params = GPUTrainingRunConfig.from_yaml(yml_file)
    main(params)
