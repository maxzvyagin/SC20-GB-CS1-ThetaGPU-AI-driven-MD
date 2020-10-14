"""
Copyright 2019 Cerebras Systems.

GPU training script for the ANL GravWave model.
"""
import logging
from typing import Optional
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

from .model import model_fn
from deepdrivemd.models.symmetric_cvae.utils import write_single_tfrecord
from deepdrivemd.models.symmetric_cvae.data import parse_function_record_predict

logger = logging.getLogger(__name__)


class TFEstimatorModel:
    def __init__(self, workdir, model_params, predict_batch_size, checkpoint_dir):
        self._tfrecords_dir = workdir.joinpath("tfrecords")
        self._tfrecords_dir.mkdir(exist_ok=True)
        self._model_params = model_params
        self._predict_batch_size = predict_batch_size
        self._checkpoint_dir = checkpoint_dir

    def get_weights_file(self) -> Optional[str]:
        """
        Returns path to latest model checkpoint or None
        """
        return tf.train.latest_checkpoint(self._checkpoint_dir)

    def preprocess(self, new_h5_files: list, dcd_files: list):
        """
        Input: new_h5_files list, dcd_files_list
        Return: blackbox `input_data` object to be used by model.predict()
        """
        # tf.data.Dataset.list_files expects a list of strings, not pathlib.Path objects!
        # as_posix() converts a Path to a string
        tfrecord_files = [
            self._tfrecords_dir.joinpath(f.with_suffix(".tfrecords").name).as_posix()
            for f in dcd_files
        ]
        logger.debug(
            f"update_dataset: Will predict from tfrecord_files={tfrecord_files}"
        )

        # Write to local node storage
        for h5_file in new_h5_files:
            write_single_tfrecord(
                h5_file=h5_file,
                initial_shape=self._model_params["h5_shape"][1:],
                final_shape=self._model_params["tfrecord_shape"][1:],
                tfrecord_dir=self._tfrecords_dir,
            )

        # Use files closure to get correct data sample
        def _data_generator():
            dtype = tf.float16 if self._model_params["mixed_precision"] else tf.float32
            list_files = tf.data.Dataset.list_files(tfrecord_files)
            dataset = tf.data.TFRecordDataset(list_files)

            # TODO: We want drop_remainder=False but this needs to be rewritten:
            dataset = dataset.batch(self._predict_batch_size, drop_remainder=True)
            parse_sample = parse_function_record_predict(
                dtype,
                self._model_params["tfrecord_shape"],
                self._model_params["input_shape"],
            )
            return dataset.map(parse_sample)

        return _data_generator

    def predict(self, input_data):
        weights_file = self.get_weights_file()
        logger.info(f"start model prediction with weights: {weights_file}")
        params = self._model_params.dict()
        params["sim_data_dir"] = self._tfrecords_dir.as_posix()
        params["data_dir"] = self._tfrecords_dir.as_posix()
        params["eval_data_dir"] = self._tfrecords_dir.as_posix()
        params["global_path"] = self._tfrecords_dir.joinpath(
            "files_seen.txt"
        ).as_posix()
        params["fraction"] = 0.0
        params["batch_size"] = self._predict_batch_size

        tf_config = tf.estimator.RunConfig()
        est = tf.estimator.Estimator(model_fn, params=params, config=tf_config,)
        gen = est.predict(
            input_fn=input_data,
            checkpoint_path=weights_file,
            yield_single_examples=True,
        )
        return np.array([list(it.values())[0] for it in gen])
