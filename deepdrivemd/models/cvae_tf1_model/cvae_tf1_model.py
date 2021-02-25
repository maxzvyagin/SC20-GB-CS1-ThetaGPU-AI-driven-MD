from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np

from deepdrivemd.models import BaseModel, ModelConfig
from .model import model_fn


class CvaeTF1ModelConfig(ModelConfig):
    h5_shape: Tuple[int, ...]
    tfrecord_shape: Tuple[int, ...]


class CvaeTF1Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.current_tfrecord_batch = None

    @property
    def _tfrecords_dir(self):
        return self.config.ssd_path.joinpath("tfrecords")

    @property
    def _checkpoint_dir(self):
        return self.config.checkpoint_path

    @staticmethod
    def write_single_tfrecord(h5_file, initial_shape, final_shape, tfrecord_dir):
        """
        Special-case: 1-to-1 mapping of h5_file to .tfrecords for lof.py
        Not needed on CS1
        """
        tfrecord_dir = Path(tfrecord_dir)
        record_name = tfrecord_dir.joinpath(
            Path(h5_file).with_suffix(".tfrecords").name
        ).as_posix()
        contact_maps = []
        with h5py.File(h5_file, "r", libver="latest", swmr=False) as f:
            for j, raw_indices in enumerate(f["contact_map"]):
                indices = raw_indices.reshape((2, -1)).astype("int16")
                # Contact matrices are binary so we don't need to store the values
                # in HDF5 format. Instead we create a vector of 1s on the fly.
                values = np.ones(indices.shape[1]).astype("byte")
                # Construct COO formated sparse matrix
                contact_map = coo_matrix(
                    (values, (indices[0], indices[1])), shape=initial_shape
                ).todense()
                contact_map = np.array(
                    contact_map[: final_shape[0], : final_shape[1]], dtype=np.float16
                )
                contact_maps.append(contact_map)
            write_record(contact_maps, record_name)
            logger.debug(f"Wrote TFRecord: {record_name}")

    def preprocess(self, h5_files: List[Path]) -> None:
        """
        Input: h5_files list, dcd_files_list
        Return: blackbox `input_data` object to be used by model.predict()
        """
        # tf.data.Dataset.list_files expects a list of strings, not pathlib.Path objects!
        # as_posix() converts a Path to a string
        existing_tfrecord_files = set(self._tf_records_dir.glob("*.tfrecords"))
        tfrecord_h5_map = {
            self._tfrecords_dir.joinpath(
                h5_file.with_suffix(".tfrecords").name
            ): h5_file
            for h5_file in h5_files
        }
        target_tfrecord_files = set(tfrecord_h5_map.keys())
        missing_tfrecord_files = target_tfrecord_files - existing_tfrecord_files

        # Write to local node storage
        for tf_record in missing_tfrecord_files:
            self.write_single_tfrecord(
                h5_file=tfrecord_h5_map[tf_record].as_posix(),
                initial_shape=self.config.h5_shape[1:],
                final_shape=self.config.tfrecord_shape[1:],
                tfrecord_dir=self.config._tfrecords_dir,
            )

        self.current_tfrecord_batch = target_tfrecord_files

    def tf_estimator_input_fn(self):
        dtype = tf.float16 if self.config.mixed_precision else tf.float32
        list_files = tf.data.Dataset.list_files(self.current_tfrecord_batch)
        dataset = tf.data.TFRecordDataset(list_files)

        # TODO: We want drop_remainder=False but this needs to be rewritten:
        dataset = dataset.batch(self._predict_batch_size, drop_remainder=True)
        parse_sample = parse_function_record_predict(
            dtype,
            self.config.tfrecord_shape,
            self.config.input_shape,
        )
        return dataset.map(parse_sample)

    def predict(self) -> np.ndarray:
        if self.current_tfrecord_batch is None:
            raise RuntimeError(
                "self.current_tfrecord_batch has not been set yet. Need to call preprocess first"
            )

        weights_file = self.get_weights_file()
        logger.info(f"start model prediction with weights: {weights_file}")
        params = self.config.dict()
        params["sim_data_dir"] = self._tfrecords_dir.as_posix()
        params["data_dir"] = self._tfrecords_dir.as_posix()
        params["eval_data_dir"] = self._tfrecords_dir.as_posix()
        params["global_path"] = self._tfrecords_dir.joinpath(
            "files_seen.txt"
        ).as_posix()
        params["fraction"] = 0.0
        params["batch_size"] = self._predict_batch_size

        tf_config = tf.estimator.RunConfig()
        est = tf.estimator.Estimator(
            model_fn,
            params=params,
            config=tf_config,
        )
        gen = est.predict(
            input_fn=self.tf_estimator_input_fn,
            checkpoint_path=weights_file,
            yield_single_examples=True,
        )
        result = np.array([list(it.values())[0] for it in gen])
        self.current_tfrecord_batch = None
        return result

    def get_weights_file(self) -> Optional[str]:
        """
        Returns path to latest model checkpoint or None
        """
        return tf.train.latest_checkpoint(self._checkpoint_dir)
