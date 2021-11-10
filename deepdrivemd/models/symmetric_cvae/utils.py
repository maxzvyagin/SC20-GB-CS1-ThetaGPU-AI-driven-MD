import tensorflow as tf
from pathlib import Path
import h5py
import numpy as np
from scipy.sparse import coo_matrix
import logging
import os
import glob
# import yaml
import shutil
import random
from tqdm import tqdm

logger = logging.getLogger(__name__)

# _curdir = os.path.dirname(os.path.abspath(__file__))
# DEFAULT_YAML_PATH = os.path.join(_curdir, "params.yaml")
DEFAULT_YAML_PATH = "params.yaml"


def get_params(params_file=DEFAULT_YAML_PATH):
    """
    Return params dict from yaml file.
    """
    with open(params_file, "r") as stream:
        params = yaml.safe_load(stream)
    return params


def preprocess_h5files(files, shape):
    os.makedirs("./npy_data_256", exist_ok=True)
    for i, h5_file_path in enumerate(files):
        with h5py.File(h5_file_path, "r", libver="latest", swmr=False) as f:
            for j, raw_indices in enumerate(f["contact_map"]):
                # Unpack indices (stored as concatenated row, col vectors)
                indices = raw_indices.reshape((2, -1)).astype("int16")
                # Contact matrices are binary so we don't need to store the values
                # in HDF5 format. Instead we create a vector of 1s on the fly.
                values = np.ones(indices.shape[1]).astype("byte")
                # Construct COO formated sparse matrix
                contact_map = coo_matrix(
                    (values, (indices[0], indices[1])), shape=shape
                ).todense()
                contact_map = contact_map[:256, :256]
                # Expand COO format to dense and add matrix to return array
                np.save(
                    f"npy_data_256/data_file_{i+1}_sample{j+1}.npy",
                    np.array(contact_map, dtype=np.float16),
                )


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_record(contact_maps, record_name):
    with tf.io.TFRecordWriter(record_name) as writer:
        for sample in contact_maps:
            feature = {"image_raw": _bytes_feature(sample.tobytes())}
            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)
            example_to_string = example.SerializeToString()
            writer.write(example_to_string)


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


def write_to_tfrecords(
    files,
    initial_shape,
    final_shape,
    num_samples,
    train_dir_path,
    eval_dir_path,
    fraction=0.2,
    padding=False
):
    os.makedirs(train_dir_path, exist_ok=True)
    os.makedirs(eval_dir_path, exist_ok=True)
    record_counter = len(glob.glob(f"{train_dir_path}/*.tfrecords"))
    initial_counter = record_counter
    contact_maps = []
    for i, h5_file_path in enumerate(files):
        with h5py.File(h5_file_path, "r", libver="latest", swmr=False) as f:
            for j, raw_indices in tqdm(enumerate(f["contact_map"])):
                indices = raw_indices.reshape((2, -1)).astype("int16")
                # Contact matrices are binary so we don't need to store the values
                # in HDF5 format. Instead we create a vector of 1s on the fly.
                values = np.ones(indices.shape[1]).astype("byte")
                # Construct COO formated sparse matrix
                contact_map = coo_matrix(
                    (values, (indices[0], indices[1])), shape=initial_shape
                ).todense()
                if padding:
                    contact_map = np.array(
                        contact_map[: final_shape[0], : final_shape[1]], dtype=bool
                    )
                    pad_size = (final_shape[0] - initial_shape[0]) // 2
                    contact_map = np.pad(contact_map, 1, mode="constant")
                    print(contact_map.shape)
                    # print(np.size(contact_map)-np.count_nonzero(contact_map))
                else:
                    contact_map = np.array(
                        contact_map[: final_shape[0], : final_shape[1]], dtype=bool
                    )
                contact_maps.append(contact_map)
                if len(contact_maps) == num_samples:
                    record_counter += 1
                    record_name = os.path.join(
                        train_dir_path,
                        f"tfrecord_{i}_sample{record_counter}.tfrecords",
                    )
                    logger.debug(f"Wrote TFRecord: {record_name}")
                    write_record(contact_maps, record_name)
                    contact_maps = []
    if len(contact_maps) > 0:
        record_counter += 1
        record_name = os.path.join(
            train_dir_path, f"tfrecord_{i}_sample{record_counter}.tfrecords"
        )
        write_record(contact_maps, record_name)
        contact_maps = []
    files_created = record_counter - initial_counter
    # choose X% between initial_counter to record_counter to move to eval_dir_path
    randomlist = random.sample(
        range(initial_counter + 1, record_counter + 1), int(fraction * files_created)
    )
    for n in randomlist:
        full_filename = glob.glob(f"{train_dir_path}/*_sample{n}.tfrecords")[0]
        file_name = full_filename[
            full_filename.find(f"{train_dir_path}/") + len(f"{train_dir_path}/") :
        ]
        shutil.move(full_filename, f"{eval_dir_path}/{file_name}")


if __name__ == "__main__":
    # preprocess_h5files(
    #     ["/cb/home/vishal/ws/toy_data/ras1_prot.h5"],
    #     [370, 370]
    # )

    files = glob.glob('/Users/mzvyagin/Documents/gordon_bell/7egq_h5_files/anda_newsim_7egq_segmentA/*')

    write_to_tfrecords(
        # files=["/Users/mzvyagin/Documents/gordon_bell/7egq_h5_files/run_7egq_0_openmm_segmentA.h5"],
        files=files,
        initial_shape=[926, 926],
        final_shape=[926, 926],
        padding=False,
        num_samples=4096,
        train_dir_path="/Users/mzvyagin/Documents/gordon_bell/7egq_h5_files/sambanova_science_segmentA_926res",
        eval_dir_path="/Users/mzvyagin/Documents/gordon_bell/7egq_h5_files/sambanova_science_segmentA_926res_eval",
    )
