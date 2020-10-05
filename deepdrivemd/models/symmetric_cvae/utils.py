import tensorflow as tf
import h5py
import numpy as np
from scipy.sparse import coo_matrix
import os
import glob
import yaml
import shutil
import random

#_curdir = os.path.dirname(os.path.abspath(__file__))
#DEFAULT_YAML_PATH = os.path.join(_curdir, "params.yaml")
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
        with h5py.File(h5_file_path, 'r', libver='latest', swmr=False) as f:
            for j, raw_indices in enumerate(f['contact_map']):
                # Unpack indices (stored as concatenated row, col vectors)
                indices = raw_indices.reshape((2, -1)).astype('int16')
                # Contact matrices are binary so we don't need to store the values
                # in HDF5 format. Instead we create a vector of 1s on the fly.
                values = np.ones(indices.shape[1]).astype('byte')
                # Construct COO formated sparse matrix
                contact_map = coo_matrix((values, (indices[0], indices[1])), shape=shape).todense()
                contact_map = contact_map[:256, :256]
                # Expand COO format to dense and add matrix to return array
                np.save(f"npy_data_256/data_file_{i+1}_sample{j+1}.npy", np.array(contact_map, dtype=np.float16))



def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_record(contact_maps, record_name):
    with tf.python_io.TFRecordWriter(record_name) as writer:
        for sample in contact_maps:
            feature={
                'image_raw': _bytes_feature(sample.tostring())
            }
            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)
            example_to_string = example.SerializeToString()
            writer.write(example_to_string)


def write_to_tfrecords(files, initial_shape, final_shape, num_samples, train_dir_path, eval_dir_path, fraction=0.2):
    os.makedirs(train_dir_path, exist_ok=True)
    os.makedirs(eval_dir_path, exist_ok=True)
    record_counter = len(glob.glob(f"{train_dir_path}/*.tfrecords"))
    initial_counter = record_counter
    contact_maps = []
    for i, h5_file_path in enumerate(files):
        with h5py.File(h5_file_path, 'r', libver='latest', swmr=False) as f:
            print(f.keys())
            for j, raw_indices in enumerate(f['contact_map']):
                indices = raw_indices.reshape((2, -1)).astype('int16')
                # Contact matrices are binary so we don't need to store the values
                # in HDF5 format. Instead we create a vector of 1s on the fly.
                values = np.ones(indices.shape[1]).astype('byte')
                # Construct COO formated sparse matrix
                contact_map = coo_matrix((values, (indices[0], indices[1])), shape=initial_shape).todense()
                contact_map = np.array(contact_map[:final_shape[0], :final_shape[1]], dtype=np.float16)
                contact_maps.append(contact_map)
                if len(contact_maps) == num_samples:
                    record_counter+=1
                    record_name = os.path.join(train_dir_path, f"tfrecord_{i}_sample{record_counter}.tfrecords")
                    print(record_name)
                    write_record(contact_maps, record_name)
                    contact_maps = []
    if len(contact_maps) > 0:
        record_counter+=1
        record_name = os.path.join(train_dir_path, f"tfrecord_{i}_sample{record_counter}.tfrecords")
        print(record_name)
        write_record(contact_maps, record_name)
        contact_maps = []
    print(f"total files: {record_counter}")
    files_created = record_counter - initial_counter
    # choose X% between initial_counter to record_counter to move to eval_dir_path
    randomlist = random.sample(range(initial_counter+1, record_counter+1), int(fraction*files_created))
    print(f"eval file numbers: {randomlist}")
    for n in randomlist:
        full_filename = glob.glob(f"{train_dir_path}/*_sample{n}.tfrecords")[0]
        file_name = full_filename[full_filename.find(f"{train_dir_path}/")+len(f"{train_dir_path}/"):]
        shutil.move(full_filename, f"{eval_dir_path}/{file_name}")



if __name__ == "__main__":
    # preprocess_h5files(
    #     ["/cb/home/vishal/ws/toy_data/ras1_prot.h5"],
    #     [370, 370]
    # )

    write_to_tfrecords(
        files=["/data/shared/vishal/toy_data/ras1_prot.h5"],
        initial_shape=[370, 370],
        final_shape=[256, 256],
        num_samples=1024,
        train_dir_path="records_256",
        eval_dir_path="eval_records_256"
    )
