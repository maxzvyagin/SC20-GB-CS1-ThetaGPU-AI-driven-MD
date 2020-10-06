"""
File containing input data functions used for the Covid CVAE model.
"""
import tensorflow as tf
import numpy as np
import glob

def get_real_datasets(data_dir):
    train_ds = tf.data.TFRecordDataset(f"{data_dir}/train.tfrecords")
    val_ds = tf.data.TFRecordDataset(f"{data_dir}/val.tfrecords")
    return train_ds, val_ds


def get_real_batch_map_fn(tfrecord_shape, input_shape, dtype):
    feature_description = {
        'data': tf.io.FixedLenFeature([1], tf.string),
    }
    def _map_fn(record):
        features = tf.io.parse_example(record, feature_description)
        data = tf.compat.v1.io.decode_raw(
            features['data'],
            tf.bool,
        )
        BS = data.shape.as_list()[0]
        x = tf.reshape(data, [BS]+tfrecord_shape)
        input_x = tf.slice(x, [0, 0, 0, 0], [BS]+input_shape)
        flat_input_x = tf.reshape(input_x, [BS, -1])
        return (
            tf.cast(input_x, dtype),
            tf.cast(flat_input_x, dtype)
        )
    return _map_fn


def train_input_fn(params):
    return input_fn(params, "train")


def val_input_fn(params):
    return input_fn(params, "val")


def input_fn(params, split):
    batch_size = params["batch_size"]
    input_shape = params["input_shape"]
    mp = params["mixed_precision"]
    dtype = tf.float16 if mp else tf.float32
    data_dir = params["data_dir"]
    tfrecord_shape = params["tfrecord_shape"]
    seed = params["data_random_seed"]

    train_ds, val_ds = get_real_datasets(data_dir)
    map_fn = get_real_batch_map_fn(tfrecord_shape, input_shape, dtype)

    if split == "train":
        ds = train_ds
    elif split == "val":
        ds = val_ds
        batch_size = 1
    else:
        raise ValueError(f"Invalid split: {split}")

    if split == "train":
        ds = ds.repeat()
        ds = ds.shuffle(1000, seed=seed)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

##########################################


# using npy
def parse_funct(input_shape, final_shape, dtype):
    def _parse_sample(raw_record):
        # dtype should match what was used in utils.py
        feature = tf.io.decode_raw(
            raw_record,
            tf.bool
        )
        batch_size = feature.shape.as_list()[0]
        image = tf.reshape(feature, [batch_size]+input_shape)
        act_img = tf.slice(image, [0, 0, 0, 0], [batch_size]+final_shape)
        flat_img = tf.reshape(act_img, [batch_size, -1])
        return tf.cast(act_img, dtype), tf.cast(flat_img, dtype)
    return _parse_sample


def simulation_input_fn(params):
    batch_size = params["batch_size"]
    input_shape = params["tfrecord_shape"]
    final_shape = params["input_shape"]
    itemsize = params["itemsize"]
    mp = params["mixed_precision"]
    dtype = tf.float16 if mp else tf.float32
    data_dir = params["data_dir"]
    list_files = glob.glob(f"{data_dir}/*.npy")
    assert len(list_files)> 10, f"{len(list_files)}, {data_dir}"
    dataset = tf.data.FixedLengthRecordDataset(
        list_files,
        record_bytes=np.prod(input_shape)*itemsize,
        header_bytes=128 # npy usual but can change
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)
    parse_sample = parse_funct(input_shape, final_shape, dtype)
    dataset = dataset.map(parse_sample)
    dataset = dataset.shuffle(
        10,
    )
    dataset = dataset.repeat()
    dataset=dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return dataset

# tfrecords

def parse_function_record(dtype, input_shape, final_shape):
    feature_description = {
        'image_raw': tf.io.FixedLenFeature([1], tf.string),
    }
    def _parse_record(record):
        features = tf.parse_example(
            record,
            features=feature_description
        )
        # dtype should match what was used in utils.py
        image = tf.decode_raw(features['image_raw'], tf.float16)
        batch_size = image.shape.as_list()[0]
        image = tf.reshape(image, [batch_size]+input_shape)
        act_img = tf.slice(image, [0, 0, 0, 0], [batch_size]+final_shape)
        flat_img = tf.reshape(act_img, [batch_size, -1])
        return tf.cast(act_img, dtype), tf.cast(flat_img, dtype)
    return _parse_record

def parse_function_record_predict(dtype, input_shape, final_shape):
    feature_description = {
        'image_raw': tf.io.FixedLenFeature([1], tf.string),
    }
    def _parse_record(record):
        features = tf.parse_example(
            record,
            features=feature_description
        )
        # dtype should match what was used in utils.py
        image = tf.decode_raw(features['image_raw'], tf.float16)
        batch_size = image.shape.as_list()[0]
        image = tf.reshape(image, [batch_size]+input_shape)
        act_img = tf.slice(image, [0, 0, 0, 0], [batch_size]+final_shape)
        return tf.cast(act_img, dtype)
    return _parse_record


def sort_string_numbers(filename):
    part2 = ".tfrecords"
    part1 = "_sample"
    num1 = filename.find(part1)+len(part1)
    num2 = filename.find(part2)
    return int(filename[num1:num2])

def simulation_tf_record_input_fn(params):
    batch_size = params["batch_size"]
    input_shape = params["tfrecord_shape"]
    final_shape = params["input_shape"]
    mp = params["mixed_precision"]
    dtype = tf.float16 if mp else tf.float32
    data_dir = params["data_dir"]
    last_n_files = params["last_n_files"]
    files = glob.glob(f"{data_dir}/*.tfrecords")
    last_n_files = sorted(files, key=sort_string_numbers)[:last_n_files]
    list_files = tf.data.Dataset.list_files(last_n_files)
    dataset = tf.data.TFRecordDataset(
        list_files
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)
    parse_sample = parse_function_record(dtype, input_shape, final_shape)
    dataset = dataset.map(parse_sample)
    dataset = dataset.shuffle(
        10,
    )
    dataset = dataset.repeat()
    dataset=dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return dataset

def simulation_tf_record_eval_input_fn(params):
    batch_size = params["batch_size"]
    input_shape = params["tfrecord_shape"]
    final_shape = params["input_shape"]
    mp = params["mixed_precision"]
    dtype = tf.float16 if mp else tf.float32
    data_dir = params["eval_data_dir"]
    last_n_files = params["last_n_files_eval"]
    files = glob.glob(f"{data_dir}/*.tfrecords")
    last_n_files = sorted(files, key=sort_string_numbers)[:last_n_files]
    list_files = tf.data.Dataset.list_files(last_n_files)
    dataset = tf.data.TFRecordDataset(
        list_files
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)
    parse_sample = parse_function_record(dtype, input_shape, final_shape)
    dataset = dataset.map(parse_sample)
    dataset = dataset.shuffle(
        10,
    )
    dataset = dataset.repeat()
    dataset=dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return dataset

if __name__ == "__main__":
    for i in range(2):
        params ={
                "batch_size": 4,
                "tfrecord_shape": [1, 256, 256],
                "itemsize": 1,
                "mixed_precision": False,
                "data_dir": "/cb/home/vishal/ws/ANL-shared/cvae_gb/records_1",
                "input_shape": [1, 192, 192]
            }
        if i == 0:
            ds=simulation_tf_record_input_fn(params)
        else:
            params["data_dir"] = "/cb/home/vishal/ws/ANL-shared/cvae_gb/npy_data_256"
            ds=simulation_input_fn(params)
        print(ds)
        dataset_iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
        inputs = dataset_iterator.get_next()
        print(inputs)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.tables_initializer())
            for i in range(10):
                outputs = sess.run(inputs)
                print(outputs[0].shape, outputs[1].shape)
