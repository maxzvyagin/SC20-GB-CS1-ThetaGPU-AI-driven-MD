
import glob
import os
from utils import write_to_tfrecords, get_params

def update_dataset(params):
    sim_data_dir = params["sim_data_dir"]
    train_data_dir = params["data_dir"]
    eval_data_dir = params["eval_data_dir"]
    files_seen_file = params["global_path"]
    tfrecord_shape = params["tfrecord_shape"]
    h5_shape = params["h5_shape"]
    fraction = params["fraction"]
    num_samples = params["samples_per_file"]
    if not os.path.exists(files_seen_file):
        with open(files_seen_file, "w") as f:
            pass

    with open(files_seen_file, "r") as f:
        files_seen = f.readlines()
    files_seen = [f.split()[0] for f in files_seen]
    print(files_seen)
    h5_files = glob.glob(f"{sim_data_dir}/*.h5")
    h5_files = list(set(h5_files)-set(files_seen))
    write_to_tfrecords(
            files=h5_files,
            initial_shape=h5_shape[1:],
            final_shape=tfrecord_shape[1:],
            num_samples=num_samples,
            train_dir_path=train_data_dir,
            eval_dir_path=eval_data_dir,
            fraction=fraction
        )
    with open(files_seen_file, "a") as f:
        for file_seen in h5_files:
            f.writelines(f"{file_seen}\n")


params = get_params()
update_dataset(params)
print("update done")
