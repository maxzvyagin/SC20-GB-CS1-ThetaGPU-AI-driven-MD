import os
import glob
import h5py
import numpy as np


def concat_h5(workdir, outfile):
    h5_files = glob.glob(os.path.join(workdir, "*.h5"))
    # Sort files by time of creation
    h5_files = sorted(h5_files, key=lambda file: os.path.getctime(file))

    fields = ["contact_map", "rmsd"]
    data = {x: [] for x in fields}

    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as f:
            for field in fields:
                data[field] = f[field][...]

    for field in data:
        data[field] = np.concatenate(data[field])

    with h5py.File(outfile, "w") as fout:

        # Create new dsets from concatenated dataset
        for field, concat_dset in data.items():

            shape = concat_dset.shape
            chunkshape = (1,) + shape[1:]
            # Create dataset
            if concat_dset.dtype != np.object:
                if np.any(np.isnan(concat_dset)):
                    raise ValueError("NaN detected in concat_dset.")
                dtype = concat_dset.dtype
            else:
                # For sparse matrix format
                dtype = h5py.vlen_dtype(np.int16)

            fout.create_dataset(
                name=field,
                shape=shape,
                dtype=dtype,
                data=concat_dset,
                chunks=chunkshape,
            )


def cleanup_h5(workdir, keep):
    h5_files = glob.glob(os.path.join(workdir, "*.h5"))
    for fname in h5_files:
        if fname != keep:
            os.remove(fname)
