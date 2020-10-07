import os
import glob
import h5py
import numpy as np
import logging

logger = logging.getLogger(__name__)


def concat_h5(workdir, outfile, fields=["contact_map"]):
    h5_files = glob.glob(os.path.join(workdir, "*.h5"))
    # Sort files by time of creation
    h5_files = list(sorted(h5_files, key=lambda file: os.path.getctime(file)))

    logger.debug(f"Collected {len(h5_files)} to concatenate")

    data = {x: [] for x in fields}

    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as f:
            for field in fields:
                try:
                    data[field].append(f[field][...])
                    logger.debug(
                        f"{os.path.dirname(h5_file)} has {field} with {len(data[field][-1])} examples"
                    )
                except KeyError:
                    raise KeyError(
                        f"Cannot access field {field}: available keys are {f.keys()}"
                    )

    for field in data:
        data[field] = np.concatenate(data[field])
        logger.debug(f"Concatenated: {field} length is {len(data[field])}")

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
    count = 0
    for fname in h5_files:
        if fname != keep:
            os.remove(fname)
            count += 1
    logger.debug(f"Cleaned up {count} .h5 files in {workdir}")
