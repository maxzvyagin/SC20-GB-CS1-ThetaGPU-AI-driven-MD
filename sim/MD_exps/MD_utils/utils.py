import os
import glob
import numpy as np

def create_md_path(label, h5_dir='h5_dir'): 
    """
    create MD simulation path based on its label (int), 
    and automatically update label if path exists. 
    """
    md_path = f'omm_runs_{label}'

    try:
        os.mkdir(md_path)
        os.mkdir(os.path.join(md_path, h5_dir))
        return md_path
    except: 
        return create_md_path(label + 1, h5_dir)

def concat_h5(md_path, outfile):
    h5_files = glob.glob(os.path.join(md_path, '*.h5'))
    h5_files = sorted(h5_files, key=lambda file: os.path.getctime(file))

    fields = ['contact_map', 'rmsd']
    data = {x: [] for x in fields}

    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            for field in fields:
                data[field] = f[field][...]

    for field in data:
        data[field] = np.concatenate(data[field])

    with h5py.File(outfile, 'w') as fout:

        # Create new dsets from concatenated dataset
        for field, concat_dset in data.items():

            shape = concat_dset.shape
            chunkshape = ((1,) + shape[1:])
            # Create dataset
            if concat_dset.dtype != np.object:
                if np.any(np.isnan(concat_dset)):
                    raise ValueError('NaN detected in concat_dset.')
                dtype = concat_dset.dtype
            else:
                # For sparse matrix format
                dtype=h5py.vlen_dtype(np.int16)

            fout.create_dataset(
                name=field,
                shape=shape,
                dtype=dtype,
                data=concat_dset,
                chunks=chunkshape
            )




