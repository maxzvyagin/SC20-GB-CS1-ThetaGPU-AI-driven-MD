import os 

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
