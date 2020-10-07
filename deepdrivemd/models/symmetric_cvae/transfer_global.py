import subprocess
import os
import sys
import yaml
from pathlib import Path

DEFAULT_YAML_PATH = "params.yaml"


def get_params(params_file=DEFAULT_YAML_PATH):
    """
    Return params dict from yaml file.
    """
    with open(params_file, "r") as stream:
        params = yaml.safe_load(stream)
    return params


medulla_endpoint = "5b66ac62-042a-11eb-8930-0a5521ff3f4b"
theta_endpoint = "08925f04-569f-11e7-bef8-22000b9a448b"


def transfer_files(file_list, target_dir):
    cmd = f"globus transfer --encrypt --batch {medulla_endpoint} {theta_endpoint}"
    transfers = "\n".join(
        f"{src} {os.path.join(target_dir, os.path.basename(src))}" for src in file_list
    )
    print("Doing transfers:\n", transfers)
    proc = subprocess.Popen(
        cmd, executable="/bin/bash", stdin=subprocess.PIPE, shell=True
    )
    proc.communicate(transfers.encode("utf-8"))


if __name__ == "__main__":
    params = get_params()
    model_dir = Path(params["model_dir"]).resolve()
    theta_gpu_path = params["theta_gpu_path"]
    with open(Path(model_dir).joinpath("checkpoint")) as fp:
        line = fp.readline()
    pattern = line[line.find('"') + 1 : -2] + "*"
    file_list = list(Path(model_dir).glob(pattern)) + [
        Path(model_dir).joinpath("checkpoint")
    ] + [Path(model_dir).joinpath("performance.json")]
    target_dir = theta_gpu_path  # dir on theta
    transfer_files(file_list, target_dir)
