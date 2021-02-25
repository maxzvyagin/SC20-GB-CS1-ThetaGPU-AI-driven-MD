"""
Copyright 2019 Cerebras Systems.

CS-1 training script for the ANL GravWave model.
"""


import argparse
import json
import os
import tensorflow as tf
import yaml

from .model import model_fn
from .data import (
    train_input_fn,
    val_input_fn,
    simulation_tf_record_input_fn,
    simulation_tf_record_eval_input_fn,
)
from .utils import get_params

from cerebras.tf.cs_estimator import CerebrasEstimator
from cerebras.tf.run_config import CSRunConfig
from cerebras.tf.cs_slurm_cluster_resolver import CSSlurmClusterResolver

######## HELPER FUNCTIONS ###########


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        default="./params.yaml",
        help="Path to .yaml file with model parameters",
    )
    parser.add_argument(
        "--model-dir", default=None, help="Directory to save / load model from"
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=None,
        help="Directory to save / load model from",
    )
    parser.add_argument(
        "--cs-ip", default=None, help="CS-1 IP address, defaults to None"
    )
    parser.add_argument(
        "--mode",
        choices=["validate_only", "compile_only", "train", "eval"],
        default=None,
        help=(
            "Can choose from validate_only, compile_only, train "
            + "or eval. Defaults to validate_only."
            + "  Validate only will only go up to kernel matching."
            + "  Compile only continues through and generate compiled"
            + "  executables."
            + "  Train will compile and train if on CS-1,"
            + "  and just train locally (CPU/GPU) if not on CS-1."
            + "  Eval will run eval locally."
        ),
    )
    return parser.parse_args()


def check_env(args):
    """
    Perform basic checks for parameters and env
    """
    if args.cs_ip is not None and args.mode != "train":
        tf.compat.v1.logging.warn("No need to specify CS-1 IP if not training")
    return


def prep_env(use_cs1, port_base=23111):
    """
    Prepare environment for SLURM distributed training
    """
    if use_cs1:
        slurm_cluster_resolver = CSSlurmClusterResolver(port_base=port_base)
        cluster_spec = slurm_cluster_resolver.cluster_spec()
        task_type, task_id = slurm_cluster_resolver.get_task_info()
        os.environ["TF_CONFIG"] = json.dumps(
            {
                "cluster": cluster_spec.as_dict(),
                "task": {"type": task_type, "index": task_id},
            }
        )
    return


###################### MAIN #####################


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    # SET UP
    args = parse_args()
    use_cs1 = args.mode == "train" and args.cs_ip is not None
    cs_ip = args.cs_ip + ":9000" if use_cs1 else None
    check_env(args)

    # Read and maybe update params
    params = get_params()
    cmd_params = list(vars(args).items())
    params.update({k: v for (k, v) in cmd_params if (k in params) and (v is not None)})
    if params["mode"] != "eval":
        params["mixed_precision"] = True
        params["metrics"] = False
    else:
        params["mixed_precision"] = False
        params["metrics"] = True

    # RUN
    prep_env(use_cs1)
    config = CSRunConfig(
        cs_ip=cs_ip,
        **params["runconfig_params"],
    )
    est = CerebrasEstimator(
        model_fn,
        params=params,
        model_dir=params["model_dir"],
        use_cs=use_cs1,
        config=config,
    )

    if params["mode"] == "train":
        est.train(input_fn=simulation_tf_record_input_fn, steps=params["train_steps"])
    elif params["mode"] == "eval":
        est.evaluate(
            input_fn=simulation_tf_record_eval_input_fn,
            steps=params["eval_steps"],
            name="validation",
        )
    else:
        est.compile(train_input_fn, validate_only=(params["mode"] == "validate_only"))


if __name__ == "__main__":
    main()
