"""
Copyright 2019 Cerebras Systems.

GPU training script for the ANL GravWave model.
"""

import argparse
import json
import os
import tensorflow as tf
import yaml

from model import model_fn
from data import val_input_fn, train_input_fn, simulation_input_fn, simulation_tf_record_input_fn
from utils import get_params

######## HELPER FUNCTIONS ###########

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        default="./params.yaml",
        help="Path to .yaml file with model parameters"
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory to save / load model from"
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=2,
        help="Directory to save / load model from"
    )
    parser.add_argument(
        "--mode",
        choices=[
            "train",
            "eval"
        ],
        default="train",
        help="Can choose from train or eval. Defaults to train."
    )

    return parser.parse_args()


###################### MAIN #####################

def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    # SET UP
    args = parse_args()

    # Read and maybe update params
    params = get_params(args.params)
    cmd_params = list(vars(args).items())
    params.update({
        k: v for (k,v) in cmd_params
        if (k in params) and (v is not None)
    })

    # RUN
    config = tf.estimator.RunConfig(
        **params["runconfig_params"],
    )
    est = tf.estimator.Estimator(
        model_fn,
        params=params,
        model_dir=params["model_dir"],
        config=config,
    )

    if params["mode"]  == "train":
        print("train_steps:", params["train_steps"])
        est.train(input_fn=simulation_tf_record_input_fn, steps=params["train_steps"])
    elif params["mode"]  == "eval":
        est.evaluate(input_fn=val_input_fn, name='validation')
    else:
        raise Exception(f"Unknown mode '{params['mode']}'")


if __name__ == '__main__':
    main()
