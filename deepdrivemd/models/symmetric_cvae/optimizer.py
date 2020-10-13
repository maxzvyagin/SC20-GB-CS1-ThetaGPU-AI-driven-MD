"""
Module to set optimizer
"""
import tensorflow as tf

_ALLOWED_OPTIMIZERS = ["sgd", "sgdm", "adam", "rmsprop"]


def get_optimizer(params):
    lr = params["learning_rate"]
    optimizer_name = params["optimizer_name"]
    if optimizer_name in _ALLOWED_OPTIMIZERS:
        if optimizer_name == "adam":
            optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=lr,
                beta1=params["beta1"],
                beta2=params["beta2"],
                epsilon=params["epsilon"],
                name="adam",
            )
        elif optimizer_name == "rmsprop":
            optimizer = tf.compat.v1.train.RMSPropOptimizer(
                learning_rate=lr,
                decay=params["decay"],
                epsilon=params["epsilon"],
                name="rmsprop",
            )
        elif optimizer_name == "sgdm":
            optimizer = tf.compat.v1.train.MomentumOptimizer(
                learning_rate=lr, momentum=params["momentum"], name="sgd_momentum",
            )
        elif optimizer_name == "sgd":
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(
                learning_rate=lr, name="sgd",
            )
        else:
            raise AssertionError(
                f"Optimizer is in allowed list {_ALLOWED_OPTIMIZERS},"
                f"but not defined, passed {optimizer_name}"
            )
    else:
        raise AssertionError(
            f"Supported optimizer are {_ALLOWED_OPTIMIZERS}," f"passed {optimizer_name}"
        )
    return optimizer
