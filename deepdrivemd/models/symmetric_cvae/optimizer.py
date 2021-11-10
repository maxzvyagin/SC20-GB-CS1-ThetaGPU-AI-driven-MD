"""
Module to set optimizer
"""
import tensorflow as tf
import horovod.tensorflow as hvd

_ALLOWED_OPTIMIZERS = ["sgd", "sgdm", "adam", "rmsprop"]


def get_optimizer(params):
    lr = params["learning_rate"] * hvd.size()
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
                learning_rate=lr,
                momentum=params["momentum"],
                name="sgd_momentum",
            )
        elif optimizer_name == "sgd":
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(
                learning_rate=lr,
                name="sgd",
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
    return hvd.DistributedOptimizer(optimizer)


class MixedPrecisionLossScaleOptimizerAdapter(
    loss_scale_optimizer.MixedPrecisionLossScaleOptimizer
):
    def __init__(self, opt, loss_scale):

        super(MixedPrecisionLossScaleOptimizerAdapter, self).__init__(
            opt, loss_scale
        )

    @property
    def loss_scale(self):
        return self._loss_scale()

    def _scale_loss(self, loss):
        loss_scale = self._loss_scale()
        if callable(loss):

            def new_loss():
                loss_val = loss()
                return loss_val * math_ops.cast(loss_scale, loss_val.dtype)

            return new_loss
        else:
            return loss * math_ops.cast(loss_scale, loss.dtype)

    def _unscale_grads(self, grads):
        loss_scale = self._loss_scale()
        loss_scale_reciprocal = 1 / loss_scale
        return [
            None if g is None else self._scale_grad(g, loss_scale_reciprocal)
            for g in grads
        ]

    def _scale_grad(self, grad, loss_scale_reciprocal):
        if isinstance(grad, ops.IndexedSlices):
            grad_vals = (
                math_ops.cast(grad.values, tf.float32) * loss_scale_reciprocal
            )
            return ops.IndexedSlices(grad_vals, grad.indices, grad.dense_shape)
        return math_ops.cast(grad, tf.float32) * loss_scale_reciprocal


def wrap_optimizer(opt, loss_scale):
    return MixedPrecisionLossScaleOptimizerAdapter(opt, loss_scale)