"""
File containing the model function and parameterizations for the
Covid CVAE model.
"""
import tensorflow as tf
from .layers import build_model
from .optimizer import get_optimizer

_REDUCTION_TYPES = ["sum", "mean"]


def model_fn(features, labels, mode, params):
    targets = labels
    loss = None
    train_op = None

    mixed_precision = params["mixed_precision"]
    fp_loss = params["full_precision_loss"]
    recon_loss_red_type = params["reconstruction_loss_reduction_type"]
    assert (
        recon_loss_red_type in _REDUCTION_TYPES
    ), f"invalid reconstruction loss reduction type: {recon_loss_red_type}"
    model_random_seed = params["model_random_seed"]
    loss_scale = params["loss_scale"]
    tf.compat.v1.set_random_seed(model_random_seed)

    training_hooks = []
    eval_metric_ops = {}
    log_metrics = params["metrics"]
    global_step = tf.compat.v1.train.get_global_step()

    if mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

    # Model output
    outputs, kl_loss, embedding_output = build_model(features, params)
    tf.compat.v1.logging.info(f"Model Outputs Shape: {outputs.get_shape()}")
    # tf.compat.v1.logging.info(f"Targets Shape: {targets.get_shape()}")

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions={"embeddings": embedding_output}
        )

    # Losses
    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        # Cast outputs, targets to FP32 if full_precision_reconstruction_loss
        if fp_loss:
            targets = tf.cast(targets, tf.float32)
            outputs = tf.cast(outputs, tf.float32)

        # Binary cross entropy loss
        bce_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
            targets,
            outputs,
            loss_collection=None,
            reduction=tf.compat.v1.losses.Reduction.NONE,
        )

        print("KL Loss: {}".format(kl_loss))
        print("BCE Loss: {}".format(bce_loss))

        if recon_loss_red_type == "sum":
            # Sum across elements
            bce_loss = tf.reduce_sum(bce_loss, axis=1)
        else:
            # Average across elements
            bce_loss = tf.reduce_mean(bce_loss, axis=1)
        # Average across batch
        bce_loss = tf.reduce_mean(bce_loss)

        # Add BCE + KL
        # tf.reduce_mean is currently needed for stack support.
        # Both losses are scalar so tf.reduce_mean does not impact the model.
        loss = tf.reduce_mean(bce_loss + kl_loss)

        if log_metrics:
            for name, tensor in [
                ("variational_losses/bce", bce_loss),
                ("variational_losses/kl", kl_loss),
            ]:
                tf.compat.v1.summary.scalar(name, tensor)
                eval_metric_ops[name] = tf.compat.v1.metrics.mean(tensor)

    # Optimizer
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Choose the right optimizer
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(tf.compat.v1.train.AdamOptimizer(
                learning_rate=params["lr"],
                beta1=params["beta1"],
                beta2=params["beta2"],
                epsilon=params["epsilon"],
                name="adam",
            ))

        # Apply loss scaling
        scaled_grads_vars = optimizer.compute_gradients(
            loss * tf.cast(loss_scale, dtype=loss.dtype),
            tf.compat.v1.trainable_variables(),
        )
        unscaled_grads_vars = [(g / loss_scale, v) for g, v in scaled_grads_vars]

        # Minimize the loss
        train_op = optimizer.apply_gradients(
            unscaled_grads_vars, global_step=global_step
        )

    logging_hook = tf.estimator.LoggingTensorHook([loss], every_n_iter=10, at_end=True)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=outputs,
        loss=loss,
        train_op=train_op,
        training_chief_hooks=[logging_hook],
        training_hooks=training_hooks,
        eval_metric_ops=eval_metric_ops,
    )
