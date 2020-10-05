import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
_REDUCTION_TYPES = ["sum", "mean"]

def conv2d(
        inputs,
        filters,
        kernel_size,
        padding,
        dilation_rate=(1, 1),
        strides=(1, 1),
        activation="relu",
        name="conv",
        use_bias=True,
):
    if strides != (1, 1) and dilation_rate != (1, 1):
        raise ValueError(
            f"both strides and dilation should not be specified as greater than 1. \
            strides set to {strides} and dilation_rate set to {dilation_rate}"
        )

    net = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        dilation_rate=dilation_rate,
        data_format="channels_first",
        activation=activation,
        name=name,
        use_bias=use_bias,
    )(inputs)

    tf.compat.v1.logging.info(f"Shape net after {net.name}: {net.get_shape()}")
    return net


def deconv2d(
        inputs,
        filters,
        kernel_size,
        padding,
        dilation_rate=(1, 1),
        strides=(1, 1),
        activation="relu",
        name="deconv",
        use_bias=True,
):
    if strides != (1, 1) and dilation_rate != (1, 1):
        raise ValueError(
            f"both strides and dilation should not be specified as greater than 1. \
            strides set to {strides} and dilation_rate set to {dilation_rate}"
        )

    net = tf.keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        dilation_rate=dilation_rate,
        data_format="channels_first",
        activation=activation,
        name=name,
        use_bias=use_bias,
    )(inputs)

    tf.compat.v1.logging.info(f"Shape net after {net.name}: {net.get_shape()}")
    return net


def dense(
        inputs,
        units,
        activation,
        name="dense",
        use_bias=True,
):
    net = tf.keras.layers.Dense(
        name=name,
        units=units,
        activation=activation,
        use_bias=use_bias,
    )(inputs)

    tf.compat.v1.logging.info(f"Shape net after {net.name}: {net.get_shape()}")
    return net


def variational_embedding(
    inputs,
    latent_ndim,
    kl_loss_reduction_type="sum",
    fp_loss=False,
    name="embedding",
):
    mean = dense(
        inputs=inputs,
        units=latent_ndim,
        activation=None,
        name="enc_dense_mean",
    )

    logvar = dense(
        inputs=inputs,
        units=latent_ndim,
        activation=None,
        name="enc_dense_logvar",
    )

    # Sample embedding
    eps = tf.random.normal(shape=mean.shape, dtype=mean.dtype, name="epsilon")
    net = mean + K.exp(0.5 * logvar) * eps
    net = tf.identity(net, name=f"{name}")
    tf.compat.v1.logging.info(f"Shape net after {net.name}: {net.get_shape()}")

    # KL loss
    if kl_loss_reduction_type == "sum":
        reduce_op = tf.reduce_sum
    else:
        reduce_op = tf.reduce_mean
    if fp_loss:
        mean = tf.cast(mean, tf.float32)
        logvar = tf.cast(logvar, tf.float32)
    # Reduce along latent_ndim
    kl_loss = -0.5 * reduce_op(
        (1 + logvar - K.square(mean) - K.exp(logvar)),
        axis=1
    )
    # Average across batch
    kl_loss = tf.reduce_mean(kl_loss)

    return net, kl_loss


def build_model(net, params):
    input_shape = params["input_shape"]
    assert len(input_shape) == 3, "Input shape must be 3-dim"

    enc_conv_kernels = params["enc_conv_kernels"]
    enc_conv_filters = params["enc_conv_filters"] # output filters
    enc_conv_strides = params["enc_conv_strides"]
    activation = params["activation"]
    assert len(enc_conv_kernels) == len(enc_conv_filters), \
        "encoder layers are misspecified: len(kernels) != len(filters)"
    assert len(enc_conv_kernels) == len(enc_conv_strides), \
        "encoder layers are misspecified: len(kernels) != len(strides)"

    dec_conv_kernels = params["dec_conv_kernels"]
    dec_conv_filters = params["dec_conv_filters"] # input filters
    dec_conv_strides = params["dec_conv_strides"]
    # Last decoder layer does not have an activation.
    dec_conv_activations = [activation] * (len(dec_conv_strides)-1) + [None]
    assert len(dec_conv_kernels) == len(dec_conv_filters), \
        "decoder layers are misspecified: len(kernels) != len(filters)"
    assert len(dec_conv_kernels) == len(dec_conv_strides), \
        "decoder layers are misspecified: len(kernels) != len(strides)"


    dense_units = params["dense_units"]
    latent_ndim = params["latent_ndim"]
    kl_loss_red_type = params["kl_loss_reduction_type"]
    assert kl_loss_red_type in _REDUCTION_TYPES, \
        f"invalid reconstruction loss reduction type: {kl_loss_red_type}"
    fp_loss = params["full_precision_loss"]

    upsample = np.product(dec_conv_strides)
    assert input_shape[1] % upsample == 0, \
        "Input shape dim1 must be divisible by decoder strides"
    assert input_shape[2] % upsample == 0, \
        "Input shape dim2 must be divisible by decoder strides"
    unflatten_filters = dec_conv_filters[0]
    unflatten_shape = (
        [unflatten_filters] + [d // upsample for d in input_shape[1:]]
    )
    unflatten_nelem = np.product(unflatten_shape)

    # Input
    tf.compat.v1.logging.info(f"Shape net after {net.name}: {net.get_shape()}")

    # Encoder conv layers
    enc_conv_activations = [activation] * (len(enc_conv_strides))
    for count, (kernel, filters, stride, activation) in enumerate(
            zip(
                enc_conv_kernels,
                enc_conv_filters,
                enc_conv_strides,
                enc_conv_activations
            )
    ):
        net = conv2d(
            inputs=net,
            filters=filters,
            kernel_size=[kernel, kernel],
            strides=(stride, stride),
            activation=activation,
            padding="SAME",
            name=f"enc_conv_{count+1}",
        )

    # Flatten
    net = tf.keras.layers.Flatten(name="flatten_encoder")(net)
    tf.compat.v1.logging.info(f"Shape net after {net.name}: {net.get_shape()}")

    # Encoder dense layers
    net = dense(
        inputs=net,
        units=dense_units,
        activation=activation,
        name=f"enc_dense_1",
    )

    # Embedding
    # Variational, sampled embedding
    # Two FCs for mean and logvar
    net, kl_loss = variational_embedding(
        inputs=net,
        latent_ndim=latent_ndim,
        kl_loss_reduction_type=kl_loss_red_type,
        fp_loss=fp_loss,
        name="embedding",
    )

    # Decoder dense layers
    net = dense(
        inputs=net,
        units=dense_units,
        activation=activation,
        name=f"dec_dense_1",
    )

    net = dense(
        inputs=net,
        units=unflatten_nelem,
        activation=activation,
        name=f"dec_dense_2",
    )

    # Unflatten
    net = tf.keras.layers.Reshape(unflatten_shape, name="unflatten")(net)
    tf.compat.v1.logging.info(f"Shape net after {net.name}: {net.get_shape()}")

    # Decoder layers
    dec_conv_filters_out = dec_conv_filters[1:] + [input_shape[0]]
    for count, (kernel, filters, stride, activation) in list(enumerate(
            zip(
                dec_conv_kernels,
                dec_conv_filters_out,
                dec_conv_strides,
                dec_conv_activations
            )
    )):
        net = deconv2d(
            inputs=net,
            filters=filters,
            kernel_size=[kernel, kernel],
            strides=(stride, stride),
            padding="SAME",
            activation=activation,
            name=f"dec_deconv_{count+1}",
        )

    # Flatten output
    net = tf.keras.layers.Flatten(name="flatten_final")(net)
    tf.compat.v1.logging.info(f"Shape net after {net.name}: {net.get_shape()}")

    return net, kl_loss
