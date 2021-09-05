import tensorflow.keras as keras

from tfim.modeling.layers import batch_norm


__all__ = ["dense", "dense_bn_relu"]


def dense_bn_relu(
    units: int,
    # kernel_initializer: str = "variance_scaling",
    kernel_initializer: str = "glorot_uniform",
    weight_decay: float = None,
) -> keras.Sequential:
    layers = keras.Sequential()
    layers.add(batch_norm(weight_decay))
    layers.add(keras.layers.ReLU())
    layers.add(
        keras.layers.Dense(
            units,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=keras.regularizers.l2(weight_decay),
        )
    )
    return layers


def dense(
    units: int,
    # kernel_initializer: str = "variance_scaling",
    kernel_initializer: str = "glorot_uniform",
    weight_decay: float = None,
) -> keras.layers.Dense:
    layers = keras.layers.Dense(
        units,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=keras.regularizers.l2(weight_decay),
    )
    return layers
