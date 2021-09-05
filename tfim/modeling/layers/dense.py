from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.regularizers import L2
from tfim.modeling.layers import batch_norm


__all__ = ["dense", "dense_bn", "dense_bn_relu", "dense_relu"]


def dense(
    units: int, kernel_initializer: str = "glorot_uniform", weight_decay: float = None,
) -> Dense:
    layers = Dense(
        units,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=L2(weight_decay),
    )
    return layers


def dense_bn(
    units: int,
    kernel_initializer: str = "glorot_uniform",
    norm_momentum: float = 0.99,
    weight_decay: float = None,
) -> Sequential:
    layers = Sequential()
    layers.add(
        Dense(
            units,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=L2(weight_decay),
        )
    )
    layers.add(batch_norm(momentum=norm_momentum, weight_decay=weight_decay))
    return layers


def dense_bn_relu(
    units: int,
    kernel_initializer: str = "glorot_uniform",
    norm_momentum: float = 0.99,
    weight_decay: float = None,
) -> Sequential:
    layers = Sequential()
    layers.add(
        Dense(
            units,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=L2(weight_decay),
        )
    )
    layers.add(batch_norm(momentum=norm_momentum, weight_decay=weight_decay))
    layers.add(ReLU())
    return layers


def dense_relu(
    units: int, kernel_initializer: str = "glorot_uniform", weight_decay: float = None,
) -> Sequential:
    layers = Sequential()
    layers.add(
        Dense(
            units,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=L2(weight_decay),
        )
    )
    layers.add(ReLU())
    return layers
