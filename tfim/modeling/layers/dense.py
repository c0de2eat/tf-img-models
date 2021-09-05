from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.regularizers import L2
from tfim.modeling.layers import batch_norm


__all__ = ["dense", "dense_bn_relu"]


def dense_bn_relu(
    units: int, kernel_initializer: str = "glorot_uniform", weight_decay: float = None,
) -> Sequential:
    layers = Sequential()
    layers.add(batch_norm(weight_decay=weight_decay))
    layers.add(ReLU())
    layers.add(
        Dense(
            units,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=L2(weight_decay),
        )
    )
    return layers


def dense(
    units: int, kernel_initializer: str = "glorot_uniform", weight_decay: float = None,
) -> Dense:
    layers = Dense(
        units,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=L2(weight_decay),
    )
    return layers
