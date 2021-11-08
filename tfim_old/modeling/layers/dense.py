from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.regularizers import L2
from tfim.modeling.layers import batch_norm


__all__ = ["dense", "dense_bn", "dense_bn_relu", "dense_relu"]


def dense(
    units: int,
    kernel_initializer: str = "glorot_uniform",
    weight_decay: float = None,
    name: str = None,
) -> Dense:
    return Dense(
        units,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=L2(weight_decay),
        name=name,
    )


def dense_bn(
    units: int,
    kernel_initializer: str = "glorot_uniform",
    weight_decay: float = None,
    name: str = None,
) -> Sequential:
    return Sequential(
        [
            Dense(
                units,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=L2(weight_decay),
            ),
            batch_norm(weight_decay=weight_decay),
        ],
        name=name,
    )


def dense_bn_relu(
    units: int,
    kernel_initializer: str = "glorot_uniform",
    weight_decay: float = None,
    name: str = None,
) -> Sequential:
    return Sequential(
        [
            Dense(
                units,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=L2(weight_decay),
            ),
            batch_norm(weight_decay=weight_decay),
            ReLU(),
        ],
        name=name,
    )


def dense_relu(
    units: int,
    kernel_initializer: str = "glorot_uniform",
    weight_decay: float = None,
    name: str = None,
) -> Sequential:
    return Sequential(
        [
            Dense(
                units,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=L2(weight_decay),
            ),
            ReLU(),
        ],
        name=name,
    )
