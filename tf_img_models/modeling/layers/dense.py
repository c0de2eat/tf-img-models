from typing import Union

from tensorflow.keras import Sequential
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.regularizers import l2

from tf_img_models.modeling.layers import batch_norm


__all__ = ["bn_relu_dense", "dense"]


def bn_relu_dense(
    units: int,
    kernel_initializer: Union[str, Initializer] = "variance_scaling",
    weight_decay: float = None,
    name: str = None,
) -> Sequential:
    layers = Sequential(name=name)
    layers.add(batch_norm(weight_decay))
    layers.add(ReLU())
    layers.add(
        Dense(
            units,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=l2(weight_decay),
        )
    )
    return layers


def dense(
    units: int,
    kernel_initializer: Union[str, Initializer] = "variance_scaling",
    weight_decay: float = None,
    name: str = None,
) -> Sequential:
    layers = Sequential(name=name)
    layers.add(
        Dense(
            units,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=l2(weight_decay),
        )
    )
    return layers
