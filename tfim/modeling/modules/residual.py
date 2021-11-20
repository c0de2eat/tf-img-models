from typing import Tuple, Union

from tensorflow.keras import Sequential
from tensorflow.keras.layers import add, ReLU

from tfim.modeling.layers import conv2d_bn, conv2d_bn_relu


__all__ = ["residual_block"]


def residual_block(
    x,
    filters: int,
    use_bottleneck: bool = True,
    *,
    strides: Union[int, Tuple[int, int]] = (1, 1),
    downsample: Union[Sequential, None] = None,
    weight_decay: float = 0.0,
    name: str = None,
):
    identity = x
    if use_bottleneck:
        x = bottleneck(filters, strides, weight_decay, name)(x)
    else:
        x = residual(filters, strides, weight_decay, name)(x)
    if downsample is not None:
        identity = downsample(identity)
    x = add([x, identity])
    x = ReLU()(x)
    return x


def bottleneck(
    filters: int,
    strides: Union[int, Tuple[int, int]],
    weight_decay: float = 0.0,
    name: str = None,
):
    return Sequential(
        [
            conv2d_bn_relu(filters, 1, weight_decay=weight_decay),
            conv2d_bn_relu(
                filters, 3, strides=strides, weight_decay=weight_decay
            ),
            conv2d_bn(
                filters * 4,
                1,
                norm_weight_zero_init=True,
                weight_decay=weight_decay,
            ),
        ],
        name=f"{name}_bottleneck",
    )


def residual(
    filters: int,
    strides: Union[int, Tuple[int, int]],
    weight_decay: float = 0.0,
    name: str = None,
):
    return Sequential(
        [
            conv2d_bn_relu(
                filters, 3, strides=strides, weight_decay=weight_decay
            ),
            conv2d_bn(
                filters,
                3,
                norm_weight_zero_init=True,
                weight_decay=weight_decay,
            ),
        ],
        name=f"{name}_residual",
    )
