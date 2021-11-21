from typing import Tuple, Union

from tensorflow.keras import Sequential
from tensorflow.keras.layers import add, ReLU

from tfim.modeling.layers import Conv2dNorm, Conv2dNormReLU


__all__ = ["residual_block"]


def residual_block(
    x,
    filters: int,
    use_bottleneck: bool = True,
    *,
    strides: Union[int, Tuple[int, int]] = (1, 1),
    normalization: str = "bn",
    downsample: Union[Sequential, None] = None,
    name: str = None,
):
    identity = x
    if use_bottleneck:
        x = Bottleneck(filters, strides, normalization, name)(x)
    else:
        x = Residual(filters, strides, normalization, name)(x)
    if downsample is not None:
        identity = downsample(identity)
    x = add([x, identity])
    x = ReLU()(x)
    return x


def Bottleneck(
    filters: int,
    strides: Union[int, Tuple[int, int]],
    normalization: str = "bn",
    name: str = None,
):
    return Sequential(
        [
            Conv2dNormReLU(filters, 1, normalization=normalization),
            Conv2dNormReLU(filters, 3, strides=strides, normalization="bn"),
            Conv2dNorm(
                filters * 4, 1, normalization="bn", norm_weight_zero_init=True
            ),
        ],
        name=f"{name}_bottleneck",
    )


def Residual(
    filters: int,
    strides: Union[int, Tuple[int, int]],
    normalization: str = "bn",
    name: str = None,
):
    return Sequential(
        [
            Conv2dNormReLU(
                filters, 3, strides=strides, normalization=normalization
            ),
            Conv2dNorm(
                filters, 3, normalization="bn", norm_weight_zero_init=True
            ),
        ],
        name=f"{name}_residual",
    )
