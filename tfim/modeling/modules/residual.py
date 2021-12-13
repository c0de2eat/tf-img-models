from typing import Tuple, Union

from tensorflow.keras import Sequential
from tensorflow.keras.layers import add

from tfim.modeling.layers import Activations, Conv2dNorm, Conv2dNormActivation


__all__ = ["residual_block"]


def residual_block(
    x,
    filters: int,
    use_bottleneck: bool = True,
    *,
    width: int = 64,
    strides: Union[int, Tuple[int, int]] = (1, 1),
    groups: int = 1,
    norm: str = "bn",
    activation: str = "relu",
    downsample: Union[Sequential, None] = None,
    name: str = None,
):
    if not use_bottleneck and groups != 1 and width != 64:
        raise ValueError("Basic block only supports `groups=1` and `width=64`")
    width = int(filters * (width / 64.0)) * groups

    identity = x
    if use_bottleneck:
        x = Bottleneck(
            filters, width, strides, groups, norm, activation, name
        )(x)
    else:
        x = Residual(filters, strides, norm, activation, name)(x)
    if downsample is not None:
        identity = downsample(identity)
    x = add([x, identity])
    x = Activations(activation)(x)
    return x


def Bottleneck(
    filters: int,
    width: int,
    strides: Union[int, Tuple[int, int]],
    groups: int,
    norm: str = "bn",
    activation: str = "relu",
    name: str = None,
):
    return Sequential(
        [
            Conv2dNormActivation(width, 1, norm=norm, activation=activation),
            Conv2dNormActivation(
                width, 3, strides=strides, groups=groups, norm="bn"
            ),
            Conv2dNorm(
                filters * 4,
                1,
                norm="bn",
                norm_weight_zero_init=True,
            ),
        ],
        name=f"{name}_bottleneck",
    )


def Residual(
    filters: int,
    strides: Union[int, Tuple[int, int]],
    norm: str = "bn",
    activation: str = "relu",
    name: str = None,
):
    return Sequential(
        [
            Conv2dNormActivation(
                filters,
                3,
                strides=strides,
                norm=norm,
                activation=activation,
            ),
            Conv2dNorm(
                filters,
                3,
                norm="bn",
                norm_weight_zero_init=True,
            ),
        ],
        name=f"{name}_residual",
    )
