from typing import Tuple, Union

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D

from tfim.modeling.layers import Activations, get_normalization


__all__ = [
    "Conv2d",
    "Conv2dActivation",
    "Conv2dNorm",
    "Conv2dNormActivation",
]


def Conv2d(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding: str = "same",
    dilation_rate: Union[int, Tuple[int, int]] = (1, 1),
    groups: int = 1,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    name: str = None,
) -> Conv2D:
    return Conv2D(
        filters,
        kernel_size,
        strides,
        padding,
        dilation_rate=dilation_rate,
        groups=groups,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        name=name,
    )


def Conv2dNorm(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding: str = "same",
    dilation_rate: Union[int, Tuple[int, int]] = (1, 1),
    groups: int = 1,
    kernel_initializer: str = "glorot_uniform",
    norm: str = "bn",
    norm_weight_zero_init: bool = False,
    name: str = None,
) -> Sequential:

    return Sequential(
        [
            Conv2d(
                filters,
                kernel_size,
                strides,
                padding,
                dilation_rate,
                groups,
                False,
                kernel_initializer,
            ),
            get_normalization(
                norm, norm_weight_zero_init=norm_weight_zero_init
            ),
        ],
        name,
    )


def Conv2dNormActivation(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding: str = "same",
    dilation_rate: Union[int, Tuple[int, int]] = (1, 1),
    groups: int = 1,
    kernel_initializer: str = "glorot_uniform",
    norm: str = "bn",
    norm_weight_zero_init: bool = False,
    activation: str = "relu",
    name: str = None,
) -> Sequential:
    conv = Conv2dNorm(
        filters,
        kernel_size,
        strides,
        padding,
        dilation_rate,
        groups,
        kernel_initializer,
        norm,
        norm_weight_zero_init,
        name,
    )
    conv.add(Activations(activation))
    return conv


def Conv2dActivation(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding: str = "same",
    dilation_rate: Union[int, Tuple[int, int]] = (1, 1),
    groups: int = 1,
    use_bias: bool = True,
    kernel_initializer: str = "glorot_uniform",
    activation: str = "relu",
    name: str = None,
) -> Sequential:
    return Sequential(
        [
            Conv2d(
                filters,
                kernel_size,
                strides,
                padding,
                dilation_rate,
                groups,
                use_bias,
                kernel_initializer,
            ),
            Activations(activation),
        ],
        name,
    )
