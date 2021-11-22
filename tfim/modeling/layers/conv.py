from typing import Tuple, Union

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, ReLU

from tfim.modeling.layers import (
    BatchNorm,
    GroupBatchNorm,
    InstanceBatchNorm,
)


__all__ = ["Conv2d", "Conv2dNorm", "Conv2dNormReLU", "Conv2dReLU"]


def Conv2d(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding="same",
    dilation_rate: Union[int, Tuple[int, int]] = (1, 1),
    groups=1,
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
    padding="same",
    dilation_rate: Union[int, Tuple[int, int]] = (1, 1),
    groups=1,
    kernel_initializer="glorot_uniform",
    normalization: str = "bn",
    norm_weight_zero_init: bool = False,
    name: str = None,
) -> Sequential:
    if normalization == "bn":
        norm = BatchNorm
    elif normalization == "gbn":
        norm = GroupBatchNorm
    elif normalization == "ibn":
        norm = InstanceBatchNorm
    else:
        raise NotImplementedError

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
            norm(norm_weight_zero_init=norm_weight_zero_init),
        ],
        name,
    )


def Conv2dNormReLU(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding="same",
    dilation_rate: Union[int, Tuple[int, int]] = (1, 1),
    groups=1,
    kernel_initializer="glorot_uniform",
    normalization: str = "bn",
    norm_weight_zero_init: bool = False,
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
        normalization,
        norm_weight_zero_init,
        name,
    )
    conv.add(ReLU())
    return conv


def Conv2dReLU(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding="same",
    dilation_rate: Union[int, Tuple[int, int]] = (1, 1),
    groups=1,
    use_bias=True,
    kernel_initializer="glorot_uniform",
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
            ReLU(),
        ],
        name,
    )
