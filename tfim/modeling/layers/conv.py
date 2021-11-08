from typing import Tuple, Union

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, ReLU
from tensorflow.keras.regularizers import L2

from .norm import batch_norm


__all__ = ["conv2d"]


def conv2d(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding="valid",
    dilation_rate: Union[int, Tuple[int, int]] = (1, 1),
    groups=1,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    weight_decay: float = None,
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
        kernel_regularizer=L2(weight_decay),
        name=name,
    )


def conv2d_bn(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding="valid",
    dilation_rate: Union[int, Tuple[int, int]] = (1, 1),
    groups=1,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    norm_weight_zero_init: bool = False,
    weight_decay: float = None,
    name: str = None,
) -> Sequential:
    return Sequential(
        [
            conv2d(
                filters,
                kernel_size,
                strides,
                padding,
                dilation_rate,
                groups,
                use_bias,
                kernel_initializer,
                weight_decay,
                name,
            ),
            batch_norm(
                norm_weight_zero_init=norm_weight_zero_init,
                weight_decay=weight_decay,
            ),
        ]
    )


def conv2d_bn_relu(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding="valid",
    dilation_rate: Union[int, Tuple[int, int]] = (1, 1),
    groups=1,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    norm_weight_zero_init: bool = False,
    weight_decay: float = None,
    name: str = None,
) -> Sequential:
    return Sequential(
        [
            conv2d(
                filters,
                kernel_size,
                strides,
                padding,
                dilation_rate,
                groups,
                use_bias,
                kernel_initializer,
                weight_decay,
                name,
            ),
            batch_norm(
                norm_weight_zero_init=norm_weight_zero_init,
                weight_decay=weight_decay,
            ),
            ReLU(),
        ]
    )


def conv2d_relu(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding="valid",
    dilation_rate: Union[int, Tuple[int, int]] = (1, 1),
    groups=1,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    norm_weight_zero_init: bool = False,
    weight_decay: float = None,
    name: str = None,
) -> Sequential:
    return Sequential(
        [
            conv2d(
                filters,
                kernel_size,
                strides,
                padding,
                dilation_rate,
                groups,
                use_bias,
                kernel_initializer,
                weight_decay,
                name,
            ),
            ReLU(),
        ]
    )
