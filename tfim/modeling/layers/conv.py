from typing import Tuple, Union

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, ReLU
from tensorflow.keras.regularizers import L2
from tfim.modeling.layers import batch_norm


__all__ = ["bn_relu_conv2d", "conv2d", "conv2d_bn", "conv2d_bn_relu"]


def bn_relu_conv2d(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    *,
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding: str = "same",
    dilation: Union[int, Tuple[int, int]] = (1, 1),
    groups: int = 1,
    kernel_initializer: str = "glorot_uniform",
    norm_weight_zero_init: bool = False,
    weight_decay: float = None,
) -> Sequential:
    layers = Sequential(
        [
            batch_norm(
                norm_weight_zero_init=norm_weight_zero_init, weight_decay=weight_decay
            ),
            ReLU(),
            Conv2D(
                filters,
                kernel_size,
                strides,
                padding,
                dilation_rate=dilation,
                groups=groups,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=L2(weight_decay),
            ),
        ]
    )
    return layers


def conv2d(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    *,
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding: str = "same",
    dilation: Union[int, Tuple[int, int]] = (1, 1),
    groups: int = 1,
    use_bias: bool = True,
    kernel_initializer: str = "glorot_uniform",
    weight_decay: float = None,
) -> Conv2D:
    return Conv2D(
        filters,
        kernel_size,
        strides,
        padding,
        dilation_rate=dilation,
        groups=groups,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=L2(weight_decay),
    )


def conv2d_bn(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    *,
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding: str = "same",
    dilation: Union[int, Tuple[int, int]] = (1, 1),
    groups: int = 1,
    kernel_initializer: str = "glorot_uniform",
    norm_weight_zero_init: bool = False,
    weight_decay: float = None,
) -> Sequential:
    layers = Sequential(
        [
            Conv2D(
                filters,
                kernel_size,
                strides,
                padding,
                dilation_rate=dilation,
                groups=groups,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=L2(weight_decay),
            ),
            batch_norm(
                norm_weight_zero_init=norm_weight_zero_init, weight_decay=weight_decay
            ),
        ]
    )
    return layers


def conv2d_bn_relu(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    *,
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding: str = "same",
    dilation: Union[int, Tuple[int, int]] = (1, 1),
    groups: int = 1,
    kernel_initializer: str = "glorot_uniform",
    norm_weight_zero_init: bool = False,
    weight_decay: float = None,
) -> Sequential:
    layers = Sequential(
        [
            Conv2D(
                filters,
                kernel_size,
                strides,
                padding,
                dilation_rate=dilation,
                groups=groups,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=L2(weight_decay),
            ),
            batch_norm(
                norm_weight_zero_init=norm_weight_zero_init, weight_decay=weight_decay
            ),
            ReLU(),
        ]
    )
    return layers
