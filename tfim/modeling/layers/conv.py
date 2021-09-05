from typing import Tuple, Union

import tensorflow.keras as keras
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
    # kernel_initializer: str = "variance_scaling",
    kernel_initializer: str = "glorot_uniform",
    norm_weight_zero_init: bool = False,
    weight_decay: float = None,
) -> keras.Sequential:
    layers = keras.Sequential(
        [
            batch_norm(weight_decay, norm_weight_zero_init),
            keras.layers.ReLU(),
            keras.layers.Conv2D(
                filters,
                kernel_size,
                strides,
                padding,
                dilation_rate=dilation,
                groups=groups,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=keras.regularizers.l2(weight_decay),
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
    # kernel_initializer: str = "variance_scaling",
    kernel_initializer: str = "glorot_uniform",
    weight_decay: float = None,
) -> keras.layers.Conv2D:
    return keras.layers.Conv2D(
        filters,
        kernel_size,
        strides,
        padding,
        dilation_rate=dilation,
        groups=groups,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=keras.regularizers.l2(weight_decay),
    )


def conv2d_bn(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    *,
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding: str = "same",
    dilation: Union[int, Tuple[int, int]] = (1, 1),
    groups: int = 1,
    # kernel_initializer: str = "variance_scaling",
    kernel_initializer: str = "glorot_uniform",
    norm_weight_zero_init: bool = False,
    weight_decay: float = None,
) -> keras.Sequential:
    layers = keras.Sequential(
        [
            keras.layers.Conv2D(
                filters,
                kernel_size,
                strides,
                padding,
                dilation_rate=dilation,
                groups=groups,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=keras.regularizers.l2(weight_decay),
            ),
            batch_norm(weight_decay, norm_weight_zero_init),
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
    # kernel_initializer: str = "variance_scaling",
    kernel_initializer: str = "glorot_uniform",
    norm_weight_zero_init: bool = False,
    weight_decay: float = None,
) -> keras.Sequential:
    layers = keras.Sequential(
        [
            keras.layers.Conv2D(
                filters,
                kernel_size,
                strides,
                padding,
                dilation_rate=dilation,
                groups=groups,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=keras.regularizers.l2(weight_decay),
            ),
            batch_norm(weight_decay, norm_weight_zero_init),
            keras.layers.ReLU(),
        ]
    )
    return layers
