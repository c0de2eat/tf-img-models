from typing import Tuple, Union

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Add
from tfim.modeling.layers import conv2d_bn, conv2d_bn_relu, conv2d_ibn_relu
from tfim.modeling.modules.attention import (
    ConvolutionalBottleneckAttentionModule,
    SqueezeExcitationModule,
)


__all__ = [
    "bottleneck_block",
    "residual_block",
]


def _common(
    x,
    identity,
    *,
    downsample: Union[Sequential, None] = None,
    # Attentions
    se: bool = False,
    se_reduction: int = 16,
    cbam: bool = False,
    cbam_channel_reduction: int = 16,
    weight_decay: float = None,
):
    if se:
        x = SqueezeExcitationModule(reduction=se_reduction, weight_decay=weight_decay)(
            x
        )
    if cbam:
        x = ConvolutionalBottleneckAttentionModule(
            channel_reduction=cbam_channel_reduction, weight_decay=weight_decay,
        )(x)
    if downsample is not None:
        identity = downsample(identity)
    x = Add()([x, identity])
    x = tf.nn.relu(x)
    return x


def bottleneck_block(
    x,
    filters: int,
    *,
    strides: Union[int, Tuple[int, int]] = 1,
    downsample: Union[Sequential, None] = None,
    # Normalizations
    ibn: bool = False,
    # Attentions
    se: bool = False,
    se_reduction: int = 16,
    cbam: bool = False,
    cbam_channel_reduction: int = 16,
    weight_decay: float = None,
    name: str = None,
):
    conv = conv2d_ibn_relu if ibn else conv2d_bn_relu
    if se or cbam:
        last_norm_weight_zero_init = False
    else:
        last_norm_weight_zero_init = True

    identity = x
    x = Sequential(
        [
            conv(filters, 1, weight_decay=weight_decay),
            conv2d_bn_relu(filters, 3, strides=strides, weight_decay=weight_decay,),
            conv2d_bn(
                filters * 4,
                1,
                norm_weight_zero_init=last_norm_weight_zero_init,
                weight_decay=weight_decay,
            ),
        ],
        name=f"{name}_f",
    )(x)
    x = _common(
        x,
        identity,
        downsample=downsample,
        se=se,
        se_reduction=se_reduction,
        cbam=cbam,
        cbam_channel_reduction=cbam_channel_reduction,
        weight_decay=weight_decay,
    )
    return x


def residual_block(
    x,
    filters: int,
    *,
    strides: Union[int, Tuple[int, int]] = 1,
    downsample: Union[Sequential, None] = None,
    # Normalizations
    ibn: bool = False,
    # Attentions
    se: bool = False,
    se_reduction: int = 16,
    cbam: bool = False,
    cbam_channel_reduction: int = 16,
    weight_decay: float = None,
    name: str = None,
):
    conv = conv2d_ibn_relu if ibn else conv2d_bn_relu
    if se or cbam:
        last_norm_weight_zero_init = False
    else:
        last_norm_weight_zero_init = True

    identity = x
    x = Sequential(
        [
            conv(filters, 3, strides=strides, weight_decay=weight_decay),
            conv2d_bn(
                filters,
                3,
                norm_weight_zero_init=last_norm_weight_zero_init,
                weight_decay=weight_decay,
            ),
        ],
        name=f"{name}_f",
    )(x)
    x = _common(
        x,
        identity,
        downsample=downsample,
        se=se,
        se_reduction=se_reduction,
        cbam=cbam,
        cbam_channel_reduction=cbam_channel_reduction,
        weight_decay=weight_decay,
    )
    return x
