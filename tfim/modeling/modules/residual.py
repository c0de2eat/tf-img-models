from typing import Tuple, Union

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer
from tfim.modeling.layers import conv2d_bn, conv2d_bn_relu
from tfim.modeling.modules.attention import ConvolutionalBottleneckAttentionModule


__all__ = [
    "BottleneckBlock",
    "BottleneckBlockCBAM",
    "ResidualBlock",
    "ResidualBlockCBAM",
]


class ResidualBlock(Layer):
    """Residual block.

    References:
    - https://arxiv.org/abs/1812.01187
    - https://arxiv.org/abs/1512.03385
    """

    expansion = 1

    def __init__(
        self,
        filters: int,
        downsample: Sequential,
        *,
        strides: Union[int, Tuple[int, int]] = 1,
        last_norm_weight_zero_init: bool = True,
        weight_decay: float = None,
        name: str = None,
    ):
        super().__init__(name=name)
        self.downsample = downsample

        self.f = Sequential(
            [
                conv2d_bn_relu(filters, 3, strides=strides, weight_decay=weight_decay),
                conv2d_bn(
                    filters,
                    3,
                    norm_weight_zero_init=last_norm_weight_zero_init,
                    weight_decay=weight_decay,
                ),
            ]
        )

    def call(self, inputs, training=None):
        x = self.f(inputs, training)
        identity = self.downsample(inputs, training)
        x = tf.add(x, identity)
        x = tf.nn.relu(x)
        return x


class ResidualBlockCBAM(ResidualBlock):
    """Residual block + CBAM.

    References:
    - https://arxiv.org/abs/1807.06521
    """

    def __init__(
        self,
        filters: int,
        downsample: Sequential,
        *,
        strides: Union[int, Tuple[int, int]] = 1,
        channel_reduction: int = 16,
        weight_decay: float = None,
        name: str = None,
    ):
        super().__init__(
            filters,
            downsample,
            strides=strides,
            last_norm_weight_zero_init=False,
            weight_decay=weight_decay,
            name=name,
        )
        self.cbam = ConvolutionalBottleneckAttentionModule(
            channel_reduction=channel_reduction, weight_decay=weight_decay,
        )

    def call(self, inputs, training=None):
        x = self.f(inputs, training)
        x = self.cbam(x, training)
        identity = self.downsample(inputs, training)
        x = tf.add(x, identity)
        x = tf.nn.relu(x)
        return x


class BottleneckBlock(Layer):
    """Bottleneck block.

    References:
    - https://arxiv.org/abs/1812.01187
    - https://arxiv.org/abs/1512.03385
    """

    expansion = 4

    def __init__(
        self,
        filters: int,
        downsample: Sequential,
        *,
        strides: Union[int, Tuple[int, int]] = 1,
        last_norm_weight_zero_init: bool = True,
        weight_decay: float = None,
        name: str = None,
    ):
        super().__init__(name=name)
        self.downsample = downsample

        self.f = Sequential(
            [
                conv2d_bn_relu(filters, 1, weight_decay=weight_decay),
                conv2d_bn_relu(filters, 3, strides=strides, weight_decay=weight_decay,),
                conv2d_bn(
                    filters * self.expansion,
                    1,
                    norm_weight_zero_init=last_norm_weight_zero_init,
                    weight_decay=weight_decay,
                ),
            ]
        )

    def call(self, inputs, training=None):
        x = self.f(inputs, training)
        identity = self.downsample(inputs, training)
        x = tf.add(x, identity)
        x = tf.nn.relu(x)
        return x


class BottleneckBlockCBAM(BottleneckBlock):
    """Bottleneck block + CBAM.

    References:
    - https://arxiv.org/abs/1807.06521
    """

    def __init__(
        self,
        filters: int,
        downsample: Sequential,
        *,
        strides: Union[int, Tuple[int, int]] = 1,
        channel_reduction: int = 16,
        weight_decay: float = None,
        name: str = None,
    ):
        super().__init__(
            filters,
            downsample,
            strides=strides,
            last_norm_weight_zero_init=False,
            weight_decay=weight_decay,
            name=name,
        )
        self.cbam = ConvolutionalBottleneckAttentionModule(
            channel_reduction=channel_reduction, weight_decay=weight_decay,
        )

    def call(self, inputs, training=None):
        x = self.f(inputs, training)
        x = self.cbam(x, training)
        identity = self.downsample(inputs, training)
        x = tf.add(x, identity)
        x = tf.nn.relu(x)
        return x
