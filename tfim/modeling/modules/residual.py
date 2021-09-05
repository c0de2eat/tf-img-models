from typing import Tuple, Union

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer
from tfim.modeling.layers import conv2d_bn, conv2d_bn_relu


__all__ = ["ResidualBlock", "BottleneckBlock"]


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
        weight_decay: float = None,
        name: str = None,
    ):
        super().__init__(name=name)
        self.downsample = downsample

        self.conv1 = conv2d_bn_relu(
            filters, 3, strides=strides, weight_decay=weight_decay
        )
        self.conv2 = conv2d_bn(
            filters, 3, norm_weight_zero_init=True, weight_decay=weight_decay
        )

    def call(self, inputs, training=None):
        x = self.conv1(inputs, training)
        x = self.conv2(x, training)
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
        weight_decay: float = None,
        name: str = None,
    ):
        super().__init__(name=name)
        self.downsample = downsample

        self.conv1 = conv2d_bn_relu(filters, 1, weight_decay=weight_decay)
        self.conv2 = conv2d_bn_relu(
            filters, 3, strides=strides, weight_decay=weight_decay,
        )
        self.conv3 = conv2d_bn(
            filters * self.expansion,
            1,
            norm_weight_zero_init=True,
            weight_decay=weight_decay,
        )

    def call(self, inputs, training=None):
        x = self.conv1(inputs, training)
        x = self.conv2(x, training)
        x = self.conv3(x, training)
        identity = self.downsample(inputs, training)
        x = tf.add(x, identity)
        x = tf.nn.relu(x)
        return x
