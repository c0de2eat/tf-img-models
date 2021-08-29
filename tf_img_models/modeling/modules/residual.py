from typing import Tuple, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer

from tf_img_models.modeling.layers import batch_norm, bn_relu_conv2d, conv2d
from tf_img_models.modeling.modules import ConvolutionalBottleneckAttentionModule

__all__ = ["ResidualBlock", "ResidualBlockCBAM", "Bottleneck", "BottleneckCBAM"]


class ResidualBlock(Layer):
    """Residual block.

    References:
    - https://arxiv.org/abs/1603.05027
    - https://arxiv.org/abs/1512.03385
    """

    expansion = 1

    def __init__(
        self,
        filters: int,
        *,
        strides: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        base_width: int = 64,
        downsample: Layer = Layer(),
        weight_decay: float = None,
        name: str = None,
    ):
        super().__init__(name=name)
        assert groups == 1 and base_width == 64, (
            f"ResidualBlock only supports groups=1 (given {groups}) and "
            f"base_width=64 (given {base_width})"
        )
        assert dilation == 1, "ResidualBlock only supports dilation=1"

        self.downsample = downsample

        self.bn = batch_norm(weight_decay)
        self.conv1 = conv2d(filters, 3, strides=strides, weight_decay=weight_decay)
        self.conv2 = bn_relu_conv2d(filters, 3, weight_decay=weight_decay)

    def call(self, inputs, training=None):
        x = self.bn(inputs, training)
        x = tf.nn.relu(x)
        identity = self.downsample(x)
        x = self.conv1(x)
        x = self.conv2(x, training)
        x = tf.add(x, identity)
        return x


class ResidualBlockCBAM(ResidualBlock):
    """Residual block + CBAM.

    References:
    - https://arxiv.org/abs/1807.06521
    """

    def __init__(
        self,
        filters: int,
        *,
        strides: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        base_width: int = 64,
        downsample: Layer = Layer(),
        convolutional_bottleneck_attention_channel_reduction: int = 16,
        weight_decay: float = None,
        name: str = None,
    ):
        super().__init__(
            filters,
            strides=strides,
            dilation=dilation,
            groups=groups,
            base_width=base_width,
            downsample=downsample,
            weight_decay=weight_decay,
            name=name,
        )
        self.cbam = ConvolutionalBottleneckAttentionModule(
            channel_reduction=convolutional_bottleneck_attention_channel_reduction,
            weight_decay=weight_decay,
        )

    def call(self, inputs, training=None):
        x = self.bn(inputs, training)
        x = tf.nn.relu(x)
        identity = self.downsample(x)
        x = self.conv1(x)
        x = self.conv2(x, training)
        x = self.cbam(x, training)
        x = tf.add(x, identity)
        return x


class Bottleneck(Layer):
    """Bottleneck block.

    Design follows [2] where `strides=2` in the 3x3 convolution instead of the first 1x1
    convolution for bottleneck block. This increases the Top1 for ~0.5, with a slight
    performance drawback of ~5% images/sec.

    References:
    - https://arxiv.org/abs/1603.05027
    - https://ngc.nvidia.com/catalog/model-scripts/nvidia
    - https://arxiv.org/abs/1512.03385
    """

    expansion = 4

    def __init__(
        self,
        filters: int,
        *,
        strides: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        base_width: int = 64,
        downsample: Layer = Layer(),
        weight_decay: float = None,
        name: str = None,
    ):
        super().__init__(name=name)
        self.downsample = downsample

        width = int(filters * (base_width / 64.0)) * groups
        self.bn = batch_norm(weight_decay)
        self.conv1 = conv2d(width, 1, weight_decay=weight_decay)
        self.conv2 = bn_relu_conv2d(
            width,
            3,
            strides=strides,
            dilation=dilation,
            groups=groups,
            weight_decay=weight_decay,
        )
        self.conv3 = bn_relu_conv2d(
            filters * self.expansion, 1, weight_decay=weight_decay
        )

    def call(self, inputs, training=None):
        x = self.bn(inputs, training)
        x = tf.nn.relu(x)
        identity = self.downsample(x)
        x = self.conv1(x)
        x = self.conv2(x, training)
        x = self.conv3(x, training)
        x = tf.add(x, identity)
        return x


class BottleneckCBAM(Bottleneck):
    """Bottleneck block + CBAM.

    References:
    - https://arxiv.org/abs/1807.06521
    """

    def __init__(
        self,
        filters: int,
        *,
        strides: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        base_width: int = 64,
        downsample: Layer = Layer(),
        convolutional_bottleneck_attention_channel_reduction: int = 16,
        weight_decay: float = None,
        name: str = None,
    ):
        super().__init__(
            filters,
            strides=strides,
            dilation=dilation,
            groups=groups,
            base_width=base_width,
            downsample=downsample,
            weight_decay=weight_decay,
            name=name,
        )
        self.cbam = ConvolutionalBottleneckAttentionModule(
            channel_reduction=convolutional_bottleneck_attention_channel_reduction,
            weight_decay=weight_decay,
        )

    def call(self, inputs, training=None):
        x = self.bn(inputs, training)
        x = tf.nn.relu(x)
        identity = self.downsample(x)
        x = self.conv1(x)
        x = self.conv2(x, training)
        x = self.conv3(x, training)
        x = self.cbam(x, training)
        x = tf.add(x, identity)
        return x
