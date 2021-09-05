import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer
from tfim.modeling.layers import conv2d_bn, conv2d_bn_relu, dense_bn, dense_bn_relu


__all__ = ["BottleneckAttentionModule"]


class ChannelAttention(Layer):
    def __init__(self, reduction: int = 16, *, weight_decay: float = None):
        super().__init__()
        self._reduction = reduction
        self._weight_decay = weight_decay

    def build(self, input_shape):
        self.f = Sequential(
            [
                dense_bn_relu(
                    input_shape[-1] // self._reduction, weight_decay=self._weight_decay
                ),
                dense_bn(input_shape[-1], weight_decay=self._weight_decay),
            ]
        )
        del self._reduction
        del self._weight_decay

    def call(self, inputs, training=None):
        x = tf.reduce_mean(inputs, axis=[1, 2])
        x = self.f(x, training)
        x = tf.expand_dims(x, [1])
        x = tf.expand_dims(x, [1])
        return x


class SpatialAttention(Layer):
    def __init__(self, reduction: int = 16, *, weight_decay: float = None):
        super().__init__()
        self._reduction = reduction
        self._weight_decay = weight_decay

    def build(self, input_shape):
        self.f = Sequential(
            [
                conv2d_bn_relu(
                    input_shape[-1] // self._reduction,
                    1,
                    weight_decay=self._weight_decay,
                ),
                conv2d_bn_relu(
                    input_shape[-1] // self._reduction,
                    3,
                    dilation=4,
                    weight_decay=self._weight_decay,
                ),
                conv2d_bn_relu(
                    input_shape[-1] // self._reduction,
                    3,
                    dilation=4,
                    weight_decay=self._weight_decay,
                ),
                conv2d_bn(1, 1, weight_decay=self._weight_decay),
            ]
        )
        del self._reduction
        del self._weight_decay

    def call(self, inputs, training=None):
        x = self.f(inputs, training)
        return x


class BottleneckAttentionModule(Layer):
    """Bottleneck Attention Module (BAM).

    References:
    - https://arxiv.org/abs/1807.06514
    """

    def __init__(
        self,
        *,
        channel_reduction: int = 16,
        spatial_reduction: int = 16,
        weight_decay: float = None,
        name: str = None
    ):
        super().__init__(name=name)
        self.channel_attention = ChannelAttention(
            channel_reduction, weight_decay=weight_decay
        )
        self.spatial_attention = SpatialAttention(
            spatial_reduction, weight_decay=weight_decay
        )

    def call(self, inputs, training=None):
        x_c = self.channel_attention(inputs, training)
        x_a = self.spatial_attention(inputs, training)
        x = tf.add(x_c, x_a)
        x = tf.nn.sigmoid(x)
        x = tf.multiply(x, inputs)
        x = tf.add(x, inputs)
        return x
