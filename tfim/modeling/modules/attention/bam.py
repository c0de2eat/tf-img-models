import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer
from tfim.modeling.layers import conv2d, conv2d_bn_relu


__all__ = ["BottleneckAttentionModule"]


class ChannelAttention(Layer):
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
                conv2d(input_shape[-1], 1, weight_decay=self._weight_decay),
            ]
        )
        del self._reduction
        del self._weight_decay

    def call(self, inputs, training=None, **kwargs):
        x = tf.reduce_mean(inputs, [1, 2], True)
        x = self.f(x, training)
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
                conv2d(1, 1, weight_decay=self._weight_decay),
            ]
        )
        del self._reduction
        del self._weight_decay

    def call(self, inputs, training=None, **kwargs):
        x = self.f(inputs, training)
        return x


class BottleneckAttentionModule(Layer):
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

    def call(self, inputs, training=None, **kwargs):
        x_c = self.channel_attention(inputs, training)
        x_s = self.spatial_attention(inputs, training)
        x = tf.add(x_c, x_s)
        x = tf.nn.sigmoid(x)
        x = tf.multiply(x, inputs)
        x = tf.add(x, inputs)
        return x
