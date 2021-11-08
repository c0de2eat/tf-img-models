import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer
from tfim.modeling.layers import conv2d, conv2d_relu


__all__ = ["ConvolutionalBottleneckAttentionModule"]


class ChannelAttention(Layer):
    def __init__(self, reduction: int = 16, *, weight_decay: float = None):
        super().__init__()
        self._reduction = reduction
        self._weight_decay = weight_decay

    def build(self, input_shape):
        self.f = Sequential(
            [
                conv2d_relu(
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
        x_avg = tf.reduce_mean(inputs, [1, 2], True)
        x_avg = self.f(x_avg, training)
        x_max = tf.reduce_max(inputs, [1, 2], True)
        x_max = self.f(x_max, training)
        x = tf.add(x_avg, x_max)
        x = tf.nn.sigmoid(x)
        x = tf.multiply(x, inputs)
        return x


class SpatialAttention(Layer):
    def __init__(self, *, weight_decay: float = None):
        super().__init__()

        self.f = conv2d(1, 7, weight_decay=weight_decay)

    def call(self, inputs, training=None, **kwargs):
        x_avg = tf.reduce_mean(inputs, -1, True)
        x_max = tf.reduce_max(inputs, -1, True)
        x = tf.concat([x_avg, x_max], -1)
        x = self.f(x)
        x = tf.nn.sigmoid(x)
        x = tf.multiply(x, inputs)
        return x


class ConvolutionalBottleneckAttentionModule(Layer):
    def __init__(self, *, channel_reduction: int = 16, weight_decay: float = None):
        super().__init__()
        self.channel_attention = ChannelAttention(
            channel_reduction, weight_decay=weight_decay
        )
        self.spatial_attention = SpatialAttention(weight_decay=weight_decay)

    def call(self, inputs, training=None, **kwargs):
        x = self.channel_attention(inputs, training)
        x = self.spatial_attention(x, training)
        return x
