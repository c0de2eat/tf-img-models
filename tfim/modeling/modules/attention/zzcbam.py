import tensorflow as tf
from tensorflow.keras.layers import Layer

from tf_img_models.modeling.layers import bn_relu_conv2d, bn_relu_dense


__all__ = ["ConvolutionalBottleneckAttentionModule"]


class ChannelAttention(Layer):
    def __init__(self, reduction: int = 16, *, weight_decay: float = None):
        super().__init__()
        self._reduction = reduction
        self._weight_decay = weight_decay

    def build(self, input_shape):
        self.dense1 = bn_relu_dense(
            input_shape[-1] // self._reduction, weight_decay=self._weight_decay
        )
        self.dense2 = bn_relu_dense(input_shape[-1], weight_decay=self._weight_decay)
        del self._reduction
        del self._weight_decay

    def call(self, inputs, training=None):
        x_avg = tf.reduce_mean(inputs, axis=[1, 2])
        x_avg = self.dense1(x_avg, training)
        x_avg = self.dense2(x_avg, training)
        x_max = tf.reduce_max(inputs, axis=[1, 2])
        x_max = self.dense1(x_max, training)
        x_max = self.dense2(x_max, training)
        x = tf.add(x_avg, x_max)
        x = tf.nn.sigmoid(x)
        x = tf.expand_dims(x, [1])
        x = tf.expand_dims(x, [1])
        x = tf.multiply(x, inputs)
        return x


class SpatialAttention(Layer):
    def __init__(self, *, weight_decay: float = None):
        super().__init__()

        self.conv1 = bn_relu_conv2d(1, 7, weight_decay=weight_decay)

    def call(self, inputs, training=None):
        x_avg = tf.reduce_mean(inputs, -1, True)
        x_max = tf.reduce_max(inputs, -1, True)
        x = tf.concat([x_avg, x_max], -1)
        x = self.conv1(x, training)
        x = tf.nn.sigmoid(x)
        x = tf.multiply(x, inputs)
        return x


class ConvolutionalBottleneckAttentionModule(Layer):
    """Convolutional Block Attention Module (CBAM).

    References:
    - https://arxiv.org/abs/1807.06521
    """

    def __init__(self, *, channel_reduction: int = 16, weight_decay: float = None):
        super().__init__()
        self.channel_attention = ChannelAttention(
            channel_reduction, weight_decay=weight_decay
        )
        self.spatial_attention = SpatialAttention(weight_decay=weight_decay)

    def call(self, inputs, training=None):
        x = self.channel_attention(inputs, training)
        x = self.spatial_attention(x, training)
        return x
