import tensorflow as tf
from tensorflow.keras.layers import Layer

from tf_img_models.modeling.layers import bn_relu_conv2d, bn_relu_dense


__all__ = ["BottleneckAttentionModule"]


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
        x = tf.reduce_mean(inputs, axis=[1, 2])
        x = self.dense1(x, training)
        x = self.dense2(x, training)
        x = tf.expand_dims(x, [1])
        x = tf.expand_dims(x, [1])
        return x


class SpatialAttention(Layer):
    def __init__(
        self, reduction: int = 16, dilation: int = 4, *, weight_decay: float = None
    ):
        super().__init__()
        self._reduction = reduction
        self._dilation = dilation
        self._weight_decay = weight_decay

    def build(self, input_shape):
        self.conv1 = bn_relu_conv2d(
            input_shape[-1] // self._reduction, 1, weight_decay=self._weight_decay
        )
        self.conv2 = bn_relu_conv2d(
            input_shape[-1] // self._reduction,
            3,
            dilation=self._dilation,
            weight_decay=self._weight_decay,
        )
        self.conv3 = bn_relu_conv2d(
            input_shape[-1] // self._reduction,
            3,
            dilation=self._dilation,
            weight_decay=self._weight_decay,
        )
        self.conv4 = bn_relu_conv2d(1, 1, weight_decay=self._weight_decay)
        del self._reduction
        del self._dilation
        del self._weight_decay

    def call(self, inputs, training=None):
        x = self.conv1(inputs, training)
        x = self.conv2(x, training)
        x = self.conv3(x, training)
        x = self.conv4(x, training)
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
        spatial_dilation: int = 4,
        weight_decay: float = None
    ):
        super().__init__()
        self.channel_attention = ChannelAttention(
            channel_reduction, weight_decay=weight_decay
        )
        self.spatial_attention = SpatialAttention(
            spatial_reduction, spatial_dilation, weight_decay=weight_decay
        )

    def call(self, inputs, training=None):
        x_c = self.channel_attention(inputs, training)
        x_a = self.spatial_attention(inputs, training)
        x = tf.add(x_c, x_a)
        x = tf.nn.sigmoid(x)
        x = tf.multiply(x, inputs)
        x = tf.add(x, inputs)
        return x
