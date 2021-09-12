import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer
from tfim.modeling.layers import conv2d, conv2d_relu


__all__ = ["SqueezeExcitationModule"]


class SqueezeExcitationModule(Layer):
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

    def call(self, inputs, training=None):
        x = tf.reduce_mean(inputs, [1, 2], True)
        x = self.f(x, training)
        x = tf.nn.sigmoid(x)
        x = tf.multiply(x, inputs)
        return x
