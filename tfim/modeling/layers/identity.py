from tensorflow.keras.layers import Layer

__all__ = ["Identity"]


class Identity(Layer):
    def call(self, inputs, training=None):
        return inputs
