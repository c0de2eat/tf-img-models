from tensorflow.keras.activations import relu, swish
from tensorflow.keras.layers import Layer


__all__ = ["Activations"]


class Activations(Layer):
    def __init__(self, name: str):
        super().__init__()
        if name == "relu":
            self.act = relu
        elif name == "swish":
            self.act = swish
        else:
            raise NotImplementedError

    def call(self, inputs):
        return self.act(inputs)
