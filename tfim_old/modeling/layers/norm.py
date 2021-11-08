import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Layer
from tensorflow.keras.regularizers import L2
from tensorflow_addons.layers import InstanceNormalization


__all__ = ["batch_norm", "instance_norm", "InstanceBatchNorm"]


def instance_norm(
    *, center: bool = True, weight_decay: float = None, name: str = None
) -> InstanceNormalization:
    return InstanceNormalization(
        center=center, gamma_regularizer=L2(weight_decay), name=name
    )


class InstanceBatchNorm(Layer):
    """Instance Batch Normalization (IBN).
    References:
    - https://arxiv.org/abs/1807.09441
    """

    def __init__(
        self,
        *,
        center: bool = True,
        weight_decay: float = None,
        name: str = None
    ):
        super().__init__(name=name)
        self.b = batch_norm(center=center, weight_decay=weight_decay)
        self.i = instance_norm(center=center, weight_decay=weight_decay)

    def call(self, inputs, training):
        b, i = tf.split(inputs, 2, -1)
        b = self.b(b, training)
        i = self.i(i)
        x = tf.concat([b, i], -1)
        return x
