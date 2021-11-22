from typing import Union

import tensorflow as tf
from tensorflow_addons.layers import GroupNormalization, InstanceNormalization
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import BatchNormalization, Layer


__all__ = ["BatchNorm", "GroupBatchNorm", "InstanceBatchNorm"]


def BatchNorm(
    *,
    momentum: float = 0.99,
    epsilon: float = 0.001,
    center: bool = True,
    scale: bool = True,
    norm_weight_zero_init: bool = False,
    moving_mean_initializer: Union[Initializer, str] = "zeros",
    moving_variance_initializer: Union[Initializer, str] = "ones",
    name: str = None,
) -> BatchNormalization:
    gamma_initializer = "zeros" if norm_weight_zero_init else "ones"
    return BatchNormalization(
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        name=name,
    )


def GroupNorm(
    *,
    groups: int = 32,
    epsilon: float = 0.001,
    center: bool = True,
    scale: bool = True,
    norm_weight_zero_init: bool = False,
    name: str = None,
) -> GroupNormalization:
    gamma_initializer = "zeros" if norm_weight_zero_init else "ones"
    return GroupNormalization(
        groups,
        epsilon=epsilon,
        center=center,
        scale=scale,
        gamma_initializer=gamma_initializer,
        name=name,
    )


class GroupBatchNorm(Layer):
    """Group Batch Normalization (GBN).

    Implementation is based on the idea of IBN.
    """

    def __init__(
        self,
        *,
        momentum: float = 0.99,
        groups: int = 32,
        epsilon: float = 0.001,
        center: bool = True,
        scale: bool = True,
        norm_weight_zero_init: bool = False,
        moving_mean_initializer: Union[Initializer, str] = "zeros",
        moving_variance_initializer: Union[Initializer, str] = "ones",
        name: str = None,
    ):
        super().__init__(name=name)
        self.b = BatchNorm(
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            norm_weight_zero_init=norm_weight_zero_init,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            name=f"{name}_bn",
        )
        self.i = GroupNorm(
            groups=groups,
            epsilon=epsilon,
            center=center,
            scale=scale,
            name=f"{name}_in",
        )

    def call(self, inputs, training):
        b, i = tf.split(inputs, 2, -1)
        b = self.b(b, training)
        i = self.i(i)
        x = tf.concat([b, i], -1)
        return x


def InstanceNorm(
    *,
    epsilon: float = 0.001,
    center: bool = True,
    scale: bool = True,
    name: str = None,
) -> InstanceNormalization:
    """
    """
    return InstanceNormalization(
        epsilon=epsilon, center=center, scale=scale, name=name
    )


class InstanceBatchNorm(Layer):
    """Instance Batch Normalization (IBN).

    References:
    - https://arxiv.org/abs/1807.09441
    """

    def __init__(
        self,
        *,
        momentum: float = 0.99,
        epsilon: float = 0.001,
        center: bool = True,
        scale: bool = True,
        norm_weight_zero_init: bool = False,
        moving_mean_initializer: Union[Initializer, str] = "zeros",
        moving_variance_initializer: Union[Initializer, str] = "ones",
        name: str = None,
    ):
        super().__init__(name=name)
        self.b = BatchNorm(
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            norm_weight_zero_init=norm_weight_zero_init,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            name=f"{name}_bn",
        )
        self.i = InstanceNorm(
            epsilon=epsilon, center=center, scale=scale, name=f"{name}_in"
        )

    def call(self, inputs, training):
        b, i = tf.split(inputs, 2, -1)
        b = self.b(b, training)
        i = self.i(i)
        x = tf.concat([b, i], -1)
        return x
