from typing import Union

import tensorflow as tf
from tensorflow_addons.layers import GroupNormalization, InstanceNormalization
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import BatchNormalization, Layer


__all__ = ["get_normalization"]


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


def InstanceNorm(
    *,
    epsilon: float = 0.001,
    center: bool = True,
    scale: bool = True,
    norm_weight_zero_init: bool = False,
    name: str = None,
) -> InstanceNormalization:
    gamma_initializer = "zeros" if norm_weight_zero_init else "ones"
    return InstanceNormalization(
        epsilon=epsilon,
        center=center,
        scale=scale,
        gamma_initializer=gamma_initializer,
        name=name,
    )


class GroupBatchNorm(Layer):
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
        self.g = GroupNorm(
            groups=groups,
            epsilon=epsilon,
            center=center,
            scale=scale,
            norm_weight_zero_init=norm_weight_zero_init,
            name=f"{name}_in",
        )

    def call(self, inputs, training):
        b, g = tf.split(inputs, 2, -1)
        b = self.b(b, training)
        g = self.g(g)
        x = tf.concat([b, g], -1)
        return x


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
            epsilon=epsilon,
            center=center,
            scale=scale,
            norm_weight_zero_init=norm_weight_zero_init,
            name=f"{name}_in",
        )

    def call(self, inputs, training):
        b, i = tf.split(inputs, 2, -1)
        b = self.b(b, training)
        i = self.i(i)
        x = tf.concat([b, i], -1)
        return x


def get_normalization(name: str, norm_weight_zero_init: bool = False):
    if name == "bn":
        return BatchNorm(norm_weight_zero_init=norm_weight_zero_init)
    elif name == "gbn":
        return GroupBatchNorm(norm_weight_zero_init=norm_weight_zero_init)
    elif name == "ibn":
        return InstanceBatchNorm(norm_weight_zero_init=norm_weight_zero_init)
    else:
        raise NotImplementedError
