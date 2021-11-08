from typing import Union

from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import L2


__all__ = ["batch_norm"]


def batch_norm(
    *,
    momentum: float = 0.99,
    epsilon: float = 0.001,
    center: bool = True,
    scale: bool = True,
    norm_weight_zero_init: bool = False,
    moving_mean_initializer: Union[Initializer, str] = "zeros",
    moving_variance_initializer: Union[Initializer, str] = "ones",
    weight_decay: float = None,
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
        gamma_regularizer=L2(weight_decay),
        name=name,
    )
