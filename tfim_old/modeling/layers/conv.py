from typing import Tuple, Union

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, ReLU
from tensorflow.keras.regularizers import L2
from tfim.modeling.layers import batch_norm, InstanceBatchNorm


__all__ = [
    "conv2d",
    "conv2d_bn",
    "conv2d_bn_relu",
    "conv2d_ibn_relu",
    "conv2d_relu",
]


def conv2d_ibn_relu(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    *,
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding: str = "same",
    dilation: Union[int, Tuple[int, int]] = (1, 1),
    groups: int = 1,
    kernel_initializer: str = "glorot_uniform",
    weight_decay: float = None,
    name: str = None,
) -> Sequential:
    return Sequential(
        [
            Conv2D(
                filters,
                kernel_size,
                strides,
                padding,
                dilation_rate=dilation,
                groups=groups,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=L2(weight_decay),
            ),
            InstanceBatchNorm(
                weight_decay=weight_decay,
            ),
            ReLU(),
        ],
        name=name,
    )
