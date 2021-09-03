from typing import Tuple, Union

from tensorflow.keras import Sequential
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Conv2D, ReLU
from tensorflow.keras.regularizers import l2

from tf_img_models.modeling.layers import batch_norm


__all__ = ["bn_relu_conv2d", "conv2d"]


def bn_relu_conv2d(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    *,
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding: str = "same",
    dilation: Union[int, Tuple[int, int]] = (1, 1),
    groups: int = 1,
    kernel_initializer: Union[str, Initializer] = "variance_scaling",
    weight_decay: float = None,
    name: str = None
):
    layers = Sequential(name=name)
    layers.add(batch_norm(weight_decay))
    layers.add(ReLU())
    layers.add(
        Conv2D(
            filters,
            kernel_size,
            strides,
            padding,
            dilation_rate=dilation,
            groups=groups,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=l2(weight_decay),
        )
    )
    return layers


def conv2d(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    *,
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding: str = "same",
    dilation: Union[int, Tuple[int, int]] = (1, 1),
    groups: int = 1,
    use_bias: bool = True,
    kernel_initializer: Union[str, Initializer] = "variance_scaling",
    weight_decay: float = None,
    name: str = None
):
    return Conv2D(
        filters,
        kernel_size,
        strides,
        padding,
        dilation_rate=dilation,
        groups=groups,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=l2(weight_decay),
        name=name,
    )
