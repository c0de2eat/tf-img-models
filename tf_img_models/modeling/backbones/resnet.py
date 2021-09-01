from typing import Tuple, Type, Union

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import InputSpec, Layer, MaxPool2D

from tf_img_models.modeling.layers import conv2d
from tf_img_models.modeling.modules import (
    Bottleneck,
    BottleneckCBAM,
    BottleneckAttentionModule,
    ResidualBlock,
    ResidualBlockCBAM,
)


class ResNet(Model):
    """ResNet.

    - Attention Layers
        - BAM: Bottleneck Attention Module
        - CBAM: Convolutional Block Attention Module

    References:
    - https://arxiv.org/abs/1807.06521
    - https://arxiv.org/abs/1807.06514
    - https://arxiv.org/abs/1603.05027
    - https://ngc.nvidia.com/catalog/model-scripts/nvidia
    - https://arxiv.org/abs/1512.03385
    """

    filters = 64

    def __init__(
        self,
        img_shape: Tuple[int, int, int],
        block: Type[Union[ResidualBlock, Bottleneck]],
        cfg: Tuple[int, int, int, int],
        *,
        groups: int = 1,
        width_per_group: int = 64,
        bottleneck_attention: bool = False,
        weight_decay: float = None,
    ):
        self.groups = groups
        self.base_width = width_per_group

        self.input_spec = InputSpec(shape=(None,) + img_shape)

        inputs = tf.keras.Input(shape=img_shape)

        x = conv2d(self.filters, 7, strides=2, weight_decay=weight_decay)(inputs)
        x = MaxPool2D(3, 2, "same")(x)

        # Layer1: 56x56
        x = self.__construct_residual_block(
            block, 64, cfg[0], 1, weight_decay=weight_decay, name="layer1"
        )(x)
        if bottleneck_attention:
            x = BottleneckAttentionModule(weight_decay=weight_decay)(x)

        # Layer2: 28x28
        x = self.__construct_residual_block(
            block, 128, cfg[1], 2, weight_decay=weight_decay, name="layer2"
        )(x)
        if bottleneck_attention:
            x = BottleneckAttentionModule(weight_decay=weight_decay)(x)

        # Layer3: 14x14
        x = self.__construct_residual_block(
            block, 256, cfg[2], 2, weight_decay=weight_decay, name="layer3"
        )(x)
        if bottleneck_attention:
            x = BottleneckAttentionModule(weight_decay=weight_decay)(x)

        # Layer4: 7x7
        x = self.__construct_residual_block(
            block, 512, cfg[3], 2, weight_decay=weight_decay, name="layer4"
        )(x)

        total_layers = 2
        n = 2 if block.expansion == 1 else 3
        for c in cfg:
            total_layers += c * n
        print(f"=> # of layers in ResNet: {total_layers}")
        super().__init__(inputs=inputs, outputs=x, name=f"ResNet{total_layers}")

    def __construct_residual_block(
        self,
        block: Type[Union[ResidualBlock, Bottleneck]],
        filters: int,
        n_layers: int,
        strides: Union[int, Tuple[int, int]],
        weight_decay: float = None,
        name: str = None,
    ) -> Sequential:
        if strides != 1 or self.filters != filters * block.expansion:
            downsample = conv2d(
                filters * block.expansion,
                1,
                strides=strides,
                weight_decay=weight_decay,
            )

        else:
            downsample = Layer()

        layers = Sequential(name=name)
        layers.add(
            block(
                filters,
                strides=strides,
                groups=self.groups,
                base_width=self.base_width,
                downsample=downsample,
                weight_decay=weight_decay,
            )
        )
        self.filters *= block.expansion

        for _ in range(1, n_layers):
            layers.add(
                block(
                    filters,
                    strides=1,
                    groups=self.groups,
                    base_width=self.base_width,
                    downsample=Layer(),
                    weight_decay=weight_decay,
                )
            )

        return layers


def resnet18(
    img_shape: Tuple[int, int, int],
    *,
    groups: int = 1,
    width_per_group: int = 64,
    bottleneck_attention: bool = False,
    convolutional_bottleneck_attention: bool = False,
    weight_decay: float = None,
) -> Model:
    """ResNet18.
    """
    model = ResNet(
        img_shape,
        ResidualBlockCBAM if convolutional_bottleneck_attention else ResidualBlock,
        (2, 2, 2, 2),
        groups=groups,
        width_per_group=width_per_group,
        bottleneck_attention=bottleneck_attention,
        weight_decay=weight_decay,
    )
    return model


def resnet34(
    img_shape: Tuple[int, int, int],
    *,
    groups: int = 1,
    width_per_group: int = 64,
    bottleneck_attention: bool = False,
    convolutional_bottleneck_attention: bool = False,
    weight_decay: float = None,
) -> Model:
    """ResNet34.
    """
    model = ResNet(
        img_shape,
        ResidualBlockCBAM if convolutional_bottleneck_attention else ResidualBlock,
        (3, 4, 6, 3),
        groups=groups,
        width_per_group=width_per_group,
        bottleneck_attention=bottleneck_attention,
        weight_decay=weight_decay,
    )
    return model


def resnet50(
    img_shape: Tuple[int, int, int],
    *,
    groups: int = 1,
    width_per_group: int = 64,
    bottleneck_attention: bool = False,
    convolutional_bottleneck_attention: bool = False,
    weight_decay: float = None,
) -> Model:
    """ResNet50.
    """
    model = ResNet(
        img_shape,
        BottleneckCBAM if convolutional_bottleneck_attention else Bottleneck,
        (3, 4, 6, 3),
        groups=groups,
        width_per_group=width_per_group,
        bottleneck_attention=bottleneck_attention,
        weight_decay=weight_decay,
    )
    return model


def resnet101(
    img_shape: Tuple[int, int, int],
    *,
    groups: int = 1,
    width_per_group: int = 64,
    bottleneck_attention: bool = False,
    convolutional_bottleneck_attention: bool = False,
    weight_decay: float = None,
) -> Model:
    """ResNet101.
    """
    model = ResNet(
        img_shape,
        BottleneckCBAM if convolutional_bottleneck_attention else Bottleneck,
        (3, 4, 23, 3),
        groups=groups,
        width_per_group=width_per_group,
        bottleneck_attention=bottleneck_attention,
        weight_decay=weight_decay,
    )
    return model


def resnet152(
    img_shape: Tuple[int, int, int],
    *,
    groups: int = 1,
    width_per_group: int = 64,
    bottleneck_attention: bool = False,
    convolutional_bottleneck_attention: bool = False,
    weight_decay: float = None,
) -> Model:
    """ResNet152.
    """
    model = ResNet(
        img_shape,
        BottleneckCBAM if convolutional_bottleneck_attention else Bottleneck,
        (3, 8, 36, 3),
        groups=groups,
        width_per_group=width_per_group,
        bottleneck_attention=bottleneck_attention,
        weight_decay=weight_decay,
    )
    return model
