from typing import Tuple, Type, Union

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import AvgPool2D, InputSpec
from tfim.modeling.layers import conv2d_bn, conv2d_bn_relu, Identity
from tfim.modeling.modules import (
    BottleneckBlock,
    BottleneckBlockCBAM,
    BottleneckAttentionModule,
    ResidualBlock,
    ResidualBlockCBAM,
)


__all__ = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "ResNet"]


class ResNet(Model):
    """ResNet.

    References:
    - https://arxiv.org/abs/2103.07579
    - https://arxiv.org/abs/1812.01187
    - https://arxiv.org/abs/1512.03385
    """

    filters = 64

    def __init__(
        self,
        inputs,
        block: Type[Union[ResidualBlock, BottleneckBlock]],
        cfg: Tuple[int, int, int, int],
        *,
        small_input: bool = False,
        bottleneck_attention: bool = False,
        weight_decay: float = None,
    ):
        self.input_spec = InputSpec(shape=(None,) + inputs.shape)

        # Stem: 56x56
        stem = Sequential(name="stem")
        if small_input:
            stem.add(conv2d_bn_relu(64, 3, weight_decay=weight_decay))
        else:
            stem.add(conv2d_bn_relu(64, 3, strides=2, weight_decay=weight_decay))
        stem.add(conv2d_bn_relu(64, 3, weight_decay=weight_decay))
        stem.add(conv2d_bn_relu(64, 3, weight_decay=weight_decay))
        x = stem(inputs)

        # Layer1: 56x56
        first_stride = 1 if small_input else 2
        x = self.__construct_residual_block(
            block, 64, cfg[0], first_stride, weight_decay=weight_decay, name="layer1"
        )(x)
        if bottleneck_attention:
            x = BottleneckAttentionModule(weight_decay=weight_decay, name="bam1",)(x)

        # Layer2: 28x28
        x = self.__construct_residual_block(
            block, 128, cfg[1], 2, weight_decay=weight_decay, name="layer2"
        )(x)
        if bottleneck_attention:
            x = BottleneckAttentionModule(weight_decay=weight_decay, name="bam2",)(x)

        # Layer3: 14x14
        x = self.__construct_residual_block(
            block, 256, cfg[2], 2, weight_decay=weight_decay, name="layer3"
        )(x)
        if bottleneck_attention:
            x = BottleneckAttentionModule(weight_decay=weight_decay, name="bam3",)(x)

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
        block: Type[Union[ResidualBlock, BottleneckBlock]],
        filters: int,
        n_layers: int,
        strides: Union[int, Tuple[int, int]],
        weight_decay: float = None,
        name: str = None,
    ) -> Sequential:
        if strides != 1 or self.filters != filters * block.expansion:
            downsample = Sequential(
                [
                    AvgPool2D(2, strides, padding="same"),
                    conv2d_bn(filters * block.expansion, 1, weight_decay=weight_decay),
                ]
            )
        else:
            downsample = Sequential([Identity()])

        layers = Sequential(name=name)
        layers.add(
            block(filters, downsample, strides=strides, weight_decay=weight_decay,)
        )
        self.filters *= block.expansion

        for _ in range(1, n_layers):
            layers.add(
                block(
                    filters,
                    Sequential([Identity()]),
                    strides=1,
                    weight_decay=weight_decay,
                )
            )

        return layers


def resnet18(
    inputs,
    *,
    small_input: bool = False,
    bottleneck_attention: bool = False,
    convolutional_bottleneck_attention: bool = False,
    weight_decay: float = None,
) -> Model:
    """ResNet18.
    """
    model = ResNet(
        inputs,
        ResidualBlockCBAM if convolutional_bottleneck_attention else ResidualBlock,
        (2, 2, 2, 2),
        small_input=small_input,
        bottleneck_attention=bottleneck_attention,
        weight_decay=weight_decay,
    )
    return model


def resnet34(
    inputs,
    *,
    small_input: bool = False,
    bottleneck_attention: bool = False,
    convolutional_bottleneck_attention: bool = False,
    weight_decay: float = None,
) -> Model:
    """ResNet34.
    """
    model = ResNet(
        inputs,
        ResidualBlockCBAM if convolutional_bottleneck_attention else ResidualBlock,
        (3, 4, 6, 3),
        small_input=small_input,
        bottleneck_attention=bottleneck_attention,
        weight_decay=weight_decay,
    )
    return model


def resnet50(
    inputs,
    *,
    small_input: bool = False,
    bottleneck_attention: bool = False,
    convolutional_bottleneck_attention: bool = False,
    weight_decay: float = None,
) -> Model:
    """ResNet50.
    """
    model = ResNet(
        inputs,
        BottleneckBlockCBAM if convolutional_bottleneck_attention else BottleneckBlock,
        (3, 4, 6, 3),
        small_input=small_input,
        bottleneck_attention=bottleneck_attention,
        weight_decay=weight_decay,
    )
    return model


def resnet101(
    inputs,
    *,
    small_input: bool = False,
    bottleneck_attention: bool = False,
    convolutional_bottleneck_attention: bool = False,
    weight_decay: float = None,
) -> Model:
    """ResNet101.
    """
    model = ResNet(
        inputs,
        BottleneckBlockCBAM if convolutional_bottleneck_attention else BottleneckBlock,
        (3, 4, 23, 3),
        small_input=small_input,
        bottleneck_attention=bottleneck_attention,
        weight_decay=weight_decay,
    )
    return model


def resnet152(
    inputs,
    *,
    small_input: bool = False,
    bottleneck_attention: bool = False,
    convolutional_bottleneck_attention: bool = False,
    weight_decay: float = None,
) -> Model:
    """ResNet152.
    """
    model = ResNet(
        inputs,
        BottleneckBlockCBAM if convolutional_bottleneck_attention else BottleneckBlock,
        (3, 8, 36, 3),
        small_input=small_input,
        bottleneck_attention=bottleneck_attention,
        weight_decay=weight_decay,
    )
    return model


if __name__ == "__main__":
    from tensorflow.keras import Input
    from tensorflow.keras.utils import plot_model

    inputs = Input((224, 224, 3))
    backbone = resnet50(inputs, bottleneck_attention=True)
    print(backbone.summary())
    plot_model(backbone, "resnet50.png", True, show_layer_names=True)
