from typing import Tuple, Type, Union

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import AvgPool2D, InputSpec, Layer, MaxPool2D
from tfim.modeling.layers import conv2d_bn_relu
from tfim.modeling.modules import ResidualBlock, BottleneckBlock


__all__ = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "ResNet"]


class ResNet(Model):
    """ResNet.

    References:
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
        weight_decay: float = None,
    ):
        self.input_spec = InputSpec(shape=(None,) + inputs.shape)

        # Stem: 56x56
        x = Sequential(
            [
                conv2d_bn_relu(32, 3, strides=2, weight_decay=weight_decay),
                conv2d_bn_relu(32, 3, weight_decay=weight_decay),
                conv2d_bn_relu(64, 3, weight_decay=weight_decay),
                MaxPool2D(3, 2, padding="same"),
            ],
            "stem",
        )(inputs)

        # Layer1: 56x56
        x = self.__construct_residual_block(
            block, 64, cfg[0], 1, weight_decay=weight_decay, name="layer1"
        )(x)
        # if bottleneck_attention:
        #     x = BottleneckAttentionModule(weight_decay=weight_decay)(x)

        # Layer2: 28x28
        x = self.__construct_residual_block(
            block, 128, cfg[1], 2, weight_decay=weight_decay, name="layer2"
        )(x)
        # if bottleneck_attention:
        #     x = BottleneckAttentionModule(weight_decay=weight_decay)(x)

        # Layer3: 14x14
        x = self.__construct_residual_block(
            block, 256, cfg[2], 2, weight_decay=weight_decay, name="layer3"
        )(x)
        # if bottleneck_attention:
        #     x = BottleneckAttentionModule(weight_decay=weight_decay)(x)

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
                    conv2d_bn_relu(
                        filters * block.expansion, 1, weight_decay=weight_decay
                    ),
                ]
            )
        else:
            downsample = Layer()

        layers = Sequential(name=name)
        layers.add(
            block(
                filters,
                strides=strides,
                downsample=downsample,
                weight_decay=weight_decay,
            )
        )
        self.filters *= block.expansion

        for _ in range(1, n_layers):
            layers.add(block(filters, strides=1, weight_decay=weight_decay,))

        return layers


def resnet18(inputs, *, weight_decay: float = None,) -> Model:
    """ResNet18.
    """
    model = ResNet(inputs, ResidualBlock, (2, 2, 2, 2), weight_decay=weight_decay,)
    return model


def resnet34(inputs, *, weight_decay: float = None,) -> Model:
    """ResNet34.
    """
    model = ResNet(inputs, ResidualBlock, (3, 4, 6, 3), weight_decay=weight_decay,)
    return model


def resnet50(inputs, *, weight_decay: float = None,) -> Model:
    """ResNet50.
    """
    model = ResNet(inputs, BottleneckBlock, (3, 4, 6, 3), weight_decay=weight_decay,)
    return model


def resnet101(inputs, *, weight_decay: float = None,) -> Model:
    """ResNet101.
    """
    model = ResNet(inputs, BottleneckBlock, (3, 4, 23, 3), weight_decay=weight_decay,)
    return model


def resnet152(inputs, *, weight_decay: float = None,) -> Model:
    """ResNet152.
    """
    model = ResNet(inputs, BottleneckBlock, (3, 8, 36, 3), weight_decay=weight_decay,)
    return model


if __name__ == "__main__":
    from tensorflow.keras import Input
    from tensorflow.keras.utils import plot_model

    inputs = Input((224, 224, 3))
    backbone = resnet50(inputs)
    print(backbone.summary())
    plot_model(backbone, "resnet50.png", True, show_layer_names=True)
