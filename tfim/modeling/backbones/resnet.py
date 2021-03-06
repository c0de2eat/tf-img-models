from typing import Tuple, Union

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import AvgPool2D, InputSpec, MaxPool2D

from tfim.modeling.layers import Conv2dNorm, Conv2dNormActivation
from tfim.modeling.modules import residual_block


__all__ = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnext50_32x4d",
    "resnet101",
    "resnet152",
    "ResNet",
]


class ResNet(Model):
    filters = 64

    def __init__(
        self,
        inputs,
        cfg: Tuple[int, int, int, int],
        width: int = 64,
        groups: int = 1,
        use_bottleneck: bool = True,
        norm: str = "bn",
        activation: str = "relu",
        small_input: bool = False,
    ):
        self.input_spec = InputSpec(shape=(None,) + inputs.shape)

        # Stem: 56x56
        stem = Sequential(name="stem")
        stride = 1 if small_input else 2
        stem.add(
            Conv2dNormActivation(
                32, 3, stride, norm="bn", activation=activation
            )
        )
        stem.add(Conv2dNormActivation(32, 3, norm="bn", activation=activation))
        stem.add(Conv2dNormActivation(64, 3, norm="bn", activation=activation))
        if not small_input:
            stem.add(MaxPool2D(3, 2, padding="same"))
        x = stem(inputs)

        # Layer1: 56x56
        x = self.__construct_residual_block(
            x,
            cfg[0],
            64,
            width,
            1,
            groups,
            use_bottleneck,
            norm,
            activation,
            "layer1",
        )

        # Layer2: 28x28
        x = self.__construct_residual_block(
            x,
            cfg[1],
            128,
            width,
            2,
            groups,
            use_bottleneck,
            norm,
            activation,
            "layer2",
        )

        # Layer3: 14x14
        x = self.__construct_residual_block(
            x,
            cfg[2],
            256,
            width,
            2,
            groups,
            use_bottleneck,
            norm,
            activation,
            "layer3",
        )

        # Layer4: 7x7
        x = self.__construct_residual_block(
            x,
            cfg[0],
            512,
            width,
            2,
            groups,
            use_bottleneck,
            norm,
            activation,
            "layer4",
        )

        total_layers = 2
        n = 3 if use_bottleneck else 2
        for c in cfg:
            total_layers += c * n
        super().__init__(
            inputs=inputs, outputs=x, name=f"ResNet{total_layers}"
        )

    def __construct_residual_block(
        self,
        x,
        n_layers: int,
        filters: int,
        width: int,
        strides: Union[int, Tuple[int, int]],
        groups: int,
        use_bottleneck: bool,
        norm: str = "bn",
        activation: str = "relu",
        name: str = None,
    ):
        expansion = 4 if use_bottleneck else 1

        if strides != 1 or self.filters != filters * expansion:
            downsample = Sequential(
                [
                    AvgPool2D(2, strides, padding="same"),
                    Conv2dNorm(filters * expansion, 1, norm="bn"),
                ],
                name=f"{name}_downsample",
            )
        else:
            downsample = None

        x = residual_block(
            x,
            filters,
            use_bottleneck,
            width=width,
            strides=strides,
            groups=groups,
            norm=norm,
            activation=activation,
            downsample=downsample,
            name=f"{name}_block_1",
        )

        self.filters *= expansion

        for idx in range(1, n_layers):
            x = residual_block(
                x,
                filters,
                use_bottleneck,
                width=width,
                strides=1,
                groups=groups,
                norm=norm,
                activation=activation,
                downsample=None,
                name=f"{name}_block_{idx + 1}",
            )

        return x


def resnet18(
    inputs,
    *,
    norm: str = "bn",
    activation: str = "relu",
    small_input: bool = False,
) -> Model:
    model = ResNet(
        inputs,
        cfg=(2, 2, 2, 2),
        use_bottleneck=False,
        norm=norm,
        activation=activation,
        small_input=small_input,
    )
    return model


def resnet34(
    inputs,
    *,
    norm: str = "bn",
    activation: str = "relu",
    small_input: bool = False,
) -> Model:
    model = ResNet(
        inputs,
        cfg=(3, 4, 6, 3),
        use_bottleneck=False,
        norm=norm,
        activation=activation,
        small_input=small_input,
    )
    return model


def resnet50(
    inputs,
    *,
    norm: str = "bn",
    activation: str = "relu",
    small_input: bool = False,
) -> Model:
    model = ResNet(
        inputs,
        cfg=(3, 4, 6, 3),
        use_bottleneck=True,
        norm=norm,
        activation=activation,
        small_input=small_input,
    )
    return model


def resnext50_32x4d(
    inputs,
    *,
    norm: str = "bn",
    activation: str = "relu",
    small_input: bool = False,
) -> Model:
    model = ResNet(
        inputs,
        cfg=(3, 4, 6, 3),
        width=4,
        groups=32,
        use_bottleneck=True,
        norm=norm,
        activation=activation,
        small_input=small_input,
    )
    return model


def resnet101(
    inputs,
    *,
    norm: str = "bn",
    activation: str = "relu",
    small_input: bool = False,
) -> Model:
    model = ResNet(
        inputs,
        cfg=(3, 4, 23, 3),
        use_bottleneck=True,
        norm=norm,
        activation=activation,
        small_input=small_input,
    )
    return model


def resnet152(
    inputs,
    *,
    norm: str = "bn",
    activation: str = "relu",
    small_input: bool = False,
) -> Model:
    model = ResNet(
        inputs,
        cfg=(3, 8, 36, 3),
        use_bottleneck=True,
        norm=norm,
        activation=activation,
        small_input=small_input,
    )
    return model
