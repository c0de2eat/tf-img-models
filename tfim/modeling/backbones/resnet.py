from typing import Tuple, Union

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import AvgPool2D, InputSpec

from tfim.modeling.layers import Conv2dNorm, Conv2dNormReLU
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
        normalization: str = "bn",
        small_input: bool = False,
    ):
        self.input_spec = InputSpec(shape=(None,) + inputs.shape)

        # Stem: 56x56
        stem = Sequential(name="stem")
        stride = 1 if small_input else 2
        stem.add(Conv2dNormReLU(64, 3, stride, normalization="bn"))
        stem.add(Conv2dNormReLU(64, 3, normalization="bn"))
        stem.add(Conv2dNormReLU(64, 3, normalization="bn"))
        x = stem(inputs)

        # Layer1: 56x56
        x = self.__construct_residual_block(
            x,
            cfg[0],
            64,
            width,
            2,
            groups,
            use_bottleneck,
            normalization,
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
            normalization,
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
            normalization,
            "layer3",
        )
        print("layer3", x.shape)

        # Layer4: 7x7
        x = self.__construct_residual_block(
            x,
            cfg[0],
            512,
            width,
            2,
            groups,
            use_bottleneck,
            normalization,
            "layer4",
        )

        total_layers = 2
        n = 3 if use_bottleneck else 2
        for c in cfg:
            total_layers += c * n
        print(f"=> # of layers in ResNet: {total_layers}")
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
        normalization: str = "bn",
        name: str = None,
    ):
        expansion = 4 if use_bottleneck else 1

        if strides != 1 or self.filters != filters * expansion:
            downsample = Sequential(
                [
                    AvgPool2D(2, strides, padding="same"),
                    Conv2dNorm(filters * expansion, 1, normalization="bn"),
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
            normalization=normalization,
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
                normalization=normalization,
                downsample=None,
                name=f"{name}_block_{idx + 1}",
            )

        return x


def resnet18(
    inputs, *, normalization: str = "bn", small_input: bool = False
) -> Model:
    model = ResNet(
        inputs,
        cfg=(2, 2, 2, 2),
        use_bottleneck=False,
        normalization=normalization,
        small_input=small_input,
    )
    return model


def resnet34(
    inputs, *, normalization: str = "bn", small_input: bool = False
) -> Model:
    model = ResNet(
        inputs,
        cfg=(3, 4, 6, 3),
        use_bottleneck=False,
        normalization=normalization,
        small_input=small_input,
    )
    return model


def resnet50(
    inputs, *, normalization: str = "bn", small_input: bool = False
) -> Model:
    model = ResNet(
        inputs,
        cfg=(3, 4, 6, 3),
        use_bottleneck=True,
        normalization=normalization,
        small_input=small_input,
    )
    return model


def resnext50_32x4d(
    inputs, *, normalization: str = "bn", small_input: bool = False
) -> Model:
    model = ResNet(
        inputs,
        cfg=(3, 4, 6, 3),
        width=4,
        groups=32,
        use_bottleneck=True,
        normalization=normalization,
        small_input=small_input,
    )
    return model


def resnet101(
    inputs, *, normalization: str = "bn", small_input: bool = False
) -> Model:
    model = ResNet(
        inputs,
        cfg=(3, 4, 23, 3),
        use_bottleneck=True,
        normalization=normalization,
        small_input=small_input,
    )
    return model


def resnet152(
    inputs, *, normalization: str = "bn", small_input: bool = False
) -> Model:
    model = ResNet(
        inputs,
        cfg=(3, 8, 36, 3),
        use_bottleneck=True,
        normalization=normalization,
        small_input=small_input,
    )
    return model
