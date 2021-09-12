from typing import Callable, Tuple, Union

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import AvgPool2D, InputSpec
from tfim.modeling.layers import conv2d_bn, conv2d_bn_relu, Identity
from tfim.modeling.modules import (
    bottleneck_block,
    BottleneckAttentionModule,
    residual_block,
)


__all__ = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "ResNet"]


class ResNet(Model):
    filters = 64

    def __init__(
        self,
        inputs,
        block: Callable,
        cfg: Tuple[int, int, int, int],
        *,
        small_input: bool = False,
        # Normalizations
        ibn: bool = False,
        # Attention blocks
        se: bool = False,
        bam: bool = False,
        cbam: bool = False,
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
            x,
            block,
            64,
            cfg[0],
            first_stride,
            # Normalizations
            ibn=ibn,
            # Attention blocks
            se=se,
            cbam=cbam,
            weight_decay=weight_decay,
            name="layer1",
        )
        if bam:
            x = BottleneckAttentionModule(weight_decay=weight_decay, name="bam1")(x)

        # Layer2: 28x28
        x = self.__construct_residual_block(
            x,
            block,
            128,
            cfg[1],
            2,
            # Normalizations
            ibn=ibn,
            # Attention blocks
            se=se,
            cbam=cbam,
            weight_decay=weight_decay,
            name="layer2",
        )
        if bam:
            x = BottleneckAttentionModule(weight_decay=weight_decay, name="bam2")(x)

        # Layer3: 14x14
        x = self.__construct_residual_block(
            x,
            block,
            256,
            cfg[2],
            2,
            # Normalizations
            ibn=ibn,
            # Attention blocks
            se=se,
            cbam=cbam,
            weight_decay=weight_decay,
            name="layer3",
        )
        if bam:
            x = BottleneckAttentionModule(weight_decay=weight_decay, name="bam3")(x)

        # Layer4: 7x7
        x = self.__construct_residual_block(
            x,
            block,
            512,
            cfg[3],
            2,
            # Normalizations
            ibn=False,
            # Attention blocks
            se=False,
            cbam=False,
            weight_decay=weight_decay,
            name="layer4",
        )

        total_layers = 2
        n = 2 if "residual" in str(block) else 3
        for c in cfg:
            total_layers += c * n
        print(f"=> # of layers in ResNet: {total_layers}")
        super().__init__(inputs=inputs, outputs=x, name=f"ResNet{total_layers}")

    def __construct_residual_block(
        self,
        x,
        block: Callable,
        filters: int,
        n_layers: int,
        strides: Union[int, Tuple[int, int]],
        # Normalizations
        ibn: bool = False,
        # Attention blocks
        se: bool = False,
        cbam: bool = False,
        weight_decay: float = None,
        name: str = None,
    ) -> Sequential:
        expansion = 1 if "residual" in str(block) else 4

        if strides != 1 or self.filters != filters * expansion:
            downsample = Sequential(
                [
                    AvgPool2D(2, strides, padding="same"),
                    conv2d_bn(filters * expansion, 1, weight_decay=weight_decay),
                ],
                name=f"{name}_downsample",
            )
        else:
            downsample = None

        x = block(
            x,
            filters,
            strides=strides,
            downsample=downsample,
            # Normalizations
            ibn=ibn,
            # Attention blocks
            se=se,
            cbam=cbam,
            weight_decay=weight_decay,
            name=f"{name}_block_1",
        )

        self.filters *= expansion

        for idx in range(1, n_layers):
            x = block(
                x,
                filters,
                strides=1,
                # Normalizations
                ibn=ibn,
                # Attention blocks
                se=se,
                cbam=cbam,
                weight_decay=weight_decay,
                name=f"{name}_block_{idx + 1}",
            )

        return x


def resnet18(
    inputs,
    *,
    small_input: bool = False,
    # Normalizations
    ibn: bool = False,
    # Attention blocks
    se: bool = False,
    bam: bool = False,
    cbam: bool = False,
    weight_decay: float = None,
) -> Model:
    model = ResNet(
        inputs,
        residual_block,
        (2, 2, 2, 2),
        small_input=small_input,
        # Normalizations
        ibn=ibn,
        # Attention blocks
        se=se,
        bam=bam,
        cbam=cbam,
        weight_decay=weight_decay,
    )
    return model


def resnet34(
    inputs,
    *,
    small_input: bool = False,
    # Normalizations
    ibn: bool = False,
    # Attention blocks
    se: bool = False,
    bam: bool = False,
    cbam: bool = False,
    weight_decay: float = None,
) -> Model:
    model = ResNet(
        inputs,
        residual_block,
        (3, 4, 6, 3),
        small_input=small_input,
        # Normalizations
        ibn=ibn,
        # Attention blocks
        se=se,
        bam=bam,
        cbam=cbam,
        weight_decay=weight_decay,
    )
    return model


def resnet50(
    inputs,
    *,
    small_input: bool = False,
    # Normalizations
    ibn: bool = False,
    # Attention blocks
    se: bool = False,
    bam: bool = False,
    cbam: bool = False,
    weight_decay: float = None,
) -> Model:
    model = ResNet(
        inputs,
        bottleneck_block,
        (3, 4, 6, 3),
        small_input=small_input,
        # Normalizations
        ibn=ibn,
        # Attention blocks
        se=se,
        bam=bam,
        cbam=cbam,
        weight_decay=weight_decay,
    )
    return model


def resnet101(
    inputs,
    *,
    small_input: bool = False,
    # Normalizations
    ibn: bool = False,
    # Attention blocks
    se: bool = False,
    bam: bool = False,
    cbam: bool = False,
    weight_decay: float = None,
) -> Model:
    model = ResNet(
        inputs,
        bottleneck_block,
        (3, 4, 23, 3),
        small_input=small_input,
        # Normalizations
        ibn=ibn,
        # Attention blocks
        se=se,
        bam=bam,
        cbam=cbam,
        weight_decay=weight_decay,
    )
    return model


def resnet152(
    inputs,
    *,
    small_input: bool = False,
    # Normalizations
    ibn: bool = False,
    # Attention blocks
    se: bool = False,
    bam: bool = False,
    cbam: bool = False,
    weight_decay: float = None,
) -> Model:
    model = ResNet(
        inputs,
        bottleneck_block,
        (3, 8, 36, 3),
        small_input=small_input,
        # Normalizations
        ibn=ibn,
        # Attention blocks
        se=se,
        bam=bam,
        cbam=cbam,
        weight_decay=weight_decay,
    )
    return model
