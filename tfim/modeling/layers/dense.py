from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import L2


__all__ = ["dense"]


def dense(
    units: int,
    kernel_initializer: str = "glorot_uniform",
    weight_decay: float = None,
    name: str = None,
) -> Dense:
    return Dense(
        units,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=L2(weight_decay),
        name=name,
    )
