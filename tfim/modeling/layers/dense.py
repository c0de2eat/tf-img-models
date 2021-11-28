from tensorflow.keras.layers import Dense as D


__all__ = ["Dense"]


def Dense(
    units: int, kernel_initializer: str = "glorot_uniform", name: str = None,
) -> D:
    return D(units, kernel_initializer=kernel_initializer, name=name)
