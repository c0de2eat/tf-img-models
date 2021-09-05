from tensorflow.keras.backend import image_data_format
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import L2


__all__ = ["batch_norm"]


def batch_norm(
    *, norm_weight_zero_init: bool = False, weight_decay: float = None
) -> BatchNormalization:
    axis = -1 if image_data_format() == "channels_last" else 1
    gamma_initializer = "zeros" if norm_weight_zero_init else "ones"
    return BatchNormalization(
        axis, gamma_initializer=gamma_initializer, gamma_regularizer=L2(weight_decay),
    )
