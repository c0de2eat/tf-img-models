from tensorflow.keras.backend import image_data_format
from tensorflow.keras.layers import BatchNormalization, Layer
from tensorflow.keras.regularizers import l2


__all__ = ["batch_norm"]


def batch_norm(weight_decay: float = None) -> Layer:
    axis = -1 if image_data_format() == "channels_last" else 1
    return BatchNormalization(axis, gamma_regularizer=l2(weight_decay))
