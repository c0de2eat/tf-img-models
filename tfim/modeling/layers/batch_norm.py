import tensorflow.keras as keras


__all__ = ["batch_norm"]


def batch_norm(
    weight_decay: float = None, norm_weight_zero_init: bool = False
) -> keras.layers.BatchNormalization:
    axis = -1 if keras.backend.image_data_format() == "channels_last" else 1
    gamma_initializer = "zeros" if norm_weight_zero_init else "ones"
    return keras.layers.BatchNormalization(
        axis,
        gamma_initializer=gamma_initializer,
        gamma_regularizer=keras.regularizers.l2(weight_decay),
    )
