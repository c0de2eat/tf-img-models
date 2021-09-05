from typing import List, Union

import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy


__all__ = ["setup_tf"]


def setup_tf(
    *,
    gpus: Union[int, List[int]],
    mixed_precision_training: bool = True,
    use_cpu: bool = False,
    debug: bool = False,
) -> tf.distribute.Strategy:
    """Setup TensorFlow environment.

    Args:
        gpus: IDs of the GPU to be used
        mixed_precision_training: Whether to use mixed precicion for faster training
        use_cpu: Whether to use CPU
        debug: Whether to use run eagerly for the ease of debugging
    """
    print("Setting up TensorFlow...")

    if debug:
        print("> Eager mode enabled for easy debug")
        tf.config.run_functions_eagerly(True)

    if use_cpu:  # Ignore GPU only settings
        print("> Running on CPU")
        return tf.distribute.OneDeviceStrategy("CPU")

    assert tf.config.list_physical_devices(
        "GPU"
    ), "GPU is not available! Please use CPU mode."

    print("> Running on GPU(s)")
    physical_gpus = tf.config.list_physical_devices("GPU")
    print(f"  >> Physical GPUs: {physical_gpus}")
    logical_gpus = tf.config.list_logical_devices("GPU")
    print(f"  >> Logical GPU: {logical_gpus}")

    gids = (gpus,) if isinstance(gpus, int) else gpus
    print(f"  >> Using GPU: {','.join([str(x) for x in gids])}")

    try:
        for gid in gids:
            tf.config.set_visible_devices(physical_gpus[gid], "GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

    if mixed_precision_training:
        print("  >> Using mixed precision")
        set_global_policy("mixed_float16")

    return tf.distribute.MirroredStrategy()
