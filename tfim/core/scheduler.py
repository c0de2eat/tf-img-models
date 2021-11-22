import math

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


__all__ = ["CosineDecayWithWarmup"]


class CosineDecayWithWarmup(LearningRateSchedule):
    """Scheduler with linear warmup + consine decay.

    Args:
        val: Initial value
        warmup_epochs: Number of epochs for warming up
        total_epochs: Total number of training epochs
        steps_per_epoch: Number of iteration of batches per epoch
    """

    def __init__(
        self,
        val: float,
        *,
        warmup_epochs: int,
        total_epochs: int,
        steps_per_epoch: int,
    ):
        super().__init__()
        self._val = val
        self._warmup_steps = steps_per_epoch * warmup_epochs
        self._total_steps = steps_per_epoch * total_epochs
        self._cosine_steps = self._total_steps - self._warmup_steps

    def __call__(self, global_step) -> tf.Tensor:
        global_step = tf.cast(global_step, dtype=tf.float32)

        warmup_val = tf.multiply(
            tf.divide(global_step, self._warmup_steps), self._val
        )
        cosine_val = tf.multiply(
            tf.divide(self._val, 2),
            tf.add(
                tf.cos(
                    tf.multiply(
                        math.pi,
                        tf.divide(
                            tf.subtract(global_step, self._warmup_steps),
                            self._cosine_steps,
                        ),
                    )
                ),
                1.0,
            ),
        )
        val = tf.where(
            tf.less_equal(global_step, self._warmup_steps),
            warmup_val,
            cosine_val,
        )
        return val

    def get_config(self):
        return {
            "cosine_steps": self._cosine_steps,
            "val": self._val,
            "total_steps": self._total_steps,
            "warmup_steps": self._warmup_steps,
        }
