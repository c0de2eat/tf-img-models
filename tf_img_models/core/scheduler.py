import math

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


__all__ = ["CosineDecayWithWarmup"]


class CosineDecayWithWarmup(LearningRateSchedule):
    def __init__(
        self, *, lr: float, warmup_epochs: int, total_epochs: int, steps_per_epoch: int,
    ):
        """Scheduler with linear warmup + consine decay.

        Args:
            lr: Base learning rate
            warmup_epochs: Number of epochs for warming up
            total_epochs: Total number of training epochs
            steps_per_epoch: Number of iteration of batches per epoch
        """
        super().__init__()
        self._lr = lr
        self._warmup_steps = steps_per_epoch * warmup_epochs
        self._total_steps = steps_per_epoch * total_epochs
        self._cosine_steps = self._total_steps - self._warmup_steps

    def __call__(self, global_step) -> tf.Tensor:
        global_step = tf.cast(global_step, dtype=tf.float32)

        warmup_lr = tf.multiply(tf.divide(global_step, self._warmup_steps), self._lr)
        cosine_lr = tf.multiply(
            tf.divide(self._lr, 2),
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
        lr = tf.where(
            tf.less_equal(global_step, self._warmup_steps), warmup_lr, cosine_lr
        )
        return lr

    def get_config(self):
        return {
            "cosine_steps": self._cosine_steps,
            "lr": self._lr,
            "total_steps": self._total_steps,
            "warmup_steps": self._warmup_steps,
        }
