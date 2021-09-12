import tensorflow as tf


__all__ = ["batchify"]


def batchify(
    dataset: tf.data.Dataset, batch_size: int, *, shuffle: bool
) -> tf.data.Dataset:
    if shuffle:
        dataset = dataset.shuffle(len(dataset))
    dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
