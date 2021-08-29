import tensorflow as tf


__all__ = ["batchify"]


def batchify(
    dataset: tf.data.Dataset, batch_size: int, *, shuffle: bool
) -> tf.data.Dataset:
    if shuffle:
        n = len(dataset)
        dataset = dataset.shuffle(n, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
