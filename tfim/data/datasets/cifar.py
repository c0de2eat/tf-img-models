import os
import pickle

import numpy as np
import tensorflow as tf


__all__ = ["cifar10", "cifar100"]


def _unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def cifar10(root):
    images, labels = [], []
    for i in range(1, 6):
        data = _unpickle(os.path.join(root, f"data_batch_{i}"))
        images.extend(data[b"data"])
        labels.extend(data[b"labels"])
    images = np.array(images, np.uint8)
    images = np.reshape(images, [-1, 3, 32, 32])
    images = np.transpose(images, [0, 2, 3, 1])
    labels = np.array(labels)
    train = tf.data.Dataset.from_tensor_slices((list(images), list(labels)))

    images, labels = [], []
    data = _unpickle(os.path.join(root, "test_batch"))
    images.extend(data[b"data"])
    labels.extend(data[b"labels"])
    images = np.array(images, np.uint8)
    images = np.reshape(images, [-1, 3, 32, 32])
    images = np.transpose(images, [0, 2, 3, 1])
    labels = np.array(labels)
    val = tf.data.Dataset.from_tensor_slices((list(images), list(labels)))

    names = _unpickle(os.path.join(root, "batches.meta"))[b"label_names"]
    names = [x.decode("utf-8") for x in names]

    return train, val, names


def cifar100(root):
    images, labels = [], []
    data = _unpickle(os.path.join(root, "train"))
    images.extend(data[b"data"])
    labels.extend(data[b"fine_labels"])
    images = np.array(images, np.uint8)
    images = np.reshape(images, [-1, 3, 32, 32])
    images = np.transpose(images, [0, 2, 3, 1])
    labels = np.array(labels)
    train = list(zip(images, labels))
    train = tf.data.Dataset.from_tensor_slices((list(images), list(labels)))

    images, labels = [], []
    data = _unpickle(os.path.join(root, "test"))
    images.extend(data[b"data"])
    labels.extend(data[b"fine_labels"])
    images = np.array(images, np.uint8)
    images = np.reshape(images, [-1, 3, 32, 32])
    images = np.transpose(images, [0, 2, 3, 1])
    labels = np.array(labels)
    val = list(zip(images, labels))
    val = tf.data.Dataset.from_tensor_slices((list(images), list(labels)))

    names = _unpickle(os.path.join(root, "meta"))[b"fine_label_names"]
    names = [x.decode("utf-8") for x in names]

    return train, val, names
