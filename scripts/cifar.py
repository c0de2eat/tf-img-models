import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import (
    CSVLogger,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.layers import Dense, GlobalAvgPool2D
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model
import tensorflow_datasets as tfds
from tfim import setup_tf, CosineDecayWithWarmup
from tfim.data import batchify
from tfim.modeling.backbones import resnet18

# os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)


def main(
    batch_size=128,
    learning_rate=3e-3,
    warmup_epochs=5,
    epochs=50,
    img_height=32,
    img_width=32,
    weight_decay=1e-4,
    data_dir="/home/xz/workspace/datasets/tfds",
    out_dir="output",
    gpus=0,
):
    os.makedirs(out_dir, exist_ok=True)
    tb_dir = os.path.join(out_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    state_dir = os.path.join(ckpt_dir, "state")
    os.makedirs(state_dir, exist_ok=True)
    model_dir = os.path.join(ckpt_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    weights_dir = os.path.join(ckpt_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    """
    scale hue, saturation, brightness [0.6, 1.4]
    pca noise normal distribution 0, 0.1
    xavier uniform init, a= sqrt(6 / (d_in + d_out))
    label smoothing
    mixup training
    RandAugment
    """

    def process_train(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.pad_to_bounding_box(img, 6, 6, img_height + 12, img_width + 12)
        img = tf.image.random_crop(img, (img_height, img_width, 3))
        img = tf.image.resize_with_pad(img, img_height, img_width)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img, label

    def process_val(img, label):
        img = tf.image.resize_with_pad(img, img_height, img_width)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img, label

    train_dataset, val_dataset = tfds.load(
        "cifar10", split=["train", "test"], data_dir=data_dir, as_supervised=True
    )

    train_dataset = train_dataset.map(process_train)
    val_dataset = val_dataset.map(process_val)
    train_loader = batchify(train_dataset, batch_size, shuffle=True)
    val_loader = batchify(val_dataset, batch_size, shuffle=False)

    s = (
        "---------------------------------------------------------\n"
        "| Dataset\t| # of samples\t| # of iterations\t|\n"
        "---------------------------------------------------------\n"
        f"| Train\t\t| {len(train_dataset)}\t\t| "
        f"{len(train_loader)}\t\t\t|\n"
    )
    s += f"| Val\t\t| {len(val_dataset)}\t\t| " f"{len(val_loader)}\t\t\t|\n"
    print(s + "".join(["-" for _ in range(57)]))

    loss_fn = SparseCategoricalCrossentropy(from_logits=True)

    strategy = setup_tf(gpus=gpus, mixed_precision_training=True)
    # nesterov accelerated gradient NAG descent
    optimizer = SGD(
        CosineDecayWithWarmup(
            learning_rate,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
            steps_per_epoch=len(train_loader),
        ),
        momentum=0.9,
        nesterov=True,
    )

    with strategy.scope():
        inputs = Input((32, 32, 3))
        feature_map = resnet18(inputs, weight_decay=weight_decay)(inputs)
        features = GlobalAvgPool2D()(feature_map)
        # if dropout is not None:
        #     features = tf.keras.layers.Dropout(dropout)(features)
        outputs = Dense(10, name="classifier")(features)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    plot_model(model, os.path.join(out_dir, "model.png"), True, show_layer_names=True)

    callbacks = [
        TensorBoard(log_dir=os.path.join(tb_dir), histogram_freq=5, write_images=True,),
        CSVLogger(os.path.join(out_dir, "training.log"), separator=",", append=False),
        ModelCheckpoint(model_dir, monitor="val_accuracy", save_best_only=True),
        ModelCheckpoint(
            weights_dir,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]

    model.fit(
        train_loader, epochs=epochs, callbacks=callbacks, validation_data=val_loader
    )

    with strategy.scope():
        model.load_weights(weights_dir)
    model.save(model_dir, include_optimizer=False)
    model.evaluate(val_loader)


if __name__ == "__main__":
    main()
