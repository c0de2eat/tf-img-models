import os
import random

import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import (
    CSVLogger,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.mixed_precision import global_policy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import tensorflow_datasets as tfds
from tfim import setup_tf, CosineDecayWithWarmup
from tfim.data import batchify
from tfim.modeling import backbones
from tfim.modeling.layers import dense


os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)
rng = tf.random.Generator.from_seed(1234)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    tb_dir = os.path.join(args.out_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    state_dir = os.path.join(ckpt_dir, "state")
    os.makedirs(state_dir, exist_ok=True)
    model_dir = os.path.join(ckpt_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    weights_dir = os.path.join(ckpt_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_name = os.path.join(weights_dir, "weights")

    img_pad_target_h = int(args.img_height * 1.25)
    img_pad_target_w = int(args.img_width * 1.25)
    img_pad_offset_h = (img_pad_target_h - args.img_height) // 2
    img_pad_offset_w = (img_pad_target_w - args.img_width) // 2

    def process_train(img, label):
        img = tf.image.pad_to_bounding_box(
            img,
            img_pad_offset_h,
            img_pad_offset_w,
            img_pad_target_h,
            img_pad_target_w,
        )
        img = tf.image.stateless_random_crop(
            img,
            (args.img_height, args.img_width, 3),
            seed=rng.uniform_full_int((2,), dtype=tf.int32),
        )
        img = tf.image.stateless_random_flip_left_right(
            img, seed=rng.uniform_full_int((2,), dtype=tf.int32)
        )
        img = tf.divide(tf.cast(img, tf.float32), 255.0)
        label = tf.one_hot(label, 10, dtype=tf.float32)
        return img, label

    def process_val(img, label):
        img = tf.image.resize_with_pad(img, args.img_height, args.img_width)
        img = tf.divide(tf.cast(img, tf.float32), 255.0)
        label = tf.one_hot(label, 10, dtype=tf.float32)
        return img, label

    train_dataset, val_dataset = tfds.load(
        "cifar10",
        split=["train", "test"],
        data_dir=args.data_dir,
        as_supervised=True,
    )

    train_dataset = train_dataset.map(process_train)
    val_dataset = val_dataset.map(process_val, tf.data.AUTOTUNE)
    train_loader = batchify(train_dataset, args.batch_size, shuffle=True)
    val_loader = batchify(val_dataset, args.batch_size, shuffle=False)

    s = (
        "---------------------------------------------------------\n"
        "| Dataset\t| # of samples\t| # of iterations\t|\n"
        "---------------------------------------------------------\n"
        f"| Train\t\t| {len(train_dataset)}\t\t| "
        f"{len(train_loader)}\t\t\t|\n"
    )
    s += f"| Val\t\t| {len(val_dataset)}\t\t| " f"{len(val_loader)}\t\t\t|\n"
    print(s + "".join(["-" for _ in range(57)]))

    strategy = setup_tf(gpus=args.gpus, mixed_precision_training=True)
    loss_fn = CategoricalCrossentropy(
        from_logits=True,
        label_smoothing=tf.convert_to_tensor(
            args.label_smoothing, dtype=global_policy().compute_dtype
        ),
    )

    optimizer = Adam(
        CosineDecayWithWarmup(
            args.learning_rate,
            warmup_epochs=args.epochs // 10,
            total_epochs=args.epochs,
            steps_per_epoch=len(train_loader),
        )
    )

    with strategy.scope():
        inputs = Input((args.img_height, args.img_width, 3))
        backbone = getattr(backbones, args.backbone)(
            inputs,
            small_input=True,
            # Normalizations
            ibn=args.instance_batch_norm,
            # Attention blocks
            se=args.squeeze_excitation,
            bam=args.bottleneck_attention,
            cbam=args.convolutional_bottleneck_attention,
            weight_decay=args.weight_decay,
        )
        feature_map = backbone(inputs)
        features = GlobalAvgPool2D()(feature_map)
        if args.dropout > 0:
            features = tf.keras.layers.Dropout(args.dropout)(features)
        outputs = dense(10, weight_decay=args.weight_decay, name="classifier")(
            features
        )
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=["accuracy"],
        )
    plot_model(
        backbone,
        os.path.join(args.out_dir, "backbone.png"),
        True,
        show_layer_names=True,
    )
    plot_model(
        model,
        os.path.join(args.out_dir, "model.png"),
        True,
        show_layer_names=True,
    )

    callbacks = [
        TensorBoard(
            log_dir=os.path.join(tb_dir),
            histogram_freq=5,
            write_images=True,
        ),
        CSVLogger(
            os.path.join(args.out_dir, "training.log"),
            separator=",",
            append=False,
        ),
        ModelCheckpoint(
            state_dir, monitor="val_accuracy", save_best_only=True
        ),
        ModelCheckpoint(
            weights_name,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]

    model.fit(
        train_loader,
        epochs=args.epochs,
        callbacks=callbacks,
        validation_data=val_loader,
    )

    with strategy.scope():
        model.load_weights(weights_name)
    model.save(model_dir, include_optimizer=False)
    model.evaluate(val_loader)

    predictions, labels = [], []
    for img, label in val_loader:
        label = tf.argmax(label, -1)
        labels.extend(label)
        outputs = model(img)
        outputs = tf.argmax(outputs, -1)
        predictions.extend(outputs)
    report = classification_report(labels, predictions)
    with open(os.path.join(args.out_dir, "report.txt"), "w+") as f:
        f.write(str(report))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # ==================================================================================
    # Data
    # ==================================================================================
    parser.add_argument(
        "-b", "--batch-size", default=64, type=int, help="Batch size per GPU."
    )
    parser.add_argument(
        "--img-height", default=224, type=int, help="Image height."
    )
    parser.add_argument(
        "--img-width", default=224, type=int, help="Image width."
    )
    # ==================================================================================

    # ==================================================================================
    # Directories
    # ==================================================================================
    parser.add_argument(
        "--data-dir", type=str, required=True, help="TFDS root directory."
    )
    parser.add_argument(
        "--out-dir", type=str, required=True, help="Output directory."
    )

    # ==================================================================================
    # Loss
    # ==================================================================================
    parser.add_argument(
        "--label-smoothing",
        default=0.1,
        type=float,
        help="Label smoothing rate.",
    )
    # ----------------------------------------------------------------------------------

    # ==================================================================================
    # Model
    # ==================================================================================
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        help="Backbone to be used.",
    )
    parser.add_argument(
        "--bottleneck-attention",
        action="store_true",
        help="Use bottleneck attention module (BAM).",
    )
    parser.add_argument(
        "--convolutional-bottleneck-attention",
        action="store_true",
        help="Use convolutional bottleneck attention module (CBAM).",
    )
    parser.add_argument(
        "--dropout", default=0.0, type=float, help="Dropout rate."
    )
    parser.add_argument(
        "--instance-batch-norm",
        action="store_true",
        help="Use instance batch norm. (IBN).",
    )
    parser.add_argument(
        "--squeeze-excitation",
        action="store_true",
        help="Use squeeze and excitation (SE) module.",
    )
    parser.add_argument(
        "-wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        help="Label smoothing rate.",
    )
    # ==================================================================================

    # ==================================================================================
    # Training
    # ==================================================================================
    parser.add_argument(
        "-e", "--epochs", default=50, type=int, help="Total training epochs."
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=0,
        type=int,
        help="GPU ID to be used for training based on `nvidia-smi`",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-3,
        type=float,
        help="Learning rate.",
    )
    # ==================================================================================

    args = parser.parse_args()
    main(args)
