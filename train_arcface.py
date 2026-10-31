"""Train a MobileNetV2 face-embedding model with ArcFace loss on LFW.

    python train_arcface.py --epochs-head 8 --epochs-ft 12

Outputs: embedder.keras, training_curves.png
(Use a GPU - Google Colab's free T4 is fine. On CPU, lower --epochs-*.)
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model

from common import IMG, load_lfw


class ArcFace(layers.Layer):
    """Additive angular margin head (Deng et al., 2019)."""

    def __init__(self, n_classes, s=30.0, m=0.5, **kw):
        super().__init__(**kw)
        self.n_classes, self.s, self.m = n_classes, s, m

    def build(self, input_shape):
        dim = input_shape[0][-1]
        self.W = self.add_weight(name="W", shape=(dim, self.n_classes),
                                 initializer="glorot_uniform", trainable=True)

    def call(self, inputs):
        emb, labels = inputs
        labels = tf.reshape(tf.cast(labels, tf.int32), [-1])
        x = tf.math.l2_normalize(emb, axis=1)
        W = tf.math.l2_normalize(self.W, axis=0)
        cos = tf.clip_by_value(x @ W, -1.0 + 1e-7, 1.0 - 1e-7)
        sin = tf.sqrt(1.0 - tf.square(cos))
        cos_m = tf.math.cos(self.m)
        sin_m = tf.math.sin(self.m)
        cos_margin = cos * cos_m - sin * sin_m                  # cos(theta + m)
        th = tf.math.cos(3.141592653589793 - self.m)
        mm = tf.math.sin(3.141592653589793 - self.m) * self.m
        cos_margin = tf.where(cos > th, cos_margin, cos - mm)   # keep monotonic
        one_hot = tf.one_hot(labels, self.n_classes)
        return self.s * (one_hot * cos_margin + (1.0 - one_hot) * cos)


def build_embedder():
    inp = layers.Input((IMG, IMG, 3), name="image")
    x = layers.Rescaling(1 / 127.5, offset=-1)(inp)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
    x = base(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    out = layers.UnitNormalization(axis=-1, name="embedding")(x)
    return Model(inp, out, name="embedder"), base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-head", type=int, default=8)
    ap.add_argument("--epochs-ft", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    d = load_lfw()
    embedder, base = build_embedder()

    img = layers.Input((IMG, IMG, 3), name="image")
    lab = layers.Input((), dtype="int32", name="label")
    logits = ArcFace(d["n_classes"], name="arcface")([embedder(img), lab])
    model = Model([img, lab], logits)

    train = ({"image": d["Xtr"], "label": d["ytr"]}, d["ytr"])
    val = ({"image": d["Xval"], "label": d["yval"]}, d["yval"])
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    hist = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    def fit(epochs, lr):
        model.compile(tf.keras.optimizers.Adam(lr), loss=loss, metrics=["accuracy"])
        h = model.fit(*train, validation_data=val, epochs=epochs, batch_size=a.batch)
        for k in hist:
            hist[k] += h.history[k]

    base.trainable = False                       # stage 1: train head only
    fit(a.epochs_head, 1e-3)

    base.trainable = True                        # stage 2: fine-tune top of backbone
    for layer in base.layers[:-40]:
        layer.trainable = False
    fit(a.epochs_ft, 1e-4)

    embedder.save("embedder.keras")
    print("Saved embedder.keras")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(hist["loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("ArcFace loss"); ax[0].legend()
    ax[1].plot(hist["accuracy"], label="train"); ax[1].plot(hist["val_accuracy"], label="val")
    ax[1].set_title("Identity accuracy (seen identities)"); ax[1].legend()
    for x in ax:
        x.set_xlabel("epoch")
    fig.tight_layout(); fig.savefig("training_curves.png", dpi=150)


if __name__ == "__main__":
    main()
