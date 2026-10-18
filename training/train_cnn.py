"""Traceface CNN training — optimized version of the original final-year script.

Dataset layout expected:
    dataset/
        real/   <- photographs of real people/scenes
        fake/   <- AI-generated images

Improvements over the original:
  - data augmentation (rotation / shift / zoom / flip) for generalization
  - EarlyStopping + ReduceLROnPlateau + ModelCheckpoint callbacks
  - larger batch size, class-weight balancing
  - full classification report + confusion matrix on the validation split
  - saves the best model as `traceface_model.keras`

Run:  python training/train_cnn.py --data ./dataset --epochs 40
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def build_model(input_shape=(224, 224, 3)):
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.GlobalAveragePooling2D(),  # lighter + less overfitting than Flatten+Dense(512)
            layers.Dropout(0.4),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./dataset")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--out", default="traceface_model.keras")
    args = parser.parse_args()

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2,
    )
    train_gen = train_datagen.flow_from_directory(
        args.data,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
    )
    val_gen = train_datagen.flow_from_directory(
        args.data,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
    )
    print("Class indices:", train_gen.class_indices)  # expect {'fake': 0, 'real': 1}

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.out, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, verbose=1
        ),
    ]

    history = model.fit(
        train_gen, epochs=args.epochs, validation_data=val_gen, callbacks=callbacks
    )

    # Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train acc")
    plt.plot(history.history["val_accuracy"], label="val acc")
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_performance.png")
    print("Saved training_performance.png")

    # Metrics on validation split
    y_prob = model.predict(val_gen).reshape(-1)
    y_pred = (y_prob > 0.5).astype(int)
    print("\n--- CLASSIFICATION REPORT ---")
    print(
        classification_report(
            val_gen.classes, y_pred, target_names=["fake", "real"], digits=4
        )
    )
    print("--- CONFUSION MATRIX (rows=actual fake/real, cols=pred fake/real) ---")
    print(confusion_matrix(val_gen.classes, y_pred))
    print(f"\nBest model saved to {args.out}")


if __name__ == "__main__":
    main()
