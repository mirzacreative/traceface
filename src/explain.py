"""Grad-CAM explainability for the Traceface CNN.

Shows *which regions* of the image pushed the model toward its verdict —
the key differentiator over black-box "real/fake" classifiers.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _last_conv_layer(model):
    """Find the deepest Conv2D layer with a 4D (feature-map) output."""
    for layer in reversed(model.layers):
        name = layer.name.lower()
        cls = layer.__class__.__name__.lower()
        if "conv" not in name and "conv" not in cls:
            continue
        try:
            if len(layer.output.shape) == 4:
                return layer
        except Exception:  # noqa: BLE001
            continue
    return None


def grad_cam_overlay(model, pil_image, preprocess_fn):
    """Build a Grad-CAM heatmap overlay for the model's prediction.

    Returns:
        (overlay PIL image, heatmap PIL image) or (None, None) on failure.
    """
    import tensorflow as tf

    conv_layer = _last_conv_layer(model)
    if conv_layer is None:
        return None, None

    grad_model = tf.keras.models.Model(
        inputs=model.inputs, outputs=[conv_layer.output, model.output]
    )
    batch = preprocess_fn(pil_image)

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(batch, training=False)
        score = preds[:, 0]

    grads = tape.gradient(score, conv_out)
    if grads is None:
        return None, None

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(weights * conv_out[0], axis=-1).numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    heat = Image.fromarray((cam * 255).astype("uint8")).resize(
        pil_image.size, Image.BILINEAR
    )

    # Colorize with a jet colormap (red = strongest influence on verdict).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("jet")
    colored = (cmap(np.asarray(heat) / 255.0)[:, :, :3] * 255).astype("uint8")
    overlay = Image.blend(pil_image.convert("RGB"), Image.fromarray(colored), 0.45)
    return overlay, heat
