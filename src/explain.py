"""Grad-CAM explainability for the Traceface CNN.

Shows *which regions* of the image pushed the model toward its verdict —
the key differentiator over black-box "real/fake" classifiers.

Implementation note: instead of reaching into the loaded model's internal
graph via ``layer.output`` (fragile across Keras versions — it raises
``AttributeError`` on Keras 3 for deserialized Sequential models), we
*replay* the model's layers on a fresh Functional ``Input``. This builds a
clean differentiable graph on every Keras version, reusing the trained
weights. The public function never raises: any failure returns
``(None, None)`` so the UI degrades gracefully instead of crashing.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _is_conv_layer(layer):
    name = layer.name.lower()
    cls = layer.__class__.__name__.lower()
    return "conv" in name or "conv" in cls


def grad_cam_overlay(model, pil_image, preprocess_fn):
    """Build a Grad-CAM heatmap overlay for the model's prediction.

    Returns:
        (overlay PIL image, heatmap PIL image) or (None, None) on failure.
        Never raises.
    """
    try:
        return _grad_cam_overlay(model, pil_image, preprocess_fn)
    except Exception:  # noqa: BLE001 - explainability must never crash the app
        return None, None


def _grad_cam_overlay(model, pil_image, preprocess_fn):
    import tensorflow as tf

    try:
        input_shape = tuple(model.input_shape[1:])
    except Exception:  # noqa: BLE001 - fall back to the standard size
        input_shape = (224, 224, 3)

    conv_idx = None
    for i, layer in enumerate(model.layers):
        if _is_conv_layer(layer):
            conv_idx = i
    if conv_idx is None:
        return None, None

    # Replay the trained layers as a Functional graph so the last conv
    # feature maps are accessible for gradients on any Keras version.
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    conv_out = None
    for i, layer in enumerate(model.layers):
        x = layer(x)
        if i == conv_idx:
            conv_out = x
    grad_model = tf.keras.Model(inputs=inputs, outputs=[conv_out, x])

    batch = preprocess_fn(pil_image)
    with tf.GradientTape() as tape:
        conv_features, preds = grad_model(batch, training=False)
        score = preds[:, 0]

    grads = tape.gradient(score, conv_features)
    if grads is None:
        return None, None

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(weights * conv_features[0], axis=-1).numpy()
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
