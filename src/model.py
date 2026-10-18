"""CNN model loading and inference helpers for Traceface.

Label convention (kept from the original training):
    flow_from_directory sorts classes alphabetically -> fake = 0, real = 1
    so a sigmoid output > 0.5 means "Real".
"""
from __future__ import annotations

import os

import streamlit as st

MODEL_FILENAME = "traceface_model.keras"
MODEL_SEARCH_PATHS = (
    MODEL_FILENAME,
    f"models/{MODEL_FILENAME}",
)

# The 110 MB weights live on Google Drive (over GitHub's 100 MB file limit).
# Set to "Anyone with the link" and paste the share link's file id below.
# The app auto-downloads it on first run — users never touch this manually.
MODEL_DRIVE_FILE_ID = "1L0zg13NeJBukgwoEpDjrrREMEboXrTQh"
MODEL_DRIVE_URL = (
    f"https://drive.google.com/uc?id={MODEL_DRIVE_FILE_ID}"
    if MODEL_DRIVE_FILE_ID
    else ""
)

INPUT_SIZE = (224, 224)

MISSING_MODEL_MSG = (
    "⚠️ **CNN model file not found.** The weights auto-download from Google Drive "
    "on first run — check your connection and restart. "
    "The forensic tools below (ELA / EXIF / metadata) still work without it."
)


def ensure_model_file():
    """Return a local path to the weights, downloading from Drive if needed."""
    for path in MODEL_SEARCH_PATHS:
        if os.path.exists(path):
            return path
    if MODEL_DRIVE_URL:
        import gdown

        dest = os.path.join("models", MODEL_FILENAME)
        os.makedirs("models", exist_ok=True)
        gdown.download(MODEL_DRIVE_URL, dest, quiet=False)
        if os.path.exists(dest):
            return dest
    return None


@st.cache_resource(show_spinner="Loading Traceface CNN...")
def load_cnn_model():
    """Load the trained Keras model (cached).

    Returns:
        tuple: (model | None, error_message | None)
    """
    try:
        import tensorflow as tf
    except ImportError:
        return None, "TensorFlow is not installed. Run `pip install -r requirements.txt`."

    path = ensure_model_file()
    if path:
        try:
            model = tf.keras.models.load_model(path)
            return model, None
        except Exception as exc:  # noqa: BLE001 - surface any load failure
            return None, f"Found `{path}` but failed to load it: {exc}"
    return None, MISSING_MODEL_MSG


def preprocess_image(pil_image, target_size=INPUT_SIZE):
    """Resize + normalize a PIL image into a (1, 224, 224, 3) batch."""
    import numpy as np

    img = pil_image.convert("RGB").resize(target_size)
    arr = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def predict_image(model, pil_image):
    """Run the CNN on a single PIL image.

    Returns:
        dict with p_real, p_fake, verdict, confidence, is_real
    """
    batch = preprocess_image(pil_image)
    p_real = float(model.predict(batch, verbose=0).reshape(-1)[0])
    p_real = max(0.0, min(1.0, p_real))
    is_real = p_real >= 0.5
    return {
        "p_real": p_real,
        "p_fake": 1.0 - p_real,
        "verdict": "Real" if is_real else "AI-Generated / Fake",
        "confidence": p_real if is_real else 1.0 - p_real,
        "is_real": is_real,
    }
