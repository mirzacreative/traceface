"""Video deepfake screening via frame sampling + temporal aggregation.

Method (honest framing, also stated in the UI):
  1. Sample up to N frames evenly across the video.
  2. Run the Traceface image CNN on every sampled frame.
  3. Aggregate with temporal statistics (mean fake probability,
     share of flagged frames, score volatility).

This is a *frame-based* analysis reusing the image model — not a dedicated
video-deepfake network — and the UI labels it as such.
"""
from __future__ import annotations

import tempfile

import numpy as np
from PIL import Image

SUPPORTED_VIDEO_TYPES = ["mp4", "mov", "avi", "webm", "mkv"]
DEFAULT_MAX_FRAMES = 24


def save_upload_to_temp(uploaded_file, suffix):
    """Persist a Streamlit upload to a temp file for cv2. Returns the path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}")
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


def extract_frames(video_path, max_frames=DEFAULT_MAX_FRAMES):
    """Sample frames evenly across the video.

    Returns:
        (frames, metadata) where frames = [{"index": int, "image": PIL}, ...]
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the video file with OpenCV.")

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    metadata = {
        "frames": count,
        "fps": round(float(fps), 2),
        "resolution": f"{width}x{height}",
        "duration_s": round(count / fps, 1) if fps else 0,
    }

    n = max(1, min(max_frames, count or 1))
    indices = np.linspace(0, max(count - 1, 0), n).astype(int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append({"index": int(idx), "image": Image.fromarray(rgb)})
    cap.release()
    return frames, metadata


def analyze_frames(model, frames):
    """Score every frame and aggregate into a video-level verdict.

    All frames are stacked into ONE batch for a single ``model.predict``
    call — far faster (and less failure-prone) than N separate predicts
    on CPU-only hosts.
    """
    from src.model import preprocess_image

    results = []
    if frames:
        batch = np.concatenate(
            [preprocess_image(item["image"]) for item in frames], axis=0
        )
        p_real = np.asarray(model.predict(batch, verbose=0)).reshape(-1)
        for item, pr in zip(frames, p_real):
            pr = float(min(1.0, max(0.0, pr)))
            results.append(
                {"frame": item["index"], "p_fake": 1.0 - pr, "p_real": pr}
            )

    p_fakes = np.array([r["p_fake"] for r in results], dtype="float64")
    mean_fake = float(p_fakes.mean()) if len(p_fakes) else 0.0
    fake_ratio = float((p_fakes >= 0.5).mean()) if len(p_fakes) else 0.0
    temporal_std = float(p_fakes.std()) if len(p_fakes) else 0.0

    # A video is flagged when the average leans fake OR a large share of
    # frames is flagged (catches partially manipulated clips).
    is_fake = (mean_fake >= 0.5) or (fake_ratio >= 0.4)
    return {
        "frames": results,
        "mean_fake": mean_fake,
        "fake_ratio": fake_ratio,
        "temporal_std": temporal_std,
        "verdict": "AI-Generated / Fake" if is_fake else "Real",
        "confidence": mean_fake if is_fake else 1.0 - mean_fake,
        "is_fake": is_fake,
    }
