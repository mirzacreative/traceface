"""Classical forensic signals: ELA, EXIF/metadata, AI-generator signatures.

These are *supporting* signals shown next to the CNN verdict — they do not
replace the model, but they make Traceface harder to fool than a bare CNN
and they work even when no model file is present.
"""
from __future__ import annotations

import io

from PIL import ExifTags, Image, ImageChops, ImageEnhance

# Keywords commonly left behind by AI image generators in metadata / PNG chunks.
AI_SIGNATURES = (
    "stable diffusion",
    "dall-e",
    "dalle",
    "midjourney",
    "firefly",
    "leonardo",
    "civitai",
    "comfyui",
    "automatic1111",
    "invokeai",
    "ideogram",
    "flux",
    "sdxl",
    "dreamstudio",
    "nightcafe",
    "starryai",
)

EXIF_TAGS_OF_INTEREST = (
    "Software",
    "Make",
    "Model",
    "DateTimeOriginal",
    "Artist",
    "HostComputer",
    "CreateDate",
)


def ela_image(pil_image, quality=90):
    """Error Level Analysis: recompress the image and amplify the difference.

    Regions that were digitally altered (or fully synthesized) tend to show
    visibly different error levels from the rest of the picture.
    """
    original = pil_image.convert("RGB")
    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)

    diff = ImageChops.difference(original, recompressed)
    extrema = diff.getextrema()
    max_diff = max(channel[1] for channel in extrema) or 1
    diff = ImageEnhance.Brightness(diff).enhance(255.0 / max_diff)
    return diff


def ela_score(pil_image, quality=90):
    """Rough 0..1 tamper *signal* from ELA brightness (supporting signal only)."""
    import numpy as np

    diff = ela_image(pil_image, quality).convert("L")
    arr = np.asarray(diff).astype("float32") / 255.0
    return float(arr.mean())


def extract_exif(pil_image):
    """Pull the interesting EXIF tags + PNG text chunks from an image."""
    info = {}
    try:
        raw = pil_image.getexif()
        if raw:
            for tag_id, value in raw.items():
                name = ExifTags.TAGS.get(tag_id, tag_id)
                if name in EXIF_TAGS_OF_INTEREST:
                    info[str(name)] = str(value).strip()
    except Exception:  # noqa: BLE001 - metadata must never crash analysis
        pass

    # PNG text chunks: tools like Automatic1111 store prompts in "parameters".
    try:
        png_info = getattr(pil_image, "info", {}) or {}
        for key in ("parameters", "Software", "prompt", "workflow", "Comment"):
            if key in png_info and png_info[key]:
                info[f"PNG:{key}"] = str(png_info[key])[:400]
    except Exception:  # noqa: BLE001
        pass
    return info


def scan_ai_signatures(exif_dict):
    """Return any known AI-generator keywords found in the metadata blob."""
    blob = " ".join(str(v) for v in exif_dict.values()).lower()
    return [sig for sig in AI_SIGNATURES if sig in blob]
