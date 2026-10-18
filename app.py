"""Traceface v2 — Hybrid AI-Media Authenticity Detector.

Stack: Keras CNN + ELA forensics + EXIF/AI-signature scan + Grad-CAM
explainability, with frame-based video deepfake screening.
"""
import streamlit as st
from PIL import Image

from src.explain import grad_cam_overlay
from src.forensics import ela_image, ela_score, extract_exif, scan_ai_signatures
from src.model import INPUT_SIZE, load_cnn_model, predict_image, preprocess_image
from src.video import (
    SUPPORTED_VIDEO_TYPES,
    analyze_frames,
    extract_frames,
    save_upload_to_temp,
)

# ============================================================
# Page config + premium theme
# ============================================================
st.set_page_config(
    page_title="Traceface — AI Media Authenticity Detector",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(1200px 600px at 15% -5%, rgba(56,130,246,0.14), transparent 60%),
        radial-gradient(1000px 500px at 90% 10%, rgba(168,85,247,0.12), transparent 60%),
        #0b1120;
    color: #e2e8f0;
    font-family: 'Inter', 'Helvetica Neue', sans-serif;
}
h1, h2, h3, h4 { font-family: 'Inter', 'Helvetica Neue', sans-serif; }

.hero {
    background: linear-gradient(135deg, rgba(30,41,59,0.9), rgba(15,23,42,0.9));
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 26px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.35);
}
.hero h1 { margin: 0; font-size: 34px; font-weight: 800; letter-spacing: -0.5px; }
.hero h1 .accent {
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { margin: 8px 0 0; color: #94a3b8; font-size: 15px; }
.hero .badges { margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }
.hero .badge {
    font-size: 12px; font-weight: 600; color: #cbd5e1;
    background: rgba(148,163,184,0.12); border: 1px solid rgba(148,163,184,0.25);
    padding: 4px 12px; border-radius: 999px;
}

.card {
    background: rgba(30,41,59,0.75);
    border: 1px solid rgba(148,163,184,0.16);
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 18px;
    box-shadow: 0 8px 28px rgba(0,0,0,0.28);
}
.verdict-real { border-left: 5px solid #22c55e; }
.verdict-fake { border-left: 5px solid #ef4444; }
.verdict-card .label { font-size: 12px; letter-spacing: 2px; color: #94a3b8; margin: 0; }
.verdict-card h2 { margin: 6px 0 2px; font-size: 38px; font-weight: 800; }
.verdict-real h2 { color: #4ade80; }
.verdict-fake h2 { color: #f87171; }
.verdict-card .sub { color: #94a3b8; font-size: 13px; margin: 0; }

.meta-row {
    display: flex; justify-content: space-between; gap: 12px;
    padding: 9px 0; border-bottom: 1px solid rgba(148,163,184,0.14);
    font-size: 14px;
}
.meta-row:last-child { border-bottom: none; }
.meta-key { color: #94a3b8; font-weight: 600; }
.meta-val { color: #e2e8f0; text-align: right; word-break: break-word; }

[data-testid="stFileUploadDropzone"] {
    background: rgba(30,41,59,0.6) !important;
    border: 2px dashed rgba(96,165,250,0.45) !important;
    border-radius: 16px !important;
    padding: 26px !important;
}
[data-testid="stFileUploadDropzone"] button {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
    color: #fff !important; font-weight: 700 !important;
    border: none !important; border-radius: 10px !important;
    padding: 8px 22px !important;
}
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(30,41,59,0.6); border-radius: 10px 10px 0 0;
    padding: 10px 22px; font-weight: 600; color: #94a3b8;
}
.stTabs [aria-selected="true"] { color: #fff !important; background: rgba(59,130,246,0.18) !important; }
img { border-radius: 12px; }
.footer { text-align: center; color: #64748b; font-size: 13px; margin-top: 34px; }
.footer a { color: #60a5fa; text-decoration: none; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Hero header
# ============================================================
st.markdown(
    """
<div class="hero">
    <h1>🕵️ Trace<span class="accent">face</span></h1>
    <p>Hybrid AI-media authenticity detector — deep-learning CNN fused with
    classical forensics (ELA + metadata), explainable heatmaps, and video screening.</p>
    <div class="badges">
        <span class="badge">🧠 Keras CNN</span>
        <span class="badge">🔬 ELA Forensics</span>
        <span class="badge">🏷️ EXIF + AI-signature scan</span>
        <span class="badge">🔥 Grad-CAM explainability</span>
        <span class="badge">🎬 Video support</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

model, model_error = load_cnn_model()

with st.sidebar:
    st.markdown("### ⚙️ System status")
    if model is not None:
        st.success("CNN model loaded ✅")
    else:
        st.warning("CNN model missing ⚠️")
    st.caption(
        "Forensic tools (ELA / EXIF) run without the model. "
        "Add `traceface_model.keras` to the project root for full AI verdicts."
    )
    st.divider()
    st.markdown("**Links**")
    st.markdown("⭐ [GitHub repository](https://github.com/mirzacreative/traceface)")
    st.markdown("📖 [Training guide](https://github.com/mirzacreative/traceface#training)")

# ============================================================
# Helpers
# ============================================================

def verdict_card(verdict, confidence, subtitle):
    cls = "verdict-real" if verdict == "Real" else "verdict-fake"
    st.markdown(
        f"""
<div class="card verdict-card {cls}">
    <p class="label">TRACEFACE VERDICT</p>
    <h2>{verdict}</h2>
    <p class="sub">{subtitle}</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.progress(min(max(float(confidence), 0.0), 1.0),
                text=f"Confidence: {confidence * 100:.1f}%")


def exif_panel(image):
    st.markdown("#### 🏷️ Metadata & AI-signature scan")
    exif = extract_exif(image)
    hits = scan_ai_signatures(exif)
    if hits:
        st.error(
            "🚨 **AI-generator signature detected in metadata:** "
            + ", ".join(f"`{h}`" for h in hits)
        )
    elif exif:
        st.success("✅ No known AI-generator signatures in metadata.")
    else:
        st.info(
            "No EXIF/metadata found. Note: AI-generated images often ship "
            "with stripped metadata — its absence is itself a weak signal."
        )
    if exif:
        rows = "".join(
            f'<div class="meta-row"><span class="meta-key">{k}</span>'
            f'<span class="meta-val">{v}</span></div>'
            for k, v in exif.items()
        )
        st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)


def forensics_panel(image, show_gradcam=True):
    st.markdown("#### 🔬 Forensic signals")
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Error Level Analysis (ELA) — bright spots = suspicious regions")
        st.image(ela_image(image), use_container_width=True)
        st.caption(f"ELA tamper signal: {ela_score(image):.3f} (supporting signal, not a verdict)")
    with col_b:
        if show_gradcam and model is not None:
            st.caption("Grad-CAM — red regions drove the CNN's verdict")
            with st.spinner("Computing explainability heatmap..."):
                overlay, _ = grad_cam_overlay(
                    model, image.convert("RGB"), preprocess_image
                )
            if overlay is not None:
                st.image(overlay, use_container_width=True)
            else:
                st.info("Heatmap unavailable for this model architecture.")
        elif show_gradcam:
            st.info("Add the CNN model file to enable Grad-CAM heatmaps.")

# ============================================================
# Tabs
# ============================================================
tab_img, tab_vid, tab_about = st.tabs(
    ["🖼️ Image Analysis", "🎬 Video Analysis", "ℹ️ How it works"]
)

# ---------------- Image ----------------
with tab_img:
    uploaded = st.file_uploader(
        "Drop an image to verify — JPG, JPEG or PNG",
        type=["jpg", "jpeg", "png"],
        key="img_uploader",
    )
    if uploaded is not None:
        image = Image.open(uploaded)
        c1, c2 = st.columns([1.1, 1])

        with c1:
            st.markdown("#### Image preview")
            st.image(image, use_container_width=True)

        with c2:
            st.markdown("#### Analysis result")
            if model is not None:
                with st.spinner("🧠 CNN scanning pixels..."):
                    pred = predict_image(model, image)
                verdict_card(
                    pred["verdict"],
                    pred["confidence"],
                    f"P(real) = {pred['p_real']:.3f} · P(fake) = {pred['p_fake']:.3f} · "
                    f"input {INPUT_SIZE[0]}×{INPUT_SIZE[1]}",
                )
                m1, m2 = st.columns(2)
                m1.metric("P(Real) 🟢", f"{pred['p_real'] * 100:.1f}%")
                m2.metric("P(Fake) 🔴", f"{pred['p_fake'] * 100:.1f}%")
            else:
                st.warning(model_error, icon="⚠️")
                st.info(
                    "Running in **forensic-only mode** — ELA, metadata and "
                    "heatmaps below still work."
                )

        st.divider()
        forensics_panel(image)
        st.divider()
        exif_panel(image)

# ---------------- Video ----------------
with tab_vid:
    st.markdown(
        "> **Frame-based video screening:** Traceface samples frames across the clip, "
        "scores each with the image CNN, and aggregates the verdict temporally."
    )
    vfile = st.file_uploader(
        "Drop a video to verify — MP4, MOV, AVI, WEBM, MKV",
        type=SUPPORTED_VIDEO_TYPES,
        key="vid_uploader",
    )
    if vfile is not None:
        st.video(vfile)
        if model is None:
            st.warning(model_error, icon="⚠️")
            st.info("Video analysis needs the CNN model file — forensic mode is image-only.")
        elif st.button("🎬 Analyze video", type="primary", use_container_width=True):
            suffix = vfile.name.split(".")[-1].lower()
            path = save_upload_to_temp(vfile, suffix)
            try:
                with st.spinner("Extracting frames..."):
                    frames, meta = extract_frames(path)
                st.markdown("#### 📦 Clip metadata")
                mc = st.columns(4)
                mc[0].metric("Duration", f"{meta['duration_s']}s")
                mc[1].metric("Resolution", meta["resolution"])
                mc[2].metric("FPS", meta["fps"])
                mc[3].metric("Frames sampled", len(frames))

                with st.spinner(f"🧠 Scoring {len(frames)} frames with the CNN..."):
                    result = analyze_frames(
                        model, frames,
                        lambda img: predict_image(model, img),
                    )
                st.divider()
                verdict_card(
                    result["verdict"],
                    result["confidence"],
                    f"Mean P(fake) = {result['mean_fake']:.3f} · "
                    f"{result['fake_ratio'] * 100:.0f}% frames flagged · "
                    f"temporal volatility σ = {result['temporal_std']:.3f}",
                )

                st.markdown("#### 📈 Frame-by-frame fake probability")
                import pandas as pd

                chart_df = pd.DataFrame(
                    {
                        "frame": [r["frame"] for r in result["frames"]],
                        "P(fake)": [r["p_fake"] for r in result["frames"]],
                    }
                ).set_index("frame")
                st.line_chart(chart_df, height=220)

                with st.expander("🖼️ Inspect sampled frames"):
                    cols = st.columns(6)
                    for i, item in enumerate(frames):
                        r = result["frames"][i]
                        tag = "🔴" if r["p_fake"] >= 0.5 else "🟢"
                        cols[i % 6].image(
                            item["image"],
                            caption=f"{tag} f{item['index']} · {r['p_fake']:.2f}",
                            use_container_width=True,
                        )
            finally:
                try:
                    import os

                    os.remove(path)
                except OSError:
                    pass

# ---------------- About ----------------
with tab_about:
    st.markdown("### 🧩 The hybrid stack — what makes Traceface different")
    a, b = st.columns(2)
    with a:
        st.markdown(
            """
<div class="card">
<b>🧠 Deep-learning CNN</b><br>
4-layer Keras convolutional network trained on real vs AI-generated faces/images.
Outputs a calibrated P(real) score.
</div>
<div class="card">
<b>🔬 ELA forensics</b><br>
Error Level Analysis recompresses the image and amplifies compression
differences — edited or synthesized regions light up.
</div>
""",
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            """
<div class="card">
<b>🏷️ Metadata + AI-signature scan</b><br>
Extracts EXIF / PNG chunks and hunts for generator fingerprints
(Stable Diffusion, Midjourney, DALL·E, ComfyUI…).
</div>
<div class="card">
<b>🔥 Grad-CAM explainability</b><br>
Instead of a black-box "fake" label, Traceface shows <i>which pixels</i>
drove the verdict.
</div>
""",
            unsafe_allow_html=True,
        )
    st.markdown("### 🎬 How video screening works")
    st.markdown(
        "1. Up to 24 frames are sampled evenly across the clip.\n"
        "2. Each frame is scored by the image CNN.\n"
        "3. The clip verdict fuses **mean fake probability** with the **share of "
        "flagged frames** — catching partially manipulated videos too.\n"
        "4. The timeline chart exposes *where* in the video the suspicion lies."
    )
    st.markdown("### ⚠️ Honest limitations")
    st.markdown(
        "- The CNN is only as good as its training data — brand-new generators can fool it.\n"
        "- Video screening is **frame-based** (it reuses the image model); it does not "
        "yet model motion-level deepfake artifacts.\n"
        "- ELA / metadata are supporting signals, never proof on their own.\n"
        "- Treat every verdict as probabilistic, not a legal judgment."
    )
    st.markdown("### 👨‍💻 Author")
    st.markdown(
        "Built by **Muhammad Atif Iqbal** ([Mirza Creative](https://mirzacreative.github.io/mirzacreative/)) "
        "as a final-year project, rebuilt as an open-source hybrid detector."
    )

st.markdown(
    """
<div class="footer">
    Traceface v2.0 · Hybrid AI-media authenticity detector ·
    <a href="https://github.com/mirzacreative/traceface">GitHub</a> ·
    Built with TensorFlow + Streamlit
</div>
""",
    unsafe_allow_html=True,
)
