# 🕵️ Traceface — Hybrid AI-Media Authenticity Detector

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/license-MIT-green)

**Is this image real — or AI-generated? Is this video authentic — or a deepfake?**
Traceface answers with a *hybrid* stack instead of a black-box classifier:

| Signal | What it does |
|---|---|
| 🧠 **Keras CNN** | 4-layer convolutional network trained on real vs AI-generated imagery → calibrated P(real) score |
| 🔬 **ELA forensics** | Error Level Analysis exposes edited / synthesized regions classical detectors miss |
| 🏷️ **Metadata + AI-signature scan** | Hunts EXIF / PNG chunks for generator fingerprints (Stable Diffusion, Midjourney, DALL·E, ComfyUI…) |
| 🔥 **Grad-CAM explainability** | Shows *which pixels* drove the verdict — not just the verdict |
| 🎬 **Video screening** | Samples frames across the clip, scores each with the CNN, aggregates temporally with a suspicion timeline |

🌐 **Landing page:** https://mirzacreative.github.io/traceface/
🚀 **Live demo:** deploy in 2 minutes — see below.

---

## ⚡ Quickstart

```bash
git clone https://github.com/mirzacreative/traceface.git
cd traceface
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501` with three tabs:
**🖼️ Image Analysis** · **🎬 Video Analysis** · **ℹ️ How it works**

> The CNN weights (`traceface_model.keras`, 110 MB) exceed GitHub's file limit, so they
> live on Google Drive — the app **auto-downloads them on first run** (cached afterwards).
> No manual setup needed. Without network access, the app runs in **forensic-only
> mode** (ELA + metadata still work).

## 🚀 Deploy the live demo (free, 2 minutes)

1. Push this repo to GitHub (already done).
2. Go to [share.streamlit.io](https://share.streamlit.io) → *New app* → select this repo → `app.py` → Deploy.
3. Copy your `https://<name>.streamlit.app` URL into the landing page (`docs/index.html` → `LIVE_APP_URL`).

## 🧪 Training

Dataset layout:

```
dataset/
    real/   # photographs
    fake/   # AI-generated images
```

```bash
python training/train_cnn.py --data ./dataset --epochs 40
```

The optimized script adds augmentation, BatchNorm, GlobalAveragePooling,
EarlyStopping, LR scheduling and checkpointing, then prints a full
classification report + confusion matrix. Best model → `traceface_model.keras`.

> Label convention: `flow_from_directory` sorts classes alphabetically, so
> `fake = 0`, `real = 1` and sigmoid output **> 0.5 ⇒ Real**. The app follows this.

## 📁 Project structure

```
traceface/
├── app.py                  # premium Streamlit dashboard
├── src/
│   ├── model.py            # model loading + inference
│   ├── forensics.py        # ELA, EXIF, AI-signature scan
│   ├── explain.py          # Grad-CAM heatmaps
│   └── video.py            # frame sampling + temporal aggregation
├── training/
│   └── train_cnn.py        # optimized training script
├── docs/                   # GitHub Pages landing page
├── requirements.txt
└── traceface_model.keras   # trained weights (auto-downloaded from Drive)
```

## ⚠️ Honest limitations

- The CNN is only as good as its training data — brand-new generators can fool it.
- Video screening is **frame-based** (reuses the image model); it doesn't yet model motion-level deepfake artifacts.
- ELA / metadata are supporting signals, never proof.
- Every verdict is probabilistic — not a legal judgment.

## 👨‍💻 Author

**Muhammad Atif Iqbal** — [Mirza Creative](https://mirzacreative.github.io/mirzacreative/)
Built as a final-year project, rebuilt as an open-source hybrid detector. MIT licensed.
