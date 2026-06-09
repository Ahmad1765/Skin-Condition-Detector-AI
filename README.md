# Skin Condition Detector AI

A webcam-based facial skin analysis tool that scores **10 skin metrics on a 0–100 scale**, inspired by Cetaphil's *MySkin* analyzer. It combines pre-trained Vision Transformer (ViT) models with computer-vision heuristics to provide real-time skin condition feedback from any standard webcam.

> Heuristic and ML estimates only. **Not a medical or cosmetic diagnostic tool.** For educational and personal use. Always consult a qualified dermatologist for skin concerns.

---

## What It Does

The program opens a live webcam preview, lets you capture a snapshot, and returns 10 skin scores with color-coded bars (green ≥80, yellow 60–79, red <60):

| Metric          | Source        | What it measures                                        |
| --------------- | ------------- | ------------------------------------------------------- |
| Hydration       | CNN (ViT)     | Predicted dryness probability from skin-type classifier |
| Blemish prone   | CNN (ViT)     | Acne severity (5-level: –1 to 3)                        |
| Redness prone   | Heuristic     | Mean `a*` channel in LAB color space on cheeks          |
| Oiliness/Shine  | CNN (ViT)     | Predicted oily probability                              |
| Dark Spots      | Heuristic     | Local-mean dark anomaly detection                       |
| Radiance        | Heuristic     | Median luminance × tone evenness                        |
| Texture         | Heuristic     | High-pass-filtered cheek roughness                      |
| Firmness        | Heuristic     | Jaw-edge sharpness + cheek tone uniformity              |
| Wrinkles        | CNN (ViT)     | Wrinkle classifier probability                          |
| Dark Circles    | Heuristic     | Under-eye vs. cheek luminance gap                       |

---

## How It Works

```
Webcam frame
   │
   ├─► YuNet face detector  ──►  5 landmarks (eyes, nose, mouth corners)
   │                                  │
   │                                  ├─► Region rectangles anchored to landmarks
   │                                  │     (forehead, cheeks, under-eyes, nose, jaw, eye corners)
   │                                  │
   │                                  └─► Heuristic scores per region
   │
   └─► Face crop  ──►  3 ViT models (acne / wrinkles / skin-type)
                            │
                            └─► CNN scores for Blemish, Wrinkles, Oiliness, Hydration

   ►► Combined 10-metric panel rendered next to the snapshot
```

### Models Used

| Model                                         | Purpose                            | License |
| --------------------------------------------- | ---------------------------------- | ------- |
| `imfarzanansari/skintelligent-acne`           | Acne severity (5 classes)          | Apache  |
| `imfarzanansari/skintelligent-wrinkles`       | Binary wrinkle classifier          | Apache  |
| `dima806/skin_types_image_detection`          | Skin type: dry / normal / oily     | Apache  |
| OpenCV YuNet (`face_detection_yunet_2023mar`) | Face detection + 5 facial landmarks (fallback) | Apache |
| MediaPipe `face_landmarker.task` (float16)    | 478 facial landmarks + skin polygon mask        | Apache |

All models are downloaded automatically on first run.

---

## Requirements

- Python 3.10+
- Webcam (built-in or USB)
- ~1 GB free disk space for model weights
- Internet connection (first run only)

### Python Dependencies

```
opencv-python
numpy
torch
transformers
pillow
huggingface_hub
mediapipe        # 468-landmark face mesh — falls back to YuNet if unavailable
streamlit        # only for the web app (app.py); CLI works without it
```

---

## Installation

```bash
git clone https://github.com/Ahmad1765/Skin-Condition-Detector-AI.git
cd Skin-Condition-Detector-AI
pip install opencv-python numpy torch transformers pillow huggingface_hub mediapipe streamlit
```

---

## Usage

### Web app (recommended — easiest to test)

```bash
streamlit run app.py
```

Opens a browser tab at `http://localhost:8501`. Use the **Webcam** tab to snap a
photo or the **Upload** tab to drop in any face photo. The grid of condition
cards renders below.

### Desktop / CLI mode

```bash
python skin_score.py                 # webcam
python skin_score.py path/to/img.jpg # static image / video
```

### Desktop controls

| Key            | Action                  |
| -------------- | ----------------------- |
| `SPACE`        | Capture and analyze     |
| `R`            | Retake (discard result) |
| `B`            | Toggle landmark boxes (preview) |
| `Up`/`Down` or `J`/`K` | Scroll result grid |
| `Q` / `ESC`    | Quit                    |

---

## First Run

On the first launch the program downloads:

1. **YuNet face detector** (~250 KB) to `~/.cache/skin_score/`
2. **3 ViT models** (~340 MB total) to `D:\hf_cache\` (or `$HF_HOME`)

Subsequent runs load cached weights in ~10 seconds.

To use a different cache location, set the `HF_HOME` environment variable before running.

---

## Tips for Accurate Scores

- Use bright, **diffuse** lighting (window light works well). Avoid harsh shadows or single-side lighting.
- Position your face **30–50 cm from the camera** and fill the frame.
- Look straight at the camera; tilt distorts measurements.
- Clean the webcam lens — smudges add noise to texture and wrinkle estimates.
- Heuristic metrics (Redness, Dark Spots, Radiance, Texture, Firmness, Dark Circles) are sensitive to lighting; CNN metrics (Hydration, Blemish, Oiliness, Wrinkles) are more stable.

---

## Project Structure

```
Skin-Condition-Detector-AI/
├── skin_score.py        # Core analysis + CLI desktop app
├── app.py               # Streamlit web app
├── README.md            # This file
└── .gitignore
```

---

## Performance

| Stage                 | Time on CPU       |
| --------------------- | ----------------- |
| First-run model download | 5–20 min (network) |
| Cached model load     | 5–10 s            |
| Per-capture analysis  | 1.2–1.8 s         |
| Live preview          | 30 fps            |

GPU is not required.

---

## Limitations

- **Heuristic metrics** drift up to ±20 points across lighting changes. This is inherent to pixel-based estimation without a controlled-lighting reference.
- **CNN metrics** were trained on dermatology datasets, not on webcam selfies. Results are best-effort indicators, not measurements.
- **No public dataset** exists for the 10 Cetaphil-style metrics. Cetaphil's analyzer uses proprietary models trained on labeled datasets; this project approximates the *concept* with open tools.
- **Single-face only.** The largest detected face is used.

---

## Disclaimer

This software is provided for educational and personal interest only. It is **not** a substitute for professional medical or dermatological advice. Do not use it to diagnose or treat any skin condition.

---

## Credits

- Acne and Wrinkles ViT models by [@imfarzanansari](https://huggingface.co/imfarzanansari)
- Skin-type classifier by [@dima806](https://huggingface.co/dima806)
- YuNet face detector by the OpenCV team
- Inspired by Cetaphil's *MySkin* by Galderma

---

## License

MIT
