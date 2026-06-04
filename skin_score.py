"""Skin Score - heuristic 10-metric skin analyzer.

Webcam preview -> SPACE to capture -> 10 scores (0-100) shown beside snapshot.
Estimates only. Not medical or cosmetic measurement.

Controls: SPACE = capture, R = retake, Q/ESC = quit.

Uses YuNet ONNX face detector (built into OpenCV) for 5-landmark detection
(eyes, nose, mouth corners). Regions are anchored to real landmarks for
accurate measurement areas.
"""
import os
import sys
import time
import urllib.request

os.environ.setdefault("HF_HOME", r"D:\hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", r"D:\hf_cache\transformers")
os.environ.setdefault("HF_HUB_CACHE", r"D:\hf_cache\hub")
os.makedirs(r"D:\hf_cache", exist_ok=True)

import cv2
import numpy as np


HAAR_DIR = cv2.data.haarcascades
FACE_CASCADE_HAAR = cv2.CascadeClassifier(HAAR_DIR + "haarcascade_frontalface_default.xml")

DEBUG = {}

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def normalize_lighting(bgr):
    """Apply CLAHE to L channel of LAB to reduce lighting variance.
    Returns a copy of bgr with normalized luminance."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = CLAHE.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
YUNET_PATH = os.path.join(
    os.path.expanduser("~"), ".cache", "skin_score", "face_detection_yunet_2023mar.onnx"
)


def ensure_yunet():
    if os.path.exists(YUNET_PATH):
        return
    os.makedirs(os.path.dirname(YUNET_PATH), exist_ok=True)
    print(f"Downloading YuNet face detector -> {YUNET_PATH}", file=sys.stderr)
    urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
    print("Model downloaded.", file=sys.stderr)


ensure_yunet()
YUNET = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), 0.6, 0.3, 5000)


print("Loading CNN models (first run downloads ~250 MB)...", file=sys.stderr)
_t = time.time()
from PIL import Image
from transformers import pipeline
ACNE_CLF = pipeline("image-classification", model="imfarzanansari/skintelligent-acne")
WRINKLES_CLF = pipeline("image-classification", model="imfarzanansari/skintelligent-wrinkles")
SKINTYPE_CLF = pipeline("image-classification", model="dima806/skin_types_image_detection")
print(f"CNN models loaded in {time.time() - _t:.1f}s", file=sys.stderr)


ACNE_LEVELS = {"level -1": 0, "level 0": 1, "level 1": 2, "level 2": 3, "level 3": 4}


def _to_dict(preds):
    return {p["label"]: float(p["score"]) for p in preds}


def cnn_predict(bgr_face_crop):
    """Run 3 ViT models on a face crop. Returns dict of 4 scores 0-100."""
    rgb = cv2.cvtColor(bgr_face_crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    acne = _to_dict(ACNE_CLF(pil))
    severity = sum(acne.get(lbl, 0.0) * idx for lbl, idx in ACNE_LEVELS.items())
    blemish_score = int(round(100 * (1 - severity / 4)))

    wrinkles = _to_dict(WRINKLES_CLF(pil))
    p_wrinkle = wrinkles.get("wrinkle", wrinkles.get("Wrinkle", 0.0))
    wrinkles_score = int(round(100 * (1 - p_wrinkle)))

    skintype = _to_dict(SKINTYPE_CLF(pil))
    p_oily = skintype.get("oily", 0.0)
    p_dry = skintype.get("dry", 0.0)
    oiliness_score = int(round(100 * (1 - p_oily)))
    hydration_score = int(round(100 * (1 - p_dry)))

    DEBUG["cnn_acne_severity"] = round(severity, 2)
    DEBUG["cnn_p_wrinkle"] = round(p_wrinkle, 3)
    DEBUG["cnn_p_oily"] = round(p_oily, 3)
    DEBUG["cnn_p_dry"] = round(p_dry, 3)

    return {
        "Blemish prone": blemish_score,
        "Wrinkles": wrinkles_score,
        "Oiliness/Shine": oiliness_score,
        "Hydration": hydration_score,
    }


METRICS = [
    "Hydration", "Blemish prone", "Redness prone", "Oiliness/Shine",
    "Dark Spots", "Radiance", "Texture", "Firmness", "Wrinkles", "Dark Circles",
]


def detect_face_and_eyes(bgr):
    """YuNet detection + landmark-anchored regions.
    Returns dict of region rects (x, y, w, h) or None if no face."""
    h, w = bgr.shape[:2]
    YUNET.setInputSize((w, h))
    _, faces = YUNET.detect(bgr)
    if faces is None or len(faces) == 0:
        return None

    face = max(faces, key=lambda f: f[2] * f[3])
    fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
    le_x, le_y = int(face[4]), int(face[5])
    re_x, re_y = int(face[6]), int(face[7])
    nose_x, nose_y = int(face[8]), int(face[9])
    ml_x, ml_y = int(face[10]), int(face[11])
    mr_x, mr_y = int(face[12]), int(face[13])

    eye_dx = re_x - le_x
    eye_dy = re_y - le_y
    eye_dist = max(20, int(np.hypot(eye_dx, eye_dy)))
    eye_mid_x = (le_x + re_x) // 2
    eye_mid_y = (le_y + re_y) // 2
    mouth_mid_x = (ml_x + mr_x) // 2
    mouth_mid_y = (ml_y + mr_y) // 2
    eye_to_mouth = max(20, mouth_mid_y - eye_mid_y)

    def rect_around(cx, cy, rw, rh):
        return (max(0, int(cx - rw / 2)), max(0, int(cy - rh / 2)),
                int(rw), int(rh))

    forehead_h = int(eye_to_mouth * 0.55)
    forehead_y = eye_mid_y - int(eye_dist * 0.45) - forehead_h
    forehead = (max(0, le_x - int(eye_dist * 0.05)),
                max(0, forehead_y),
                int(eye_dist * 1.1),
                forehead_h)

    cheek_w = int(eye_dist * 0.55)
    cheek_h = int(eye_to_mouth * 0.55)
    cheek_y = eye_mid_y + int(eye_to_mouth * 0.25)
    left_cheek = rect_around(le_x - int(eye_dist * 0.05), cheek_y + cheek_h // 2, cheek_w, cheek_h)
    right_cheek = rect_around(re_x + int(eye_dist * 0.05), cheek_y + cheek_h // 2, cheek_w, cheek_h)

    ue_w = int(eye_dist * 0.32)
    ue_h = int(eye_dist * 0.14)
    ue_off = int(eye_dist * 0.30)
    left_under_eye = rect_around(le_x, le_y + ue_off, ue_w, ue_h)
    right_under_eye = rect_around(re_x, re_y + ue_off, ue_w, ue_h)

    nose = rect_around(nose_x, (eye_mid_y + nose_y) // 2,
                       int(eye_dist * 0.45), int(eye_to_mouth * 0.55))

    mouth_w = int((mr_x - ml_x) * 1.4)
    mouth_h = int(eye_dist * 0.35)
    mouth = rect_around(mouth_mid_x, mouth_mid_y, mouth_w, mouth_h)

    jaw_w = int(eye_dist * 2.4)
    jaw_h = int(eye_to_mouth * 0.6)
    jaw_y = mouth_mid_y + int(eye_to_mouth * 0.35)
    jaw = rect_around(mouth_mid_x, jaw_y, jaw_w, jaw_h)

    ec_w = int(eye_dist * 0.32)
    ec_h = int(eye_dist * 0.32)
    lec_cx = le_x - int(eye_dist * 0.18)
    rec_cx = re_x + int(eye_dist * 0.18)
    left_eye_corner = rect_around(lec_cx, le_y, ec_w, ec_h)
    right_eye_corner = rect_around(rec_cx, re_y, ec_w, ec_h)

    return {
        "face": (fx, fy, fw, fh),
        "forehead": forehead,
        "left_cheek": left_cheek,
        "right_cheek": right_cheek,
        "left_under_eye": left_under_eye,
        "right_under_eye": right_under_eye,
        "nose": nose,
        "mouth": mouth,
        "jaw": jaw,
        "left_eye_corner": left_eye_corner,
        "right_eye_corner": right_eye_corner,
        "landmarks": {
            "left_eye": (le_x, le_y),
            "right_eye": (re_x, re_y),
            "nose_tip": (nose_x, nose_y),
            "mouth_left": (ml_x, ml_y),
            "mouth_right": (mr_x, mr_y),
        },
    }


def rect_to_mask(rect, shape):
    h, w = shape[:2]
    x, y, rw, rh = rect
    x = max(0, x); y = max(0, y)
    x2 = min(w, x + rw); y2 = min(h, y + rh)
    mask = np.zeros((h, w), dtype=np.uint8)
    if x2 > x and y2 > y:
        mask[y:y2, x:x2] = 255
    return mask


def union(*masks):
    out = masks[0].copy()
    for m in masks[1:]:
        out |= m
    return out


def clip01(v):
    return float(max(0.0, min(1.0, v)))


def to_score(value, low, high, invert=False):
    if high == low:
        return 50
    t = (value - low) / (high - low)
    t = clip01(t)
    if invert:
        t = 1 - t
    return int(round(t * 100))


def score_hydration(bgr, mask):
    """Skin micro-roughness via high-pass filter on L channel."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    smooth = cv2.bilateralFilter(L, 7, 30, 30)
    detail = L - smooth
    vals = detail[mask > 0]
    if vals.size < 50:
        return 50
    roughness = float(np.std(vals))
    DEBUG["hydration_roughness_std"] = round(roughness, 2)
    return to_score(roughness, 2.0, 6.0, invert=True)


def score_redness(bgr, mask):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32) - 128
    vals = a[mask > 0]
    if vals.size < 50:
        return 50
    mean_a = float(vals.mean())
    DEBUG["redness_mean_a"] = round(mean_a, 2)
    return to_score(mean_a, 5, 30, invert=True)


def score_oiliness(bgr, mask):
    """Fraction of very-bright pixels (specular highlights) on T-zone."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    pixels = gray[mask > 0]
    if pixels.size < 50:
        return 50
    bright_frac = float((pixels > 235).mean())
    DEBUG["oiliness_bright_frac"] = bright_frac
    return to_score(bright_frac, 0.005, 0.35, invert=True)


def score_texture(bgr, mask):
    """Texture irregularity via Laplacian std on edge-preserving smoothed image.
    Bilateral filter removes lighting, keeps pores/lines/bumps."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.bilateralFilter(gray, 7, 30, 30)
    detail = gray - smooth
    vals = detail[mask > 0]
    if vals.size < 50:
        return 50
    rough = float(np.std(vals))
    DEBUG["texture_rough_std"] = round(rough, 2)
    return to_score(rough, 2.0, 5.5, invert=True)


def score_wrinkles(bgr, masks_list):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 30, 80)
    combined = np.zeros_like(edges)
    for m in masks_list:
        combined |= m
    region_edges = edges[combined > 0]
    if region_edges.size < 50:
        return 50
    edge_frac = float((region_edges > 0).mean())
    DEBUG["wrinkles_edge_frac"] = round(edge_frac, 4)
    return to_score(edge_frac, 0.05, 0.18, invert=True)


def score_dark_circles(bgr, under_eye_mask, cheek_mask):
    """Tighter range: small gaps are normal (everyone has some).
    Use median to ignore eyelash/iris pixels that may leak in."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    eye_vals = L[under_eye_mask > 0]
    cheek_vals = L[cheek_mask > 0]
    if eye_vals.size < 30 or cheek_vals.size < 30:
        return 50
    gap = float(np.median(cheek_vals) - np.median(eye_vals))
    DEBUG["dark_circles_gap"] = round(gap, 2)
    return to_score(gap, 5, 35, invert=True)


def score_dark_spots(bgr, mask):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    blurred = cv2.GaussianBlur(L, (25, 25), 0)
    diff = blurred.astype(np.int16) - L.astype(np.int16)
    vals = diff[mask > 0]
    if vals.size < 50:
        return 50
    dark_frac = float((vals > 15).mean())
    return to_score(dark_frac, 0.005, 0.10, invert=True)


def score_blemish(bgr, mask):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    a = lab[:, :, 1].astype(np.float32) - 128
    blurred_L = cv2.GaussianBlur(L, (25, 25), 0)
    dark_diff = blurred_L.astype(np.int16) - L.astype(np.int16)
    blurred_a = cv2.GaussianBlur(a, (25, 25), 0)
    red_diff = a - blurred_a
    anomalies = ((dark_diff > 15) | (red_diff > 8)) & (mask > 0)
    region_size = int((mask > 0).sum())
    if region_size < 1000:
        return 50
    frac = float(anomalies.sum() / region_size)
    return to_score(frac, 0.005, 0.12, invert=True)


def score_radiance(bgr, mask):
    """Brightness + tone evenness. Median is robust to background pixels."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    vals = L[mask > 0]
    if vals.size < 50:
        return 50
    median_l = float(np.median(vals))
    p25, p75 = np.percentile(vals, [25, 75])
    iqr = float(p75 - p25)
    DEBUG["radiance_median_L"] = round(median_l, 1)
    DEBUG["radiance_iqr"] = round(iqr, 1)
    bright_score = to_score(median_l, 110, 220)
    even_score = to_score(iqr, 10, 45, invert=True)
    return int(round(0.5 * bright_score + 0.5 * even_score))


def score_firmness(bgr, jaw_mask, cheek_mask):
    """Proxy: jaw edge sharpness + cheek tone evenness.
    Broader ranges so a normal face lands near 60-80, not 38."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    vals = mag[jaw_mask > 0]
    if vals.size < 50:
        return 50
    sharp = float(np.median(vals))
    DEBUG["firmness_jaw_sharp"] = round(sharp, 2)
    sharp_score = to_score(sharp, 5, 28)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    cheek_vals = L[cheek_mask > 0]
    if cheek_vals.size < 50:
        return sharp_score
    p25, p75 = np.percentile(cheek_vals, [25, 75])
    cheek_iqr = float(p75 - p25)
    DEBUG["firmness_cheek_iqr"] = round(cheek_iqr, 2)
    tone_even = to_score(cheek_iqr, 12, 60, invert=True)
    return int(round(0.5 * sharp_score + 0.5 * tone_even))


def analyze(bgr):
    DEBUG.clear()
    regs = detect_face_and_eyes(bgr)
    if regs is None:
        return None, None

    lc = rect_to_mask(regs["left_cheek"], bgr.shape)
    rc = rect_to_mask(regs["right_cheek"], bgr.shape)
    cheeks = union(lc, rc)

    lue = rect_to_mask(regs["left_under_eye"], bgr.shape)
    rue = rect_to_mask(regs["right_under_eye"], bgr.shape)
    under_eyes = union(lue, rue)

    fh = rect_to_mask(regs["forehead"], bgr.shape)
    nose = rect_to_mask(regs["nose"], bgr.shape)
    mouth = rect_to_mask(regs["mouth"], bgr.shape)
    jaw = rect_to_mask(regs["jaw"], bgr.shape)
    lec = rect_to_mask(regs["left_eye_corner"], bgr.shape)
    rec = rect_to_mask(regs["right_eye_corner"], bgr.shape)

    face_skin = union(cheeks, fh, nose)
    fh_nose = union(fh, nose)

    scores = {
        "Hydration":      score_hydration(bgr, cheeks),
        "Blemish prone":  score_blemish(bgr, face_skin),
        "Redness prone":  score_redness(bgr, cheeks),
        "Oiliness/Shine": score_oiliness(bgr, fh_nose),
        "Dark Spots":     score_dark_spots(bgr, face_skin),
        "Radiance":       score_radiance(bgr, face_skin),
        "Texture":        score_texture(bgr, cheeks),
        "Firmness":       score_firmness(bgr, jaw, cheeks),
        "Wrinkles":       score_wrinkles(bgr, [fh, lec, rec, mouth]),
        "Dark Circles":   score_dark_circles(bgr, under_eyes, cheeks),
    }

    fx, fy, fw, fh_ = regs["face"]
    pad = int(0.10 * max(fw, fh_))
    H, W = bgr.shape[:2]
    cx0 = max(0, fx - pad); cy0 = max(0, fy - pad)
    cx1 = min(W, fx + fw + pad); cy1 = min(H, fy + fh_ + pad)
    face_crop = bgr[cy0:cy1, cx0:cx1]
    if face_crop.size > 0:
        try:
            cnn_scores = cnn_predict(face_crop)
            scores.update(cnn_scores)
        except Exception as e:
            print(f"CNN inference failed: {e}", file=sys.stderr)

    return scores, regs


def color_for_score(s):
    if s >= 80:
        return (60, 180, 75)
    if s >= 60:
        return (0, 200, 230)
    return (60, 60, 230)


def draw_regions_overlay(canvas, regs):
    colors = {
        "forehead": (200, 120, 50),
        "left_cheek": (50, 200, 120),
        "right_cheek": (50, 200, 120),
        "left_under_eye": (200, 50, 200),
        "right_under_eye": (200, 50, 200),
        "nose": (50, 200, 200),
        "mouth": (200, 200, 50),
        "jaw": (100, 100, 220),
        "left_eye_corner": (220, 100, 100),
        "right_eye_corner": (220, 100, 100),
    }
    for name, color in colors.items():
        x, y, w, h = regs[name]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)
    lm = regs.get("landmarks", {})
    for pt in lm.values():
        cv2.circle(canvas, pt, 3, (0, 255, 255), -1)


def draw_panel(canvas, scores, x0, panel_w):
    h = canvas.shape[0]
    cv2.rectangle(canvas, (x0, 0), (x0 + panel_w, h), (32, 32, 32), -1)
    cv2.putText(canvas, "Skin Scores", (x0 + 16, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    y = 80
    for name in METRICS:
        s = scores[name]
        color = color_for_score(s)
        cv2.putText(canvas, name, (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{s}", (x0 + panel_w - 55, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA)
        bar_x0 = x0 + 16
        bar_x1 = x0 + panel_w - 70
        bar_y = y + 8
        cv2.rectangle(canvas, (bar_x0, bar_y), (bar_x1, bar_y + 8), (60, 60, 60), -1)
        fill_w = int((bar_x1 - bar_x0) * s / 100)
        cv2.rectangle(canvas, (bar_x0, bar_y), (bar_x0 + fill_w, bar_y + 8), color, -1)
        y += 42

    cv2.putText(canvas, "Heuristic estimates (not medical)",
                (x0 + 16, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "R = retake   Q/ESC = quit",
                (x0 + 16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


def draw_preview_overlay(frame, faces_ok):
    cv2.putText(frame, "Press SPACE to capture", (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    tip = "Face locked. Hold still." if faces_ok else "Good light + look straight + fill the frame"
    cv2.putText(frame, tip, (16, frame.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def main(source=0):
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if isinstance(source, int) else cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open source: {source}", file=sys.stderr)
        return 1
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        print(f"Capture: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
              f"@ {cap.get(cv2.CAP_PROP_FPS):.0f}fps", file=sys.stderr)

    window = "Skin Score (SPACE=capture  R=retake  Q/ESC=quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    PANEL_W = 360
    state = "PREVIEW"
    snapshot = None
    snapshot_regs = None
    scores = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if state == "PREVIEW":
            preview_regs = detect_face_and_eyes(frame)
            if preview_regs is not None:
                fx, fy, fw, fh = preview_regs["face"]
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 220, 0), 2)
                draw_regions_overlay(frame, preview_regs)
            draw_preview_overlay(frame, preview_regs is not None)
            display = frame
        else:
            h, w = snapshot.shape[:2]
            canvas = np.zeros((h, w + PANEL_W, 3), dtype=np.uint8)
            canvas[:, :w] = snapshot
            draw_regions_overlay(canvas, snapshot_regs)
            draw_panel(canvas, scores, w, PANEL_W)
            display = canvas

        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if state == "PREVIEW" and key == ord(" "):
            print("Analyzing...", file=sys.stderr)
            t0 = time.time()
            s, regs = analyze(frame)
            if s is None:
                print("No face detected. Retake.", file=sys.stderr)
                continue
            snapshot = frame.copy()
            scores = s
            snapshot_regs = regs
            state = "RESULT"
            print(f"Done in {(time.time() - t0) * 1000:.0f}ms", file=sys.stderr)
            for k, v in scores.items():
                print(f"  {k:>16}: {v}", file=sys.stderr)
            print("--- raw values ---", file=sys.stderr)
            for k, v in DEBUG.items():
                print(f"  {k:>26}: {v}", file=sys.stderr)
        elif state == "RESULT" and key == ord("r"):
            state = "PREVIEW"
            snapshot = None
            snapshot_regs = None
            scores = None

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    sys.exit(main(src))
