"""Skin Score - heuristic 10-metric skin analyzer.

Webcam preview -> SPACE to capture -> 10 scores (0-100) shown beside snapshot.
Estimates only. Not medical or cosmetic measurement.

Controls (preview): SPACE = capture, B = toggle landmark boxes, Q/ESC = quit.
Controls (result):  Up/Down or J/K = scroll, R = retake, Q/ESC = quit.

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
    Returns a copy of bgr with normalized luminance.
    NOTE: do not apply before texture/dark-spot/blemish scoring —
    CLAHE inflates local contrast and breaks those metrics."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = CLAHE.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def skin_color_mask(bgr, loose=False):
    """YCrCb-based skin filter. Excludes hair, eyebrows, glasses, lips,
    background that may fall inside region rects. Tone-tolerant range.
    `loose=True` widens the range — used for overlay visualization so
    issue pixels (often darker/redder than median skin) still pass."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Cr = ycc[:, :, 1]
    Cb = ycc[:, :, 2]
    if loose:
        skin = ((Cr >= 128) & (Cr <= 190) & (Cb >= 73) & (Cb <= 140))
    else:
        skin = ((Cr >= 133) & (Cr <= 178) & (Cb >= 77) & (Cb <= 132))
    return skin.astype(np.uint8) * 255

FACEMESH_LEFT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
FACEMESH_RIGHT_EYE_IDX = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
FACEMESH_LEFT_EYEBROW_IDX = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
FACEMESH_RIGHT_EYEBROW_IDX = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
FACEMESH_LIPS_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
                    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
FACEMESH_FACE_OVAL_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
                          152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
FACEMESH_NOSTRILS_IDX = [49, 48, 219, 218, 64, 60, 75, 4, 305, 290, 294, 279, 278, 438, 439, 459]

MP_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MP_FACE_LANDMARKER_PATH = os.path.join(
    os.path.expanduser("~"), ".cache", "skin_score", "face_landmarker.task"
)


def _ensure_mp_landmarker_model():
    if os.path.exists(MP_FACE_LANDMARKER_PATH):
        return
    os.makedirs(os.path.dirname(MP_FACE_LANDMARKER_PATH), exist_ok=True)
    print(f"Downloading MediaPipe FaceLandmarker model -> {MP_FACE_LANDMARKER_PATH}",
          file=sys.stderr)
    urllib.request.urlretrieve(MP_FACE_LANDMARKER_URL, MP_FACE_LANDMARKER_PATH)
    print("Model downloaded.", file=sys.stderr)


try:
    import mediapipe as _mp
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision

    _ensure_mp_landmarker_model()
    _mp_options = _mp_vision.FaceLandmarkerOptions(
        base_options=_mp_python.BaseOptions(model_asset_path=MP_FACE_LANDMARKER_PATH),
        running_mode=_mp_vision.RunningMode.IMAGE,
        num_faces=1,
    )
    MP_FACE_LANDMARKER = _mp_vision.FaceLandmarker.create_from_options(_mp_options)
    HAVE_MEDIAPIPE = True
    print("MediaPipe FaceLandmarker enabled (478 landmarks)", file=sys.stderr)
except Exception as _e:
    HAVE_MEDIAPIPE = False
    MP_FACE_LANDMARKER = None
    _mp = None
    print(f"MediaPipe unavailable, using YuNet 5-landmark fallback ({_e})",
          file=sys.stderr)


def detect_face_mp(bgr):
    """MediaPipe FaceLandmarker (Tasks API) — 478 landmarks → precise region
    polygons and a pixel-accurate face-skin polygon mask
    (face oval minus eyes, brows, lips, nostrils)."""
    if not HAVE_MEDIAPIPE:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
    result = MP_FACE_LANDMARKER.detect(mp_image)
    if not result.face_landmarks:
        return None
    H, W = bgr.shape[:2]
    lms = result.face_landmarks[0]
    pts = np.array([(lm.x * W, lm.y * H) for lm in lms], dtype=np.float32)

    def pi(idx):
        return tuple(pts[idx].astype(int))

    le = pi(33)
    re_ = pi(263)
    nose_tip = pi(1)
    ml = pi(61)
    mr = pi(291)
    chin = pi(152)
    forehead_top = pi(10)

    oval_pts = pts[FACEMESH_FACE_OVAL_IDX].astype(np.int32)
    x0, y0 = oval_pts.min(axis=0)
    x1, y1 = oval_pts.max(axis=0)
    fx, fy, fw, fh = int(x0), int(y0), int(x1 - x0), int(y1 - y0)

    skin_poly_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(skin_poly_mask, [oval_pts], 255)

    for idx_list in (FACEMESH_LEFT_EYE_IDX, FACEMESH_RIGHT_EYE_IDX,
                     FACEMESH_LEFT_EYEBROW_IDX, FACEMESH_RIGHT_EYEBROW_IDX,
                     FACEMESH_LIPS_IDX, FACEMESH_NOSTRILS_IDX):
        region_pts = pts[idx_list].astype(np.int32)
        if len(region_pts) >= 3:
            hull = cv2.convexHull(region_pts)
            cv2.fillPoly(skin_poly_mask, [hull], 0)

    eye_dist = max(20, int(np.hypot(re_[0] - le[0], re_[1] - le[1])))
    eye_mid = ((le[0] + re_[0]) // 2, (le[1] + re_[1]) // 2)
    mouth_mid = ((ml[0] + mr[0]) // 2, (ml[1] + mr[1]) // 2)
    eye_to_mouth = max(20, mouth_mid[1] - eye_mid[1])

    def rect_around(cx, cy, rw, rh):
        return (max(0, int(cx - rw / 2)), max(0, int(cy - rh / 2)),
                int(rw), int(rh))

    fh_y_top = max(0, forehead_top[1] + 5)
    fh_y_bot = eye_mid[1] - int(eye_dist * 0.35)
    fh_h = max(20, fh_y_bot - fh_y_top)
    forehead = (max(0, eye_mid[0] - int(eye_dist * 0.75)), fh_y_top,
                int(eye_dist * 1.5), fh_h)

    cheek_h = int(eye_to_mouth * 0.55)
    cheek_w = int(eye_dist * 0.60)
    cheek_cy = eye_mid[1] + int(eye_to_mouth * 0.50)
    left_cheek = rect_around(le[0] - int(eye_dist * 0.05), cheek_cy, cheek_w, cheek_h)
    right_cheek = rect_around(re_[0] + int(eye_dist * 0.05), cheek_cy, cheek_w, cheek_h)

    ue_w = int(eye_dist * 0.50)
    ue_h = int(eye_dist * 0.18)
    ue_off = int(eye_dist * 0.22)
    left_under_eye = rect_around(le[0] + int(eye_dist * 0.15), le[1] + ue_off, ue_w, ue_h)
    right_under_eye = rect_around(re_[0] - int(eye_dist * 0.15), re_[1] + ue_off, ue_w, ue_h)

    nose = rect_around(nose_tip[0], (eye_mid[1] + nose_tip[1]) // 2,
                       int(eye_dist * 0.45), int(eye_to_mouth * 0.55))

    mouth_w = int(np.hypot(mr[0] - ml[0], mr[1] - ml[1]) * 1.35)
    mouth = rect_around(mouth_mid[0], mouth_mid[1], mouth_w, int(eye_dist * 0.35))

    jaw = rect_around(mouth_mid[0], mouth_mid[1] + int(eye_to_mouth * 0.45),
                      int(eye_dist * 2.4), int(eye_to_mouth * 0.65))

    ec_w = int(eye_dist * 0.32)
    ec_h = int(eye_dist * 0.32)
    left_eye_corner = rect_around(le[0] - int(eye_dist * 0.18), le[1], ec_w, ec_h)
    right_eye_corner = rect_around(re_[0] + int(eye_dist * 0.18), re_[1], ec_w, ec_h)

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
            "left_eye": le,
            "right_eye": re_,
            "nose_tip": nose_tip,
            "mouth_left": ml,
            "mouth_right": mr,
        },
        "skin_poly_mask": skin_poly_mask,
        "source": "mediapipe",
    }


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
    """Face detection + landmark-anchored regions.
    Prefers MediaPipe FaceMesh (468 landmarks, polygon skin mask).
    Falls back to YuNet (5 landmarks, rect-only) if unavailable.
    Returns dict of region rects (x, y, w, h) or None if no face."""
    if HAVE_MEDIAPIPE:
        result = detect_face_mp(bgr)
        if result is not None:
            return result

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


def detect_lip_pixels(bgr, mouth_rect):
    """Return mask of lip pixels inside mouth_rect: high LAB a* (red), lower L.
    Used to strip lip edges from the mouth region so wrinkles aren't faked
    by the lip border line. Tone-tolerant: thresholds applied to relative
    deviation inside the rect, not absolute LAB."""
    h, w = bgr.shape[:2]
    x, y, rw, rh = mouth_rect
    x = max(0, x); y = max(0, y)
    x2 = min(w, x + rw); y2 = min(h, y + rh)
    mask = np.zeros((h, w), dtype=np.uint8)
    if x2 <= x or y2 <= y:
        return mask
    crop = bgr[y:y2, x:x2]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32) - 128
    L = lab[:, :, 0].astype(np.float32)
    a_med = float(np.median(a))
    L_med = float(np.median(L))
    lip = (a > a_med + 4) & (L < L_med + 5)
    sub = mask[y:y2, x:x2]
    sub[lip] = 255
    mask[y:y2, x:x2] = sub
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


def score_redness(bgr, mask, baseline_a=None):
    """Redness via LAB a-channel. With baseline_a (forehead median),
    measure cheek deviation from same person's baseline — tone-invariant."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32) - 128
    vals = a[mask > 0]
    if vals.size < 50:
        return 50
    mean_a = float(vals.mean())
    if baseline_a is not None:
        delta = mean_a - baseline_a
        DEBUG["redness_delta_a"] = round(delta, 2)
        return to_score(delta, -2, 12, invert=True)
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
    """Texture irregularity via high-pass detail std on bilateral-smoothed gray.
    Wider range so normal pores don't drag everyone to 30."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.bilateralFilter(gray, 7, 30, 30)
    detail = gray - smooth
    vals = detail[mask > 0]
    if vals.size < 50:
        return 50
    rough = float(np.std(vals))
    DEBUG["texture_rough_std"] = round(rough, 2)
    return to_score(rough, 1.5, 7.0, invert=True)


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


def score_dark_spots(bgr, mask, baseline_L=None):
    """Local-darker-than-surround fraction. Threshold scales with skin
    brightness — darker skin needs a smaller absolute L delta to qualify."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    L_smooth = cv2.medianBlur(L, 5)
    blurred = cv2.GaussianBlur(L_smooth, (25, 25), 0)
    diff = blurred.astype(np.int16) - L_smooth.astype(np.int16)
    vals = diff[mask > 0]
    if vals.size < 50:
        return 50
    if baseline_L is not None:
        thr = int(max(10, min(20, baseline_L * 0.09)))
    else:
        thr = 15
    dark_frac = float((vals > thr).mean())
    DEBUG["dark_spots_thr"] = thr
    DEBUG["dark_spots_frac"] = round(dark_frac, 4)
    return to_score(dark_frac, 0.01, 0.18, invert=True)


def score_blemish(bgr, mask, baseline_L=None):
    """Local-darker OR locally-redder pixel fraction. Dark threshold scales
    with skin brightness so the metric isn't tone-biased."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    a = lab[:, :, 1].astype(np.float32) - 128
    
    L_smooth = cv2.medianBlur(L, 5)
    blurred_L = cv2.GaussianBlur(L_smooth, (25, 25), 0)
    dark_diff = blurred_L.astype(np.int16) - L_smooth.astype(np.int16)
    
    a_smooth = cv2.GaussianBlur(a, (5, 5), 0)
    blurred_a = cv2.GaussianBlur(a_smooth, (25, 25), 0)
    red_diff = a_smooth - blurred_a
    
    if baseline_L is not None:
        dark_thr = int(max(10, min(20, baseline_L * 0.09)))
    else:
        dark_thr = 15
    anomalies = ((dark_diff > dark_thr) | (red_diff > 8)) & (mask > 0)
    region_size = int((mask > 0).sum())
    if region_size < 1000:
        return 50
    frac = float(anomalies.sum() / region_size)
    DEBUG["blemish_dark_thr"] = dark_thr
    DEBUG["blemish_frac"] = round(frac, 4)
    return to_score(frac, 0.01, 0.18, invert=True)


def score_radiance(bgr, mask, baseline_L=None):
    """Brightness + tone evenness. With baseline_L, brightness is judged
    relative to the person's own forehead so dark + light skin both
    land near mid-scale when lighting is good."""
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
    if baseline_L is not None:
        delta = median_l - baseline_L
        DEBUG["radiance_delta_L"] = round(delta, 1)
        bright_score = to_score(delta, -15, 8)
    else:
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


ISSUE_COLORS = {
    "Dark Spots":     (60, 60, 220),
    "Blemish prone":  (60, 90, 220),
    "Redness prone":  (90, 90, 220),
    "Oiliness/Shine": (60, 200, 220),
    "Wrinkles":       (200, 130, 210),
    "Texture":        (170, 200, 100),
    "Hydration":      (160, 160, 160),
    "Dark Circles":   (210, 170, 230),
    "Radiance":       (200, 220, 220),
    "Firmness":       (150, 130, 220),
}

ISSUE_STYLE = {
    "Dark Spots":     "spot",
    "Blemish prone":  "spot",
    "Redness prone":  "blob",
    "Oiliness/Shine": "blob",
    "Wrinkles":       "line",
    "Texture":        "line",
    "Hydration":      "blob",
    "Dark Circles":   "blob",
    "Radiance":       "blob",
    "Firmness":       "zone",
}

ISSUE_ORDER = [
    "Dark Spots", "Blemish prone", "Redness prone", "Oiliness/Shine",
    "Dark Circles", "Wrinkles", "Texture", "Hydration",
    "Radiance", "Firmness",
]


def severity_for_score(s):
    """Reference-app tiers: <80 needs attention, 80-94 average, 95+ good."""
    if s >= 95:
        return "Good", (80, 170, 80)
    if s >= 80:
        return "Average", (60, 180, 230)
    return "Needs Attention", (60, 60, 220)


def build_clean_skin_mask(bgr, regs):
    """Tight face-skin mask: skin color in face bbox MINUS eyes, eyebrows,
    lips, nostrils. So overlays never land on eyes/hair/lips/glasses.
    If MediaPipe provided a polygon-precise mask, intersect with that
    instead of relying solely on YCrCb color filtering."""
    H, W = bgr.shape[:2]
    skin = skin_color_mask(bgr, loose=True)
    fx, fy, fw, fh = regs["face"]
    face_bound = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(face_bound, (fx, fy), (fx + fw, fy + fh), 255, -1)
    skin = skin & face_bound

    poly = regs.get("skin_poly_mask")
    if poly is not None:
        return poly & skin

    lm = regs.get("landmarks", {})
    le = lm.get("left_eye"); re_ = lm.get("right_eye")
    if le and re_:
        eye_dist = max(20, int(np.hypot(re_[0] - le[0], re_[1] - le[1])))
        eye_r = int(eye_dist * 0.22)
        cv2.circle(skin, le, eye_r, 0, -1)
        cv2.circle(skin, re_, eye_r, 0, -1)
        brow_h = int(eye_dist * 0.35)
        brow_w = int(eye_dist * 0.55)
        cv2.rectangle(skin,
                      (le[0] - brow_w // 2, le[1] - brow_h - int(eye_dist * 0.15)),
                      (le[0] + brow_w // 2, le[1] - int(eye_dist * 0.10)), 0, -1)
        cv2.rectangle(skin,
                      (re_[0] - brow_w // 2, re_[1] - brow_h - int(eye_dist * 0.15)),
                      (re_[0] + brow_w // 2, re_[1] - int(eye_dist * 0.10)), 0, -1)

    ml = lm.get("mouth_left"); mr = lm.get("mouth_right")
    if ml and mr:
        mouth_w = max(20, int(np.hypot(mr[0] - ml[0], mr[1] - ml[1])))
        cx = (ml[0] + mr[0]) // 2
        cy = (ml[1] + mr[1]) // 2
        cv2.ellipse(skin, (cx, cy), (int(mouth_w * 0.85), int(mouth_w * 0.35)),
                    0, 0, 360, 0, -1)

    nose_tip = lm.get("nose_tip")
    if nose_tip and le and re_:
        eye_dist = max(20, int(np.hypot(re_[0] - le[0], re_[1] - le[1])))
        nr = int(eye_dist * 0.10)
        cv2.circle(skin, (nose_tip[0] - int(eye_dist * 0.10), nose_tip[1]), nr, 0, -1)
        cv2.circle(skin, (nose_tip[0] + int(eye_dist * 0.10), nose_tip[1]), nr, 0, -1)

    return skin


def smooth_mask(mask, kind="blob"):
    """Clean a raw threshold mask for visualization.
    - blob: morph close+open, blur, threshold → soft organic shapes
    - spot: morph open small → kills speckle noise, keeps tight components
    - line: morph close thin → connects edge fragments along skin lines
    - zone: large blur → soft area indicator
    """
    if mask is None or not mask.any():
        return mask
    if kind == "blob":
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.GaussianBlur(m, (21, 21), 0)
        m = (m > 110).astype(np.uint8) * 255
        return m
    if kind == "spot":
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if kind == "line":
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if kind == "zone":
        m = cv2.GaussianBlur(mask, (31, 31), 0)
        return (m > 60).astype(np.uint8) * 255
    return mask


def build_issue_overlays(bgr, viz_masks, baseline_L, baseline_a):
    """Per-issue binary masks for visualization.
    Thresholds match the scoring functions so overlays only fire where the
    score would also penalize — fewer false positives, only real issues."""
    H, W = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    a = lab[:, :, 1].astype(np.float32) - 128
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    face_skin = viz_masks["face_skin"]
    cheeks = viz_masks["cheeks"]
    fh_nose = viz_masks["fh_nose"]
    under_eyes = viz_masks["under_eyes"]
    wrinkle_regions = viz_masks["wrinkle_regions"]
    jaw = viz_masks["jaw"]

    out = {}

    L_smooth = cv2.medianBlur(L, 5)
    blurred_L = cv2.GaussianBlur(L_smooth, (25, 25), 0)
    dark_diff = blurred_L.astype(np.int16) - L_smooth.astype(np.int16)
    if baseline_L is not None:
        dark_thr = int(max(10, min(20, baseline_L * 0.09)))
    else:
        dark_thr = 15
    out["Dark Spots"] = ((dark_diff > dark_thr) & (face_skin > 0)).astype(np.uint8) * 255

    a_smooth = cv2.GaussianBlur(a, (5, 5), 0)
    blurred_a = cv2.GaussianBlur(a_smooth, (25, 25), 0)
    red_diff = a_smooth - blurred_a
    out["Blemish prone"] = (((dark_diff > dark_thr) | (red_diff > 8)) & (face_skin > 0)).astype(np.uint8) * 255

    ba = baseline_a if baseline_a is not None else 0
    out["Redness prone"] = ((a - ba > 7) & (cheeks > 0)).astype(np.uint8) * 255

    bright_thr = 235
    out["Oiliness/Shine"] = ((gray > bright_thr) & (fh_nose > 0)).astype(np.uint8) * 255

    g2 = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(g2, 35, 90)
    out["Wrinkles"] = cv2.bitwise_and(edges, wrinkle_regions)

    smooth_g = cv2.bilateralFilter(gray.astype(np.float32), 7, 30, 30)
    detail_g = np.abs(gray.astype(np.float32) - smooth_g)
    out["Texture"] = ((detail_g > 3.5) & (cheeks > 0)).astype(np.uint8) * 255

    smooth_L = cv2.bilateralFilter(L.astype(np.float32), 7, 30, 30)
    detail_L = np.abs(L.astype(np.float32) - smooth_L)
    out["Hydration"] = ((detail_L > 3.5) & (cheeks > 0)).astype(np.uint8) * 255

    out["Dark Circles"] = under_eyes.copy()

    if baseline_L is not None:
        out["Radiance"] = ((L < baseline_L - 12) & (face_skin > 0)).astype(np.uint8) * 255
    else:
        out["Radiance"] = ((L < 115) & (face_skin > 0)).astype(np.uint8) * 255

    out["Firmness"] = jaw.copy()
    return out


def _blend_mask_color(img, mask, color, alpha):
    """Alpha-blend solid color into img where mask>0. mask may be soft (0-255)."""
    if mask is None or not mask.any():
        return
    m = mask.astype(np.float32) / 255.0
    m *= alpha
    m3 = m[..., None]
    overlay = np.full_like(img, color, dtype=np.uint8)
    img[...] = (img.astype(np.float32) * (1 - m3) + overlay.astype(np.float32) * m3).astype(np.uint8)


def draw_overlay_blob(img, mask, color, alpha=0.55):
    """Soft filled blob — for area conditions (Dark Circles, Hydration, etc).
    No hard contour, edges feather out via Gaussian blur on the mask."""
    if mask is None or not mask.any():
        return
    soft = cv2.GaussianBlur(mask, (25, 25), 0)
    _blend_mask_color(img, soft, color, alpha)


def draw_overlay_spots(img, mask, color, max_spots=18, min_area=8, max_area_ratio=0.04):
    """Per-component bounding boxes — for discrete spot conditions
    (Dark Spots, Blemish prone). Matches reference app style: small red squares."""
    if mask is None or not mask.any():
        return
    H, W = img.shape[:2]
    max_area = int(H * W * max_area_ratio)
    nb, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for i in range(1, nb):
        x, y, w, h, area = stats[i]
        if area < min_area or area > max_area:
            continue
        candidates.append((area, x, y, w, h))
    candidates.sort(reverse=True)
    for _area, x, y, w, h in candidates[:max_spots]:
        pad = 3
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)


def draw_overlay_lines(img, mask, color, alpha=0.7):
    """Thin line draw — for wrinkles and texture. Skeletonize via erosion
    then overlay as colored lines."""
    if mask is None or not mask.any():
        return
    soft = cv2.GaussianBlur(mask, (3, 3), 0)
    _blend_mask_color(img, soft, color, alpha)


def draw_overlay_zone(img, mask, color, alpha=0.30):
    """Soft area highlight — for Firmness (whole jaw zone)."""
    if mask is None or not mask.any():
        return
    soft = cv2.GaussianBlur(mask, (41, 41), 0)
    _blend_mask_color(img, soft, color, alpha)


def apply_overlay(img, mask, condition):
    """Dispatch to the right overlay style for this condition."""
    color = ISSUE_COLORS[condition]
    style = ISSUE_STYLE[condition]
    smooth = smooth_mask(mask, kind=style)
    if style == "spot":
        draw_overlay_spots(img, smooth, color)
    elif style == "blob":
        draw_overlay_blob(img, smooth, color)
    elif style == "line":
        draw_overlay_lines(img, smooth, color)
    elif style == "zone":
        draw_overlay_zone(img, smooth, color)


def draw_rounded_rect(img, p1, p2, color, radius=14, thickness=-1):
    """Filled or outlined rounded rectangle. Antialiased."""
    x1, y1 = p1
    x2, y2 = p2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)


def draw_dashed_bar(img, x, y, width, height, score, n_dashes=18):
    """Reference-app style dashed progress bar.
    Red 0-60%, yellow 60-85%, green 85-100%."""
    gap = 3
    dash_w = (width - gap * (n_dashes - 1)) // n_dashes
    filled = int(round(score / 100 * n_dashes))
    for i in range(n_dashes):
        x0 = x + i * (dash_w + gap)
        pos = (i + 0.5) / n_dashes
        if i < filled:
            if pos < 0.6:
                color = (90, 90, 230)
            elif pos < 0.85:
                color = (70, 180, 230)
            else:
                color = (90, 180, 90)
        else:
            color = (220, 220, 220)
        cv2.rectangle(img, (x0, y), (x0 + dash_w, y + height), color, -1)


def face_head_crop(snapshot, face_rect, target_aspect=1.0):
    """Crop snapshot to head region (face + headroom + shoulders).
    Returns (cropped_bgr, crop_offset_xy) so masks can be sliced the same way."""
    H, W = snapshot.shape[:2]
    fx, fy, fw, fh = face_rect
    cx = fx + fw // 2
    cy = fy + int(fh * 0.55)
    crop_w = int(fw * 1.9)
    crop_h = int(crop_w / target_aspect)
    x0 = max(0, cx - crop_w // 2)
    y0 = max(0, cy - crop_h // 2)
    x1 = min(W, x0 + crop_w)
    y1 = min(H, y0 + crop_h)
    if x1 - x0 < crop_w:
        x0 = max(0, x1 - crop_w)
    if y1 - y0 < crop_h:
        y0 = max(0, y1 - crop_h)
    return snapshot[y0:y1, x0:x1].copy(), (x0, y0)


def render_card(face_thumb, mask_crop, condition, score, card_w=320, card_h=450):
    """One condition card. face_thumb is already cropped to head;
    mask_crop is at the same crop coords as face_thumb."""
    card = np.full((card_h, card_w, 3), 245, dtype=np.uint8)
    draw_rounded_rect(card, (0, 0), (card_w, card_h), (255, 255, 255), radius=18, thickness=-1)
    draw_rounded_rect(card, (0, 0), (card_w - 1, card_h - 1), (225, 225, 230), radius=18, thickness=1)

    pad = 16
    thumb_w = card_w - pad * 2
    fh_h, fh_w = face_thumb.shape[:2]
    scale = thumb_w / fh_w
    thumb_h_render = int(fh_h * scale)
    thumb_resized = cv2.resize(face_thumb, (thumb_w, thumb_h_render), interpolation=cv2.INTER_AREA)
    if mask_crop is not None and mask_crop.any():
        mask_resized = cv2.resize(mask_crop, (thumb_w, thumb_h_render), interpolation=cv2.INTER_NEAREST)
        apply_overlay(thumb_resized, mask_resized, condition)
    card[pad:pad + thumb_h_render, pad:pad + thumb_w] = thumb_resized

    text_y = pad + thumb_h_render + 32
    cv2.putText(card, f"{condition} score", (pad, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 110, 120), 1, cv2.LINE_AA)
    score_text = f"{score}"
    (sw_text, _), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)
    cv2.putText(card, score_text, (card_w - pad - sw_text - 36, text_y + 6),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (25, 25, 30), 2, cv2.LINE_AA)
    cv2.putText(card, "/100", (card_w - pad - 34, text_y + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 150), 1, cv2.LINE_AA)

    sev_label, _ = severity_for_score(score)
    cv2.putText(card, sev_label, (pad, text_y + 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (25, 25, 30), 1, cv2.LINE_AA)

    bar_y = text_y + 64
    draw_dashed_bar(card, pad, bar_y, card_w - pad * 2, 6, score)
    cv2.putText(card, "0", (pad, bar_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 160), 1, cv2.LINE_AA)
    cv2.putText(card, "100", (card_w - pad - 22, bar_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 160), 1, cv2.LINE_AA)

    return card


def render_card_grid(snapshot, regs, scores, overlays, cols=5, gap=18):
    """Grid of condition cards. Crops face once and reuses for all cards."""
    face_thumb, (ox, oy) = face_head_crop(snapshot, regs["face"], target_aspect=1.0)
    th_h, th_w = face_thumb.shape[:2]

    cards = []
    for name in METRICS:
        if name not in scores:
            continue
        s = scores[name]
        full_mask = overlays.get(name) if overlays else None
        mask_crop = None
        if full_mask is not None:
            mask_crop = full_mask[oy:oy + th_h, ox:ox + th_w]
        cards.append(render_card(face_thumb, mask_crop, name, s))
    if not cards:
        return None
    rows = (len(cards) + cols - 1) // cols
    cw, ch = cards[0].shape[1], cards[0].shape[0]
    margin = 28
    title_h = 64
    W = cols * cw + (cols + 1) * gap + margin * 2
    H = title_h + rows * ch + (rows + 1) * gap + margin
    canvas = np.full((H, W, 3), 240, dtype=np.uint8)
    cv2.putText(canvas, "Your detailed skin scores", (margin + gap, title_h - 12),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (25, 25, 30), 2, cv2.LINE_AA)
    for i, card in enumerate(cards):
        r, c = divmod(i, cols)
        x = margin + gap + c * (cw + gap)
        y = title_h + gap + r * (ch + gap)
        canvas[y:y + ch, x:x + cw] = card
    return canvas


def analyze(bgr):
    DEBUG.clear()
    regs = detect_face_and_eyes(bgr)
    if regs is None:
        return None, None, None

    skin = skin_color_mask(bgr)

    def filt(m):
        return m & skin

    lc = filt(rect_to_mask(regs["left_cheek"], bgr.shape))
    rc = filt(rect_to_mask(regs["right_cheek"], bgr.shape))
    cheeks = union(lc, rc)

    lue = filt(rect_to_mask(regs["left_under_eye"], bgr.shape))
    rue = filt(rect_to_mask(regs["right_under_eye"], bgr.shape))
    under_eyes = union(lue, rue)

    fh = filt(rect_to_mask(regs["forehead"], bgr.shape))
    nose = filt(rect_to_mask(regs["nose"], bgr.shape))
    mouth_raw = rect_to_mask(regs["mouth"], bgr.shape)
    lips = detect_lip_pixels(bgr, regs["mouth"])
    mouth = filt(mouth_raw & (~lips))
    jaw = filt(rect_to_mask(regs["jaw"], bgr.shape))
    lec = filt(rect_to_mask(regs["left_eye_corner"], bgr.shape))
    rec = filt(rect_to_mask(regs["right_eye_corner"], bgr.shape))

    face_skin = union(cheeks, fh, nose)
    fh_nose = union(fh, nose)

    lab_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    fh_L_vals = lab_full[:, :, 0][fh > 0]
    fh_a_vals = lab_full[:, :, 1][fh > 0].astype(np.float32) - 128
    if fh_L_vals.size >= 50:
        baseline_L = float(np.median(fh_L_vals))
        baseline_a = float(np.median(fh_a_vals))
        DEBUG["baseline_fh_L"] = round(baseline_L, 1)
        DEBUG["baseline_fh_a"] = round(baseline_a, 2)
    else:
        baseline_L = None
        baseline_a = None

    wrinkle_regions = union(fh, lec, rec, mouth)

    scores = {
        "Hydration":      score_hydration(bgr, cheeks),
        "Blemish prone":  score_blemish(bgr, face_skin, baseline_L),
        "Redness prone":  score_redness(bgr, cheeks, baseline_a),
        "Oiliness/Shine": score_oiliness(bgr, fh_nose),
        "Dark Spots":     score_dark_spots(bgr, face_skin, baseline_L),
        "Radiance":       score_radiance(bgr, face_skin, baseline_L),
        "Texture":        score_texture(bgr, cheeks),
        "Firmness":       score_firmness(bgr, jaw, cheeks),
        "Wrinkles":       score_wrinkles(bgr, [fh, lec, rec, mouth]),
        "Dark Circles":   score_dark_circles(bgr, under_eyes, cheeks),
    }

    clean_skin = build_clean_skin_mask(bgr, regs)
    def filt_clean(m):
        return m & clean_skin

    lc_v = filt_clean(rect_to_mask(regs["left_cheek"], bgr.shape))
    rc_v = filt_clean(rect_to_mask(regs["right_cheek"], bgr.shape))
    cheeks_v = union(lc_v, rc_v)
    fh_v = filt_clean(rect_to_mask(regs["forehead"], bgr.shape))
    nose_v = filt_clean(rect_to_mask(regs["nose"], bgr.shape))
    lue_v = rect_to_mask(regs["left_under_eye"], bgr.shape)
    rue_v = rect_to_mask(regs["right_under_eye"], bgr.shape)
    under_eyes_v = union(lue_v, rue_v)
    mouth_v = filt_clean(rect_to_mask(regs["mouth"], bgr.shape) & (~lips))
    jaw_v = filt_clean(rect_to_mask(regs["jaw"], bgr.shape))
    lec_v = filt_clean(rect_to_mask(regs["left_eye_corner"], bgr.shape))
    rec_v = filt_clean(rect_to_mask(regs["right_eye_corner"], bgr.shape))

    viz_masks = {
        "face_skin": union(cheeks_v, fh_v, nose_v),
        "cheeks": cheeks_v,
        "fh_nose": union(fh_v, nose_v),
        "under_eyes": under_eyes_v,
        "wrinkle_regions": union(fh_v, lec_v, rec_v, mouth_v),
        "jaw": jaw_v,
    }
    overlays = build_issue_overlays(bgr, viz_masks, baseline_L, baseline_a)

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

    return scores, regs, overlays


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


def draw_panel(canvas, scores, x0, panel_w, issue_view=None):
    h = canvas.shape[0]
    cv2.rectangle(canvas, (x0, 0), (x0 + panel_w, h), (32, 32, 32), -1)
    cv2.putText(canvas, "Skin Scores", (x0 + 16, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    y = 80
    for name in METRICS:
        s = scores[name]
        color = color_for_score(s)
        active = (issue_view == name) or (issue_view == "all" and name in ISSUE_COLORS)
        if active:
            cv2.rectangle(canvas, (x0 + 8, y - 16),
                          (x0 + panel_w - 8, y + 20), (60, 60, 60), -1)
        label_color = ISSUE_COLORS.get(name, (220, 220, 220)) if issue_view == name else (220, 220, 220)
        try:
            idx = ISSUE_ORDER.index(name) + 1
            label = f"{idx}. {name}"
        except ValueError:
            label = f"   {name}"
        cv2.putText(canvas, label, (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, label_color, 1, cv2.LINE_AA)
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
                (x0 + 16, h - 64), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "N/P=cycle  A=all  H=hide  1-9,0=issue",
                (x0 + 16, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "R=retake  B=boxes  Q/ESC=quit",
                (x0 + 16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)


def draw_preview_overlay(frame, faces_ok, show_boxes=True):
    cv2.putText(frame, "Press SPACE to capture", (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    tip = "Face locked. Hold still." if faces_ok else "Good light + look straight + fill the frame"
    cv2.putText(frame, tip, (16, frame.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    hint = "B = hide boxes" if show_boxes else "B = show boxes"
    cv2.putText(frame, hint, (16, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


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

    window = "Skin Analysis"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    state = "PREVIEW"
    snapshot = None
    snapshot_regs = None
    snapshot_overlays = None
    scores = None
    show_boxes = False
    grid_canvas = None
    scroll_y = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if state == "PREVIEW":
            preview_regs = detect_face_and_eyes(frame)
            if preview_regs is not None and show_boxes:
                fx, fy, fw, fh = preview_regs["face"]
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 220, 0), 2)
                draw_regions_overlay(frame, preview_regs)
            draw_preview_overlay(frame, preview_regs is not None, show_boxes)
            display = frame
        else:
            view_h = 900
            view_w = grid_canvas.shape[1]
            gh = grid_canvas.shape[0]
            max_scroll = max(0, gh - view_h)
            scroll_y = max(0, min(scroll_y, max_scroll))
            if gh <= view_h:
                display = grid_canvas
            else:
                display = grid_canvas[scroll_y:scroll_y + view_h].copy()
                track_x = view_w - 14
                cv2.rectangle(display, (track_x, 10), (track_x + 6, view_h - 10),
                              (220, 220, 225), -1)
                thumb_h = max(40, int(view_h * view_h / gh))
                thumb_y = 10 + int((view_h - 20 - thumb_h) * scroll_y / max(1, max_scroll))
                cv2.rectangle(display, (track_x, thumb_y),
                              (track_x + 6, thumb_y + thumb_h), (130, 130, 140), -1)
                cv2.putText(display, "Up/Down=scroll  R=retake  Q=quit",
                            (24, view_h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (90, 90, 100), 1, cv2.LINE_AA)

        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("b"):
            show_boxes = not show_boxes
        if state == "RESULT":
            if key == 82 or key == ord("k"):
                scroll_y -= 80
            elif key == 84 or key == ord("j"):
                scroll_y += 80
        if state == "PREVIEW" and key == ord(" "):
            N_FRAMES = 8
            print(f"Capturing {N_FRAMES} frames...", file=sys.stderr)
            t0 = time.time()
            collected = []
            last_frame = None
            last_regs = None
            last_overlays = None
            for i in range(N_FRAMES):
                ok2, f = cap.read()
                if not ok2:
                    continue
                s_i, regs_i, ov_i = analyze(f)
                if s_i is None:
                    continue
                collected.append(s_i)
                last_frame = f
                last_regs = regs_i
                last_overlays = ov_i
            if not collected:
                print("No face detected in any frame. Retake.", file=sys.stderr)
                continue
            scores = {k: int(round(sum(d[k] for d in collected) / len(collected)))
                      for k in collected[0].keys()}
            snapshot = last_frame.copy()
            snapshot_regs = last_regs
            snapshot_overlays = last_overlays
            grid_canvas = render_card_grid(snapshot, snapshot_regs, scores,
                                           snapshot_overlays, cols=5)
            scroll_y = 0
            state = "RESULT"
            print(f"Done in {(time.time() - t0) * 1000:.0f}ms "
                  f"({len(collected)}/{N_FRAMES} frames used)", file=sys.stderr)
            for k, v in scores.items():
                print(f"  {k:>16}: {v}", file=sys.stderr)
            print("--- raw values (last frame) ---", file=sys.stderr)
            for k, v in DEBUG.items():
                print(f"  {k:>26}: {v}", file=sys.stderr)
        elif state == "RESULT" and key == ord("r"):
            state = "PREVIEW"
            snapshot = None
            snapshot_regs = None
            snapshot_overlays = None
            scores = None
            grid_canvas = None
            scroll_y = 0

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    sys.exit(main(src))
