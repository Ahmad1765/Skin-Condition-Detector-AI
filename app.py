"""Streamlit web app wrapper for skin_score.

Run with:  streamlit run app.py
Then open the URL it prints (typically http://localhost:8501).
"""
import io
import sys
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image


@st.cache_resource(show_spinner="Loading models (one-time, ~250MB on first run)...")
def get_skin_score_module():
    import skin_score as _ss
    return _ss


st.set_page_config(
    page_title="Skin Analysis",
    page_icon=":sparkles:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1400px; }
    h1 { font-weight: 700; letter-spacing: -0.5px; }
    .stMetric { background: white; border-radius: 14px; padding: 12px; }
    div[data-testid="stImage"] img { border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Skin Analysis")
st.caption(
    "Multi-frame heuristic + CNN skin scoring. "
    "10 metrics. Per-condition overlays. Heuristic estimates only — not medical."
)

ss = get_skin_score_module()

with st.sidebar:
    st.header("Settings")
    cols_per_row = st.slider("Cards per row", 2, 5, 3)
    use_super_res = st.checkbox("Enable AI Super-Resolution", False,
                                help="Upscales the image using FSRCNN before analysis to detect micro-details. May increase processing time.")
    show_debug = st.checkbox("Show raw DEBUG values", False)
    if ss.HAVE_MEDIAPIPE:
        st.success("MediaPipe FaceMesh: 468 landmarks")
    else:
        st.warning("MediaPipe unavailable. Using YuNet 5-landmark fallback.")


def _to_bgr(img_file):
    pil = Image.open(img_file).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _render_grid_native(snapshot, regs, scores, overlays, cols=3):
    """Streamlit-native grid using columns. Better than rendering one giant image."""
    face_thumb, (ox, oy) = ss.face_head_crop(snapshot, regs["face"], target_aspect=1.0)
    th_h, th_w = face_thumb.shape[:2]
    metrics = [m for m in ss.METRICS if m in scores]
    rows = (len(metrics) + cols - 1) // cols
    idx = 0
    for _r in range(rows):
        row_cols = st.columns(cols, gap="small")
        for c in range(cols):
            if idx >= len(metrics):
                break
            name = metrics[idx]
            score = scores[name]
            mask = overlays.get(name) if overlays else None
            mask_crop = mask[oy:oy + th_h, ox:ox + th_w] if mask is not None else None
            with row_cols[c]:
                card = ss.render_card(face_thumb, mask_crop, name, score,
                                      card_w=380, card_h=470)
                st.image(_to_rgb(card), use_container_width=True)
            idx += 1


tab_camera, tab_upload = st.tabs(["Webcam", "Upload"])

with tab_camera:
    cam = st.camera_input("Center your face, good lighting, look straight",
                          key="camera_input")
    img_file = cam

with tab_upload:
    upl = st.file_uploader("Upload a clear face photo",
                           type=["jpg", "jpeg", "png", "bmp", "webp"],
                           key="upload_input")
    if upl is not None:
        img_file = upl

if img_file is not None:
    bgr = _to_bgr(img_file)

    if use_super_res:
        bgr = ss.upsample_image(bgr)

    progress = st.empty()
    progress.info("Analyzing...")
    t0 = time.time()

    # Single pass. The old "virtual frame" loop re-analyzed the SAME still
    # image with added Gaussian noise — that noise directly inflated the
    # LBP/edge-based metrics it was supposed to stabilize.
    scores, last_regs, last_overlays, proc_bgr = ss.analyze(bgr)

    if scores is None:
        progress.empty()
        st.error("No face detected. Try a clearer photo with the face centered and well-lit.")
    else:
        elapsed = time.time() - t0
        progress.success(f"Done in {elapsed:.1f}s")

        st.subheader("Score summary")
        score_cols = st.columns(5)
        for i, name in enumerate(ss.METRICS):
            if name not in scores:
                continue
            s = scores[name]
            label, _color = ss.severity_for_score(s)
            with score_cols[i % 5]:
                st.metric(label=name, value=f"{s}/100", delta=label,
                          delta_color="off" if label == "Average"
                          else ("normal" if label == "Good" else "inverse"))

        st.subheader("Your detailed skin scores")
        _render_grid_native(proc_bgr, last_regs, scores, last_overlays, cols=cols_per_row)

        if show_debug:
            with st.expander("DEBUG raw values (last frame)"):
                st.json({k: float(v) for k, v in ss.DEBUG.items()})

        st.download_button(
            label="Download analyzed image",
            data=cv2.imencode(".png",
                              ss.render_card_grid(proc_bgr, last_regs, scores,
                                                  last_overlays, cols=cols_per_row))[1].tobytes(),
            file_name="skin_analysis.png",
            mime="image/png",
        )

st.markdown("---")
st.caption(
    "Tip: good lighting + face filling the frame + looking straight at the camera "
    "gives the most accurate scores. Heuristic estimates only — not medical."
)
