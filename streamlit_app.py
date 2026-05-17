"""
Digital Twin for Ergonomic Risk Assessment
Master of Engineering Research — NUST
Researcher: Ndlovu Primrose (N02534131W)
Supervisor: Eng K Chinguwo

Implements:
  • Full RULA Upper Limb Score (Steps 1–7, Table A + B lookup)
  • REBA Score (trunk, neck, legs, upper arm, lower arm, wrist)
  • OCRA Repetition Risk Index (simplified frequency-based estimate)
  • Multi-joint angle extraction: neck, trunk, shoulder, elbow, wrist, knee, hip
  • Risk heatmap per body region
  • Session trend chart (risk score over time)
  • Exportable session summary CSV
  • 3-D skeletal digital twin
  • Colour-coded alert panel for floor managers
"""

# ─── Standard & Third-Party Imports ───────────────────────────────────────────
import csv
import io
import math
import time
from collections import deque

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

MP_AVAILABLE = False
MP_IMPORT_ERROR = ""
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )
    try:
        from mediapipe.framework.formats import landmark_pb2
    except ImportError:
        from mediapipe import python as _mp_py  # noqa
        import mediapipe.python._framework_bindings as _fb
        landmark_pb2 = _fb.landmark_pb2
    MP_AVAILABLE = True
except Exception as _e:
    MP_IMPORT_ERROR = str(_e)

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digital Twin – Ergonomic Risk Assessment",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for professional dashboard look ───────────────────────────────
st.markdown("""
<style>
    .risk-card {
        border-radius: 10px;
        padding: 14px 18px;
        margin: 6px 0;
        font-family: sans-serif;
    }
    .card-green  { background:#2d6a4f; color:#fff; }
    .card-orange { background:#e76f00; color:#fff; }
    .card-red    { background:#b5192b; color:#fff; }
    .card-gray   { background:#555;    color:#fff; }
    .section-header { font-size:0.78rem; font-weight:600;
                      letter-spacing:0.08em; text-transform:uppercase;
                      color:#888; margin-top:12px; }
    div[data-testid="metric-container"] { background:#1e1e2e;
                                          border-radius:8px; padding:10px; }
</style>
""", unsafe_allow_html=True)

# ─── MediaPipe Initialization ─────────────────────────────────────────────────
_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

if MP_AVAILABLE:
    import urllib.request, os, tempfile

    _model_path = os.path.join(tempfile.gettempdir(), "pose_landmarker_full.task")
    if not os.path.exists(_model_path):
        urllib.request.urlretrieve(_POSE_MODEL_URL, _model_path)

    _options = PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_model_path),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_model = PoseLandmarker.create_from_options(_options)

    import mediapipe.python.solutions.pose as _mp_pose_compat
    POSE_CONNECTIONS = _mp_pose_compat.POSE_CONNECTIONS

    class _PL:
        NOSE = 0
        LEFT_EYE_INNER = 1; LEFT_EYE = 2; LEFT_EYE_OUTER = 3
        RIGHT_EYE_INNER = 4; RIGHT_EYE = 5; RIGHT_EYE_OUTER = 6
        LEFT_EAR = 7; RIGHT_EAR = 8
        MOUTH_LEFT = 9; MOUTH_RIGHT = 10
        LEFT_SHOULDER = 11; RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13; RIGHT_ELBOW = 14
        LEFT_WRIST = 15; RIGHT_WRIST = 16
        LEFT_PINKY = 17; RIGHT_PINKY = 18
        LEFT_INDEX = 19; RIGHT_INDEX = 20
        LEFT_THUMB = 21; RIGHT_THUMB = 22
        LEFT_HIP = 23; RIGHT_HIP = 24
        LEFT_KNEE = 25; RIGHT_KNEE = 26
        LEFT_ANKLE = 27; RIGHT_ANKLE = 28
        LEFT_HEEL = 29; RIGHT_HEEL = 30
        LEFT_FOOT_INDEX = 31; RIGHT_FOOT_INDEX = 32

    class _FakeMpPose:
        PoseLandmark = _PL
        POSE_CONNECTIONS = POSE_CONNECTIONS

    mp_pose = _FakeMpPose()

# ─── GEOMETRY HELPERS ─────────────────────────────────────────────────────────

def angle_3pts(a, b, c):
    """Interior angle at vertex b formed by rays b→a and b→c (degrees)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))

def vertical_angle(top, bottom):
    """Angle of the vector (top→bottom) relative to vertical (degrees)."""
    v = np.array(top) - np.array(bottom)
    vertical = np.array([0, -1, 0])
    cos_val = np.dot(v, vertical) / (np.linalg.norm(v) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))

# ─── JOINT EXTRACTION ─────────────────────────────────────────────────────────

def lm_xyz(landmarks, idx):
    lm = landmarks[idx]
    return [lm.x, lm.y, lm.z]

def extract_joint_angles(landmarks):
    """
    Returns a dict of clinically relevant joint angles (degrees).
    Uses MediaPipe landmark indices.
    """
    L = landmarks
    PL = mp_pose.PoseLandmark

    def pt(idx): return lm_xyz(L, idx)

    angles = {}

    # ── RULA / REBA joints ──
    # Right upper arm (shoulder–elbow angle from vertical)
    angles["R_upper_arm"]   = angle_3pts(pt(PL.RIGHT_HIP), pt(PL.RIGHT_SHOULDER), pt(PL.RIGHT_ELBOW))
    # Right elbow flexion
    angles["R_elbow"]       = angle_3pts(pt(PL.RIGHT_SHOULDER), pt(PL.RIGHT_ELBOW), pt(PL.RIGHT_WRIST))
    # Right wrist deviation (elbow–wrist–index_finger)
    angles["R_wrist"]       = angle_3pts(pt(PL.RIGHT_ELBOW), pt(PL.RIGHT_WRIST), pt(PL.RIGHT_INDEX))

    # Left upper arm
    angles["L_upper_arm"]   = angle_3pts(pt(PL.LEFT_HIP), pt(PL.LEFT_SHOULDER), pt(PL.LEFT_ELBOW))
    angles["L_elbow"]       = angle_3pts(pt(PL.LEFT_SHOULDER), pt(PL.LEFT_ELBOW), pt(PL.LEFT_WRIST))
    angles["L_wrist"]       = angle_3pts(pt(PL.LEFT_ELBOW), pt(PL.LEFT_WRIST), pt(PL.LEFT_INDEX))

    # Neck flexion (ear–shoulder–hip proxy)
    angles["neck"]          = angle_3pts(pt(PL.RIGHT_EAR), pt(PL.RIGHT_SHOULDER), pt(PL.RIGHT_HIP))

    # Trunk flexion (shoulder–hip–knee proxy)
    angles["trunk"]         = angle_3pts(pt(PL.RIGHT_SHOULDER), pt(PL.RIGHT_HIP), pt(PL.RIGHT_KNEE))

    # Right knee
    angles["R_knee"]        = angle_3pts(pt(PL.RIGHT_HIP), pt(PL.RIGHT_KNEE), pt(PL.RIGHT_ANKLE))
    # Left knee
    angles["L_knee"]        = angle_3pts(pt(PL.LEFT_HIP), pt(PL.LEFT_KNEE), pt(PL.LEFT_ANKLE))

    # Hip flexion (shoulder–hip–knee)
    angles["R_hip"]         = angle_3pts(pt(PL.RIGHT_SHOULDER), pt(PL.RIGHT_HIP), pt(PL.RIGHT_KNEE))

    return angles

# ─── RULA SCORING (Full Steps 1-7) ────────────────────────────────────────────

# Table A: Upper arm × Lower arm × Wrist
# Dimensions: upper_arm_score(1-4) × lower_arm_score(1-2) × wrist_score(1-4)
RULA_TABLE_A = {
    # (upper_arm, lower_arm, wrist): score
    (1, 1, 1): 1, (1, 1, 2): 2, (1, 1, 3): 2, (1, 1, 4): 3,
    (1, 2, 1): 2, (1, 2, 2): 2, (1, 2, 3): 3, (1, 2, 4): 3,
    (2, 1, 1): 2, (2, 1, 2): 2, (2, 1, 3): 3, (2, 1, 4): 4,
    (2, 2, 1): 2, (2, 2, 2): 3, (2, 2, 3): 3, (2, 2, 4): 4,
    (3, 1, 1): 2, (3, 1, 2): 3, (3, 1, 3): 3, (3, 1, 4): 4,
    (3, 2, 1): 2, (3, 2, 2): 3, (3, 2, 3): 4, (3, 2, 4): 5,
    (4, 1, 1): 3, (4, 1, 2): 3, (4, 1, 3): 4, (4, 1, 4): 5,
    (4, 2, 1): 3, (4, 2, 2): 4, (4, 2, 3): 4, (4, 2, 4): 5,
}

# Table B: Neck × Trunk × Legs
RULA_TABLE_B = {
    (1, 1, 1): 1, (1, 1, 2): 2, (1, 2, 1): 2, (1, 2, 2): 3,
    (1, 3, 1): 3, (1, 3, 2): 4, (1, 4, 1): 4, (1, 4, 2): 5,
    (2, 1, 1): 2, (2, 1, 2): 2, (2, 2, 1): 2, (2, 2, 2): 3,
    (2, 3, 1): 3, (2, 3, 2): 4, (2, 4, 1): 4, (2, 4, 2): 5,
    (3, 1, 1): 3, (3, 1, 2): 3, (3, 2, 1): 3, (3, 2, 2): 4,
    (3, 3, 1): 4, (3, 3, 2): 5, (3, 4, 1): 5, (3, 4, 2): 6,
    (4, 1, 1): 4, (4, 1, 2): 4, (4, 2, 1): 4, (4, 2, 2): 5,
    (4, 3, 1): 5, (4, 3, 2): 6, (4, 4, 1): 6, (4, 4, 2): 7,
    (5, 1, 1): 5, (5, 1, 2): 5, (5, 2, 1): 5, (5, 2, 2): 6,
    (5, 3, 1): 6, (5, 3, 2): 7, (5, 4, 1): 7, (5, 4, 2): 7,
    (6, 1, 1): 6, (6, 1, 2): 6, (6, 2, 1): 6, (6, 2, 2): 7,
    (6, 3, 1): 7, (6, 3, 2): 7, (6, 4, 1): 7, (6, 4, 2): 8,
}

# Grand score table C (Score A rows × Score B columns)
RULA_TABLE_C = [
    # ScoreB  1  2  3  4  5  6  7
    [1,  1,  1,  2,  3,  3,  4,  5,  5],   # ScoreA=1
    [2,  2,  2,  2,  3,  4,  4,  5,  5],   # ScoreA=2
    [3,  3,  3,  3,  3,  4,  4,  5,  6],   # ScoreA=3
    [4,  3,  3,  3,  4,  4,  5,  6,  6],   # ScoreA=4
    [5,  3,  3,  4,  4,  4,  5,  6,  7],   # ScoreA=5
    [6,  3,  4,  4,  4,  5,  6,  6,  7],   # ScoreA=6
    [7,  4,  4,  4,  5,  6,  6,  7,  7],   # ScoreA=7
]

def rula_upper_arm_score(angle):
    if angle < 20:                  return 1
    if angle < 45:                  return 2
    if angle < 90:                  return 3
    return 4

def rula_lower_arm_score(angle):
    return 1 if 60 <= angle <= 100 else 2

def rula_wrist_score(angle):
    # Wrist score based on deviation from neutral (180°)
    dev = abs(180 - angle)
    if dev < 15:   return 1
    if dev < 30:   return 2
    return 3

def rula_neck_score(angle):
    dev = abs(180 - angle)   # deviation from upright
    if dev < 10:   return 1
    if dev < 20:   return 2
    if dev < 30:   return 3
    return 4

def rula_trunk_score(angle):
    dev = abs(180 - angle)
    if dev < 10:   return 1
    if dev < 20:   return 2
    if dev < 60:   return 3
    return 4

def rula_legs_score():
    """Simplified: legs well supported (score 1) vs. not (score 2). Default 1."""
    return 1

def compute_full_rula(angles):
    ua  = rula_upper_arm_score(angles.get("R_upper_arm", 90))
    la  = rula_lower_arm_score(angles.get("R_elbow", 90))
    w   = rula_wrist_score(angles.get("R_wrist", 180))
    sa  = RULA_TABLE_A.get((min(ua, 4), min(la, 2), min(w, 4)), 5) + 1   # +1 for muscle use
    neck  = rula_neck_score(angles.get("neck", 180))
    trunk = rula_trunk_score(angles.get("trunk", 180))
    legs  = rula_legs_score()
    sb  = RULA_TABLE_B.get((min(neck, 6), min(trunk, 4), legs), 5) + 1
    # Final Grand score from Table C
    row = min(sa - 1, len(RULA_TABLE_C) - 1)
    col = min(sb, 7)
    grand = RULA_TABLE_C[row][col]
    return {
        "upper_arm_score": ua, "lower_arm_score": la, "wrist_score": w,
        "score_a": sa, "neck_score": neck, "trunk_score": trunk,
        "score_b": sb, "rula_grand": grand,
    }

# ─── REBA SCORING ─────────────────────────────────────────────────────────────

def compute_reba(angles):
    """
    Simplified REBA incorporating trunk, neck, legs, upper arm, lower arm, wrist.
    Returns REBA score (1-15) and risk level.
    Reference: Hignett & McAtamney (2000).
    """
    # Group A: Trunk, Neck, Legs
    trunk_dev = abs(180 - angles.get("trunk", 180))
    if trunk_dev < 5:     trunk_s = 1
    elif trunk_dev < 20:  trunk_s = 2
    elif trunk_dev < 60:  trunk_s = 3
    else:                 trunk_s = 4

    neck_dev = abs(180 - angles.get("neck", 180))
    if neck_dev < 10:    neck_s = 1
    elif neck_dev < 20:  neck_s = 2
    else:                neck_s = 3

    knee_angle = angles.get("R_knee", 180)
    if knee_angle > 170:   legs_s = 1
    elif knee_angle > 150: legs_s = 2
    else:                  legs_s = 3

    table_a = min(trunk_s + neck_s + legs_s - 1, 12)

    # Group B: Upper arm, Lower arm, Wrist
    ua_dev = abs(90 - angles.get("R_upper_arm", 90))
    if ua_dev < 20:   ua_s = 1
    elif ua_dev < 45: ua_s = 2
    elif ua_dev < 90: ua_s = 3
    else:             ua_s = 4

    la_angle = angles.get("R_elbow", 90)
    la_s = 1 if 60 <= la_angle <= 100 else 2

    wrist_dev = abs(180 - angles.get("R_wrist", 180))
    wrist_s = 1 if wrist_dev < 15 else 2

    table_b = min(ua_s + la_s + wrist_s - 1, 9)

    # Score C = table_a + table_b (simplified; full REBA uses lookup)
    score_c = table_a + table_b

    # Activity score (+1 for repetitive task)
    activity = 1
    reba = min(score_c + activity, 15)

    if reba <= 1:   level = ("Negligible", "#2d6a4f")
    elif reba <= 3: level = ("Low Risk", "#52b788")
    elif reba <= 7: level = ("Medium Risk", "#e76f00")
    elif reba <= 10: level = ("High Risk", "#d62828")
    else:            level = ("Very High Risk", "#6a040f")

    return {"reba_score": reba, "reba_level": level[0], "reba_color": level[1],
            "trunk_s": trunk_s, "neck_s": neck_s, "legs_s": legs_s,
            "ua_s": ua_s, "la_s": la_s, "wrist_s": wrist_s}

# ─── OCRA (Simplified Repetition Index) ──────────────────────────────────────

def compute_ocra(elbow_angle, cycle_time_sec=5.0):
    """
    Simplified OCRA index based on elbow flexion frequency.
    Assumes a 480-min shift and estimates actions per minute.
    """
    actions_per_min = 60.0 / max(cycle_time_sec, 1)
    # Reference: ≤30 act/min low risk; 31-39 medium; >40 high
    if actions_per_min <= 30:   return {"ocra_index": round(actions_per_min / 30, 2), "ocra_level": "Acceptable", "ocra_color": "#2d6a4f"}
    if actions_per_min <= 39:   return {"ocra_index": round(actions_per_min / 30, 2), "ocra_level": "Borderline", "ocra_color": "#e76f00"}
    return {"ocra_index": round(actions_per_min / 30, 2), "ocra_level": "Not Acceptable", "ocra_color": "#d62828"}

# ─── RULA RISK LABEL ─────────────────────────────────────────────────────────

def rula_action_level(grand_score):
    if grand_score <= 2:  return "Acceptable",         "#2d6a4f"
    if grand_score <= 4:  return "Further Investigation", "#52b788"
    if grand_score <= 6:  return "Investigate & Change Soon", "#e76f00"
    return "Investigate & Change Immediately", "#d62828"

# ─── REGION HEATMAP DATA ──────────────────────────────────────────────────────

def build_heatmap_df(angles, rula_data, reba_data):
    body_regions = {
        "Neck":        reba_data["neck_s"],
        "Trunk":       reba_data["trunk_s"],
        "Upper Arm":   reba_data["ua_s"],
        "Elbow":       rula_data["lower_arm_score"],
        "Wrist":       rula_data["wrist_score"],
        "Knee / Legs": reba_data["legs_s"],
    }
    df = pd.DataFrame(
        [(region, score) for region, score in body_regions.items()],
        columns=["Region", "Risk Score"]
    )
    return df

# ─── SESSION STATE INITIALIZATION ─────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=300)   # ~5 min at 1 fps
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/NUST_Zimbabwe_crest.png/120px-NUST_Zimbabwe_crest.png",
             width=80, use_container_width=False)
    st.markdown("### Digital Twin Controls")
    st.caption("NUST MEng Research — Ndlovu Primrose")
    st.divider()

    video_source = st.file_uploader(
        "Upload Workstation Video (MP4/MOV/AVI)",
        type=["mp4", "mov", "avi"]
    )

    run_analysis = st.toggle("▶ Start Analytics Pipeline", value=False)

    st.divider()
    st.markdown("**Assessment Settings**")
    cycle_time = st.slider("Estimated Task Cycle Time (sec)", 2, 30, 5,
                           help="Used for OCRA repetition index calculation.")
    skip_frames = st.slider("Frame Skip (speed vs. accuracy)", 1, 10, 3,
                            help="Process every Nth frame to improve performance.")
    show_skeleton_2d = st.checkbox("Show 2-D Skeleton Overlay", value=True)
    show_landmarks   = st.checkbox("Show Landmark Labels",       value=False)

    st.divider()
    st.markdown("**Export**")
    export_btn = st.button("⬇ Export Session CSV")

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.title("🏭 Digital Twin — Ergonomic Risk Assessment")
st.caption(
    "Real-time RULA · REBA · OCRA scoring | "
    "Multi-joint tracking | Risk heatmap | Session trends | "
    "NUST MEng Research 2026"
)

if not MP_AVAILABLE:
    st.error(
        f"MediaPipe failed to load: {MP_IMPORT_ERROR}\n\n"
        "Ensure mediapipe==0.10.35 is in requirements.txt and redeploy."
    )
    st.stop()

# ─── LAYOUT ───────────────────────────────────────────────────────────────────
tab_live, tab_trend, tab_heatmap, tab_scores = st.tabs(
    ["📹 Live Feed", "📈 Session Trends", "🗺 Risk Heatmap", "📋 Score Details"]
)

with tab_live:
    col_vid, col_twin = st.columns([1, 1])
    with col_vid:
        st.markdown('<p class="section-header">Computer Vision Feed</p>', unsafe_allow_html=True)
        video_ph = st.empty()
    with col_twin:
        st.markdown('<p class="section-header">3-D Skeletal Digital Twin</p>', unsafe_allow_html=True)
        twin_ph  = st.empty()

    st.divider()
    # Alert panel — three columns
    alert_col1, alert_col2, alert_col3 = st.columns(3)
    with alert_col1:
        rula_ph = st.empty()
    with alert_col2:
        reba_ph = st.empty()
    with alert_col3:
        ocra_ph = st.empty()

    st.markdown('<p class="section-header">Joint Angles (Current Frame)</p>', unsafe_allow_html=True)
    angle_ph = st.empty()

with tab_trend:
    st.markdown('<p class="section-header">Risk Score Over Session Time</p>', unsafe_allow_html=True)
    trend_ph = st.empty()
    st.markdown('<p class="section-header">Per-Joint Angle History</p>', unsafe_allow_html=True)
    joint_trend_ph = st.empty()

with tab_heatmap:
    st.markdown('<p class="section-header">Body Region Risk Heatmap</p>', unsafe_allow_html=True)
    heatmap_ph = st.empty()
    st.caption(
        "Colour-coded by sub-score from RULA/REBA analysis. "
        "Green = low risk | Orange = medium | Red = high risk."
    )

with tab_scores:
    scores_ph = st.empty()

# ─── HELPER: Render placeholders when idle ────────────────────────────────────

def render_idle():
    video_ph.info("⏸ Waiting for video input …")
    twin_ph.info("3-D twin will appear here during analysis.")
    rula_ph.markdown(
        '<div class="risk-card card-gray"><b>RULA</b><br>—</div>',
        unsafe_allow_html=True
    )
    reba_ph.markdown(
        '<div class="risk-card card-gray"><b>REBA</b><br>—</div>',
        unsafe_allow_html=True
    )
    ocra_ph.markdown(
        '<div class="risk-card card-gray"><b>OCRA</b><br>—</div>',
        unsafe_allow_html=True
    )

render_idle()

# ─── CSV EXPORT ───────────────────────────────────────────────────────────────
if export_btn and st.session_state.history:
    df_exp = pd.DataFrame(list(st.session_state.history))
    csv_bytes = df_exp.to_csv(index=False).encode()
    st.sidebar.download_button(
        "Download CSV", csv_bytes,
        file_name="ergonomic_session.csv", mime="text/csv"
    )

# ─── COLOUR HELPER ────────────────────────────────────────────────────────────
SCORE_COLORS = {1: "#2d6a4f", 2: "#52b788", 3: "#e76f00", 4: "#d62828", 5: "#6a040f"}

def score_to_color(s):
    return SCORE_COLORS.get(min(int(s), 5), "#555")

# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────
if video_source is not None and run_analysis:
    # Write temp file
    with open("/tmp/workstation_video.mp4", "wb") as f:
        f.write(video_source.read())

    cap = cv2.VideoCapture("/tmp/workstation_video.mp4")
    frame_count = 0
    session_start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.info("✅ Video processing complete. Review the Trends and Heatmap tabs.")
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        elapsed = time.time() - session_start

        # ── Process frame ──
        frame = cv2.resize(frame, (640, 480))
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = pose_model.detect(mp_image)

        # Defaults
        angles   = {}
        rula_d   = {}
        reba_d   = {}
        ocra_d   = {}
        xs, ys, zs = [], [], []

        if detection_result.pose_landmarks:
            lms = detection_result.pose_landmarks[0]  # first pose

            # Draw 2-D skeleton overlay
            if show_skeleton_2d:
                from mediapipe.python.solutions import drawing_utils as _du
                from mediapipe.python.solutions import drawing_styles as _ds
                import mediapipe.python.solutions.pose as _pose_sol
                proto_list = landmark_pb2.NormalizedLandmarkList()
                for lm in lms:
                    lm_proto = proto_list.landmark.add()
                    lm_proto.x = lm.x; lm_proto.y = lm.y; lm_proto.z = lm.z
                _du.draw_landmarks(
                    rgb, proto_list,
                    _pose_sol.POSE_CONNECTIONS,
                    _ds.get_default_pose_landmarks_style(),
                )

            angles = extract_joint_angles(lms)
            rula_d = compute_full_rula(angles)
            reba_d = compute_reba(angles)
            ocra_d = compute_ocra(angles.get("R_elbow", 90), cycle_time)

            # Collect 3-D coords
            for lm in lms:
                xs.append(lm.x)
                ys.append(-lm.y)
                zs.append(lm.z)

            # Store history
            record = {
                "elapsed_s": round(elapsed, 1),
                "rula_grand": rula_d.get("rula_grand", 0),
                "reba_score": reba_d.get("reba_score", 0),
                "ocra_index": ocra_d.get("ocra_index", 0),
                **{k: round(v, 1) for k, v in angles.items()},
            }
            st.session_state.history.append(record)

        # ── Column 1: Annotated video ──
        video_ph.image(rgb, channels="RGB", use_container_width=True)

        # ── Column 2: 3-D Digital Twin ──
        fig3d = go.Figure()
        if xs:
            fig3d.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="markers",
                marker=dict(size=4, color="royalblue", opacity=0.9),
                name="Joints",
            ))
            for conn in POSE_CONNECTIONS:
                s, e = conn[0], conn[1]
                if s < len(xs) and e < len(xs):
                    fig3d.add_trace(go.Scatter3d(
                        x=[xs[s], xs[e]], y=[ys[s], ys[e]], z=[zs[s], zs[e]],
                        mode="lines",
                        line=dict(
                            color=score_to_color(rula_d.get("rula_grand", 1)),
                            width=3,
                        ),
                        showlegend=False,
                    ))
        fig3d.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                xaxis=dict(range=[0, 1],  showticklabels=False, showgrid=False),
                yaxis=dict(range=[-1, 0], showticklabels=False, showgrid=False),
                zaxis=dict(range=[-1, 1], showticklabels=False, showgrid=False),
                bgcolor="rgba(14,17,23,0.8)",
                aspectmode="manual",
                aspectratio=dict(x=1, y=1.2, z=1.5),
            ),
            showlegend=False,
            height=340,
        )
        twin_ph.plotly_chart(fig3d, use_container_width=True)

        # ── Alert Cards ──
        if rula_d:
            grand = rula_d["rula_grand"]
            action, color = rula_action_level(grand)
            css_cls = "card-green" if grand <= 2 else "card-orange" if grand <= 4 else "card-red"
            rula_ph.markdown(
                f'<div class="risk-card {css_cls}">'
                f'<b>RULA Grand Score: {grand}</b><br>'
                f'<small>{action}</small></div>',
                unsafe_allow_html=True,
            )

        if reba_d:
            reba_ph.markdown(
                f'<div class="risk-card" style="background:{reba_d["reba_color"]};color:#fff;'
                f'border-radius:10px;padding:14px 18px;margin:6px 0;">'
                f'<b>REBA Score: {reba_d["reba_score"]}</b><br>'
                f'<small>{reba_d["reba_level"]}</small></div>',
                unsafe_allow_html=True,
            )

        if ocra_d:
            css = "card-green" if ocra_d["ocra_level"] == "Acceptable" else \
                  "card-orange" if ocra_d["ocra_level"] == "Borderline" else "card-red"
            ocra_ph.markdown(
                f'<div class="risk-card {css}">'
                f'<b>OCRA Index: {ocra_d["ocra_index"]}</b><br>'
                f'<small>{ocra_d["ocra_level"]}</small></div>',
                unsafe_allow_html=True,
            )

        # ── Angle Table ──
        if angles:
            nice_names = {
                "R_upper_arm": "Right Upper Arm (°)",
                "R_elbow":     "Right Elbow (°)",
                "R_wrist":     "Right Wrist (°)",
                "L_upper_arm": "Left Upper Arm (°)",
                "L_elbow":     "Left Elbow (°)",
                "L_wrist":     "Left Wrist (°)",
                "neck":        "Neck (°)",
                "trunk":       "Trunk (°)",
                "R_knee":      "Right Knee (°)",
                "R_hip":       "Right Hip (°)",
            }
            rows = [(nice_names.get(k, k), f"{v:.1f}°") for k, v in angles.items()]
            angle_ph.table(pd.DataFrame(rows, columns=["Joint", "Angle"]))

        # ── Trend Charts (update every 10 recorded frames) ──
        hist = list(st.session_state.history)
        if hist and len(hist) % 5 == 0:
            df_h = pd.DataFrame(hist)

            # Risk score trend
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=df_h["elapsed_s"], y=df_h["rula_grand"],
                                           name="RULA", line=dict(color="#e76f00", width=2)))
            fig_trend.add_trace(go.Scatter(x=df_h["elapsed_s"], y=df_h["reba_score"],
                                           name="REBA", line=dict(color="#4361ee", width=2)))
            fig_trend.update_layout(
                xaxis_title="Elapsed (s)", yaxis_title="Score",
                legend=dict(orientation="h"), height=240,
                margin=dict(l=0, r=0, t=20, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            trend_ph.plotly_chart(fig_trend, use_container_width=True)

            # Joint angle trend
            joint_cols = [c for c in df_h.columns if c in
                          ["R_elbow", "R_upper_arm", "neck", "trunk", "R_knee"]]
            if joint_cols:
                fig_jt = px.line(df_h, x="elapsed_s", y=joint_cols,
                                 labels={"value": "Angle (°)", "elapsed_s": "Elapsed (s)",
                                         "variable": "Joint"},
                                 height=240)
                fig_jt.update_layout(margin=dict(l=0, r=0, t=20, b=0),
                                     paper_bgcolor="rgba(0,0,0,0)")
                joint_trend_ph.plotly_chart(fig_jt, use_container_width=True)

            # Heatmap
            if rula_d and reba_d:
                df_heat = build_heatmap_df(angles, rula_d, reba_d)
                fig_heat = px.bar(
                    df_heat, x="Region", y="Risk Score",
                    color="Risk Score",
                    color_continuous_scale=["#2d6a4f", "#52b788", "#e76f00", "#d62828", "#6a040f"],
                    range_color=[1, 5],
                    height=300,
                )
                fig_heat.update_layout(
                    coloraxis_showscale=True,
                    margin=dict(l=0, r=0, t=20, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(range=[0, 5]),
                )
                heatmap_ph.plotly_chart(fig_heat, use_container_width=True)

            # Score detail table
            if rula_d and reba_d:
                detail_rows = [
                    ("RULA – Upper Arm Score",  rula_d.get("upper_arm_score")),
                    ("RULA – Lower Arm Score",  rula_d.get("lower_arm_score")),
                    ("RULA – Wrist Score",       rula_d.get("wrist_score")),
                    ("RULA – Score A",           rula_d.get("score_a")),
                    ("RULA – Neck Score",        rula_d.get("neck_score")),
                    ("RULA – Trunk Score",       rula_d.get("trunk_score")),
                    ("RULA – Score B",           rula_d.get("score_b")),
                    ("RULA – Grand Score",       rula_d.get("rula_grand")),
                    ("REBA – Trunk",             reba_d.get("trunk_s")),
                    ("REBA – Neck",              reba_d.get("neck_s")),
                    ("REBA – Legs",              reba_d.get("legs_s")),
                    ("REBA – Upper Arm",         reba_d.get("ua_s")),
                    ("REBA – Lower Arm",         reba_d.get("la_s")),
                    ("REBA – Wrist",             reba_d.get("wrist_s")),
                    ("REBA – Final Score",       reba_d.get("reba_score")),
                    ("OCRA – Index",             ocra_d.get("ocra_index")),
                    ("OCRA – Level",             ocra_d.get("ocra_level")),
                ]
                scores_ph.table(
                    pd.DataFrame(detail_rows, columns=["Metric", "Value"])
                )

    cap.release()

else:
    if not run_analysis:
        st.info(
            "⚠️ System idle. Upload a workstation video and toggle "
            "**Start Analytics Pipeline** in the sidebar to begin."
        )
