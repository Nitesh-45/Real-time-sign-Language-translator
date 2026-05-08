import time

import cv2
import streamlit as st

from config import CAMERA_INDEX, CONFIDENCE_THRESHOLD, ENCODER_PATH, MODEL_PATH
from src.hand_tracker import HandTracker
from src.predictor import SignPredictor
from src.quiz_mode import QuizManager
from src.sentence_builder import SentenceBuilder
from src.text_to_speech import TextToSpeech
from src.utils import (
    draw_status_box,
    ensure_directories,
    format_sign_name,
    get_emoji,
    load_supported_labels,
)


APP_NAME = "Real-Time Sign Language Translator"


st.set_page_config(
    page_title=APP_NAME,
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_custom_css(theme: str = "Dark") -> None:
    """Apply dashboard styling with optional light theme overrides."""
    st.markdown(
        """
        <style>
        :root {
            --bg-0: #070b14;
            --bg-1: #0d1322;
            --card: rgba(18, 27, 46, 0.86);
            --card-strong: rgba(25, 37, 62, 0.95);
            --line: rgba(148, 163, 184, 0.20);
            --text: #e5eefc;
            --muted: #9aa8bd;
            --accent: #38bdf8;
            --accent-2: #8b5cf6;
            --good: #22c55e;
            --warn: #f59e0b;
            --bad: #ef4444;
            --shadow: 0 18px 48px rgba(0, 0, 0, 0.22);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.14), transparent 30rem),
                radial-gradient(circle at top right, rgba(139, 92, 246, 0.18), transparent 28rem),
                linear-gradient(180deg, #070b14 0%, #0b1020 100%);
            color: var(--text);
            font-family: "Segoe UI", Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .block-container {
            padding-top: 1.35rem;
            padding-bottom: 2.5rem;
        }

        p, label, span, div {
            letter-spacing: 0;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a1020 0%, #111827 100%);
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] * {
            color: #dbeafe;
        }

        div[data-testid="stMetric"] {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: var(--shadow);
            min-height: 112px;
        }

        div[data-testid="stMetric"] * {
            color: var(--text);
        }

        .hero {
            padding: 34px 36px;
            border-radius: 28px;
            background:
                linear-gradient(135deg, rgba(14, 165, 233, 0.95), rgba(124, 58, 237, 0.92)),
                linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0));
            box-shadow: 0 24px 70px rgba(56, 189, 248, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.18);
            margin-bottom: 22px;
            position: relative;
            overflow: hidden;
        }

        .hero::after {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(120deg, rgba(255,255,255,0.16), transparent 34%),
                repeating-linear-gradient(90deg, rgba(255,255,255,0.06) 0 1px, transparent 1px 56px);
            opacity: 0.32;
            pointer-events: none;
        }

        .hero h1 {
            font-size: 2.75rem;
            line-height: 1.08;
            margin: 0 0 10px 0;
            color: white;
            letter-spacing: 0;
            position: relative;
            z-index: 1;
        }

        .hero p {
            color: rgba(255,255,255,0.88);
            font-size: 1.08rem;
            margin: 0;
            max-width: 920px;
            position: relative;
            z-index: 1;
        }

        .section-title {
            font-size: 1.18rem;
            font-weight: 800;
            color: var(--text);
            margin: 22px 0 12px 0;
        }

        .card {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 20px;
            box-shadow: var(--shadow);
            box-sizing: border-box;
            height: 100%;
            color: var(--text);
        }

        .feature-card {
            min-height: 190px;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }

        .metric-card {
            min-height: 132px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .prediction-card {
            min-height: 250px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .dashboard-card {
            background:
                linear-gradient(180deg, rgba(30, 41, 59, 0.92), rgba(15, 23, 42, 0.90));
            border: 1px solid rgba(125, 211, 252, 0.18);
            border-radius: 20px;
            padding: 18px;
            box-shadow: var(--shadow);
            min-height: 132px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-sizing: border-box;
        }

        .dashboard-icon {
            font-size: 1.8rem;
            margin-bottom: 10px;
        }

        .dashboard-label {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .dashboard-value {
            color: var(--text);
            font-size: 1.75rem;
            font-weight: 950;
            margin-top: 4px;
            line-height: 1.15;
        }

        .compact-card {
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 14px 16px;
            margin-bottom: 10px;
        }

        .feature-title {
            font-size: 1.04rem;
            font-weight: 800;
            color: var(--text);
            margin-bottom: 6px;
            line-height: 1.25;
        }

        .muted {
            color: var(--muted);
            font-size: 0.94rem;
            line-height: 1.50;
        }

        .big-emoji {
            font-size: 4.2rem;
            line-height: 1;
            margin: 8px 0 10px 0;
        }

        .big-label {
            font-size: 2.0rem;
            font-weight: 900;
            color: var(--text);
            margin: 4px 0;
            line-height: 1.14;
            overflow-wrap: anywhere;
        }

        .sentence-box {
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.16), rgba(139, 92, 246, 0.13));
            border: 1px solid rgba(125, 211, 252, 0.30);
            border-radius: 18px;
            padding: 18px;
            color: #e0f2fe;
            font-size: 1.05rem;
            font-weight: 650;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            font-weight: 800;
            font-size: 0.90rem;
            border: 1px solid transparent;
            margin: 4px 0 12px 0;
        }

        .status-success {
            color: #bbf7d0;
            background: rgba(34, 197, 94, 0.14);
            border-color: rgba(34, 197, 94, 0.34);
        }

        .status-warning {
            color: #fde68a;
            background: rgba(245, 158, 11, 0.15);
            border-color: rgba(245, 158, 11, 0.38);
        }

        .status-error {
            color: #fecaca;
            background: rgba(239, 68, 68, 0.14);
            border-color: rgba(239, 68, 68, 0.38);
        }

        .status-info {
            color: #bae6fd;
            background: rgba(14, 165, 233, 0.14);
            border-color: rgba(14, 165, 233, 0.34);
        }

        .workflow {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
            margin-top: 10px;
        }

        .workflow-step {
            background: rgba(15, 23, 42, 0.86);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 12px 14px;
            font-weight: 800;
            color: #f8fafc;
        }

        .workflow-arrow {
            color: #67e8f9;
            font-size: 1.25rem;
            font-weight: 900;
        }

        .process-card {
            text-align: center;
            background: rgba(15, 23, 42, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 18px;
            padding: 16px 10px;
            min-height: 120px;
            box-shadow: 0 14px 36px rgba(0, 0, 0, 0.18);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-sizing: border-box;
        }

        .process-icon {
            font-size: 2rem;
            line-height: 1;
            margin-bottom: 8px;
        }

        .process-label {
            color: var(--text);
            font-size: 0.92rem;
            font-weight: 850;
            line-height: 1.22;
        }

        .process-arrow-card {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 120px;
            color: #67e8f9;
            font-size: 1.55rem;
            font-weight: 950;
        }

        .sign-chip {
            display: flex;
            align-items: center;
            gap: 10px;
            background: rgba(15, 23, 42, 0.88);
            border: 1px solid var(--line);
            border-radius: 15px;
            padding: 12px 14px;
            margin-bottom: 10px;
            font-weight: 800;
            min-height: 54px;
            box-sizing: border-box;
            color: var(--text);
        }

        .sidebar-brand {
            padding: 10px 0 18px 0;
        }

        .sidebar-title {
            font-size: 1.25rem;
            font-weight: 900;
            color: var(--text);
            margin-bottom: 4px;
            line-height: 1.22;
        }

        .confidence-wrap {
            margin: 12px 0 14px 0;
        }

        .confidence-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            color: var(--text);
            font-weight: 800;
            margin-bottom: 8px;
        }

        .confidence-track {
            height: 14px;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.94);
            border: 1px solid rgba(148, 163, 184, 0.20);
            overflow: hidden;
        }

        .confidence-fill {
            height: 100%;
            border-radius: 999px;
            transition: width 0.2s ease;
        }

        .confidence-green { background: linear-gradient(90deg, #22c55e, #86efac); }
        .confidence-yellow { background: linear-gradient(90deg, #f59e0b, #fde047); }
        .confidence-red { background: linear-gradient(90deg, #ef4444, #fb7185); }

        .empty-state {
            background: rgba(15, 23, 42, 0.84);
            border: 1px dashed rgba(148, 163, 184, 0.36);
            border-radius: 20px;
            padding: 20px;
            color: var(--text);
            min-height: 132px;
            box-sizing: border-box;
        }

        .empty-state-icon {
            font-size: 2rem;
            margin-bottom: 8px;
        }

        .empty-state-title {
            font-size: 1.05rem;
            font-weight: 900;
            color: var(--text);
            margin-bottom: 4px;
        }

        .stButton > button {
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.24);
            font-weight: 800;
            padding: 0.62rem 0.9rem;
        }

        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #38bdf8, #8b5cf6);
        }

        code {
            border-radius: 12px;
        }

        pre, code {
            font-family: "Cascadia Code", Consolas, "Courier New", monospace;
        }

        @media (max-width: 900px) {
            .hero h1 { font-size: 2.0rem; }
            .big-label { font-size: 1.55rem; }
            .big-emoji { font-size: 3.2rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if theme != "Light":
        return

    st.markdown(
        """
        <style>
        :root {
            --bg-0: #f8fafc;
            --bg-1: #eef2ff;
            --card: rgba(255, 255, 255, 0.92);
            --card-strong: rgba(255, 255, 255, 0.98);
            --line: rgba(15, 23, 42, 0.13);
            --text: #0f172a;
            --muted: #526071;
            --accent: #0284c7;
            --accent-2: #7c3aed;
            --shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(14, 165, 233, 0.13), transparent 28rem),
                radial-gradient(circle at top right, rgba(124, 58, 237, 0.11), transparent 26rem),
                linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
            color: var(--text);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%);
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] *,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: #0f172a !important;
        }

        .stApp,
        .stApp p,
        .stApp label,
        .stApp span,
        .stApp div {
            color: #0f172a;
        }

        .hero h1,
        .hero p {
            color: #ffffff !important;
        }

        div[data-testid="stMetric"],
        .card,
        .dashboard-card,
        .compact-card,
        .process-card,
        .sign-chip,
        .empty-state {
            background: rgba(255, 255, 255, 0.92);
            border-color: var(--line);
            box-shadow: var(--shadow);
        }

        div[data-testid="stMetric"] * {
            color: #0f172a;
        }

        .workflow-step,
        .confidence-track {
            background: rgba(241, 245, 249, 0.95);
            border-color: var(--line);
        }

        .section-title,
        .feature-title,
        .big-label,
        .dashboard-value,
        .process-label,
        .empty-state-title,
        .sidebar-title,
        .confidence-top,
        .sign-chip {
            color: #0f172a;
        }

        .muted,
        .dashboard-label {
            color: #526071;
        }

        .status-success {
            color: #166534;
            background: rgba(34, 197, 94, 0.13);
            border-color: rgba(22, 101, 52, 0.24);
        }

        .status-warning {
            color: #92400e;
            background: rgba(245, 158, 11, 0.15);
            border-color: rgba(146, 64, 14, 0.25);
        }

        .status-error {
            color: #991b1b;
            background: rgba(239, 68, 68, 0.12);
            border-color: rgba(153, 27, 27, 0.24);
        }

        .status-info {
            color: #075985;
            background: rgba(14, 165, 233, 0.12);
            border-color: rgba(7, 89, 133, 0.22);
        }

        .sentence-box {
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.10), rgba(124, 58, 237, 0.08));
            border-color: rgba(14, 165, 233, 0.20);
            color: #075985;
        }

        .confidence-track {
            background: #e2e8f0;
        }

        .confidence-green { background: linear-gradient(90deg, #16a34a, #22c55e); }
        .confidence-yellow { background: linear-gradient(90deg, #d97706, #f59e0b); }
        .confidence-red { background: linear-gradient(90deg, #dc2626, #ef4444); }

        .stButton > button {
            background: rgba(255, 255, 255, 0.94);
            color: #0f172a !important;
            border-color: rgba(15, 23, 42, 0.16);
        }

        .stButton > button * {
            color: #0f172a !important;
        }

        [data-testid="stWidgetLabel"] *,
        div[data-testid="stRadio"] *,
        div[data-testid="stSelectbox"] *,
        div[data-baseweb="radio"] *,
        div[data-baseweb="select"] *,
        div[data-baseweb="slider"] *,
        [data-testid="stMarkdownContainer"],
        [data-testid="stCaptionContainer"] {
            color: #0f172a !important;
        }

        div[data-testid="stSelectbox"],
        div[data-testid="stSelectbox"] > div,
        div[data-testid="stSelectbox"] div[data-baseweb="select"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] div,
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] [role="button"],
        div[data-baseweb="select"] [role="combobox"] {
            background: #ffffff !important;
            color: #0f172a !important;
            border-color: rgba(15, 23, 42, 0.18) !important;
        }

        div[data-testid="stSelectbox"] svg,
        div[data-baseweb="select"] svg {
            color: #0f172a !important;
            fill: #0f172a !important;
        }

        div[data-baseweb="popover"],
        div[data-baseweb="popover"] *,
        ul[role="listbox"],
        ul[role="listbox"] *,
        li[role="option"],
        li[role="option"] * {
            background: #ffffff !important;
            color: #0f172a !important;
        }

        li[role="option"]:hover,
        li[role="option"]:hover * {
            background: #e0f2fe !important;
            color: #0f172a !important;
        }

        div[data-testid="stRadio"] label,
        div[data-testid="stRadio"] label *,
        div[role="radiogroup"] label,
        div[role="radiogroup"] label * {
            color: #0f172a !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"],
        div[data-testid="stRadio"] label,
        div[data-baseweb="radio"],
        div[data-baseweb="radio"] label {
            background: transparent !important;
        }

        div[data-testid="stRadio"] label:hover,
        div[data-baseweb="radio"]:hover {
            background: rgba(224, 242, 254, 0.85) !important;
            border-radius: 12px;
        }

        div[data-testid="stRadio"] input[type="radio"] {
            accent-color: #0284c7 !important;
            background-color: #ffffff !important;
            border: 2px solid #0284c7 !important;
            box-shadow: none !important;
        }

        div[data-testid="stRadio"] input[type="radio"]:checked {
            accent-color: #0284c7 !important;
            background-color: #0284c7 !important;
            border-color: #0284c7 !important;
        }

        div[data-testid="stRadio"] [role="radio"],
        div[data-testid="stRadio"] input[type="radio"] + div,
        div[data-testid="stRadio"] input[type="radio"] + div *,
        div[role="radiogroup"] label > div:first-child,
        div[role="radiogroup"] label > div:first-child *,
        div[data-baseweb="radio"] > div:first-child {
            background: #ffffff !important;
            border-color: #0284c7 !important;
            color: #0f172a !important;
        }

        div[data-testid="stRadio"] [role="radio"][aria-checked="true"],
        div[data-testid="stRadio"] input[type="radio"]:checked + div,
        div[data-testid="stRadio"] input[type="radio"]:checked + div *,
        div[role="radiogroup"] label[aria-checked="true"] > div:first-child,
        div[data-baseweb="radio"][aria-checked="true"] > div:first-child {
            background: #0284c7 !important;
            border-color: #0284c7 !important;
            color: #ffffff !important;
        }

        div[data-testid="stRadio"] input[type="radio"] + div::before,
        div[data-testid="stRadio"] input[type="radio"] + div::after,
        div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child::before,
        div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child::after {
            background: #ffffff !important;
            border-color: #0284c7 !important;
            box-shadow: none !important;
        }

        div[data-testid="stRadio"] input[type="radio"]:checked + div::before,
        div[data-testid="stRadio"] input[type="radio"]:checked + div::after {
            background: #0284c7 !important;
            border-color: #0284c7 !important;
        }

        input,
        textarea {
            background: #ffffff !important;
            color: #0f172a !important;
            border-color: rgba(15, 23, 42, 0.18) !important;
        }

        ::placeholder {
            color: #64748b !important;
        }

        pre, code {
            background: #f1f5f9 !important;
            color: #0f172a !important;
            border-color: rgba(15, 23, 42, 0.12) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_predictor(confidence_threshold: float) -> SignPredictor:
    """Load the predictor once per confidence threshold."""
    return SignPredictor(confidence_threshold=confidence_threshold)


@st.cache_resource(show_spinner=False)
def load_tts() -> TextToSpeech:
    """Load the local text-to-speech engine once."""
    return TextToSpeech()


def init_session_state() -> None:
    """Create stateful objects used across Streamlit reruns."""
    if "sentence_builder" not in st.session_state:
        st.session_state.sentence_builder = SentenceBuilder()
    if "quiz_manager" not in st.session_state:
        st.session_state.quiz_manager = QuizManager()
    if "live_last_status" not in st.session_state:
        st.session_state.live_last_status = "idle"
    if "live_last_label" not in st.session_state:
        st.session_state.live_last_label = None
    if "live_last_confidence" not in st.session_state:
        st.session_state.live_last_confidence = 0.0
    if "live_camera_running" not in st.session_state:
        st.session_state.live_camera_running = False
    if "live_speech_status" not in st.session_state:
        st.session_state.live_speech_status = ""
    if "quiz_last_label" not in st.session_state:
        st.session_state.quiz_last_label = None
    if "quiz_last_confidence" not in st.session_state:
        st.session_state.quiz_last_confidence = 0.0
    if "quiz_feedback" not in st.session_state:
        st.session_state.quiz_feedback = "Ready when you are."
    if "quiz_limit" not in st.session_state:
        st.session_state.quiz_limit = 5
    if "quiz_completed_questions" not in st.session_state:
        st.session_state.quiz_completed_questions = 0
    if "quiz_finished" not in st.session_state:
        st.session_state.quiz_finished = False
    if "quiz_camera_running" not in st.session_state:
        st.session_state.quiz_camera_running = False
    if "ui_theme" not in st.session_state:
        st.session_state.ui_theme = "Dark"


def frame_to_rgb(frame):
    """Convert an OpenCV BGR frame to RGB for Streamlit display."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def open_camera(camera_index: int = CAMERA_INDEX):
    """Open webcam safely and return None if unavailable."""
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        return None
    return camera


def get_live_camera():
    """Return a persistent Live Translator camera for the current Streamlit session."""
    camera = st.session_state.get("live_camera_obj")
    if camera is not None and camera.isOpened():
        return camera

    camera = open_camera()
    if camera is None:
        st.session_state.live_camera_obj = None
        return None

    st.session_state.live_camera_obj = camera
    return camera


def get_live_tracker() -> HandTracker:
    """Return a persistent Live Translator hand tracker for the current session."""
    tracker = st.session_state.get("live_tracker_obj")
    if tracker is None:
        tracker = HandTracker(draw_landmarks=True)
        st.session_state.live_tracker_obj = tracker
    return tracker


def release_live_resources() -> None:
    """Release Live Translator camera/tracker resources when the user stops."""
    camera = st.session_state.get("live_camera_obj")
    if camera is not None:
        try:
            camera.release()
        except Exception:
            pass
    st.session_state.live_camera_obj = None

    tracker = st.session_state.get("live_tracker_obj")
    if tracker is not None:
        try:
            tracker.close()
        except Exception:
            pass
    st.session_state.live_tracker_obj = None


def get_quiz_camera():
    """Return a persistent Quiz Mode camera for the current Streamlit session."""
    camera = st.session_state.get("quiz_camera_obj")
    if camera is not None and camera.isOpened():
        return camera

    camera = open_camera()
    if camera is None:
        st.session_state.quiz_camera_obj = None
        return None

    st.session_state.quiz_camera_obj = camera
    return camera


def get_quiz_tracker() -> HandTracker:
    """Return a persistent Quiz Mode hand tracker for the current session."""
    tracker = st.session_state.get("quiz_tracker_obj")
    if tracker is None:
        tracker = HandTracker(draw_landmarks=True)
        st.session_state.quiz_tracker_obj = tracker
    return tracker


def release_quiz_resources() -> None:
    """Release Quiz Mode camera/tracker resources when the user stops."""
    camera = st.session_state.get("quiz_camera_obj")
    if camera is not None:
        try:
            camera.release()
        except Exception:
            pass
    st.session_state.quiz_camera_obj = None

    tracker = st.session_state.get("quiz_tracker_obj")
    if tracker is not None:
        try:
            tracker.close()
        except Exception:
            pass
    st.session_state.quiz_tracker_obj = None


def model_files_exist() -> bool:
    """Check whether the trained model and label encoder are present."""
    return MODEL_PATH.exists() and ENCODER_PATH.exists()


def render_header(title: str, subtitle: str) -> None:
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_card(icon: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="card feature-card">
            <div style="font-size:2.15rem; margin-bottom:12px; line-height:1;">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="muted">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, note: str = "") -> None:
    note_html = f'<div class="muted">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="card metric-card">
            <div class="muted">{title}</div>
            <div class="big-label">{value}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_cards() -> None:
    """Render compact demo statistics for the home dashboard."""
    items = [
        ("🤟", "Supported Signs", str(len(load_supported_labels()))),
        ("📷", "Input", "Webcam"),
        ("💬", "Output", "Text + Voice"),
        ("⚡", "Mode", "Real-Time"),
    ]
    cols = st.columns(4)
    for col, (icon, label, value) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="dashboard-card">
                    <div class="dashboard-icon">{icon}</div>
                    <div class="dashboard-label">{label}</div>
                    <div class="dashboard-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_process_diagram() -> None:
    """Render the camera-to-voice process as Streamlit columns."""
    steps = [
        ("📷", "Camera"),
        ("✋", "Hand Landmarks"),
        ("🧠", "ML Prediction"),
        ("💬", "Sentence"),
        ("🔊", "Voice"),
    ]
    columns = st.columns([1, 0.18, 1, 0.18, 1, 0.18, 1, 0.18, 1])
    step_index = 0
    for index, column in enumerate(columns):
        with column:
            if index % 2 == 0:
                icon, label = steps[step_index]
                step_index += 1
                st.markdown(
                    f"""
                    <div class="process-card">
                        <div class="process-icon">{icon}</div>
                        <div class="process-label">{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="process-arrow-card">→</div>', unsafe_allow_html=True)


def confidence_visual(confidence: float) -> tuple[str, str]:
    """Return CSS class and readable label for confidence score."""
    confidence = float(confidence or 0.0)
    if confidence >= 0.80:
        return "confidence-green", "Strong"
    if confidence >= 0.50:
        return "confidence-yellow", "Moderate"
    return "confidence-red", "Weak"


def render_confidence_meter(confidence: float) -> None:
    """Render a color-coded confidence meter."""
    confidence = min(max(float(confidence or 0.0), 0.0), 1.0)
    class_name, label = confidence_visual(confidence)
    st.markdown(
        f"""
        <div class="confidence-wrap">
            <div class="confidence-top">
                <span>Confidence</span>
                <span>{confidence:.0%} · {label}</span>
            </div>
            <div class="confidence-track">
                <div class="confidence-fill {class_name}" style="width:{confidence * 100:.1f}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(icon: str, title: str, message: str) -> None:
    """Render a clean empty/error state card instead of raw warnings."""
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-state-icon">{icon}</div>
            <div class="empty-state-title">{title}</div>
            <div class="muted">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(status: str, text: str) -> None:
    class_name = {
        "success": "status-success",
        "warning": "status-warning",
        "error": "status-error",
        "info": "status-info",
    }.get(status, "status-info")
    st.markdown(
        f'<div class="status-badge {class_name}">{text}</div>',
        unsafe_allow_html=True,
    )


def render_sentence_box(sentence: str) -> None:
    display_sentence = sentence or "Generated sentence will appear here."
    st.markdown(
        f"""
        <div class="sentence-box">
            {display_sentence}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_card(label, confidence: float, status: str) -> None:
    sign_name = format_sign_name(label)
    emoji = get_emoji(label)
    st.markdown(
        f"""
        <div class="card prediction-card">
            <div class="muted">Detected Sign</div>
            <div class="big-emoji">{emoji}</div>
            <div class="big-label">{sign_name}</div>
            <div class="muted">Confidence: {confidence:.0%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_confidence_meter(confidence)
    if status == "success":
        render_status_badge("success", "✅ Sign detected")
    elif status == "low_confidence":
        render_status_badge("warning", "⚠️ Low confidence")
    elif status == "model_missing":
        render_status_badge("error", "🧠 Model missing")
    elif status == "no_hand":
        render_status_badge("error", "❌ No hand detected")
    else:
        render_status_badge("info", "🤟 Waiting for gesture")


def render_model_missing_message(key_suffix: str) -> None:
    """Show clean model-missing instructions."""
    render_status_badge("error", "🧠 Model missing")
    render_empty_state(
        "🧠",
        "Model files are missing",
        "Collect samples, train the model, then reload this page.",
    )
    st.code(
        "python src/dataset_collector.py --label water --max-samples 50 --auto-save\n"
        "python src/dataset_collector.py --label help --max-samples 50 --auto-save\n"
        "python src/train_model.py",
        language="bash",
    )
    if st.button("🔄 Reload Model", key=f"reload_model_{key_suffix}"):
        load_predictor.clear()
        st.rerun()


def prediction_message(prediction: dict) -> str:
    """Convert predictor status to a user-facing message."""
    status = prediction.get("status")
    if status == "success":
        return "Gesture detected"
    if status == "low_confidence":
        return "Gesture confidence is low. Keep your hand steady."
    if status == "model_missing":
        return "Model files are missing."
    return "No hand detected."


def render_sidebar() -> str:
    """Render minimal sidebar brand and navigation."""
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <div class="sidebar-title">🤟 {APP_NAME}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigation",
            [
                "🏠 Home",
                "🎥 Live Translator",
                "📦 Dataset Guide",
                "🎯 Quiz Mode",
                "ℹ️ About Project",
            ],
            label_visibility="collapsed",
        )
        st.write("")
        theme_options = ["Dark", "Light"]
        current_theme = st.session_state.get("ui_theme", "Dark")
        selected_theme = st.selectbox(
            "Theme",
            theme_options,
            index=theme_options.index(current_theme),
            format_func=lambda value: "🌙 Dark" if value == "Dark" else "☀️ Light",
        )
        if selected_theme != current_theme:
            st.session_state.ui_theme = selected_theme
            st.rerun()
    return page


def render_home() -> None:
    render_header(
        "Real-Time Sign Language Translator",
        "Convert useful hand signs into text, speech, emojis, and learning feedback.",
    )

    st.markdown('<div class="section-title">Project Snapshot</div>', unsafe_allow_html=True)
    render_dashboard_cards()

    st.markdown('<div class="section-title">Core Features</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    features = [
        ("🎥", "Real-time Gesture Detection", "Detect trained signs directly from webcam input using hand landmarks."),
        ("📝", "Sentence Formation", "Convert detected signs into simple useful sentence output."),
        ("🔊", "Text-to-Speech", "Speak the generated sentence locally using pyttsx3."),
        ("🎯", "Quiz Learning Mode", "Practice beginner signs with target prompts, score, and accuracy."),
    ]
    for col, feature in zip(cols, features):
        with col:
            render_feature_card(*feature)

    st.markdown('<div class="section-title">Demo Workflow</div>', unsafe_allow_html=True)
    render_process_diagram()

    st.markdown('<div class="section-title">Current Model Status</div>', unsafe_allow_html=True)
    status_cols = st.columns(3)
    with status_cols[0]:
        render_metric_card("Model File", "Found" if MODEL_PATH.exists() else "Missing")
    with status_cols[1]:
        render_metric_card("Label Encoder", "Found" if ENCODER_PATH.exists() else "Missing")
    with status_cols[2]:
        render_metric_card("Supported Signs", str(len(load_supported_labels())))


def render_live_translator() -> None:
    render_header(
        "🎥 Live Translator",
        "Keep your hand clearly visible inside the camera frame and perform one trained sign.",
    )

    builder: SentenceBuilder = st.session_state.sentence_builder

    if not model_files_exist():
        render_model_missing_message("live_files")
        return

    top_cols = st.columns([2, 1])
    with top_cols[0]:
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.30,
            max_value=0.95,
            value=float(CONFIDENCE_THRESHOLD),
            step=0.05,
        )
    with top_cols[1]:
        frame_limit = st.slider("Camera batch", 30, 240, 90, step=30)

    predictor = load_predictor(confidence_threshold)
    if not predictor.model_loaded:
        render_model_missing_message("live_loader")
        if predictor.last_error:
            render_status_badge("error", f"🧠 {predictor.last_error}")
        return

    controls = st.columns([1, 1, 1, 1, 1])
    with controls[0]:
        if st.button(
            "🎥 Start Webcam",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.live_camera_running,
        ):
            release_live_resources()
            st.session_state.live_camera_running = True
            st.session_state.live_last_status = "idle"
    with controls[1]:
        if st.button(
            "⏹️ Stop Camera",
            use_container_width=True,
            disabled=not st.session_state.live_camera_running,
        ):
            st.session_state.live_camera_running = False
            st.session_state.live_last_status = "idle"
            release_live_resources()
            st.rerun()
    with controls[2]:
        if st.button("🔊 Speak Sentence", use_container_width=True):
            tts = load_tts()
            st.session_state.live_speech_status = tts.speak(builder.get_sentence())
    with controls[3]:
        if st.button("🧹 Clear Sentence", use_container_width=True):
            builder.clear_sentence()
            st.session_state.live_last_label = None
            st.session_state.live_last_confidence = 0.0
            st.session_state.live_last_status = "idle"
            st.success("Sentence cleared.")
    with controls[4]:
        if st.button("🔇 Stop Voice", use_container_width=True):
            st.session_state.live_speech_status = load_tts().stop()

    if st.session_state.live_speech_status:
        st.info(st.session_state.live_speech_status)

    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.markdown('<div class="section-title">Webcam Feed</div>', unsafe_allow_html=True)
        video_slot = st.empty()
        instruction_slot = st.empty()
    with right:
        st.markdown('<div class="section-title">Prediction Panel</div>', unsafe_allow_html=True)
        prediction_slot = st.empty()
        sentence_slot = st.empty()

    with prediction_slot.container():
        render_prediction_card(
            st.session_state.live_last_label,
            float(st.session_state.live_last_confidence),
            st.session_state.live_last_status,
        )
    with sentence_slot.container():
        st.markdown('<div class="section-title">Generated Sentence</div>', unsafe_allow_html=True)
        render_sentence_box(builder.get_sentence())

    if not st.session_state.live_camera_running:
        with instruction_slot.container():
            render_empty_state(
                "🎥",
                "Camera not started",
                "Click Start Webcam to begin. Use Stop Camera to end manually.",
            )
        return

    tracker = get_live_tracker()
    camera = get_live_camera()
    if camera is None:
        st.session_state.live_camera_running = False
        release_live_resources()
        with instruction_slot.container():
            render_empty_state(
                "📷",
                "Webcam unavailable",
                "Check camera permission, close other camera apps, or update CAMERA_INDEX.",
            )
        return

    read_failures = 0

    try:
        for _ in range(frame_limit):
            ok, frame = camera.read()
            if not ok or frame is None:
                read_failures += 1
                with instruction_slot.container():
                    render_empty_state(
                        "📷",
                        "Waiting for camera frame",
                        "The webcam missed a frame. Keeping the camera session alive.",
                    )
                if read_failures >= 20:
                    with instruction_slot.container():
                        render_empty_state(
                            "📷",
                            "Camera stream interrupted",
                            "Restart webcam or close any other app using the camera.",
                        )
                    st.session_state.live_camera_running = False
                    release_live_resources()
                    break
                time.sleep(0.05)
                continue

            read_failures = 0

            result = tracker.process_frame(frame)
            label = None
            confidence = 0.0
            status = "no_hand"

            if result.keypoints is None:
                message = "No hand detected"
                overlay_text = message
                label = st.session_state.live_last_label
                confidence = float(st.session_state.live_last_confidence)
            else:
                prediction = predictor.predict_from_landmarks(result.keypoints)
                label = prediction["label"]
                confidence = float(prediction["confidence"])
                status = prediction["status"]
                message = prediction_message(prediction)

                if status == "success":
                    builder.add_word(label)

                overlay_text = (
                    f"{format_sign_name(label)} ({confidence:.0%})"
                    if label
                    else message
                )

            if status == "success":
                st.session_state.live_last_label = label
                st.session_state.live_last_confidence = confidence
            st.session_state.live_last_status = status

            draw_status_box(result.annotated_frame, overlay_text)
            # Older Streamlit versions use use_column_width for st.image.
            video_slot.image(frame_to_rgb(result.annotated_frame), channels="RGB", use_column_width=True)

            with instruction_slot.container():
                if status == "success":
                    render_status_badge("success", "✅ Sign detected")
                elif status == "low_confidence":
                    render_empty_state(
                        "⚠️",
                        "Low confidence",
                        "Hold the gesture steady and keep the hand inside the frame.",
                    )
                else:
                    render_empty_state(
                        "✋",
                        "No hand detected",
                        "Show one hand clearly inside the camera frame.",
                    )

            with prediction_slot.container():
                render_prediction_card(label, confidence, status)
            with sentence_slot.container():
                st.markdown('<div class="section-title">Generated Sentence</div>', unsafe_allow_html=True)
                render_sentence_box(builder.get_sentence())

            time.sleep(0.03)
    finally:
        pass

    if st.session_state.live_camera_running:
        st.rerun()


def render_dataset_collector_guide() -> None:
    render_header(
        "📦 Dataset Guide",
        "Collect clean landmark samples for each sign, then train the classifier from terminal.",
    )

    st.markdown('<div class="section-title">Supported Signs</div>', unsafe_allow_html=True)
    labels = load_supported_labels()
    sign_cols = st.columns(4)
    for index, label in enumerate(labels):
        with sign_cols[index % 4]:
            st.markdown(
                f'<div class="sign-chip"><span style="font-size:1.5rem;">{get_emoji(label)}</span><span>{format_sign_name(label)}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Collection Commands</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.code(
            "python src/dataset_collector.py --label water\n"
            "python src/dataset_collector.py --label water --max-samples 50 --auto-save",
            language="bash",
        )
    with col2:
        st.code(
            "python src/dataset_collector.py --label help\n"
            "python src/dataset_collector.py --label help --max-samples 50 --auto-save",
            language="bash",
        )

    rec_cols = st.columns(2)
    with rec_cols[0]:
        render_metric_card("Minimum", "50 samples", "Per sign")
    with rec_cols[1]:
        render_metric_card("Better", "100 samples", "Per sign for stronger demo")

    st.markdown('<div class="section-title">Step-by-Step Process</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Choose sign", "Pick one supported label such as water, help, or thank_you."),
        ("2", "Run collector command", "Use terminal from the project root folder."),
        ("3", "Press S or use auto-save", "Manual mode saves with S; auto-save records when hand is detected."),
        ("4", "Train model", "Run python src/train_model.py after collecting samples."),
        ("5", "Run app", "Refresh Streamlit and test Live Translator or Quiz Mode."),
    ]
    step_cols = st.columns(5)
    for col, (number, title, text) in zip(step_cols, steps):
        with col:
            render_feature_card(number, title, text)

    st.markdown('<div class="section-title">Training Command</div>', unsafe_allow_html=True)
    st.code("python src/train_model.py", language="bash")

    model_cols = st.columns(2)
    with model_cols[0]:
        render_metric_card("gesture_model.pkl", "Found" if MODEL_PATH.exists() else "Missing")
    with model_cols[1]:
        render_metric_card("label_encoder.pkl", "Found" if ENCODER_PATH.exists() else "Missing")


def render_quiz_score(quiz: QuizManager) -> None:
    score = quiz.get_score()
    completed = min(
        int(st.session_state.get("quiz_completed_questions", 0)),
        int(st.session_state.get("quiz_limit", 5)),
    )
    limit = max(int(st.session_state.get("quiz_limit", 5)), 1)
    cols = st.columns(4)
    cols[0].metric("Progress", f"{completed}/{limit}")
    cols[1].metric("Correct", score["correct_answers"])
    cols[2].metric("Wrong / Skips", score["wrong_attempts"])
    cols[3].metric("Accuracy", f"{score['accuracy']:.1f}%")
    st.caption("Question completion")
    st.progress(min(max(completed / limit, 0.0), 1.0))
    st.caption("Accuracy")
    st.progress(min(max(float(score["accuracy"]) / 100, 0.0), 1.0))


def reset_quiz_session(quiz: QuizManager, feedback: str = "Ready when you are.") -> None:
    """Reset quiz manager and app-level learner session state."""
    release_quiz_resources()
    quiz.reset_quiz()
    st.session_state.quiz_last_label = None
    st.session_state.quiz_last_confidence = 0.0
    st.session_state.quiz_feedback = feedback
    st.session_state.quiz_completed_questions = 0
    st.session_state.quiz_finished = False
    st.session_state.quiz_camera_running = False


def render_quiz_final_result(quiz: QuizManager) -> None:
    """Show a clean final score card after the fixed learner session ends."""
    score = quiz.get_score()
    limit = int(st.session_state.get("quiz_limit", 5))
    st.markdown(
        f"""
        <div class="card prediction-card">
            <div class="muted">Learner Session Complete</div>
            <div class="big-emoji">🏁</div>
            <div class="big-label">Final Score</div>
            <div class="muted">
                Completed {limit}/{limit} target signs ·
                Correct answers: {score["correct_answers"]} ·
                Wrong/skipped: {score["wrong_attempts"]} ·
                Accuracy: {float(score["accuracy"]):.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def skip_quiz_question(quiz: QuizManager) -> None:
    """Skip the current target sign and move the learner session forward."""
    if st.session_state.quiz_finished:
        return

    skipped_target = quiz.get_current_question()
    quiz.check_answer(None, 0.0)
    st.session_state.quiz_completed_questions += 1
    st.session_state.quiz_last_label = None
    st.session_state.quiz_last_confidence = 0.0

    completed = int(st.session_state.quiz_completed_questions)
    limit = int(st.session_state.quiz_limit)

    if completed >= limit:
        st.session_state.quiz_finished = True
        st.session_state.quiz_camera_running = False
        release_quiz_resources()
        st.session_state.quiz_feedback = (
            f"Skipped {format_sign_name(skipped_target)}. Quiz complete."
        )
        return

    next_target = quiz.next_question()
    st.session_state.quiz_feedback = (
        f"Skipped {format_sign_name(skipped_target)}. "
        f"Question {completed + 1}/{limit}: {format_sign_name(next_target)} {get_emoji(next_target)}"
    )


def render_quiz_mode() -> None:
    render_header(
        "🎯 Quiz Mode",
        "Perform the target sign in front of the webcam and build confidence through practice.",
    )

    quiz: QuizManager = st.session_state.quiz_manager

    if not model_files_exist():
        render_model_missing_message("quiz_files")
        return

    predictor = load_predictor(CONFIDENCE_THRESHOLD)
    if not predictor.model_loaded:
        render_model_missing_message("quiz_loader")
        if predictor.last_error:
            render_status_badge("error", f"🧠 {predictor.last_error}")
        return

    score_slot = st.empty()
    with score_slot.container():
        render_quiz_score(quiz)

    target = quiz.get_current_question()
    top_cols = st.columns([1, 1], gap="large")
    with top_cols[0]:
        target_slot = st.empty()
    with top_cols[1]:
        prediction_slot = st.empty()

    def update_quiz_cards(target_label: str, predicted_label: str | None, confidence: float) -> None:
        """Refresh quiz target and live prediction cards."""
        with target_slot.container():
            st.markdown(
                f"""
                <div class="card prediction-card">
                    <div class="muted">Target Sign</div>
                    <div class="big-emoji">{get_emoji(target_label)}</div>
                    <div class="big-label">{format_sign_name(target_label)}</div>
                    <div class="muted">Perform this sign clearly in front of the webcam.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with prediction_slot.container():
            st.markdown(
                f"""
                <div class="card prediction-card">
                    <div class="muted">Live Prediction</div>
                    <div class="big-emoji">{get_emoji(predicted_label)}</div>
                    <div class="big-label">{format_sign_name(predicted_label)}</div>
                    <div class="muted">Confidence: {float(confidence):.0%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_confidence_meter(confidence)

    update_quiz_cards(
        target,
        st.session_state.quiz_last_label,
        float(st.session_state.quiz_last_confidence),
    )

    controls = st.columns([1, 1, 1, 1, 1, 2])
    with controls[0]:
        if st.button(
            "🎥 Start Webcam",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.quiz_camera_running or st.session_state.quiz_finished,
        ):
            release_quiz_resources()
            st.session_state.quiz_camera_running = True
            st.session_state.quiz_feedback = "Camera started. Perform the target sign."
    with controls[1]:
        if st.button(
            "⏹️ Stop Camera",
            use_container_width=True,
            disabled=not st.session_state.quiz_camera_running,
        ):
            st.session_state.quiz_camera_running = False
            st.session_state.quiz_feedback = "Camera stopped."
            release_quiz_resources()
            st.rerun()
    with controls[2]:
        if st.button("🔄 Reset Quiz", use_container_width=True):
            reset_quiz_session(quiz)
            st.rerun()
    with controls[3]:
        if st.button(
            "⏭️ Skip Sign",
            use_container_width=True,
            disabled=st.session_state.quiz_finished,
        ):
            was_running = st.session_state.quiz_camera_running
            skip_quiz_question(quiz)
            if not st.session_state.quiz_finished:
                st.session_state.quiz_camera_running = was_running
            st.rerun()
    with controls[4]:
        selected_limit = st.selectbox(
            "Question limit",
            [5, 10, 12],
            index=[5, 10, 12].index(int(st.session_state.quiz_limit)),
        )
        if selected_limit != st.session_state.quiz_limit:
            st.session_state.quiz_limit = selected_limit
            reset_quiz_session(quiz, f"New learner session set to {selected_limit} questions.")
            st.rerun()
    with controls[5]:
        quiz_frame_limit = st.slider("Camera batch", 30, 240, 90, step=30)

    feedback_slot = st.empty()
    with feedback_slot.container():
        render_status_badge("info", f"ℹ️ {st.session_state.quiz_feedback}")

    video_slot = st.empty()

    if st.session_state.quiz_finished:
        with feedback_slot.container():
            render_status_badge("success", "✅ Quiz complete. Review your score or reset to practice again.")
        with video_slot.container():
            render_quiz_final_result(quiz)
        return

    if not st.session_state.quiz_camera_running:
        with video_slot.container():
            render_empty_state(
                "🎥",
                "Camera not started",
                "Click Start Webcam when you are ready. Use Stop Camera to end manually.",
            )
        return

    tracker = get_quiz_tracker()
    camera = get_quiz_camera()
    if camera is None:
        st.session_state.quiz_camera_running = False
        release_quiz_resources()
        with feedback_slot.container():
            render_empty_state(
                "📷",
                "Webcam unavailable",
                "Check camera permission, close other camera apps, or update CAMERA_INDEX.",
            )
        return

    stable_label = None
    stable_count = 0
    last_scored_pair = None
    last_score_time = 0.0
    answer_cooldown_seconds = 1.2
    read_failures = 0

    try:
        for _ in range(quiz_frame_limit):
            ok, frame = camera.read()
            if not ok or frame is None:
                read_failures += 1
                with feedback_slot.container():
                    render_empty_state(
                        "📷",
                        "Waiting for camera frame",
                        "The webcam missed a frame. Keeping quiz mode alive.",
                    )
                if read_failures >= 20:
                    with feedback_slot.container():
                        render_empty_state(
                            "📷",
                            "Camera stream interrupted",
                            "Restart webcam or close any other app using the camera.",
                        )
                    st.session_state.quiz_camera_running = False
                    release_quiz_resources()
                    break
                time.sleep(0.05)
                continue

            read_failures = 0

            result = tracker.process_frame(frame)
            label = None
            confidence = 0.0
            status = "no_hand"

            if result.keypoints is None:
                overlay_text = "No hand detected"
                with feedback_slot.container():
                    render_empty_state(
                        "✋",
                        "No hand detected",
                        "Show one hand clearly inside the camera frame.",
                    )
                stable_label = None
                stable_count = 0
                last_scored_pair = None
            else:
                prediction = predictor.predict_from_landmarks(result.keypoints)
                label = prediction["label"]
                confidence = float(prediction["confidence"])
                status = prediction["status"]
                overlay_text = (
                    f"{format_sign_name(label)} ({confidence:.0%})"
                    if label
                    else prediction_message(prediction)
                )

                if status == "low_confidence":
                    with feedback_slot.container():
                        render_empty_state(
                            "⚠️",
                            "Low confidence",
                            "Hold the target sign steady and try again.",
                        )
                elif status == "success":
                    with feedback_slot.container():
                        render_status_badge("info", f"👀 Detected {format_sign_name(label)}")

            st.session_state.quiz_last_label = label
            st.session_state.quiz_last_confidence = confidence
            update_quiz_cards(quiz.get_current_question(), label, confidence)

            if label and status == "success":
                if label == stable_label:
                    stable_count += 1
                else:
                    stable_label = label
                    stable_count = 1
            else:
                stable_label = None
                stable_count = 0

            draw_status_box(result.annotated_frame, overlay_text)
            # Older Streamlit versions use use_column_width for st.image.
            video_slot.image(frame_to_rgb(result.annotated_frame), channels="RGB", use_column_width=True)

            current_target = quiz.get_current_question()
            current_pair = (current_target, label)
            ready_to_score = (
                label
                and status == "success"
                and stable_count >= 3
                and current_pair != last_scored_pair
                and time.time() - last_score_time >= answer_cooldown_seconds
            )

            if ready_to_score:
                answer = quiz.check_answer(label, confidence)
                last_scored_pair = current_pair
                last_score_time = time.time()
                with score_slot.container():
                    render_quiz_score(quiz)
                if answer.correct:
                    st.session_state.quiz_completed_questions += 1
                    completed = int(st.session_state.quiz_completed_questions)
                    limit = int(st.session_state.quiz_limit)
                    next_target = answer.next_question or quiz.get_current_question()
                    with score_slot.container():
                        render_quiz_score(quiz)

                    if completed >= limit:
                        st.session_state.quiz_finished = True
                        st.session_state.quiz_camera_running = False
                        release_quiz_resources()
                        st.session_state.quiz_feedback = "Quiz complete. Final score is ready."
                        with feedback_slot.container():
                            render_status_badge("success", "✅ Quiz complete. Final score is ready.")
                            render_quiz_final_result(quiz)
                        break

                    st.session_state.quiz_feedback = (
                        f"Correct! Question {completed + 1}/{limit}: "
                        f"{format_sign_name(next_target)} {get_emoji(next_target)}"
                    )
                    with feedback_slot.container():
                        render_status_badge("success", f"✅ {st.session_state.quiz_feedback}")
                    update_quiz_cards(next_target, label, confidence)
                else:
                    st.session_state.quiz_feedback = "Try again"
                    with feedback_slot.container():
                        render_status_badge("error", "❌ Try again")
                stable_label = None
                stable_count = 0

            time.sleep(0.03)
    finally:
        pass

    if st.session_state.quiz_camera_running and not st.session_state.quiz_finished:
        st.rerun()


def render_about_project() -> None:
    render_header(
        "ℹ️ About Project",
        "",
    )

    sections = [
        ("🧩 Problem Statement", "Many beginners cannot understand basic hand signs from webcam-based demos."),
        ("🎯 Project Objective", "Detect hand signs, convert them into text and emojis, speak the generated sentence, and provide quiz-based practice."),
        ("💡 Proposed Solution", "Use OpenCV for camera input, MediaPipe for hand landmarks, RandomForest for classification, and Streamlit for the user interface."),
        ("✨ Main Features", "Live translator, dataset guide, quiz mode, emoji mapping, confidence score, sentence builder, and text-to-speech."),
        ("🚀 Future Scope", "More signs, better dataset, deep learning models, multilingual speech output, mobile app, and progress tracking."),
    ]

    for index in range(0, len(sections), 2):
        cols = st.columns(2)
        for col, (title, text) in zip(cols, sections[index : index + 2]):
            with col:
                render_feature_card(title.split()[0], " ".join(title.split()[1:]), text)


def main() -> None:
    ensure_directories()
    init_session_state()
    load_custom_css(st.session_state.ui_theme)

    page = render_sidebar()

    if page == "🏠 Home":
        render_home()
    elif page == "🎥 Live Translator":
        render_live_translator()
    elif page == "📦 Dataset Guide":
        render_dataset_collector_guide()
    elif page == "🎯 Quiz Mode":
        render_quiz_mode()
    else:
        render_about_project()


if __name__ == "__main__":
    main()
