# app_streamlit.py
# Streamlit app for Driver Drowsiness Detection (MediaPipe + EAR) with live webcam
# Runs in browser with WebRTC via streamlit-webrtc. Mobile-friendly.

import time
import threading
from collections import deque
from dataclasses import dataclass,field
from typing import Optional, Deque, Tuple, Dict

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoTransformerBase

# -----------------------------
# Landmark indices and utilities
# -----------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def euclidean_dist(a, b):
    return float(np.linalg.norm(a - b))

def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    """
    eye_pts ordered as [p1, p2, p3, p4, p5, p6]
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 ||p1 - p4||)
    """
    p1, p2, p3, p4, p5, p6 = eye_pts
    A = euclidean_dist(p2, p6)
    B = euclidean_dist(p3, p5)
    C = euclidean_dist(p1, p4) + 1e-6
    return (A + B) / (2.0 * C)

def get_eye_points(landmarks, indices, w, h) -> np.ndarray:
    pts = []
    for idx in indices:
        lm = landmarks[idx]
        pts.append(np.array([lm.x * w, lm.y * h], dtype=np.float32))
    return np.stack(pts, axis=0)

def draw_label(img, text, org=(10, 30), color=(0, 255, 0)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

# -----------------------------
# Shared state between threads
# -----------------------------
@dataclass
class SharedMetrics:
    lock: threading.Lock
    last_ear: Optional[float] = None
    last_fps: float = 0.0
    last_alert: bool = False
    closed_frames: int = 0
    # ear_series: Deque[Tuple[float, float]] = deque(maxlen=300)  # (ts, ear)
    # log_lines: Deque[str] = deque(maxlen=200)
    ear_series: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=300))  # (ts, ear)
    log_lines: Deque[str] = field(default_factory=lambda: deque(maxlen=200))


    def push_ear(self, ear: float):
        with self.lock:
            self.last_ear = float(ear)
            self.ear_series.append((time.time(), float(ear)))

    def set_fps(self, fps: float):
        with self.lock:
            self.last_fps = float(fps)

    def set_alert(self, alert: bool, closed_frames: int):
        with self.lock:
            self.last_alert = bool(alert)
            self.closed_frames = int(closed_frames)

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        with self.lock:
            self.log_lines.append(f"[{ts}] {msg}")

shared = SharedMetrics(lock=threading.Lock())

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="Driver Drowsiness Detection (Streamlit)",
    page_icon="😴",
    layout="wide",
)

st.title("😴 Driver Drowsiness Detection — Streamlit")
st.caption("MediaPipe FaceMesh + EAR in the browser (WebRTC). Raspberry Pi friendly.")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    frame_size = st.selectbox("Frame size", ["640x480", "960x540", "480x360"], index=0)
    w, h = map(int, frame_size.split("x"))
    ear_thresh = st.slider("EAR threshold", 0.15, 0.35, 0.24, 0.005)
    min_frames = st.slider("Min consecutive frames (alert)", 3, 20, 9, 1)
    smooth_win = st.slider("Smoothing window (frames)", 3, 20, 10, 1)
    show_landmarks = st.checkbox("Draw eye landmarks", True)
    enable_demo = st.checkbox("Demo mode (upload a video if no webcam)")

    st.markdown("---")
    st.write("**Tips**")
    st.write("• If FPS is low, pick a smaller frame size.")
    st.write("• Tune threshold and min frames to reduce false alerts.")

# placeholders for metrics and chart
col1, col2, col3, col4 = st.columns(4)
ear_box = col1.metric("EAR", "—")
fps_box = col2.metric("FPS", "—")
closed_box = col3.metric("Closed frames", "—")
alert_box = col4.metric("Alert", "—")

chart = st.line_chart(
    data={"EAR": []},
    height=180,
)
st.markdown("#### 📜 Logs")
log_area = st.empty()

# -----------------------------
# MediaPipe shared objects
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh

# -----------------------------
# Video transformer
# -----------------------------
class DrowsinessTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.closed_frames = 0
        self.ear_buf: Deque[float] = deque(maxlen=int(max(smooth_win, 3)))
        self.last_time = time.time()
        self.fps = 0.0

    def _moving_average(self, val: float, k: int) -> float:
        self.ear_buf.append(val)
        return sum(self.ear_buf) / len(self.ear_buf)

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (w, h))

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        ear_text = "N/A"
        alert = False

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            left = get_eye_points(lm, LEFT_EYE, w, h)
            right = get_eye_points(lm, RIGHT_EYE, w, h)
            ear_val = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
            ear_smooth = self._moving_average(ear_val, int(max(smooth_win, 3)))

            shared.push_ear(ear_smooth)
            ear_text = f"{ear_smooth:.3f}"

            if show_landmarks:
                for p in np.vstack([left, right]).astype(int):
                    cv2.circle(img, tuple(p), 1, (255, 0, 0), -1)

            # drowsiness logic
            if ear_smooth < ear_thresh:
                self.closed_frames += 1
            else:
                self.closed_frames = 0

            if self.closed_frames >= min_frames:
                alert = True
                cv2.rectangle(img, (8, 8), (330, 48), (0, 0, 255), thickness=-1)
                draw_label(img, "DROWSINESS DETECTED!", (12, 40), (255, 255, 255))
                # log only when state switches to alert
                shared.log("Drowsiness alert (EAR {:.3f}, frames {})".format(ear_smooth, self.closed_frames))

        # FPS
        now = time.time()
        dt = max(now - self.last_time, 1e-6)
        self.last_time = now
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        shared.set_fps(self.fps)
        shared.set_alert(alert, self.closed_frames)

        draw_label(img, f"EAR: {ear_text}", (10, 22), (0, 255, 0))
        draw_label(img, f"Closed frames: {self.closed_frames}", (10, 45), (0, 255, 255))
        draw_label(img, f"FPS: {self.fps:.1f}", (10, 68), (255, 255, 255))

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -----------------------------
# WebRTC configuration (STUN)
# -----------------------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# -----------------------------
# Start/Stop streamer
# -----------------------------
st.markdown("### 🎥 Live Video")
info = st.empty()

# If demo mode, let user upload a sample video as fallback
uploaded_file = None
if enable_demo:
    uploaded_file = st.file_uploader("Upload a short driving/face video (optional demo)", type=["mp4", "mov", "avi"])

# Start detection
webrtc_ctx = webrtc_streamer(
    key="drowsiness",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    #video_transformer_factory=DrowsinessTransformer,
    video_processor_factory=DrowsinessTransformer, # updated name in streamlit-webrtc v0.54.0
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# -----------------------------
# UI update loop (runs while playing)
# -----------------------------
# For demo mode fallback: if no webcam stream, we show processed frames from uploaded video
def play_demo_video(file_buffer):
    """Simple demo loop reading uploaded video and running detection on CPU, displayed with st.image."""
    if not file_buffer:
        st.info("Upload a video above to run demo mode.")
        return
    st.info("Demo mode: playing uploaded video locally (no WebRTC).")
    bytes_data = file_buffer.read()
    np_data = np.frombuffer(bytes_data, np.uint8)
    cap = cv2.VideoCapture(cv2.imdecode(np_data, cv2.IMREAD_COLOR))
    # The above line won't open a VideoCapture directly from bytes. We'll write to temp file.
    tmp_path = f"/tmp/demo_{int(time.time())}.mp4"
    with open(tmp_path, "wb") as f:
        f.write(bytes_data)
    cap = cv2.VideoCapture(tmp_path)

    img_box = st.empty()
    closed_frames = 0
    ear_buf = deque(maxlen=int(max(smooth_win, 3)))

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    last = time.time()
    fps = 0.0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (w, h))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        ear_text = "N/A"
        alert = False

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            left = get_eye_points(lm, LEFT_EYE, w, h)
            right = get_eye_points(lm, RIGHT_EYE, w, h)
            ear_val = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
            ear_buf.append(ear_val)
            ear_smooth = sum(ear_buf) / len(ear_buf)
            ear_text = f"{ear_smooth:.3f}"

            if show_landmarks:
                for p in np.vstack([left, right]).astype(int):
                    cv2.circle(frame, tuple(p), 1, (255, 0, 0), -1)

            if ear_smooth < ear_thresh:
                closed_frames += 1
            else:
                closed_frames = 0

            if closed_frames >= min_frames:
                alert = True
                cv2.rectangle(frame, (8, 8), (330, 48), (0, 0, 255), -1)
                draw_label(frame, "DROWSINESS DETECTED!", (12, 40), (255, 255, 255))

        now = time.time()
        dt = max(now - last, 1e-6)
        last = now
        fps = 0.9 * fps + 0.1 * (1.0 / dt)

        draw_label(frame, f"EAR: {ear_text}", (10, 22), (0, 255, 0))
        draw_label(frame, f"Closed frames: {closed_frames}", (10, 45), (0, 255, 255))
        draw_label(frame, f"FPS: {fps:.1f}", (10, 68), (255, 255, 255))

        img_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        # Slow down lightly to avoid CPU spike
        time.sleep(0.01)

    cap.release()
    face_mesh.close()
    st.success("Demo finished.")

# Main live updates
last_chart_update = 0.0
while True:
    if webrtc_ctx.state.playing:
        # Update metrics from shared state
        with shared.lock:
            last_ear = shared.last_ear
            last_fps = shared.last_fps
            last_alert = shared.last_alert
            closed_frames_state = shared.closed_frames
            # Update chart every ~0.2s to save CPU
            now = time.time()
            if now - last_chart_update > 0.2 and last_ear is not None:
                chart.add_rows({"EAR": [last_ear]})
                last_chart_update = now
            # Logs
            logs_text = "\n".join(shared.log_lines)

        ear_box.metric("EAR", f"{last_ear:.3f}" if last_ear is not None else "—")
        fps_box.metric("FPS", f"{last_fps:.1f}")
        closed_box.metric("Closed frames", f"{closed_frames_state}")
        alert_box.metric("Alert", "Yes" if last_alert else "No")
        log_area.text(logs_text if logs_text else "No events yet …")

        time.sleep(0.05)  # keep UI responsive without hogging CPU
    else:
        # If stopped and demo is enabled, allow playback of an uploaded sample video
        if enable_demo:
            play_demo_video(uploaded_file)
        break
