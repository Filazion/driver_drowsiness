import io
import time
from collections import deque
from threading import Thread

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response

from src.utils import LEFT_EYE, RIGHT_EYE, eye_aspect_ratio, draw_label, moving_average, FpsCounter

app = Flask(__name__, template_folder="templates")

# Config (tweak to taste / environment)
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
EAR_THRESH = 0.24
MIN_FRAMES = 9
SMOOTH = 10

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

fps_counter = FpsCounter()
ear_buf = deque([], maxlen=SMOOTH)
closed_frames = 0

def get_eye_points(landmarks, indices, w, h):
    pts = []
    for idx in indices:
        lm = landmarks[idx]
        pts.append(np.array([lm.x * w, lm.y * h], dtype=np.float32))
    return np.stack(pts, axis=0)

def gen_frames():
    global closed_frames
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        result = face_mesh.process(rgb)

        ear_text = "N/A"
        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            left = get_eye_points(lm, LEFT_EYE, w, h)
            right = get_eye_points(lm, RIGHT_EYE, w, h)
            ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
            ear_smooth = moving_average(ear, ear_buf, maxlen=SMOOTH)
            ear_text = f"{ear_smooth:.3f}"

            for p in np.vstack([left, right]).astype(int):
                cv2.circle(frame, tuple(p), 1, (255, 0, 0), -1)

            if ear_smooth < EAR_THRESH:
                closed_frames += 1
            else:
                closed_frames = 0

            if closed_frames >= MIN_FRAMES:
                draw_label(frame, "DROWSINESS DETECTED!", (10, 80), (0, 0, 255))

        fps = fps_counter.tick()
        draw_label(frame, f"EAR: {ear_text}", (10, 30), (0, 255, 0))
        draw_label(frame, f"Closed frames: {closed_frames}", (10, 55), (0, 255, 255))
        draw_label(frame, f"FPS: {fps:.1f}", (10, 105), (255, 255, 255))

        # Encode as JPEG and yield as multipart
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        jpg = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

    cap.release()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    # Run: python -m app.webapp
    app.run(host="0.0.0.0", port=5000, debug=True)
