import argparse
import sys
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

# Pygame is only used for alert sound
try:
    import pygame
    PYGAME_OK = True
except Exception:
    PYGAME_OK = False

from utils import LEFT_EYE, RIGHT_EYE, eye_aspect_ratio, FpsCounter, draw_label, moving_average

def get_eye_points(landmarks, indices, w, h):
    pts = []
    for idx in indices:
        lm = landmarks[idx]
        pts.append(np.array([lm.x * w, lm.y * h], dtype=np.float32))
    return np.stack(pts, axis=0)

def main(args):
    # --- init pygame sound (optional) ---
    if PYGAME_OK and args.alert_wav:
        pygame.mixer.init()
        pygame.mixer.music.load(args.alert_wav)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,  # better iris/eye landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # --- camera ---
    cap = cv2.VideoCapture(args.src, cv2.CAP_V4L2 if args.v4l2 else 0)
    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps:
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        print("ERROR: Cannot open camera.", file=sys.stderr)
        return

    fps_counter = FpsCounter()
    ear_buf = deque([], maxlen=10)  # smoothing

    closed_frames = 0
    alert_on = False
    last_alert_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # For speed on Pi: downscale processing copy, but keep display at input size
        proc = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = proc.shape[:2]

        # --- mediapipe inference ---
        result = face_mesh.process(proc)
        ear = None

        if result.multi_face_landmarks:
            face_lm = result.multi_face_landmarks[0].landmark

            left_eye_pts = get_eye_points(face_lm, LEFT_EYE, w, h)
            right_eye_pts = get_eye_points(face_lm, RIGHT_EYE, w, h)

            left_ear = eye_aspect_ratio(left_eye_pts)
            right_ear = eye_aspect_ratio(right_eye_pts)
            ear = (left_ear + right_ear) / 2.0

            ear_smooth = moving_average(ear, ear_buf, maxlen=args.smooth)
            draw_label(frame, f"EAR: {ear_smooth:.3f}", (10, 30), (0, 255, 0))

            # draw simple eye contours for visual feedback
            for p in np.vstack([left_eye_pts, right_eye_pts]).astype(int):
                cv2.circle(frame, tuple(p), 1, (255, 0, 0), -1)

            # --- drowsiness logic ---
            if ear_smooth < args.ear_thresh:
                closed_frames += 1
            else:
                closed_frames = 0
                alert_on = False

            draw_label(frame, f"Closed frames: {closed_frames}", (10, 60), (0, 255, 255))

            if closed_frames >= args.min_frames:
                draw_label(frame, "DROWSINESS DETECTED!", (10, 90), (0, 0, 255))
                if PYGAME_OK and args.alert_wav:
                    # rate-limit sound to avoid overlaps
                    now = time.time()
                    if now - last_alert_time > 1.5:
                        pygame.mixer.music.play()
                        last_alert_time = now
                alert_on = True

        fps = fps_counter.tick()
        draw_label(frame, f"FPS: {fps:.1f}", (10, 120), (255, 255, 255))

        cv2.imshow("Driver Drowsiness (EAR)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=0, help="Try to set camera FPS (0=ignore)")
    parser.add_argument("--v4l2", action="store_true", help="Force V4L2 backend (Linux/Pi)")
    parser.add_argument("--ear_thresh", type=float, default=0.24, help="EAR threshold for eye closed")
    parser.add_argument("--min_frames", type=int, default=9, help="Consecutive frames below threshold to trigger alert")
    parser.add_argument("--smooth", type=int, default=10, help="Moving-average window length for EAR")
    parser.add_argument("--alert_wav", type=str, default=r"C:\Users\Philmon\Documents\Projects\driver-drowsiness-detection\assets\alert.wav", help="Path to alert sound (optional)")
    args = parser.parse_args()
    main(args)
