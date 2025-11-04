import time
import numpy as np
import cv2

# --- FaceMesh eye landmark indices (left/right) ---
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def euclidean_dist(a, b):
    return np.linalg.norm(a - b)

def eye_aspect_ratio(eye_pts):
    """
    eye_pts is a (6, 2) array in pixel coords ordered as:
    [p1, p2, p3, p4, p5, p6] = [outer, top-inner, top-outer, inner, bottom-inner, bottom-outer]
    """
    p1, p2, p3, p4, p5, p6 = eye_pts
    A = euclidean_dist(p2, p6)
    B = euclidean_dist(p3, p5)
    C = euclidean_dist(p1, p4)
    ear = (A + B) / (2.0 * C + 1e-6)
    return ear

class FpsCounter:
    def __init__(self):
        self.last = time.time()
        self.fps = 0.0
        self.smooth = 0.9  # EWMA

    def tick(self):
        now = time.time()
        dt = now - self.last
        self.last = now
        instant = 1.0 / dt if dt > 0 else 0.0
        self.fps = self.smooth * self.fps + (1 - self.smooth) * instant
        return self.fps

def draw_label(img, text, org=(10, 30), color=(0, 255, 0)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

def moving_average(val, buf, maxlen=10):
    buf.append(val)
    if len(buf) > maxlen:
        buf.pop(0)
    return sum(buf) / len(buf)
