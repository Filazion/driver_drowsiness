import platform
import subprocess
import sys

def run_desktop():
    cmd = [
        sys.executable, "src/detect_drowsiness.py",
        "--width", "960", "--height", "540",
        "--ear_thresh", "0.24",
        "--min_frames", "9",
        "--smooth", "10",
        "--alert_wav", "assets/alert.wav"
    ]
    subprocess.run(cmd, check=False)

def run_pi():
    # Pi-friendly defaults: lower res to improve FPS and CPU load
    cmd = [
        sys.executable, "src/detect_drowsiness.py",
        "--width", "640", "--height", "480",
        "--v4l2",
        "--ear_thresh", "0.26",      # slightly higher; tune on device
        "--min_frames", "7",         # fewer frames for more sensitivity
        "--smooth", "8",
        "--alert_wav", "assets/alert.wav"
    ]
    subprocess.run(cmd, check=False)

if __name__ == "__main__":
    if "arm" in platform.machine() or "aarch64" in platform.machine():
        run_pi()
    else:
        run_desktop()
