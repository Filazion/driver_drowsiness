# 😴 Driver Drowsiness Detection (Streamlit + MediaPipe)

A **real-time web-based driver drowsiness detection system** built with **Streamlit**, **MediaPipe FaceMesh**, and **OpenCV**.  
This project analyzes **eye aspect ratio (EAR)** to detect signs of fatigue and visually alerts users when prolonged eye closure is detected.  
The app runs **entirely in the browser** (via WebRTC), making it accessible from **desktop, mobile, and Raspberry Pi** devices.

---

## 🚀 Features

- 🧠 **Real-time eye and face landmark detection** using MediaPipe  
- 👁️ **EAR-based drowsiness detection** with adjustable sensitivity  
- 📊 **Live metrics dashboard** (EAR, FPS, closed-frame counter, alert status)  
- 📈 **Real-time EAR graph** and detection logs  
- 🎥 **Web-based interface** — runs directly in the browser using Streamlit + WebRTC  
- 💻 **Cross-platform support** — works on Windows, macOS, Linux, and Raspberry Pi  
- 🧪 **Demo mode** — upload a short video if webcam is unavailable  

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| Frontend | Streamlit + streamlit-webrtc |
| Computer Vision | OpenCV, MediaPipe FaceMesh |
| Programming Language | Python 3.8+ |
| Visualization | Streamlit Charts, Metrics |
| Optional Deployments | Streamlit Cloud / Hugging Face Spaces |

---

## 📂 Project Structure

```text
driver-drowsiness/
├─ app_streamlit.py          # Main Streamlit application
├─ src/                      # Core logic (for standalone scripts)
│  ├─ detect_drowsiness.py
│  ├─ main.py
│  └─ utils.py
├─ assets/                   # Optional assets (alert sound, screenshots)
│  └─ demo_placeholder.png
├─ requirements.txt
├─ setup.sh                  # (optional, for cloud environments)
├─ runtime.txt               # (optional, for Hugging Face)
├─ Procfile                  # (optional, for Render.com)
└─ README.md
