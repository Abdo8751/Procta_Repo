"""Open a webcam with sensible quality settings.

Plain cv2.VideoCapture(0) usually gives 640x480, and on Windows its default
backend (Media Foundation) often delivers a soft, dark or laggy picture. Asking
for MJPG at 1280x720 through DirectShow normally fixes that on laptop webcams.
"""

import sys

import cv2


def open_camera(index=0, width=1280, height=720, fps=30):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
    for be in backends:
        cap = cv2.VideoCapture(index, be)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            print(f"Camera {index}: {w}x{h} via {cap.getBackendName()}"
                  + ("  (camera refused the requested size - try --width 1920 --height 1080 or 640 480)"
                     if (w, h) != (width, height) else ""))
            return cap
        cap.release()
    sys.exit(f"Could not open camera {index}. Close other apps using it (Teams, Zoom, browser) or try --camera 1.")
