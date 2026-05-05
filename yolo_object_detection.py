"""
ELEC 424 / COMP 553 - Object Recognition
YOLOv8 nano running on Raspberry Pi 5 CPU.

What this does:
    Reads frames from the USB camera, runs YOLOv8n inference every 3rd frame
    to keep CPU headroom for the stream server, annotates the frame with
    bounding boxes and class labels, and streams the result as MJPEG on port 8080.

Performance on RPi5 (no GPU):
    ~6-7 FPS at 320x320 input, ~130-200ms per inference.
    Inference every 3rd frame keeps the stream smooth without dropping to a crawl.

View the stream:
    Open http://<pi-ip>:8080 in a browser while this script is running.
    Only one process can use /dev/video0 at a time — make sure monster.py is not running.
"""

import cv2
import time
import threading
import socketserver
import http.server
import numpy as np
from ultralytics import YOLO

# ── Load YOLOv8 nano ──────────────────────────────────────────────────────────
# YOLOv8n is the smallest/fastest variant — ~6MB weights, trades some accuracy for speed.
# On first run it downloads the weights file automatically. Subsequent runs use the cache.
# We use ultralytics (not yolov5) because it works cleanly with modern PyTorch on the Pi.
print("Loading YOLOv8n model...")
model = YOLO('yolov8n.pt')
print("Model loaded.")

# ── Camera ────────────────────────────────────────────────────────────────────
# 320x320 matches YOLOv8's expected input size, so no extra resize step is needed.
CAMERA_DEVICE = "/dev/video0"
cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit(1)

print("Camera opened.")

# ── Shared frame for the MJPEG stream ─────────────────────────────────────────
# The main loop writes to latest_frame; the stream thread reads from it.
# frame_lock ensures we never read a half-written frame.
latest_frame = None
frame_lock   = threading.Lock()


# ── MJPEG stream server ────────────────────────────────────────────────────────
# Serves annotated frames as a multipart JPEG stream — any browser can view it.
# IMPORTANT: grab the lock only long enough to copy the frame reference, then
# release it before encoding or sleeping. Holding the lock during slow operations
# blocks the main inference loop and kills performance.
class StreamHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        while True:
            with frame_lock:
                f = latest_frame        # grab reference only, release lock immediately
            if f is None:
                time.sleep(0.05)        # no frame ready yet, wait briefly
                continue
            _, jpg = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 75])  # encode outside the lock
            try:
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                                 + jpg.tobytes() + b'\r\n')
            except Exception:
                break                   # client disconnected
            time.sleep(0.1)            # ~10 FPS to the browser — enough to see what's happening

    def log_message(self, *args):
        pass   # suppress per-request log noise in the terminal

socketserver.TCPServer.allow_reuse_address = True
server = socketserver.TCPServer(('', 8080), StreamHandler)
threading.Thread(target=server.serve_forever, daemon=True).start()
print("Stream: http://10.42.0.1:8080")

# ── Inference loop ─────────────────────────────────────────────────────────────
# We run YOLO inference on every 3rd frame and reuse the last annotated result
# for the frames in between. This keeps the stream fluid at ~10 FPS while
# inference itself only runs at ~3 FPS — a good trade-off on a CPU-only Pi.
print("Running — Ctrl+C to stop\n")

frame_count    = 0
fps_timer      = time.time()
current_fps    = 0.0
last_annotated = None   # holds the most recent YOLO-annotated frame

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            break

        frame_count += 1

        if frame_count % 3 == 0:
            t0      = time.time()
            # imgsz=320 matches our capture resolution — no internal resize needed.
            # verbose=False suppresses the per-inference console output from ultralytics.
            results = model(frame, imgsz=320, verbose=False)
            inf_ms  = (time.time() - t0) * 1000
            # results[0].plot() returns a BGR numpy array with bounding boxes,
            # class names, and confidence scores drawn on the frame.
            last_annotated = results[0].plot()

            # Print a stats line every 30 inference frames so we can monitor FPS and detection count
            if frame_count % 30 == 0:
                elapsed     = time.time() - fps_timer
                current_fps = 30.0 / elapsed
                fps_timer   = time.time()
                n_det = len(results[0].boxes)
                print(f"frame={frame_count}  FPS={current_fps:.2f}  inf={inf_ms:.0f}ms"
                      f"  detections={n_det}")

        # Don't push anything to the stream until we have at least one annotated frame
        if last_annotated is None:
            continue

        # Stamp the current FPS onto the frame so it's visible in the browser stream
        display = last_annotated.copy()
        cv2.putText(display, f"FPS: {current_fps:.1f}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        with frame_lock:
            latest_frame = display

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    cap.release()
    print("Done.")
