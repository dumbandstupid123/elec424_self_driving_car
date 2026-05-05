# COMP 424 / ELEC 424 - Final Project
# Inspired by: User raja_961, "Autonomous Lane-Keeping Car Using Raspberry Pi and OpenCV"
# Instructables. URL: https://www.instructables.com/Autonomous-Lane-Keeping-Car-Using-Raspberry-Pi-and/
# Built for Raspberry Pi 5 using lgpio for hardware PWM.
#
# Overview:
#   The car follows a dark-blue tape lane on the floor using an HSV color filter lols.
#   A PD controller converts the lateral error (pixels off-center) into a servo
#   steering command. The ESC drives the motor at a fixed throttle. When the camera
#   sees a red stop box, the car pauses, then resumes. After two red boxes it stops
#   permanently. A live MJPEG stream is served on port 8080 for monitoring.

import cv2
import numpy as np
import time
import os
import csv
import threading
import socketserver
import http.server
import lgpio

# GPIO pin assignments (BCM numbering)
# ESC  → GPIO18 (PWM0), Servo → GPIO19 (PWM1)
ESC_PIN   = 18
SERVO_PIN = 19

SHOW_DISPLAY  = False                  # no monitor attached to the Pi
CAMERA_DEVICE = "/dev/video0"          # USB camera
ENCODER_PATH  = "/sys/module/encoder_driver/parameters/speed_rpm"  # kernel module sysfs

# ── Blue tape HSV range ───────────────────────────────────────────────────────
# These values were tuned in the actual lab environment under its lighting.
# If detection breaks, scp /tmp/frame_raw.jpg from the Pi and re-tune here.
# H: 95-135 covers blue; S: 60-255 rejects washed-out whites; V: 20-160 rejects bright reflections.
BLUE_LOWER = np.array([95,  60,  20], dtype="uint8")
BLUE_UPPER = np.array([135, 255, 160], dtype="uint8")

# ── Red stop-box HSV range ────────────────────────────────────────────────────
# Red wraps around 0° in HSV, so we need two separate ranges to catch it.
# Lower range: 0-10 (red on the low end), Upper range: 160-180 (red on the high end).
RED_LOWER1 = np.array([0,   100, 100], dtype="uint8")
RED_UPPER1 = np.array([10,  255, 255], dtype="uint8")
RED_LOWER2 = np.array([160, 100, 100], dtype="uint8")
RED_UPPER2 = np.array([180, 255, 255], dtype="uint8")
RED_AREA_THRESHOLD = 1000   # minimum contour area (px²) to count as a real red box, not a red pixel speck

# ── GPIO / PWM setup ─────────────────────────────────────────────────────────
# RPi5 uses gpiochip4 for the GPIO header; older Pi OS images use gpiochip0.
# We try chip 4 first, fall back to chip 0 if it fails.
print("Initializing ESC on GPIO18 and Servo on GPIO19...")

_PWM_FREQ = 50   # standard RC PWM frequency: 50 Hz = 20 ms period
try:
    _h = lgpio.gpiochip_open(4)
    lgpio.gpio_claim_output(_h, ESC_PIN, 0)
    lgpio.gpio_free(_h, ESC_PIN)
except lgpio.error:
    lgpio.gpiochip_close(_h)
    _h = lgpio.gpiochip_open(0)

lgpio.gpio_claim_output(_h, ESC_PIN,   0)
lgpio.gpio_claim_output(_h, SERVO_PIN, 0)

# Start both at neutral (7.5% duty = 1.5 ms pulse) so the ESC doesn't freak out on startup
lgpio.tx_pwm(_h, ESC_PIN,   _PWM_FREQ, 7.5)
lgpio.tx_pwm(_h, SERVO_PIN, _PWM_FREQ, 7.5)


def _val_to_duty(value):
    # Converts a normalized value [-1.0, 1.0] to a PWM duty cycle percentage.
    # -1.0 → 1.0 ms pulse → 5.0% duty (full reverse / full left)
    #  0.0 → 1.5 ms pulse → 7.5% duty (neutral / center)
    # +1.0 → 2.0 ms pulse → 10.0% duty (full forward / full right)
    pulse_ms = 1.5 + value * 0.5
    return (pulse_ms / 20.0) * 100.0

def set_esc(value):
    # Clamp to [-1, 1] before sending — the ESC will ignore out-of-range pulses anyway,
    # but clamping keeps the math clean.
    lgpio.tx_pwm(_h, ESC_PIN, _PWM_FREQ, _val_to_duty(max(-1.0, min(1.0, value))))

def set_servo(value):
    lgpio.tx_pwm(_h, SERVO_PIN, _PWM_FREQ, _val_to_duty(max(-1.0, min(1.0, value))))

def neutral():
    # Stop the motor and center the steering — call this on any stop or shutdown.
    set_esc(0)
    set_servo(0)


# ── PD controller gains ───────────────────────────────────────────────────────
# error = (tape_x - frame_center_x), in pixels. Frame is 160px wide, so max error ≈ ±80.
# steering output = Kp*error + Kd*d(error)/dt, clamped to [-1, 1].
#
# Kp = 0.020 → full servo deflection at ~50px error (good for tight turns).
# Kd = 0.0   → derivative term disabled. When enabled (e.g. 0.004), it amplifies
#              frame-to-frame noise in the centroid position into rapid servo twitching.
#              Leave at 0 unless you have a very stable, low-noise detection signal.
Kp_steer       = 0.020
Kd_steer       = 0.004
prev_steer_error = 0
STEER_TRIM     = 0.0   # add a small offset (e.g. 0.05) if the car drifts left or right on a straight

# ── Throttle ─────────────────────────────────────────────────────────────────
# 0.20 ≈ slow but controllable. Increase toward 0.25-0.30 for more speed.
# Don't go above 0.35 unless you enjoy watching the car demolish the course.
current_esc_throttle = 0.20

# ── State machine ─────────────────────────────────────────────────────────────
# DRIVING  → normal lane-following
# STOP_CD  → red box just detected, counting down 10 frames before stopping
#            (avoids acting on a single noisy frame)
# STOP1    → first stop: hold neutral for 3 seconds, then resume
# STOP2    → second stop: hold neutral forever (end of course)
state            = "DRIVING"
stop_count       = 0
cooldown_end_time = 0
stop_cd          = 0

# ── Data logging ──────────────────────────────────────────────────────────────
# Every 3rd frame is logged to run_data.csv for post-run analysis and rubric plots.
csv_file   = open("run_data.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Error", "P_Response", "D_Response", "Steer_Val", "Throttle", "Blue_px"])


def read_encoder_speed():
    # Reads wheel RPM from the optical encoder kernel module.
    # Returns 0.0 if the module isn't loaded or the car is stopped.
    # Values above 5000 RPM are physically impossible for this car — treat as stale reads.
    try:
        with open(ENCODER_PATH, "r") as f:
            val = float(f.read().strip())
            return 0.0 if val > 5000 else val
    except Exception:
        return 0.0


# ── Camera setup ──────────────────────────────────────────────────────────────
# The USB camera sometimes takes a moment to appear after boot, so we retry up to 10 times.
print("Opening camera...")
cap = None
for _attempt in range(10):
    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    if cap.isOpened():
        break
    cap.release()
    print(f"  Camera not ready, retry {_attempt+1}/10...")
    time.sleep(1)

if not cap.isOpened():
    print("ERROR: Could not open", CAMERA_DEVICE, "after 10 attempts")
    neutral()
    csv_file.close()
    exit(1)

print("Camera opened:", CAMERA_DEVICE)
# 160x120 is intentionally small — fast to process, and the tape is visible at this resolution.
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

# Discard the first 20 frames — cameras often output black frames during warm-up.
for _ in range(20):
    cap.read()

# Shared frame for the MJPEG stream — written by the main loop, read by the stream thread.
latest_frame = None
frame_lock   = threading.Lock()


# ── MJPEG stream server ───────────────────────────────────────────────────────
# Serves a live debug view at http://<pi-ip>:8080 so you can watch what the car sees.
# IMPORTANT: grab the lock only long enough to copy the frame pointer, then release it
# before doing any slow work (encoding, sleeping). Holding the lock while encoding
# would block the main control loop and make the car sluggish.
class StreamHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        while True:
            with frame_lock:
                f = latest_frame        # just grab the reference, don't hold the lock longer
            if f is None:
                time.sleep(0.05)        # no frame yet, wait and try again
                continue
            _, jpg = cv2.imencode('.jpg', f)   # encode outside the lock
            try:
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
            except Exception:
                break                   # client disconnected, exit the loop cleanly
            time.sleep(0.05)            # ~20 FPS stream, keeps the Pi from melting

    def log_message(self, format, *args):
        pass   # suppress per-request HTTP log spam in the terminal

socketserver.TCPServer.allow_reuse_address = True
server = socketserver.TCPServer(('', 8080), StreamHandler)
threading.Thread(target=server.serve_forever, daemon=True).start()
print("Video stream: http://10.42.0.1:8080")


def get_road_center(frame):
    """
    Finds the lateral position of the blue tape lane marker in the frame.

    Returns:
        road_center_x  – x-pixel of tape centroid (or None if not found)
        blue_px        – number of blue pixels detected (useful for debugging)
        debug          – annotated copy of the frame for the live stream

    Detection strategy:
        1. Convert to HSV and mask for the blue tape color.
        2. Only look at the bottom 40% of the frame — the tape on the floor
           appears here, and ignoring the top half avoids picking up blue objects
           in the background (like jeans, walls, etc.).
        3. Use the image moment centroid as the tape position (fast and accurate
           when the tape is clearly visible).
        4. If too few pixels are found for a reliable centroid, fall back to
           Canny + Hough lines on the blue mask edges.
        5. If way too many pixels are blue (>1200), it's a false positive from
           the environment — ignore and return None.
    """
    h, w = frame.shape[:2]
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Bottom 40% of the frame — this is where the floor tape lives relative to the camera
    roi_y = int(h * 0.60)
    roi   = hsv[roi_y:, :]

    blue_mask = cv2.inRange(roi, BLUE_LOWER, BLUE_UPPER)
    blue_px   = cv2.countNonZero(blue_mask)

    debug       = frame.copy()
    road_center = None

    # If more than 1200 pixels match the blue range, it's almost certainly
    # a false positive (e.g. a blue wall, someone wearing a blue shirt).
    BLUE_MAX = 1200

    if 30 <= blue_px <= BLUE_MAX:
        # Primary method: centroid of the blue pixel distribution.
        # m["m10"]/m["m00"] gives the x-coordinate of the center of mass.
        m = cv2.moments(blue_mask)
        if m["m00"] > 0:
            cx = int(m["m10"] / m["m00"])
            road_center = cx
            cv2.circle(debug, (cx, roi_y + (h - roi_y)//2), 5, (0, 255, 255), -1)

    elif blue_px > BLUE_MAX:
        pass   # too many blue pixels — likely a false positive, skip to Hough fallback

    else:
        # Fallback: when the centroid has too few pixels to trust, try to find
        # the tape edges using Canny + Hough and average their x-positions.
        edges = cv2.Canny(blue_mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, minLineLength=8, maxLineGap=5)
        if lines is not None:
            xs = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                xs.extend([x1, x2])
            if xs:
                road_center = int(sum(xs) / len(xs))

    # Draw a vertical centerline (blue) and the detected tape position (red) on the debug frame.
    cv2.line(debug, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
    if road_center is not None:
        cv2.line(debug, (road_center, roi_y), (road_center, h), (0, 0, 255), 2)

    return road_center, blue_px, debug


def process_frame(frame, frame_num):
    global prev_steer_error, state, stop_count, cooldown_end_time
    global current_esc_throttle, latest_frame, stop_cd

    h, w         = frame.shape[:2]
    current_time = time.time()

    # ── Red stop-box detection ────────────────────────────────────────────────
    # Only run every 3rd frame to save CPU — red box detection doesn't need to be
    # frame-perfect since we have a 10-frame countdown before acting on it anyway.
    if frame_num % 3 == 0:
        hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = (cv2.inRange(hsv_full, RED_LOWER1, RED_UPPER1) |
                    cv2.inRange(hsv_full, RED_LOWER2, RED_UPPER2))
        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        red_detected = any(cv2.contourArea(c) > RED_AREA_THRESHOLD for c in contours)

        if state == "DRIVING" and red_detected and current_time > cooldown_end_time:
            stop_count += 1
            stop_cd     = 10     # count down 10 frames before actually stopping
            state       = "STOP_CD"
            print("Red Box", stop_count, "Detected!")

        # Decrement the countdown; once it hits zero, commit to the stop
        if state == "STOP_CD":
            stop_cd -= 1
            if stop_cd <= 0:
                neutral()
                if stop_count == 1:
                    state             = "STOP1"
                    cooldown_end_time = current_time + 3.0   # stop for 3 seconds
                else:
                    state = "STOP2"   # second red box: stop for good

    # ── STOP1: pause 3 seconds then resume driving ────────────────────────────
    if state == "STOP1":
        neutral()
        if current_time > cooldown_end_time:
            print("Resuming driving...")
            state             = "DRIVING"
            cooldown_end_time = current_time + 4.0   # 4-second grace period before red can trigger again
        with frame_lock:
            latest_frame = frame.copy()
        return frame

    # ── STOP2: course complete, sit still ─────────────────────────────────────
    if state == "STOP2":
        neutral()   # we're done here. go home. get some sleep.
        with frame_lock:
            latest_frame = frame.copy()
        return frame

    # ── Lane detection + PD steering ─────────────────────────────────────────
    road_center, blue_px, debug = get_road_center(frame)

    # Save raw and debug snapshots at frame 5 — useful for tuning HSV values offline.
    # scp pi@<ip>:/tmp/frame_raw.jpg . and open in any image viewer.
    if frame_num == 5:
        cv2.imwrite("/tmp/frame_raw.jpg",   frame)
        cv2.imwrite("/tmp/frame_debug.jpg", debug)

    if road_center is None:
        # Tape lost — slow way down and hold the last known steering angle.
        # Don't keep full throttle when flying blind; that's how you end up in the wall.
        if frame_num % 10 == 0:
            print(f"frame={frame_num} TAPE LOST  blue_px={blue_px}")
        set_esc(0.10)
        with frame_lock:
            latest_frame = debug
        return frame

    # error > 0 → tape is to the right of center → need to steer right
    # error < 0 → tape is to the left of center  → need to steer left
    error      = road_center - (w // 2)
    derivative = error - prev_steer_error   # change in error since last frame
    p_response = Kp_steer * error
    d_response = Kd_steer * derivative      # zero when Kd=0.0 (currently disabled)

    steering_val     = p_response + d_response
    prev_steer_error = error

    # Negate steering because positive error (tape right) means we need to turn right,
    # and the servo convention on this car is: positive value = turn right.
    # STEER_TRIM corrects any mechanical bias if the car doesn't drive straight at steer=0.
    set_servo(-steering_val + STEER_TRIM)
    set_esc(current_esc_throttle)

    # Print a status line every 30 frames (~1 second at ~30 FPS) to avoid terminal spam
    if frame_num % 30 == 0:
        print(f"frame={frame_num}  blue_px={blue_px}  err={error:+.0f}  steer_val={steering_val:+.3f}")

    # Log to CSV every 3rd frame for post-run plots
    if frame_num % 3 == 0:
        csv_writer.writerow([frame_num, error, p_response, d_response,
                             steering_val, current_esc_throttle, blue_px])

    with frame_lock:
        latest_frame = debug

    return frame


def main():
    # Arm the ESC: hold neutral for 3 seconds so it initializes properly.
    # Skipping this step will cause the ESC to ignore throttle commands.
    print("\nArming ESC (3 s)...")
    set_esc(0)
    time.sleep(3)
    print("ESC armed — starting navigation")
    print("Stream: http://10.42.0.1:8080")
    print("--- AUTONOMOUS NAVIGATION STARTED ---\n")

    try:
        MAX_FRAMES  = 5000   # safety cap — at ~30 FPS this is about 2.5 minutes of driving
        frame_count = 0

        while frame_count < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Camera read failed at frame", frame_count)
                break
            process_frame(frame, frame_count)
            frame_count += 1

        print("\nMax frames reached.")

    except KeyboardInterrupt:
        print("\nStopping.")

    finally:
        # Always clean up GPIO properly — if you Ctrl+Z instead of Ctrl+C the GPIO stays
        # claimed and the next run will crash. If that happens, just reboot the Pi.
        print("Shutting down...")
        neutral()
        lgpio.tx_pwm(_h, ESC_PIN,   0, 0)
        lgpio.tx_pwm(_h, SERVO_PIN, 0, 0)
        lgpio.gpiochip_close(_h)
        cap.release()
        csv_file.close()
        print("Data saved to run_data.csv")


if __name__ == "__main__":
    main()
