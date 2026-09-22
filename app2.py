#!/usr/bin/env python3
import os
import csv
import time
import atexit
import threading
import zipfile
import shutil
import uuid
from datetime import datetime

import cv2
import numpy as np
import serial
from flask import Flask, Response, jsonify, render_template_string, request, send_file, session
from picamera2 import Picamera2

# ============================================================
# CAR4 CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SERIAL_PORT = "/dev/ttyUSB0"          # Change to /dev/ttyUSB1 if required
BAUD_RATE = 115200

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAPTURE_INTERVAL = 0.5

MIN_SPEED = 0
MAX_SPEED = 255
DEFAULT_SPEED = 100

ROI_START_RATIO = 0.50
BLACK_V_MIN = 0
BLACK_V_MAX = 95
BLACK_S_MIN = 0
BLACK_S_MAX = 255
MIN_CONTOUR_AREA = 100

CAR_CENTER_X = CAMERA_WIDTH // 2

# ============================================================
# DATA FOLDERS
# ============================================================
IMAGE_ROOT = os.path.join(BASE_DIR, "captured_images")
SESSION_ROOT = os.path.join(BASE_DIR, "sessions")
LOG_ROOT = os.path.join(BASE_DIR, "logs")

os.makedirs(IMAGE_ROOT, exist_ok=True)
os.makedirs(SESSION_ROOT, exist_ok=True)
os.makedirs(LOG_ROOT, exist_ok=True)

# These are kept as compatibility/summary downloads for the active session.
DATASET_CSV = os.path.join(BASE_DIR, "labels.csv")
MOTOR_LOG_CSV = os.path.join(LOG_ROOT, "motor_log.csv")

DATASET_ZIP = os.path.join(BASE_DIR, "car4_cnn_dataset.zip")
IMAGES_ZIP = os.path.join(BASE_DIR, "car4_images.zip")

# One ZIP operation at a time so two browser clicks cannot corrupt the file.
zip_lock = threading.Lock()
data_lock = threading.Lock()

# ============================================================
# FLASK
# ============================================================
app = Flask(__name__)
app.secret_key = os.environ.get("CAR4_SECRET_KEY", "car4-local-secret-change-me")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False
app.config["SESSION_PERMANENT"] = False

# ============================================================
# GLOBAL STATE
# ============================================================
camera = None
camera_lock = threading.Lock()
arduino = None
serial_lock = threading.Lock()
running = True

current_command = "STOP"
current_speed = 0
last_capture_time = 0.0
image_counter = 0
active_session_id = None

latest_lane_data = {
    "lane_detected": False,
    "car_center_x": CAR_CENTER_X,
    "lane_center_x": 0,
    "left_distance": 0,
    "right_distance": 0,
    "position_error": 0,
    "lane_angle": 0.0,
    "lane_left_x": 0,
    "lane_right_x": 0,
    "decision": "STOP",
}

CSV_HEADER = [
    "image_filename", "direction", "command", "speed", "lane_detected",
    "car_center_x", "lane_center_x", "left_distance", "right_distance",
    "position_error", "lane_angle", "lane_left_x", "lane_right_x",
    "left_motor_pwm", "right_motor_pwm", "timestamp"
]

MOTOR_HEADER = [
    "command", "speed", "left_motor_pwm", "right_motor_pwm", "timestamp"
]

# ============================================================
# SESSION MANAGEMENT
# ============================================================
def new_session_id():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"session_{stamp}_{uuid.uuid4().hex[:6]}"


def session_dir(session_id):
    return os.path.join(SESSION_ROOT, session_id)


def session_images_dir(session_id):
    return os.path.join(session_dir(session_id), "images")


def session_labels_path(session_id):
    return os.path.join(session_dir(session_id), "labels.csv")


def session_motor_path(session_id):
    return os.path.join(session_dir(session_id), "motor_log.csv")


def ensure_session_files(session_id):
    os.makedirs(session_images_dir(session_id), exist_ok=True)
    labels = session_labels_path(session_id)
    motor = session_motor_path(session_id)
    if not os.path.exists(labels):
        with open(labels, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)
    if not os.path.exists(motor):
        with open(motor, "w", newline="") as f:
            csv.writer(f).writerow(MOTOR_HEADER)


def start_new_session(force=False):
    global active_session_id, image_counter, last_capture_time
    sid = new_session_id()
    with data_lock:
        ensure_session_files(sid)
        active_session_id = sid
        image_counter = 0
        last_capture_time = 0.0
    return sid


def get_or_create_browser_session():
    global active_session_id
    sid = session.get("car4_session_id")
    if not sid:
        sid = new_session_id()
        session["car4_session_id"] = sid
        with data_lock:
            ensure_session_files(sid)
        active_session_id = sid
        return sid, True
    with data_lock:
        ensure_session_files(sid)
    active_session_id = sid
    return sid, False


def clear_all_logs_and_sessions():
    """Delete all captured images, session CSVs, compatibility CSVs and ZIPs."""
    global active_session_id, image_counter, last_capture_time
    send_motor_command("STOP", 0)
    with data_lock:
        for path in (IMAGE_ROOT, SESSION_ROOT):
            if os.path.isdir(path):
                for name in os.listdir(path):
                    full = os.path.join(path, name)
                    try:
                        if os.path.isdir(full):
                            shutil.rmtree(full)
                        else:
                            os.remove(full)
                    except Exception as exc:
                        print(f"[CLEAR WARNING] {full}: {exc}")
        os.makedirs(IMAGE_ROOT, exist_ok=True)
        os.makedirs(SESSION_ROOT, exist_ok=True)

        for path in (DATASET_CSV, MOTOR_LOG_CSV, DATASET_ZIP, IMAGES_ZIP):
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except Exception as exc:
                print(f"[CLEAR WARNING] {path}: {exc}")

        active_session_id = new_session_id()
        image_counter = 0
        last_capture_time = 0.0
        initialize_compatibility_csvs()
        ensure_session_files(active_session_id)

        # New browser session cookie is assigned by route /new_session as well.
        print(f"[CLEAR] All old session data removed. New session: {active_session_id}")
        return active_session_id


def initialize_compatibility_csvs():
    if not os.path.exists(DATASET_CSV):
        with open(DATASET_CSV, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)
    if not os.path.exists(MOTOR_LOG_CSV):
        with open(MOTOR_LOG_CSV, "w", newline="") as f:
            csv.writer(f).writerow(MOTOR_HEADER)


def list_sessions():
    result = []
    if not os.path.isdir(SESSION_ROOT):
        return result
    for name in sorted(os.listdir(SESSION_ROOT), reverse=True):
        path = session_dir(name)
        if os.path.isdir(path):
            result.append(name)
    return result

# Initialize compatibility CSVs.
initialize_compatibility_csvs()

# Initialize one clean session only when the app starts if none exist.
existing = list_sessions()
if existing:
    active_session_id = existing[0]
    ensure_session_files(active_session_id)
else:
    start_new_session()

# ============================================================
# ARDUINO
# ============================================================
def connect_arduino():
    global arduino
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"[INFO] Arduino connected: {SERIAL_PORT}")
    except Exception as error:
        arduino = None
        print(f"[WARNING] Arduino connection failed: {error}")
        print("[WARNING] Camera/dashboard can still run. Check SERIAL_PORT if motors do not move.")


def send_motor_command(command, speed=0):
    global current_command, current_speed
    command = str(command).upper()
    if command == "STOP":
        speed = 0
    speed = int(max(MIN_SPEED, min(MAX_SPEED, int(speed))))

    if command == "F":
        serial_command = f"F,{speed}\n"
    elif command == "B":
        serial_command = f"B,{speed}\n"
    elif command == "L":
        serial_command = f"L,{speed}\n"
    elif command == "R":
        serial_command = f"R,{speed}\n"
    else:
        command = "STOP"
        speed = 0
        serial_command = "STOP\n"

    with serial_lock:
        try:
            if arduino is not None and arduino.is_open:
                arduino.write(serial_command.encode())
                arduino.flush()
        except Exception as error:
            print(f"[WARNING] Serial error: {error}")

    current_command = command
    current_speed = speed
    return serial_command.strip()


def calculate_motor_values(command, speed):
    command = command.upper()
    speed = int(speed)
    if command in ("F", "B"):
        return speed, speed
    if command == "L":
        return 0, speed
    if command == "R":
        return speed, 0
    return 0, 0


def log_motor_command(command, speed, session_id=None):
    sid = session_id or active_session_id
    if not sid:
        return
    ensure_session_files(sid)
    left_pwm, right_pwm = calculate_motor_values(command, speed)
    timestamp = datetime.now().isoformat(timespec="milliseconds")
    row = [command, speed, left_pwm, right_pwm, timestamp]
    try:
        with data_lock:
            with open(session_motor_path(sid), "a", newline="") as f:
                csv.writer(f).writerow(row)
            # Maintain a compatibility copy for the active session.
            with open(MOTOR_LOG_CSV, "a", newline="") as f:
                csv.writer(f).writerow(row)
    except Exception as error:
        print(f"[WARNING] Motor log error: {error}")

# ============================================================
# CAMERA
# ============================================================
def initialize_camera():
    global camera
    try:
        camera = Picamera2()
        config = camera.create_video_configuration(
            main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}
        )
        camera.configure(config)
        camera.start()
        time.sleep(2)
        print("[INFO] OV5647 camera started")
    except Exception as error:
        camera = None
        print(f"[ERROR] Camera initialization failed: {error}")

# ============================================================
# BLACK TAPE DETECTION
# ============================================================
def detect_black_lane(frame):
    height, width = frame.shape[:2]
    output = frame.copy()
    roi_start_y = int(height * ROI_START_RATIO)
    roi = frame[roi_start_y:height, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    lower_black = np.array([0, BLACK_S_MIN, BLACK_V_MIN])
    upper_black = np.array([180, BLACK_S_MAX, BLACK_V_MAX])
    mask = cv2.inRange(hsv, lower_black, upper_black)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]

    lane_data = {
        "lane_detected": False,
        "car_center_x": CAR_CENTER_X,
        "lane_center_x": 0,
        "left_distance": 0,
        "right_distance": 0,
        "position_error": 0,
        "lane_angle": 0.0,
        "lane_left_x": 0,
        "lane_right_x": 0,
        "decision": current_command,
    }

    cv2.line(output, (0, roi_start_y), (width, roi_start_y), (255, 255, 0), 2)

    if not valid:
        cv2.rectangle(output, (10, 10), (350, 55), (0, 0, 0), -1)
        cv2.putText(output, "LANE NOT DETECTED", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        return output, lane_data

    largest = max(valid, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    lane_left_x = x
    lane_right_x = x + w
    lane_center_x = int((lane_left_x + lane_right_x) / 2)
    position_error = CAR_CENTER_X - lane_center_x

    left_distance = max(0, CAR_CENTER_X - lane_left_x)
    right_distance = max(0, lane_right_x - CAR_CENTER_X)

    points = largest.reshape(-1, 2)
    lane_angle = 0.0
    if len(points) >= 2:
        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        vx = float(vx)
        vy = float(vy)
        lane_angle = float(round(np.degrees(np.arctan2(vx, vy)) if abs(vy) > 1e-4 else 90.0, 2))

    shifted = largest + np.array([0, roi_start_y])
    cv2.drawContours(output, [shifted], -1, (0, 255, 0), 3)
    cv2.line(output, (lane_left_x, roi_start_y), (lane_left_x, height), (255, 0, 0), 2)
    cv2.line(output, (lane_right_x, roi_start_y), (lane_right_x, height), (255, 0, 0), 2)
    cv2.line(output, (lane_center_x, roi_start_y), (lane_center_x, height), (0, 255, 255), 3)
    cv2.line(output, (CAR_CENTER_X, roi_start_y), (CAR_CENTER_X, height), (255, 0, 255), 3)
    cv2.circle(output, (CAR_CENTER_X, height - 25), 9, (255, 0, 255), -1)
    cv2.circle(output, (lane_center_x, height - 25), 8, (0, 255, 255), -1)
    cv2.line(output, (CAR_CENTER_X, height - 55), (lane_center_x, height - 55), (255, 255, 255), 3)

    lane_data = {
        "lane_detected": True,
        "car_center_x": CAR_CENTER_X,
        "lane_center_x": lane_center_x,
        "left_distance": left_distance,
        "right_distance": right_distance,
        "position_error": position_error,
        "lane_angle": lane_angle,
        "lane_left_x": lane_left_x,
        "lane_right_x": lane_right_x,
        "decision": current_command,
    }

    cv2.rectangle(output, (10, 10), (355, 185), (0, 0, 0), -1)
    info = [
        "LANE: DETECTED",
        f"Car4 center: {CAR_CENTER_X}px",
        f"Lane center: {lane_center_x}px",
        f"Left distance: {left_distance}px",
        f"Right distance: {right_distance}px",
        f"Position error: {position_error}px",
        f"Lane angle: {lane_angle:.2f} deg",
        f"Command: {current_command}",
    ]
    y_text = 32
    for text in info:
        cv2.putText(output, text, (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        y_text += 21
    return output, lane_data


def get_direction_label(command):
    return {
        "F": "forward",
        "B": "backward",
        "L": "left",
        "R": "right",
        "STOP": "stop",
    }.get(command, "stop")

# ============================================================
# DATASET CAPTURE
# ============================================================
def capture_dataset_image(command, speed, frame, lane_data):
    global image_counter, active_session_id
    sid = active_session_id
    if not sid:
        sid = start_new_session()
    ensure_session_files(sid)

    direction = get_direction_label(command)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_counter += 1
    filename = f"{image_counter:06d}_{direction}_speed_{speed}_{timestamp}.jpg"
    image_path = os.path.join(session_images_dir(sid), filename)

    ok = cv2.imwrite(
        image_path,
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    if not ok:
        print(f"[WARNING] Failed to save {filename}")
        return

    # Also place a copy in the legacy captured_images folder for compatibility.
    try:
        shutil.copy2(image_path, os.path.join(IMAGE_ROOT, filename))
    except Exception as exc:
        print(f"[WARNING] Legacy image copy failed: {exc}")

    left_pwm, right_pwm = calculate_motor_values(command, speed)
    timestamp_csv = datetime.now().isoformat(timespec="milliseconds")
    row = [
        filename, direction, command, speed,
        int(lane_data["lane_detected"]),
        lane_data["car_center_x"], lane_data["lane_center_x"],
        lane_data["left_distance"], lane_data["right_distance"],
        lane_data["position_error"], lane_data["lane_angle"],
        lane_data["lane_left_x"], lane_data["lane_right_x"],
        left_pwm, right_pwm, timestamp_csv,
    ]
    try:
        with data_lock:
            with open(session_labels_path(sid), "a", newline="") as f:
                csv.writer(f).writerow(row)
            with open(DATASET_CSV, "a", newline="") as f:
                csv.writer(f).writerow(row)
    except Exception as error:
        print(f"[WARNING] CSV error: {error}")
        return

    print(f"[DATASET] {sid} | {filename} | {direction} | speed={speed} | lane={lane_data['lane_detected']}")

# ============================================================
# CAMERA STREAM
# ============================================================
def generate_frames():
    global latest_lane_data, last_capture_time
    while running:
        if camera is None:
            time.sleep(0.2)
            continue
        try:
            with camera_lock:
                frame = camera.capture_array()
            overlay, lane_data = detect_black_lane(frame)
            latest_lane_data = lane_data
            now = time.time()
            if now - last_capture_time >= CAPTURE_INTERVAL:
                capture_dataset_image(current_command, current_speed, frame, lane_data)
                last_capture_time = now

            bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            success, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
        except Exception as error:
            print(f"[WARNING] Camera stream error: {error}")
            time.sleep(0.2)

# ============================================================
# ZIP HELPERS
# ============================================================
def _zip_add_if_exists(zf, path, arcname):
    if os.path.isfile(path):
        zf.write(path, arcname)


def create_dataset_zip():
    """Fast ZIP: JPEGs/CSVs are already compressed, so store them without recompressing."""
    temp_zip = DATASET_ZIP + ".tmp"
    print("[ZIP] Creating complete dataset ZIP...")
    with zip_lock:
        try:
            if os.path.exists(temp_zip):
                os.remove(temp_zip)
            count = 0
            with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_STORED) as zf:
                # All sessions. Older CSVs remain available until Clear Logs is pressed.
                for sid in sorted(list_sessions()):
                    root = session_dir(sid)
                    labels = session_labels_path(sid)
                    motor = session_motor_path(sid)
                    _zip_add_if_exists(zf, labels, f"sessions/{sid}/labels.csv")
                    _zip_add_if_exists(zf, motor, f"sessions/{sid}/motor_log.csv")
                    img_dir = session_images_dir(sid)
                    if os.path.isdir(img_dir):
                        for filename in sorted(os.listdir(img_dir)):
                            p = os.path.join(img_dir, filename)
                            if os.path.isfile(p):
                                zf.write(p, f"sessions/{sid}/images/{filename}")
                                count += 1

                # Current compatibility copies, useful for simple CNN scripts.
                _zip_add_if_exists(zf, DATASET_CSV, "labels.csv")
                _zip_add_if_exists(zf, MOTOR_LOG_CSV, "motor_log.csv")

            os.replace(temp_zip, DATASET_ZIP)
            size_mb = os.path.getsize(DATASET_ZIP) / (1024 * 1024)
            print(f"[ZIP] Complete dataset ready: {count} images, {size_mb:.2f} MB")
            return DATASET_ZIP
        except Exception:
            try:
                if os.path.exists(temp_zip):
                    os.remove(temp_zip)
            except Exception:
                pass
            raise


def create_images_zip():
    temp_zip = IMAGES_ZIP + ".tmp"
    print("[ZIP] Creating images-only ZIP...")
    with zip_lock:
        try:
            if os.path.exists(temp_zip):
                os.remove(temp_zip)
            count = 0
            with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_STORED) as zf:
                for sid in sorted(list_sessions()):
                    img_dir = session_images_dir(sid)
                    if os.path.isdir(img_dir):
                        for filename in sorted(os.listdir(img_dir)):
                            p = os.path.join(img_dir, filename)
                            if os.path.isfile(p):
                                zf.write(p, f"sessions/{sid}/{filename}")
                                count += 1
            os.replace(temp_zip, IMAGES_ZIP)
            size_mb = os.path.getsize(IMAGES_ZIP) / (1024 * 1024)
            print(f"[ZIP] Images ZIP ready: {count} images, {size_mb:.2f} MB")
            return IMAGES_ZIP
        except Exception:
            try:
                if os.path.exists(temp_zip):
                    os.remove(temp_zip)
            except Exception:
                pass
            raise

# ============================================================
# HTML DASHBOARD
# ============================================================
HTML_PAGE = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Car4 Lane Control</title>
<style>
*{box-sizing:border-box} body{margin:0;padding:20px;background:#111;color:#fff;font-family:Arial,sans-serif;text-align:center}.container{max-width:1000px;margin:auto}h1{margin-top:0;font-size:30px}.camera-box,.status,.download-panel,.session-panel{background:#222;padding:15px;border-radius:12px;margin-bottom:15px}.camera-box img{width:100%;max-width:640px;border-radius:8px;display:block;margin:auto}.status-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.status-item{background:#333;padding:12px;border-radius:8px;font-size:16px}.status-item strong{display:block;margin-top:5px;font-size:19px}.control-panel{margin:20px auto;display:grid;grid-template-columns:repeat(3,1fr);gap:12px;max-width:500px}button{min-height:70px;border:0;border-radius:12px;font-size:19px;font-weight:bold;color:#fff;cursor:pointer;user-select:none;touch-action:none}button:active{transform:scale(.95)}.forward{background:#27ae60}.left,.right{background:#3498db}.backward{background:#e67e22}.stop{background:#e74c3c}.clear{background:#8e44ad;min-height:50px}.new-session{background:#16a085;min-height:50px}.speed-panel{margin:20px 0}.speed-panel input{width:80%;max-width:700px}.speed-number{font-size:28px;font-weight:bold}.download-panel a{display:inline-block;margin:6px;padding:13px 18px;background:#555;color:#fff;text-decoration:none;border-radius:8px;font-size:15px}.download-panel a:hover{background:#777}.warning{color:#f1c40f;margin:15px 0;font-size:17px}.session-id{word-break:break-all;color:#2ecc71}@media(max-width:600px){body{padding:10px}.status-grid{grid-template-columns:1fr}button{min-height:60px;font-size:17px}.download-panel a{width:90%}}
</style>
</head>
<body>
<div class="container">
<h1>Car4 Lane Control Dashboard</h1>
<div class="session-panel">
<h2>Current Browser Session</h2>
<div class="session-id" id="session_id">---</div>
<p>Opening the dashboard in a new browser session creates a new dataset CSV and motor CSV. Old sessions stay in the Complete Dataset ZIP until Clear Logs is pressed.</p>
<button class="new-session" id="new_session_button">START NEW SESSION</button>
<button class="clear" id="clear_button">CLEAR ALL LOGS + IMAGES</button>
</div>
<div class="camera-box"><img src="/video_feed" alt="Car4 Camera"></div>
<div class="status"><h2>Lane Information</h2><div class="status-grid">
<div class="status-item">Lane detected<strong id="lane_detected">---</strong></div>
<div class="status-item">Car4 center<strong id="car_center_x">---</strong></div>
<div class="status-item">Lane center<strong id="lane_center_x">---</strong></div>
<div class="status-item">Left distance<strong id="left_distance">---</strong></div>
<div class="status-item">Right distance<strong id="right_distance">---</strong></div>
<div class="status-item">Position error<strong id="position_error">---</strong></div>
<div class="status-item">Lane angle<strong id="lane_angle">---</strong></div>
<div class="status-item">Command<strong id="current_command">STOP</strong></div>
<div class="status-item">Speed<strong id="current_speed">0</strong></div>
</div></div>
<div class="speed-panel"><h2>Speed</h2><div class="speed-number"><span id="speed_value">100</span></div><input type="range" id="speed_slider" min="0" max="255" value="100"></div>
<div class="control-panel">
<div></div><button id="forward_button" class="forward">FORWARD</button><div></div>
<button id="left_button" class="left">LEFT</button><button id="stop_button" class="stop">STOP</button><button id="right_button" class="right">RIGHT</button>
<div></div><button id="backward_button" class="backward">BACKWARD</button><div></div>
</div>
<div class="warning">Press and hold a movement button. Release it to STOP Car4.</div>
<div class="download-panel"><h2>Dataset Downloads</h2>
<a href="/download_dataset">Download Complete Dataset ZIP</a>
<a href="/download_images">Download Images ZIP</a>
<a href="/download_labels">Download Current Session Labels CSV</a>
<a href="/download_motor_log">Download Current Session Motor CSV</a>
</div>
</div>
<script>
let selectedSpeed=100,movementActive=false,activeCommand=null;
const slider=document.getElementById('speed_slider'), speedValue=document.getElementById('speed_value');
slider.addEventListener('input',()=>{selectedSpeed=parseInt(slider.value);speedValue.innerText=selectedSpeed});
function sendCommand(command,speed){fetch('/motor',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command,speed})}).catch(e=>console.log(e));}
function startMovement(command){if(movementActive&&activeCommand===command)return;movementActive=true;activeCommand=command;sendCommand(command,selectedSpeed)}
function stopMovement(){if(!movementActive&&activeCommand===null){sendCommand('STOP',0);return}movementActive=false;activeCommand=null;sendCommand('STOP',0)}
function setupMovementButton(id,cmd){const b=document.getElementById(id);b.addEventListener('mousedown',e=>{e.preventDefault();startMovement(cmd)});b.addEventListener('mouseup',e=>{e.preventDefault();stopMovement()});b.addEventListener('mouseleave',()=>stopMovement());b.addEventListener('touchstart',e=>{e.preventDefault();startMovement(cmd)},{passive:false});b.addEventListener('touchend',e=>{e.preventDefault();stopMovement()},{passive:false});b.addEventListener('touchcancel',e=>{e.preventDefault();stopMovement()},{passive:false})}
setupMovementButton('forward_button','F');setupMovementButton('left_button','L');setupMovementButton('right_button','R');setupMovementButton('backward_button','B');
document.getElementById('stop_button').addEventListener('click',()=>{movementActive=false;activeCommand=null;sendCommand('STOP',0)});
document.addEventListener('mouseup',()=>{if(movementActive)stopMovement()});window.addEventListener('blur',()=>{movementActive=false;activeCommand=null;sendCommand('STOP',0)});
window.addEventListener('beforeunload',()=>{navigator.sendBeacon('/motor_stop',new Blob([JSON.stringify({command:'STOP',speed:0})],{type:'application/json'}))});
function updateStatus(){fetch('/status').then(r=>r.json()).then(d=>{document.getElementById('lane_detected').innerText=d.lane_detected?'YES':'NO';document.getElementById('car_center_x').innerText=d.car_center_x+' px';document.getElementById('lane_center_x').innerText=d.lane_center_x+' px';document.getElementById('left_distance').innerText=d.left_distance+' px';document.getElementById('right_distance').innerText=d.right_distance+' px';document.getElementById('position_error').innerText=d.position_error+' px';document.getElementById('lane_angle').innerText=d.lane_angle+' deg';document.getElementById('current_command').innerText=d.current_command;document.getElementById('current_speed').innerText=d.current_speed;document.getElementById('session_id').innerText=d.session_id||'---'}).catch(e=>console.log(e))}
setInterval(updateStatus,300);updateStatus();
document.getElementById('new_session_button').addEventListener('click',()=>{if(!confirm('Start a new browser session? The current session data will remain for download.'))return;fetch('/new_session',{method:'POST'}).then(r=>r.json()).then(d=>{document.getElementById('session_id').innerText=d.session_id;alert('New session started: '+d.session_id)}).catch(e=>alert(e))});
document.getElementById('clear_button').addEventListener('click',()=>{if(!confirm('Delete ALL old images, labels CSVs, motor CSVs and ZIP files? This cannot be undone.'))return;fetch('/clear_logs',{method:'POST'}).then(r=>r.json()).then(d=>{if(d.success){document.getElementById('session_id').innerText=d.session_id;alert('All old data removed. New clean session started.')}else{alert(d.error||'Clear failed')}}).catch(e=>alert(e))});
</script>
</body>
</html>
"""

# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def index():
    sid, created = get_or_create_browser_session()
    return render_template_string(HTML_PAGE)

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/motor", methods=["POST"])
def motor_control():
    sid = session.get("car4_session_id") or active_session_id
    data = request.get_json(silent=True) or {}
    command = str(data.get("command", "STOP")).upper()
    try:
        speed = int(data.get("speed", 0))
    except Exception:
        speed = 0
    if command not in ["F", "B", "L", "R", "STOP"]:
        command = "STOP"
    if command == "STOP":
        speed = 0
    serial_command = send_motor_command(command, speed)
    log_motor_command(command, speed, sid)
    return jsonify({"success": True, "command": command, "speed": speed, "serial_command": serial_command})

@app.route("/motor_stop", methods=["POST"])
def motor_stop():
    sid = session.get("car4_session_id") or active_session_id
    send_motor_command("STOP", 0)
    log_motor_command("STOP", 0, sid)
    return jsonify({"success": True, "command": "STOP"})

@app.route("/status")
def status():
    data = dict(latest_lane_data)
    data["current_command"] = current_command
    data["current_speed"] = current_speed
    data["session_id"] = session.get("car4_session_id") or active_session_id
    data["session_count"] = len(list_sessions())
    return jsonify(data)

@app.route("/new_session", methods=["POST"])
def new_session_route():
    global active_session_id
    sid = new_session_id()
    session.clear()
    session["car4_session_id"] = sid
    ensure_session_files(sid)
    active_session_id = sid
    return jsonify({"success": True, "session_id": sid})

@app.route("/clear_logs", methods=["POST"])
def clear_logs_route():
    try:
        sid = clear_all_logs_and_sessions()
        session.clear()
        session["car4_session_id"] = sid
        return jsonify({"success": True, "session_id": sid})
    except Exception as error:
        print(f"[CLEAR ERROR] {error}")
        return jsonify({"success": False, "error": str(error)}), 500

@app.route("/download_dataset")
def download_dataset():
    try:
        zip_path = create_dataset_zip()
        return send_file(zip_path, mimetype="application/zip", as_attachment=True,
                         download_name="car4_cnn_dataset.zip", max_age=0)
    except Exception as error:
        print(f"[DOWNLOAD ERROR] {error}")
        return f"<h2>Dataset download failed</h2><p>{error}</p>", 500

@app.route("/download_images")
def download_images():
    try:
        zip_path = create_images_zip()
        return send_file(zip_path, mimetype="application/zip", as_attachment=True,
                         download_name="car4_images.zip", max_age=0)
    except Exception as error:
        print(f"[IMAGE DOWNLOAD ERROR] {error}")
        return f"<h2>Image download failed</h2><p>{error}</p>", 500

@app.route("/download_labels")
def download_labels():
    sid = session.get("car4_session_id") or active_session_id
    ensure_session_files(sid)
    return send_file(session_labels_path(sid), mimetype="text/csv", as_attachment=True,
                     download_name=f"labels_{sid}.csv", max_age=0)

@app.route("/download_motor_log")
def download_motor_log():
    sid = session.get("car4_session_id") or active_session_id
    ensure_session_files(sid)
    return send_file(session_motor_path(sid), mimetype="text/csv", as_attachment=True,
                     download_name=f"motor_log_{sid}.csv", max_age=0)

# ============================================================
# SAFE SHUTDOWN
# ============================================================
def shutdown():
    global running
    running = False
    print("[INFO] Shutting down Car4...")
    try:
        send_motor_command("STOP", 0)
    except Exception:
        pass
    try:
        if camera is not None:
            camera.stop()
    except Exception:
        pass
    try:
        if arduino is not None and arduino.is_open:
            arduino.close()
    except Exception:
        pass
    print("[INFO] Car4 stopped safely.")

atexit.register(shutdown)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n==========================================")
    print("        CAR4 LANE CONTROL SYSTEM")
    print("==========================================")
    print(f"[INFO] Project: {BASE_DIR}")
    print(f"[INFO] Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"[INFO] Arduino: {SERIAL_PORT}")
    print(f"[INFO] Sessions: {SESSION_ROOT}")
    print(f"[INFO] Current active session: {active_session_id}\n")

    initialize_camera()
    connect_arduino()

    print("[INFO] Dashboard starting...")
    print("[INFO] Open: http://PI_IP_ADDRESS:5000\n")

    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False, use_reloader=False)
