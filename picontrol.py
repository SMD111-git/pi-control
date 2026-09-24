#!/usr/bin/env python3

import os
import csv
import time
import atexit
import threading
import zipfile
import shutil
from datetime import datetime

import cv2
import numpy as np
import RPi.GPIO as GPIO

from flask import (
    Flask,
    Response,
    jsonify,
    render_template_string,
    request,
    send_file,
)

from picamera2 import Picamera2


# ============================================================
# CAR4 - SINGLE FILE / SINGLE DATASET
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# CAMERA CONFIGURATION
# ============================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

CAPTURE_INTERVAL = 0.5

# Don't save a frame while the car is stationary (command == STOP).
# Stops between button presses used to flood the dataset with
# near-identical "stopped" images that add nothing to training.
CAPTURE_ONLY_WHEN_MOVING = True

# ============================================================
# SPEED CONFIGURATION
# ============================================================

MIN_SPEED = 0
MAX_SPEED = 255
DEFAULT_SPEED = 100

# ============================================================
# TURN SMOOTHING
# ============================================================
# Old behavior: L/R pivoted in place (one wheel forward, one backward).
# That's abrupt -- the car snaps from moving to spinning to stopped,
# which is exactly what caused the jerky start/stop pattern and the
# pile of near-duplicate STOP frames.
#
# New behavior: L/R now curve forward -- both wheels drive forward,
# but the inside wheel runs slower (TURN_RATIO) instead of reversing.
# The car keeps moving the whole time, so the turn is smooth and every
# frame captured during it is a useful "car mid-turn" example instead
# of a stationary one.
#
# TURN_RATIO: 0.0 = old hard pivot, 1.0 = no turn at all (straight).
# 0.35 is a gentle, still-visible curve -- tune this to your track.
TURN_RATIO = 0.35

# ============================================================
# LANE DETECTION CONFIGURATION
# ============================================================

ROI_START_RATIO = 0.50

BLACK_V_MIN = 0
BLACK_V_MAX = 95

BLACK_S_MIN = 0
BLACK_S_MAX = 255

MIN_CONTOUR_AREA = 100

CAR_CENTER_X = CAMERA_WIDTH // 2


# ============================================================
# MOTOR / GPIO CONFIGURATION
# ============================================================

# LEFT MOTOR
PWMA = 6
AIN2 = 5
AIN1 = 4
STBY_LEFT = 24

# RIGHT MOTOR
PWMB = 16
BIN1 = 20
BIN2 = 21
STBY_RIGHT = 26

LEFT_SCALE = 1.00
RIGHT_SCALE = 0.875

INVERT_LEFT = False
INVERT_RIGHT = True

GPIO_PWM_FREQ = 1000


# ============================================================
# GPIO INITIALIZATION
# ============================================================

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO_PINS = (PWMA, AIN1, AIN2, STBY_LEFT, PWMB, BIN1, BIN2, STBY_RIGHT)

for pin in GPIO_PINS:
    GPIO.setup(pin, GPIO.OUT)

pwm_a = GPIO.PWM(PWMA, GPIO_PWM_FREQ)
pwm_b = GPIO.PWM(PWMB, GPIO_PWM_FREQ)

pwm_a.start(0)
pwm_b.start(0)

GPIO.output(STBY_LEFT, GPIO.HIGH)
GPIO.output(STBY_RIGHT, GPIO.HIGH)

motor_gpio_lock = threading.Lock()

print(f"[GPIO] LEFT: PWMA={PWMA} AIN1={AIN1} AIN2={AIN2} STBY={STBY_LEFT}")
print(f"[GPIO] RIGHT: PWMB={PWMB} BIN1={BIN1} BIN2={BIN2} STBY={STBY_RIGHT}")


# ============================================================
# DATA STORAGE
# ============================================================

DATA_ROOT = os.path.join(BASE_DIR, "car4_data")
IMAGE_FOLDER = os.path.join(DATA_ROOT, "images")

os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

LABELS_CSV = os.path.join(DATA_ROOT, "labels.csv")
MOTOR_LOG_CSV = os.path.join(DATA_ROOT, "motor_log.csv")

DATASET_ZIP = os.path.join(BASE_DIR, "car4_dataset.zip")

zip_lock = threading.Lock()
data_lock = threading.Lock()


# ============================================================
# CSV HEADERS
# ============================================================

CSV_HEADER = [
    "image_filename", "direction", "command", "speed", "lane_detected",
    "car_center_x", "lane_center_x", "left_distance", "right_distance",
    "position_error", "lane_angle", "lane_left_x", "lane_right_x",
    "left_motor_pwm", "right_motor_pwm", "timestamp",
]

MOTOR_HEADER = ["command", "speed", "left_motor_pwm", "right_motor_pwm", "timestamp"]


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)
app.secret_key = "car4-local-dashboard-secret"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False
app.config["SESSION_PERMANENT"] = False


# ============================================================
# GLOBAL STATE
# ============================================================

camera = None
camera_lock = threading.Lock()
running = True

current_command = "STOP"
current_speed = 0

last_capture_time = 0.0
image_counter = 0
current_recording_id = None

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


# ============================================================
# RECORDING ID
# ============================================================

def create_recording_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def start_new_recording():
    global current_recording_id, image_counter, last_capture_time
    current_recording_id = create_recording_id()
    image_counter = 0
    last_capture_time = 0.0
    print(f"[RECORDING] New recording: {current_recording_id}")
    return current_recording_id


# ============================================================
# CSV INITIALIZATION
# ============================================================

def initialize_csv_files():
    if not os.path.exists(LABELS_CSV):
        with open(LABELS_CSV, "w", newline="") as file:
            csv.writer(file).writerow(CSV_HEADER)
    if not os.path.exists(MOTOR_LOG_CSV):
        with open(MOTOR_LOG_CSV, "w", newline="") as file:
            csv.writer(file).writerow(MOTOR_HEADER)


initialize_csv_files()

if current_recording_id is None:
    start_new_recording()


# ============================================================
# MOTOR FUNCTIONS
# ============================================================

def _set_channel(in1, in2, pwm_channel, duty_percent):
    duty_percent = max(-100, min(100, duty_percent))

    if duty_percent > 0:
        GPIO.output(in1, GPIO.HIGH)
        GPIO.output(in2, GPIO.LOW)
    elif duty_percent < 0:
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
    else:
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)

    pwm_channel.ChangeDutyCycle(abs(duty_percent))


def _drive_left(duty_percent):
    duty = duty_percent * LEFT_SCALE * (-1 if INVERT_LEFT else 1)
    _set_channel(AIN1, AIN2, pwm_a, duty)


def _drive_right(duty_percent):
    duty = duty_percent * RIGHT_SCALE * (-1 if INVERT_RIGHT else 1)
    _set_channel(BIN1, BIN2, pwm_b, duty)


def apply_motor_gpio(command, speed_0_255):
    speed_0_255 = max(MIN_SPEED, min(MAX_SPEED, int(speed_0_255)))
    duty = (speed_0_255 / 255.0) * 100.0

    if command == "F":
        left, right = duty, duty
    elif command == "B":
        left, right = -duty, -duty
    elif command == "L":
        # Smooth curve left: both wheels forward, left (inside) wheel slower.
        left, right = duty * TURN_RATIO, duty
    elif command == "R":
        # Smooth curve right: both wheels forward, right (inside) wheel slower.
        left, right = duty, duty * TURN_RATIO
    else:
        left, right = 0, 0

    with motor_gpio_lock:
        GPIO.output(STBY_LEFT, GPIO.HIGH)
        GPIO.output(STBY_RIGHT, GPIO.HIGH)
        _drive_left(left)
        _drive_right(right)

    return left, right


def stop_motor_gpio():
    with motor_gpio_lock:
        _set_channel(AIN1, AIN2, pwm_a, 0)
        _set_channel(BIN1, BIN2, pwm_b, 0)


def calculate_motor_values(command, speed):
    speed = int(speed)
    if command == "F":
        return speed, speed
    if command == "B":
        return -speed, -speed
    if command == "L":
        return int(speed * TURN_RATIO), speed
    if command == "R":
        return speed, int(speed * TURN_RATIO)
    return 0, 0


def send_motor_command(command, speed=0):
    global current_command, current_speed

    command = str(command).upper()
    if command not in ("F", "B", "L", "R", "STOP"):
        command = "STOP"

    if command == "STOP":
        speed = 0
        stop_motor_gpio()
        left_pwm, right_pwm = 0, 0
    else:
        speed = int(max(MIN_SPEED, min(MAX_SPEED, int(speed))))
        left_pwm, right_pwm = apply_motor_gpio(command, speed)

    current_command = command
    current_speed = speed

    return f"{command},{speed} (L{left_pwm:.0f} R{right_pwm:.0f})"


# ============================================================
# MOTOR LOGGING
# ============================================================

def log_motor_command(command, speed):
    left_pwm, right_pwm = calculate_motor_values(command, speed)
    timestamp = datetime.now().isoformat(timespec="milliseconds")
    row = [command, speed, left_pwm, right_pwm, timestamp]
    try:
        with data_lock:
            with open(MOTOR_LOG_CSV, "a", newline="") as file:
                csv.writer(file).writerow(row)
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
        print("[INFO] Camera started")
    except Exception as error:
        camera = None
        print(f"[ERROR] Camera initialization failed: {error}")


# ============================================================
# LANE DETECTION
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
        cv2.rectangle(output, (10, 10), (360, 60), (0, 0, 0), -1)
        cv2.putText(output, "LANE NOT DETECTED", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
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
        lane_angle = float(round(np.degrees(np.arctan2(vx, vy)), 2)) if abs(vy) > 1e-4 else 90.0

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

    cv2.rectangle(output, (10, 10), (370, 190), (0, 0, 0), -1)
    info = [
        "LANE: DETECTED",
        f"Car center: {CAR_CENTER_X}px",
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


# ============================================================
# DIRECTION
# ============================================================

def get_direction_label(command):
    return {"F": "forward", "B": "backward", "L": "left", "R": "right", "STOP": "stop"}.get(command, "stop")


# ============================================================
# DATASET CAPTURE
# ============================================================

def capture_dataset_image(command, speed, frame, lane_data):
    global image_counter

    direction = get_direction_label(command)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_counter += 1
    filename = f"{image_counter:06d}_{direction}_speed_{speed}_{timestamp}.jpg"
    image_path = os.path.join(IMAGE_FOLDER, filename)

    success = cv2.imwrite(
        image_path,
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )

    if not success:
        print(f"[WARNING] Failed to save {filename}")
        return

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
            with open(LABELS_CSV, "a", newline="") as file:
                csv.writer(file).writerow(row)
    except Exception as error:
        print(f"[WARNING] CSV error: {error}")
        return

    print(f"[DATASET] {filename} | {direction} | speed={speed} | lane={lane_data['lane_detected']}")


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
            should_capture = (now - last_capture_time) >= CAPTURE_INTERVAL

            # Skip saving frames while the car is stationary -- these add
            # nothing to a driving dataset and were flooding it with
            # near-duplicate STOP images between button presses.
            if CAPTURE_ONLY_WHEN_MOVING and current_command == "STOP":
                should_capture = False

            if should_capture:
                capture_dataset_image(current_command, current_speed, frame, lane_data)
                last_capture_time = now

            bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            success, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + encoded.tobytes()
                + b"\r\n"
            )

        except Exception as error:
            print(f"[WARNING] Camera stream error: {error}")
            time.sleep(0.2)


# ============================================================
# ZIP CREATION
# ============================================================

def create_dataset_zip():
    temp_zip = DATASET_ZIP + ".tmp"
    print("[ZIP] Creating complete CAR4 dataset...")

    with zip_lock:
        try:
            if os.path.exists(temp_zip):
                os.remove(temp_zip)

            image_count = 0

            with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                if os.path.isdir(IMAGE_FOLDER):
                    for filename in sorted(os.listdir(IMAGE_FOLDER)):
                        filepath = os.path.join(IMAGE_FOLDER, filename)
                        if os.path.isfile(filepath):
                            zf.write(filepath, f"images/{filename}")
                            image_count += 1

                if os.path.isfile(LABELS_CSV):
                    zf.write(LABELS_CSV, "labels.csv")

                if os.path.isfile(MOTOR_LOG_CSV):
                    zf.write(MOTOR_LOG_CSV, "motor_log.csv")

                readme = f"""CAR4 DATASET

Generated: {datetime.now().isoformat()}
Images: {image_count}

Files:
images/       Captured camera images
labels.csv    Image labels and lane information
motor_log.csv Motor command history

Each image is linked to its command through the image_filename
column in labels.csv.

Commands: F=Forward, B=Backward, L=Left, R=Right, STOP=Stop
Speed range: 0-255
Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT}
Capture interval: {CAPTURE_INTERVAL} seconds
Capture only while moving: {CAPTURE_ONLY_WHEN_MOVING}
Turn ratio (inside wheel speed during L/R): {TURN_RATIO}
"""
                zf.writestr("README.txt", readme)

            os.replace(temp_zip, DATASET_ZIP)
            size_mb = os.path.getsize(DATASET_ZIP) / (1024 * 1024)
            print(f"[ZIP] Dataset ready: {image_count} images, {size_mb:.2f} MB")
            return DATASET_ZIP

        except Exception:
            try:
                if os.path.exists(temp_zip):
                    os.remove(temp_zip)
            except Exception:
                pass
            raise


# ============================================================
# CLEAR DATA
# ============================================================

def clear_all_data():
    global image_counter, last_capture_time

    send_motor_command("STOP", 0)

    with data_lock:
        if os.path.isdir(IMAGE_FOLDER):
            for filename in os.listdir(IMAGE_FOLDER):
                filepath = os.path.join(IMAGE_FOLDER, filename)
                try:
                    if os.path.isfile(filepath):
                        os.remove(filepath)
                    elif os.path.isdir(filepath):
                        shutil.rmtree(filepath)
                except Exception as error:
                    print(f"[CLEAR WARNING] {filepath}: {error}")

        for csv_file in (LABELS_CSV, MOTOR_LOG_CSV):
            try:
                if os.path.isfile(csv_file):
                    os.remove(csv_file)
            except Exception as error:
                print(f"[CLEAR WARNING] {csv_file}: {error}")

        try:
            if os.path.isfile(DATASET_ZIP):
                os.remove(DATASET_ZIP)
        except Exception as error:
            print(f"[CLEAR WARNING] ZIP: {error}")

        os.makedirs(DATA_ROOT, exist_ok=True)
        os.makedirs(IMAGE_FOLDER, exist_ok=True)
        initialize_csv_files()

        image_counter = 0
        last_capture_time = 0.0
        new_id = start_new_recording()

        return new_id


# ============================================================
# HTML DASHBOARD
# ============================================================

HTML_PAGE = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CAR4 Lane Control</title>
<style>
*{box-sizing:border-box}
body{margin:0;padding:20px;background:#101114;color:white;font-family:Arial,Helvetica,sans-serif;text-align:center}
.container{width:100%;max-width:1000px;margin:auto}
h1{margin-top:0;font-size:30px}
.panel{background:#202228;padding:16px;border-radius:14px;margin-bottom:16px}
.camera-box img{width:100%;max-width:640px;display:block;margin:auto;border-radius:10px;background:black}
.status-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
.status-item{background:#30333b;padding:12px;border-radius:9px;font-size:15px}
.status-item strong{display:block;margin-top:6px;font-size:20px}
.speed-number{font-size:30px;font-weight:bold;margin-bottom:10px}
.speed-panel input{width:90%;max-width:700px}
.control-panel{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;max-width:500px;margin:20px auto}
button{border:none;border-radius:12px;min-height:70px;color:white;font-size:18px;font-weight:bold;cursor:pointer;user-select:none;touch-action:none}
button:active{transform:scale(0.95)}
.forward{background:#27ae60}
.left,.right{background:#3498db}
.backward{background:#e67e22}
.stop{background:#e74c3c}
.new-session{background:#16a085;min-height:50px;width:100%;margin-bottom:10px}
.clear{background:#8e44ad;min-height:50px;width:100%}
.test{background:#2980b9;min-height:50px;width:100%;margin-bottom:12px}
.download-panel a{display:block;background:#555;color:white;text-decoration:none;padding:14px;margin:8px;border-radius:9px}
.download-panel a:hover{background:#777}
.recording{color:#2ecc71;font-size:17px;word-break:break-all}
.warning{color:#f1c40f;margin:15px}
@media(max-width:600px){body{padding:10px}.status-grid{grid-template-columns:1fr}button{min-height:60px;font-size:16px}}
</style>
</head>
<body>
<div class="container">

<h1>CAR4 Lane Control Dashboard</h1>

<div class="panel">
<h2>Current Recording</h2>
<div class="recording" id="recording_id">---</div>
<p>All images and logs are stored in one location. There are no session folders.</p>
<button class="new-session" id="new_recording_button">START NEW RECORDING</button>
<button class="clear" id="clear_button">CLEAR ALL DATA</button>
</div>

<div class="panel camera-box">
<img src="/video_feed" alt="CAR4 Camera" />
</div>

<div class="panel">
<h2>Lane Information</h2>
<div class="status-grid">
<div class="status-item">Lane detected<strong id="lane_detected">---</strong></div>
<div class="status-item">Car center<strong id="car_center_x">---</strong></div>
<div class="status-item">Lane center<strong id="lane_center_x">---</strong></div>
<div class="status-item">Left distance<strong id="left_distance">---</strong></div>
<div class="status-item">Right distance<strong id="right_distance">---</strong></div>
<div class="status-item">Position error<strong id="position_error">---</strong></div>
<div class="status-item">Lane angle<strong id="lane_angle">---</strong></div>
<div class="status-item">Command<strong id="current_command">STOP</strong></div>
<div class="status-item">Speed<strong id="current_speed">0</strong></div>
<div class="status-item">Images<strong id="image_count">0</strong></div>
</div>
</div>

<div class="panel speed-panel">
<h2>Speed</h2>
<div class="speed-number" id="speed_value">100</div>
<input type="range" id="speed_slider" min="0" max="255" value="100">
</div>

<div class="panel">
<div class="control-panel">
<div></div>
<button id="forward_button" class="forward">FORWARD</button>
<div></div>
<button id="left_button" class="left">LEFT</button>
<button id="stop_button" class="stop">STOP</button>
<button id="right_button" class="right">RIGHT</button>
<div></div>
<button id="backward_button" class="backward">BACKWARD</button>
<div></div>
</div>
<div class="warning">Press and hold a movement button. Release it to STOP CAR4.</div>
</div>

<div class="panel">
<button id="gpio_test_button" class="test">RUN GPIO SELF-TEST</button>
</div>

<div class="panel download-panel">
<h2>Dataset</h2>
<a href="/download_dataset">DOWNLOAD COMPLETE CAR4 DATASET ZIP</a>
<a href="/download_labels">DOWNLOAD LABELS CSV</a>
<a href="/download_motor_log">DOWNLOAD MOTOR LOG CSV</a>
</div>

</div>

<script>
let selectedSpeed = 100;
const slider = document.getElementById("speed_slider");
const speedValue = document.getElementById("speed_value");
slider.addEventListener("input", () => {
  selectedSpeed = parseInt(slider.value);
  speedValue.innerText = selectedSpeed;
});

let movementActive = false;
let activeCommand = null;

function sendCommand(command, speed) {
  fetch("/motor", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ command: command, speed: speed })
  }).catch(error => console.log(error));
}

function startMovement(command) {
  if (movementActive && activeCommand === command) return;
  movementActive = true;
  activeCommand = command;
  sendCommand(command, selectedSpeed);
}

function stopMovement() {
  movementActive = false;
  activeCommand = null;
  sendCommand("STOP", 0);
}

function setupMovementButton(id, command) {
  const button = document.getElementById(id);
  button.addEventListener("mousedown", e => { e.preventDefault(); startMovement(command); });
  button.addEventListener("mouseup", e => { e.preventDefault(); stopMovement(); });
  button.addEventListener("mouseleave", () => stopMovement());
  button.addEventListener("touchstart", e => { e.preventDefault(); startMovement(command); }, { passive: false });
  button.addEventListener("touchend", e => { e.preventDefault(); stopMovement(); }, { passive: false });
  button.addEventListener("touchcancel", e => { e.preventDefault(); stopMovement(); }, { passive: false });
}

setupMovementButton("forward_button", "F");
setupMovementButton("left_button", "L");
setupMovementButton("right_button", "R");
setupMovementButton("backward_button", "B");

document.getElementById("stop_button").addEventListener("click", () => stopMovement());
document.addEventListener("mouseup", () => { if (movementActive) stopMovement(); });
window.addEventListener("blur", () => { movementActive = false; activeCommand = null; sendCommand("STOP", 0); });
window.addEventListener("beforeunload", () => {
  navigator.sendBeacon("/motor_stop", new Blob([JSON.stringify({ command: "STOP", speed: 0 })], { type: "application/json" }));
});

function updateStatus() {
  fetch("/status").then(r => r.json()).then(data => {
    document.getElementById("lane_detected").innerText = data.lane_detected ? "YES" : "NO";
    document.getElementById("car_center_x").innerText = data.car_center_x + " px";
    document.getElementById("lane_center_x").innerText = data.lane_center_x + " px";
    document.getElementById("left_distance").innerText = data.left_distance + " px";
    document.getElementById("right_distance").innerText = data.right_distance + " px";
    document.getElementById("position_error").innerText = data.position_error + " px";
    document.getElementById("lane_angle").innerText = data.lane_angle + " deg";
    document.getElementById("current_command").innerText = data.current_command;
    document.getElementById("current_speed").innerText = data.current_speed;
    document.getElementById("recording_id").innerText = data.recording_id || "---";
    document.getElementById("image_count").innerText = data.image_count;
  }).catch(error => console.log(error));
}
setInterval(updateStatus, 300);
updateStatus();

document.getElementById("new_recording_button").addEventListener("click", () => {
  if (!confirm("Start a new recording? Existing images and CSV data will remain.")) return;
  fetch("/new_recording", { method: "POST" })
    .then(r => r.json())
    .then(data => { document.getElementById("recording_id").innerText = data.recording_id; alert("New recording started."); })
    .catch(error => alert(error));
});

document.getElementById("clear_button").addEventListener("click", () => {
  if (!confirm("DELETE ALL CAR4 IMAGES, CSV LOGS AND ZIP DATA? This cannot be undone.")) return;
  fetch("/clear_data", { method: "POST" })
    .then(r => r.json())
    .then(data => {
      if (data.success) { document.getElementById("recording_id").innerText = data.recording_id; alert("All data cleared."); }
      else { alert(data.error || "Clear failed."); }
    })
    .catch(error => alert(error));
});

document.getElementById("gpio_test_button").addEventListener("click", () => {
  fetch("/gpio_test", { method: "POST" })
    .then(r => r.json())
    .then(data => alert(data.message || "GPIO test complete."))
    .catch(error => alert(error));
});
</script>
</body>
</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/motor", methods=["POST"])
def motor_control():
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

    result = send_motor_command(command, speed)
    log_motor_command(command, speed)

    return jsonify({"success": True, "command": command, "speed": speed, "message": result})


@app.route("/motor_stop", methods=["POST"])
def motor_stop():
    send_motor_command("STOP", 0)
    log_motor_command("STOP", 0)
    return jsonify({"success": True})


@app.route("/gpio_test", methods=["POST"])
def gpio_test():
    try:
        print("[GPIO TEST] Left motor - 2 seconds")
        with motor_gpio_lock:
            GPIO.output(STBY_LEFT, GPIO.HIGH)
            GPIO.output(STBY_RIGHT, GPIO.HIGH)
            _drive_left(60)
            _drive_right(0)
        time.sleep(2)
        stop_motor_gpio()
        time.sleep(0.5)

        print("[GPIO TEST] Right motor - 2 seconds")
        with motor_gpio_lock:
            GPIO.output(STBY_LEFT, GPIO.HIGH)
            GPIO.output(STBY_RIGHT, GPIO.HIGH)
            _drive_left(0)
            _drive_right(60)
        time.sleep(2)
        stop_motor_gpio()

        return jsonify({
            "success": True,
            "message": "GPIO test complete. Left wheel and then right wheel were tested.",
        })

    except Exception as error:
        stop_motor_gpio()
        return jsonify({"success": False, "message": f"GPIO test failed: {error}"}), 500


@app.route("/status")
def status():
    data = dict(latest_lane_data)
    data["current_command"] = current_command
    data["current_speed"] = current_speed
    data["recording_id"] = current_recording_id

    try:
        data["image_count"] = len([x for x in os.listdir(IMAGE_FOLDER) if x.lower().endswith(".jpg")])
    except Exception:
        data["image_count"] = 0

    return jsonify(data)


@app.route("/new_recording", methods=["POST"])
def new_recording():
    send_motor_command("STOP", 0)
    recording_id = start_new_recording()
    return jsonify({"success": True, "recording_id": recording_id})


@app.route("/clear_data", methods=["POST"])
def clear_data_route():
    try:
        recording_id = clear_all_data()
        return jsonify({"success": True, "recording_id": recording_id})
    except Exception as error:
        print(f"[CLEAR ERROR] {error}")
        return jsonify({"success": False, "error": str(error)}), 500


@app.route("/download_dataset")
def download_dataset():
    try:
        zip_path = create_dataset_zip()
        return send_file(zip_path, mimetype="application/zip", as_attachment=True,
                          download_name="car4_dataset.zip", max_age=0)
    except Exception as error:
        print(f"[DOWNLOAD ERROR] {error}")
        return f"<h2>Dataset download failed</h2><p>{error}</p>", 500


@app.route("/download_labels")
def download_labels():
    initialize_csv_files()
    return send_file(LABELS_CSV, mimetype="text/csv", as_attachment=True,
                      download_name="labels.csv", max_age=0)


@app.route("/download_motor_log")
def download_motor_log():
    initialize_csv_files()
    return send_file(MOTOR_LOG_CSV, mimetype="text/csv", as_attachment=True,
                      download_name="motor_log.csv", max_age=0)


# ============================================================
# SAFE SHUTDOWN
# ============================================================

def shutdown():
    global running
    running = False
    print("[INFO] Shutting down CAR4...")

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
        pwm_a.stop()
        pwm_b.stop()
        GPIO.output(STBY_LEFT, GPIO.LOW)
        GPIO.output(STBY_RIGHT, GPIO.LOW)
        GPIO.cleanup()
    except Exception:
        pass

    print("[INFO] CAR4 stopped safely.")


atexit.register(shutdown)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print()
    print("==========================================")
    print("       CAR4 LANE CONTROL SYSTEM")
    print("==========================================")
    print(f"[INFO] Project: {BASE_DIR}")
    print(f"[INFO] Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print("[INFO] Motors: Direct Raspberry Pi GPIO")
    print("[INFO] Arduino: NONE")
    print("[INFO] Session folders: NONE")
    print(f"[INFO] Data folder: {DATA_ROOT}")
    print(f"[INFO] Images: {IMAGE_FOLDER}")
    print(f"[INFO] Dataset ZIP: {DATASET_ZIP}")
    print(f"[INFO] Recording: {current_recording_id}")
    print(f"[INFO] Turn ratio (smooth curve, inside wheel): {TURN_RATIO}")
    print(f"[INFO] Capture only while moving: {CAPTURE_ONLY_WHEN_MOVING}")
    print()

    initialize_camera()

    print("[INFO] Dashboard starting...")
    print("[INFO] Open from another device:")
    print("       http://PI_IP_ADDRESS:5000")
    print()

    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False, use_reloader=False)
