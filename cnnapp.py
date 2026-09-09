#!/usr/bin/env python3

import os
import csv
import time
import atexit
import threading
import zipfile
from datetime import datetime

import cv2
import numpy as np
import serial

from flask import (
    Flask,
    Response,
    jsonify,
    render_template_string,
    request,
    send_file
)

from picamera2 import Picamera2


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Change this if Arduino is connected to another port
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

# Camera resolution
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Image capture interval
CAPTURE_INTERVAL = 0.5

# Motor speed range
MIN_SPEED = 0
MAX_SPEED = 255
DEFAULT_SPEED = 100

# Black tape threshold
# These values may need adjustment depending on lighting
BLACK_V_MIN = 0
BLACK_V_MAX = 100
BLACK_S_MIN = 0
BLACK_S_MAX = 255

# Region of interest
# Only the lower part of the image is used for lane detection
ROI_START_RATIO = 0.50

# Minimum contour area
MIN_CONTOUR_AREA = 80

# Car center
# Initially assume the camera image center represents Car4 center
CAR_CENTER_X = CAMERA_WIDTH // 2


# ============================================================
# FOLDERS AND FILES
# ============================================================

IMAGE_FOLDER = os.path.join(BASE_DIR, "captured_images")
LOG_FOLDER = os.path.join(BASE_DIR, "logs")

DATASET_CSV = os.path.join(BASE_DIR, "labels.csv")
MOTOR_LOG_CSV = os.path.join(LOG_FOLDER, "motor_log.csv")

DATASET_ZIP = os.path.join(BASE_DIR, "car4_cnn_dataset.zip")

os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# GLOBAL VARIABLES
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
    "decision": "STOP"
}


# ============================================================
# CSV INITIALIZATION
# ============================================================

def initialize_csv_files():
    """
    Create CSV files with headers if they do not already exist.
    """

    if not os.path.exists(DATASET_CSV):
        with open(DATASET_CSV, "w", newline="") as file:
            writer = csv.writer(file)

            writer.writerow([
                "image_filename",
                "direction",
                "command",
                "speed",
                "lane_detected",
                "car_center_x",
                "lane_center_x",
                "left_distance",
                "right_distance",
                "position_error",
                "lane_angle",
                "lane_left_x",
                "lane_right_x",
                "left_motor_pwm",
                "right_motor_pwm",
                "timestamp"
            ])

    if not os.path.exists(MOTOR_LOG_CSV):
        with open(MOTOR_LOG_CSV, "w", newline="") as file:
            writer = csv.writer(file)

            writer.writerow([
                "command",
                "speed",
                "left_motor_pwm",
                "right_motor_pwm",
                "timestamp"
            ])


initialize_csv_files()


# ============================================================
# ARDUINO SERIAL
# ============================================================

def connect_arduino():
    """
    Connect to Arduino motor controller.
    """

    global arduino

    try:
        arduino = serial.Serial(
            SERIAL_PORT,
            BAUD_RATE,
            timeout=1
        )

        time.sleep(2)

        print(f"[INFO] Arduino connected: {SERIAL_PORT}")

    except Exception as error:
        arduino = None
        print(f"[WARNING] Arduino connection failed: {error}")
        print("[WARNING] Camera dashboard will still run.")
        print("[WARNING] Check SERIAL_PORT in app.py.")


def send_motor_command(command, speed=0):
    """
    Send command to Arduino.

    Commands:
        F,100
        B,100
        L,100
        R,100
        STOP
    """

    global current_command
    global current_speed

    command = command.upper()

    if command == "STOP":
        speed = 0

    speed = int(max(MIN_SPEED, min(MAX_SPEED, speed)))

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
                arduino.write(serial_command.encode("utf-8"))

        except Exception as error:
            print(f"[WARNING] Serial write failed: {error}")

    current_command = command
    current_speed = speed

    return serial_command.strip()


def calculate_motor_values(command, speed):
    """
    Estimate left and right motor PWM values.

    This is for dataset logging.
    Actual motor mixing is handled by Arduino.
    """

    command = command.upper()

    if command == "F":
        return speed, speed

    if command == "B":
        return speed, speed

    if command == "L":
        return 0, speed

    if command == "R":
        return speed, 0

    return 0, 0


def log_motor_command(command, speed):
    """
    Save motor command in motor_log.csv.
    """

    left_pwm, right_pwm = calculate_motor_values(command, speed)

    timestamp = datetime.now().isoformat(timespec="milliseconds")

    with open(MOTOR_LOG_CSV, "a", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            command,
            speed,
            left_pwm,
            right_pwm,
            timestamp
        ])


# ============================================================
# CAMERA INITIALIZATION
# ============================================================

def initialize_camera():
    """
    Initialize OV5647 camera using Picamera2.
    """

    global camera

    try:
        camera = Picamera2()

        configuration = camera.create_video_configuration(
            main={
                "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
                "format": "RGB888"
            }
        )

        camera.configure(configuration)
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
    """
    Detect black tape in the lower part of the image.

    Returns:
        overlay_frame
        lane_data
    """

    height, width = frame.shape[:2]

    output = frame.copy()

    # Region of interest
    roi_start_y = int(height * ROI_START_RATIO)

    roi = frame[roi_start_y:height, :]

    # Convert RGB to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    # Detect dark/black pixels
    lower_black = np.array([
        0,
        BLACK_S_MIN,
        BLACK_V_MIN
    ])

    upper_black = np.array([
        180,
        BLACK_S_MAX,
        BLACK_V_MAX
    ])

    mask = cv2.inRange(
        hsv,
        lower_black,
        upper_black
    )

    # Remove small noise
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area >= MIN_CONTOUR_AREA:
            valid_contours.append(contour)

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
        "decision": "STOP"
    }

    # Draw ROI line
    cv2.line(
        output,
        (0, roi_start_y),
        (width, roi_start_y),
        (0, 255, 255),
        2
    )

    if not valid_contours:
        cv2.putText(
            output,
            "LANE NOT DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        return output, lane_data

    # Combine all valid contours
    all_points = np.vstack(valid_contours)

    x_values = all_points[:, :, 0].flatten()
    y_values = all_points[:, :, 1].flatten()

    lane_left_x = int(np.min(x_values))
    lane_right_x = int(np.max(x_values))

    lane_center_x = int(
        (lane_left_x + lane_right_x) / 2
    )

    # Position error
    position_error = CAR_CENTER_X - lane_center_x

    # Distances from car center to lane boundaries
    left_distance = CAR_CENTER_X - lane_left_x
    right_distance = lane_right_x - CAR_CENTER_X

    # Calculate approximate lane angle
    moments = cv2.moments(all_points)

    if moments["m00"] != 0:
        centroid_x = int(
            moments["m10"] / moments["m00"]
        )

        centroid_y = int(
            moments["m01"] / moments["m00"]
        )

        # Convert ROI coordinates to full-image coordinates
        centroid_y += roi_start_y

        angle_radians = np.arctan2(
            CAR_CENTER_X - centroid_x,
            height - centroid_y
        )

        lane_angle = float(
            np.degrees(angle_radians)
        )

    else:
        centroid_x = lane_center_x
        centroid_y = height
        lane_angle = 0.0

    # Draw detected black lane contour
    contour_offset = np.array(
        [0, roi_start_y]
    )

    for contour in valid_contours:
        shifted_contour = contour + contour_offset

        cv2.drawContours(
            output,
            [shifted_contour],
            -1,
            (0, 255, 0),
            2
        )

    # Draw lane boundaries
    cv2.line(
        output,
        (lane_left_x, roi_start_y),
        (lane_left_x, height),
        (255, 0, 0),
        2
    )

    cv2.line(
        output,
        (lane_right_x, roi_start_y),
        (lane_right_x, height),
        (255, 0, 0),
        2
    )

    # Draw lane center
    cv2.line(
        output,
        (lane_center_x, roi_start_y),
        (lane_center_x, height),
        (0, 255, 255),
        2
    )

    # Draw Car4 center
    cv2.line(
        output,
        (CAR_CENTER_X, roi_start_y),
        (CAR_CENTER_X, height),
        (255, 0, 255),
        2
    )

    cv2.circle(
        output,
        (CAR_CENTER_X, height - 25),
        8,
        (255, 0, 255),
        -1
    )

    # Draw position error
    cv2.arrowedLine(
        output,
        (CAR_CENTER_X, height - 50),
        (lane_center_x, height - 50),
        (0, 255, 255),
        3,
        tipLength=0.2
    )

    lane_data = {
        "lane_detected": True,
        "car_center_x": CAR_CENTER_X,
        "lane_center_x": lane_center_x,
        "left_distance": max(0, left_distance),
        "right_distance": max(0, right_distance),
        "position_error": position_error,
        "lane_angle": round(lane_angle, 2),
        "lane_left_x": lane_left_x,
        "lane_right_x": lane_right_x,
        "decision": current_command
    }

    # Information panel
    cv2.rectangle(
        output,
        (10, 10),
        (330, 175),
        (0, 0, 0),
        -1
    )

    information = [
        f"Lane: DETECTED",
        f"Car center: {CAR_CENTER_X}px",
        f"Lane center: {lane_center_x}px",
        f"Left distance: {max(0, left_distance)}px",
        f"Right distance: {max(0, right_distance)}px",
        f"Error: {position_error}px",
        f"Angle: {lane_angle:.2f} deg",
        f"Command: {current_command}"
    ]

    y_position = 32

    for text in information:
        cv2.putText(
            output,
            text,
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

        y_position += 20

    return output, lane_data


# ============================================================
# IMAGE CAPTURE AND DATASET
# ============================================================

def get_direction_label(command):
    """
    Convert Arduino command into CNN direction label.
    """

    labels = {
        "F": "forward",
        "B": "backward",
        "L": "left",
        "R": "right",
        "STOP": "stop"
    }

    return labels.get(command, "stop")


def capture_dataset_image(command, speed, frame, lane_data):
    """
    Save image and CSV label.

    The image is saved with:
        direction
        speed
        timestamp
    """

    global image_counter

    direction = get_direction_label(command)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    image_counter += 1

    filename = (
        f"{image_counter:06d}_"
        f"{direction}_"
        f"speed_{speed}_"
        f"{timestamp}.jpg"
    )

    image_path = os.path.join(
        IMAGE_FOLDER,
        filename
    )

    # Save RGB frame using OpenCV
    cv2.imwrite(
        image_path,
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    )

    left_pwm, right_pwm = calculate_motor_values(
        command,
        speed
    )

    csv_timestamp = datetime.now().isoformat(
        timespec="milliseconds"
    )

    with open(DATASET_CSV, "a", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            filename,
            direction,
            command,
            speed,
            int(lane_data["lane_detected"]),
            lane_data["car_center_x"],
            lane_data["lane_center_x"],
            lane_data["left_distance"],
            lane_data["right_distance"],
            lane_data["position_error"],
            lane_data["lane_angle"],
            lane_data["lane_left_x"],
            lane_data["lane_right_x"],
            left_pwm,
            right_pwm,
            csv_timestamp
        ])

    print(
        f"[DATASET] {filename} | "
        f"{direction} | "
        f"speed={speed} | "
        f"lane={lane_data['lane_detected']}"
    )


# ============================================================
# CAMERA FRAME GENERATOR
# ============================================================

def generate_frames():
    """
    Generate MJPEG camera stream.
    """

    global latest_lane_data
    global last_capture_time

    while running:

        if camera is None:
            time.sleep(0.1)
            continue

        try:
            with camera_lock:
                frame = camera.capture_array()

            # Detect lane and draw overlay
            overlay, lane_data = detect_black_lane(frame)

            latest_lane_data = lane_data

            # Capture dataset image periodically
            current_time = time.time()

            if (
                current_time - last_capture_time
                >= CAPTURE_INTERVAL
            ):

                capture_dataset_image(
                    current_command,
                    current_speed,
                    frame,
                    lane_data
                )

                last_capture_time = current_time

            # Convert RGB to BGR
            bgr_frame = cv2.cvtColor(
                overlay,
                cv2.COLOR_RGB2BGR
            )

            success, encoded_image = cv2.imencode(
                ".jpg",
                bgr_frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    85
                ]
            )

            if not success:
                continue

            frame_bytes = encoded_image.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

        except Exception as error:
            print(f"[WARNING] Camera frame error: {error}")
            time.sleep(0.1)


# ============================================================
# DATASET ZIP
# ============================================================

def create_dataset_zip():
    """
    Create ZIP containing:
        captured images
        labels.csv
        motor_log.csv
    """

    if os.path.exists(DATASET_ZIP):
        os.remove(DATASET_ZIP)

    with zipfile.ZipFile(
        DATASET_ZIP,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zip_file:

        if os.path.exists(DATASET_CSV):
            zip_file.write(
                DATASET_CSV,
                "labels.csv"
            )

        if os.path.exists(MOTOR_LOG_CSV):
            zip_file.write(
                MOTOR_LOG_CSV,
                "motor_log.csv"
            )

        if os.path.exists(IMAGE_FOLDER):

            for filename in os.listdir(IMAGE_FOLDER):

                file_path = os.path.join(
                    IMAGE_FOLDER,
                    filename
                )

                if os.path.isfile(file_path):
                    zip_file.write(
                        file_path,
                        os.path.join(
                            "images",
                            filename
                        )
                    )

    return DATASET_ZIP


# ============================================================
# HTML DASHBOARD
# ============================================================

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>Car4 Lane Control Dashboard</title>

<style>

body {
    margin: 0;
    padding: 20px;
    background: #111;
    color: white;
    font-family: Arial, sans-serif;
    text-align: center;
}

h1 {
    margin-top: 0;
}

.container {
    max-width: 1000px;
    margin: auto;
}

.camera-box {
    background: #222;
    padding: 10px;
    border-radius: 12px;
}

.camera-box img {
    width: 100%;
    max-width: 640px;
    border-radius: 8px;
}

.status {
    margin-top: 15px;
    padding: 15px;
    background: #222;
    border-radius: 12px;
}

.status-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
}

.status-item {
    background: #333;
    padding: 10px;
    border-radius: 8px;
}

.control-panel {
    margin-top: 20px;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    max-width: 450px;
    margin-left: auto;
    margin-right: auto;
}

button {
    min-height: 75px;
    font-size: 20px;
    font-weight: bold;
    border: none;
    border-radius: 12px;
    cursor: pointer;
    user-select: none;
    -webkit-user-select: none;
    touch-action: none;
}

button:active {
    transform: scale(0.95);
}

.forward {
    background: #27ae60;
    color: white;
}

.backward {
    background: #e67e22;
    color: white;
}

.left {
    background: #3498db;
    color: white;
}

.right {
    background: #3498db;
    color: white;
}

.stop {
    background: #e74c3c;
    color: white;
}

.speed-panel {
    margin-top: 20px;
}

input[type="range"] {
    width: 80%;
}

.download-panel {
    margin-top: 20px;
}

.download-panel a {
    display: inline-block;
    margin: 5px;
    padding: 12px;
    background: #555;
    color: white;
    text-decoration: none;
    border-radius: 8px;
}

.warning {
    color: #f1c40f;
    margin-top: 15px;
}

</style>

</head>

<body>

<div class="container">

<h1>Car4 Lane-Control Dashboard</h1>

<div class="camera-box">

<img src="/video_feed"
     alt="Car4 Camera Stream">

</div>

<div class="status">

<h2>Lane Information</h2>

<div class="status-grid">

<div class="status-item">
Lane detected:
<strong id="lane_detected">---</strong>
</div>

<div class="status-item">
Car4 center:
<strong id="car_center_x">---</strong>
</div>

<div class="status-item">
Lane center:
<strong id="lane_center_x">---</strong>
</div>

<div class="status-item">
Left distance:
<strong id="left_distance">---</strong>
</div>

<div class="status-item">
Right distance:
<strong id="right_distance">---</strong>
</div>

<div class="status-item">
Position error:
<strong id="position_error">---</strong>
</div>

<div class="status-item">
Lane angle:
<strong id="lane_angle">---</strong>
</div>

<div class="status-item">
Current command:
<strong id="current_command">STOP</strong>
</div>

<div class="status-item">
Current speed:
<strong id="current_speed">0</strong>
</div>

</div>

</div>

<div class="speed-panel">

<h2>Speed: <span id="speed_value">100</span></h2>

<input type="range"
       id="speed_slider"
       min="0"
       max="255"
       value="100">

</div>

<div class="control-panel">

<div></div>

<button id="forward_button"
        class="forward">
FORWARD
</button>

<div></div>

<button id="left_button"
        class="left">
LEFT
</button>

<button id="stop_button"
        class="stop">
STOP
</button>

<button id="right_button"
        class="right">
RIGHT
</button>

<div></div>

<button id="backward_button"
        class="backward">
BACKWARD
</button>

<div></div>

</div>

<div class="warning">
Press and hold a movement button.
Release it to stop Car4.
</div>

<div class="download-panel">

<h2>Dataset Downloads</h2>

<a href="/download_dataset">
Download Complete Dataset ZIP
</a>

<a href="/download_labels">
Download Labels CSV
</a>

<a href="/download_motor_log">
Download Motor Log CSV
</a>

</div>

</div>


<script>

let selectedSpeed = 100;
let movementActive = false;


// ------------------------------------------------------------
// Speed slider
// ------------------------------------------------------------

const speedSlider = document.getElementById("speed_slider");
const speedValue = document.getElementById("speed_value");

speedSlider.addEventListener("input", function() {
    selectedSpeed = parseInt(this.value);
    speedValue.innerText = selectedSpeed;
});


// ------------------------------------------------------------
// Start movement
// ------------------------------------------------------------

function startMovement(command) {

    if (movementActive && command === window.activeCommand) {
        return;
    }

    movementActive = true;
    window.activeCommand = command;

    fetch("/motor", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            command: command,
            speed: selectedSpeed
        })
    });

}


// ------------------------------------------------------------
// Stop movement
// ------------------------------------------------------------

function stopMovement() {

    if (!movementActive) {
        return;
    }

    movementActive = false;
    window.activeCommand = "STOP";

    fetch("/motor", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            command: "STOP",
            speed: 0
        })
    });

}


// ------------------------------------------------------------
// Configure press-and-hold button
// ------------------------------------------------------------

function setupMovementButton(buttonId, command) {

    const button = document.getElementById(buttonId);

    button.addEventListener("mousedown", function(event) {
        event.preventDefault();
        startMovement(command);
    });

    button.addEventListener("mouseup", function(event) {
        event.preventDefault();
        stopMovement();
    });

    button.addEventListener("mouseleave", function() {
        stopMovement();
    });

    button.addEventListener("touchstart", function(event) {
        event.preventDefault();
        startMovement(command);
    }, { passive: false });

    button.addEventListener("touchend", function(event) {
        event.preventDefault();
        stopMovement();
    }, { passive: false });

    button.addEventListener("touchcancel", function(event) {
        event.preventDefault();
        stopMovement();
    }, { passive: false });

}


// ------------------------------------------------------------
// Button setup
// ------------------------------------------------------------

setupMovementButton(
    "forward_button",
    "F"
);

setupMovementButton(
    "backward_button",
    "B"
);

setupMovementButton(
    "left_button",
    "L"
);

setupMovementButton(
    "right_button",
    "R"
);


// STOP button
document.getElementById("stop_button").addEventListener(
    "click",
    function() {
        stopMovement();
    }
);


// Safety stop when mouse is released anywhere
document.addEventListener("mouseup", function() {
    stopMovement();
});


// Safety stop when browser loses focus
window.addEventListener("blur", function() {
    stopMovement();
});


// Safety stop when page closes
window.addEventListener("beforeunload", function() {
    navigator.sendBeacon(
        "/motor_stop",
        new Blob(
            [JSON.stringify({
                command: "STOP",
                speed: 0
            })],
            { type: "application/json" }
        )
    );
});


// ------------------------------------------------------------
// Update lane information
// ------------------------------------------------------------

function updateStatus() {

    fetch("/status")
        .then(response => response.json())
        .then(data => {

            document.getElementById("lane_detected").innerText =
                data.lane_detected ? "YES" : "NO";

            document.getElementById("car_center_x").innerText =
                data.car_center_x + " px";

            document.getElementById("lane_center_x").innerText =
                data.lane_center_x + " px";

            document.getElementById("left_distance").innerText =
                data.left_distance + " px";

            document.getElementById("right_distance").innerText =
                data.right_distance + " px";

            document.getElementById("position_error").innerText =
                data.position_error + " px";

            document.getElementById("lane_angle").innerText =
                data.lane_angle + " deg";

            document.getElementById("current_command").innerText =
                data.current_command;

            document.getElementById("current_speed").innerText =
                data.current_speed;

        })
        .catch(error => {
            console.log("Status error:", error);
        });

}

setInterval(updateStatus, 300);

</script>

</body>

</html>
"""


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/motor", methods=["POST"])
def motor_control():

    data = request.get_json(silent=True) or {}

    command = str(
        data.get("command", "STOP")
    ).upper()

    speed = int(
        data.get("speed", 0)
    )

    if command not in ["F", "B", "L", "R", "STOP"]:
        command = "STOP"

    if command == "STOP":
        speed = 0

    serial_command = send_motor_command(
        command,
        speed
    )

    log_motor_command(
        command,
        speed
    )

    return jsonify({
        "success": True,
        "command": command,
        "speed": speed,
        "serial_command": serial_command
    })


@app.route("/motor_stop", methods=["POST"])
def motor_stop():

    send_motor_command("STOP", 0)
    log_motor_command("STOP", 0)

    return jsonify({
        "success": True,
        "command": "STOP"
    })


@app.route("/status")
def status():

    data = dict(latest_lane_data)

    data["current_command"] = current_command
    data["current_speed"] = current_speed

    return jsonify(data)


@app.route("/download_dataset")
def download_dataset():

    zip_path = create_dataset_zip()

    return send_file(
        zip_path,
        as_attachment=True,
        download_name="car4_cnn_dataset.zip"
    )


@app.route("/download_labels")
def download_labels():

    return send_file(
        DATASET_CSV,
        as_attachment=True,
        download_name="labels.csv"
    )


@app.route("/download_motor_log")
def download_motor_log():

    return send_file(
        MOTOR_LOG_CSV,
        as_attachment=True,
        download_name="motor_log.csv"
    )


# ============================================================
# SAFE SHUTDOWN
# ============================================================

def shutdown():

    global running

    running = False

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

    print("[INFO] Car4 stopped safely")


atexit.register(shutdown)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("======================================")
    print("       CAR4 LANE CONTROL SERVER       ")
    print("======================================")

    print(f"[INFO] Base directory: {BASE_DIR}")
    print(f"[INFO] Serial port: {SERIAL_PORT}")
    print(f"[INFO] Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"[INFO] Dataset folder: {IMAGE_FOLDER}")

    initialize_camera()
    connect_arduino()

    print("[INFO] Starting Flask server")
    print("[INFO] Open http://PI_IP_ADDRESS:5000")

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
        debug=False,
        use_reloader=False
    )
