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
from flask import Flask, Response, jsonify, render_template_string, request, send_file
from picamera2 import Picamera2


# ============================================================
# CAR4 CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- Arduino ----------------
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

# ---------------- Camera ----------------
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAPTURE_INTERVAL = 0.5

# ---------------- Motor ----------------
MIN_SPEED = 0
MAX_SPEED = 255
DEFAULT_SPEED = 100

# ---------------- Lane Detection ----------------
ROI_START_RATIO = 0.50

BLACK_V_MIN = 0
BLACK_V_MAX = 95

BLACK_S_MIN = 0
BLACK_S_MAX = 255

MIN_CONTOUR_AREA = 100

CAR_CENTER_X = CAMERA_WIDTH // 2


# ============================================================
# DATA FOLDERS / FILES
# ============================================================

IMAGE_ROOT = os.path.join(
    BASE_DIR,
    "captured_images"
)

DATASET_CSV = os.path.join(
    BASE_DIR,
    "labels.csv"
)

MOTOR_LOG_CSV = os.path.join(
    BASE_DIR,
    "motor_log.csv"
)

DATASET_ZIP = os.path.join(
    BASE_DIR,
    "car4_cnn_dataset.zip"
)

IMAGES_ZIP = os.path.join(
    BASE_DIR,
    "car4_images.zip"
)

os.makedirs(
    IMAGE_ROOT,
    exist_ok=True
)


# ============================================================
# LOCKS
# ============================================================

zip_lock = threading.Lock()
data_lock = threading.Lock()
camera_lock = threading.Lock()
serial_lock = threading.Lock()


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "CAR4_SECRET_KEY",
    "car4-local-secret-change-me"
)


# ============================================================
# GLOBAL STATE
# ============================================================

camera = None
arduino = None

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

    "decision": "STOP",
}


# ============================================================
# CSV HEADERS
# ============================================================

CSV_HEADER = [
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
    "timestamp",
]


MOTOR_HEADER = [
    "command",
    "speed",
    "left_motor_pwm",
    "right_motor_pwm",
    "timestamp",
]


# ============================================================
# CSV INITIALIZATION
# ============================================================

def initialize_csv_files():

    if not os.path.exists(DATASET_CSV):

        with open(
            DATASET_CSV,
            "w",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                CSV_HEADER
            )


    if not os.path.exists(MOTOR_LOG_CSV):

        with open(
            MOTOR_LOG_CSV,
            "w",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                MOTOR_HEADER
            )


initialize_csv_files()


# ============================================================
# ARDUINO CONNECTION
# ============================================================

def connect_arduino():

    global arduino

    try:

        arduino = serial.Serial(
            SERIAL_PORT,
            BAUD_RATE,
            timeout=1
        )

        time.sleep(2)

        print(
            f"[INFO] Arduino connected: "
            f"{SERIAL_PORT}"
        )

    except Exception as error:

        arduino = None

        print(
            f"[WARNING] Arduino connection failed: "
            f"{error}"
        )

        print(
            "[WARNING] Camera/dashboard can still run."
        )


# ============================================================
# SEND MOTOR COMMAND
# ============================================================

def send_motor_command(
    command,
    speed=0
):

    global current_command
    global current_speed

    command = str(
        command
    ).upper()


    if command == "STOP":

        speed = 0


    try:

        speed = int(speed)

    except Exception:

        speed = 0


    speed = max(
        MIN_SPEED,
        min(MAX_SPEED, speed)
    )


    if command == "F":

        serial_command = (
            f"F,{speed}\n"
        )

    elif command == "B":

        serial_command = (
            f"B,{speed}\n"
        )

    elif command == "L":

        serial_command = (
            f"L,{speed}\n"
        )

    elif command == "R":

        serial_command = (
            f"R,{speed}\n"
        )

    else:

        command = "STOP"

        speed = 0

        serial_command = "STOP\n"


    with serial_lock:

        try:

            if (
                arduino is not None
                and arduino.is_open
            ):

                arduino.write(
                    serial_command.encode()
                )

                arduino.flush()

        except Exception as error:

            print(
                f"[WARNING] Serial error: "
                f"{error}"
            )


    current_command = command
    current_speed = speed

    return serial_command.strip()


# ============================================================
# MOTOR PWM VALUES
# ============================================================

def calculate_motor_values(
    command,
    speed
):

    command = command.upper()

    speed = int(speed)


    if command in ("F", "B"):

        return speed, speed


    if command == "L":

        return 0, speed


    if command == "R":

        return speed, 0


    return 0, 0


# ============================================================
# MOTOR LOGGING
# ============================================================

def log_motor_command(
    command,
    speed
):

    left_pwm, right_pwm = (
        calculate_motor_values(
            command,
            speed
        )
    )


    timestamp = datetime.now().isoformat(
        timespec="milliseconds"
    )


    row = [
        command,
        speed,
        left_pwm,
        right_pwm,
        timestamp
    ]


    try:

        with data_lock:

            with open(
                MOTOR_LOG_CSV,
                "a",
                newline=""
            ) as f:

                csv.writer(f).writerow(
                    row
                )

    except Exception as error:

        print(
            f"[WARNING] Motor log error: "
            f"{error}"
        )


# ============================================================
# CAMERA INITIALIZATION
# ============================================================

def initialize_camera():

    global camera

    try:

        camera = Picamera2()


        config = camera.create_video_configuration(

            main={
                "size": (
                    CAMERA_WIDTH,
                    CAMERA_HEIGHT
                ),
                "format": "RGB888"
            }

        )


        camera.configure(
            config
        )


        camera.start()

        time.sleep(2)


        print(
            "[INFO] OV5647 camera started"
        )


    except Exception as error:

        camera = None

        print(
            f"[ERROR] Camera initialization failed: "
            f"{error}"
        )


# ============================================================
# BLACK TAPE / LANE DETECTION
# ============================================================

def detect_black_lane(frame):

    height, width = frame.shape[:2]

    output = frame.copy()


    roi_start_y = int(
        height * ROI_START_RATIO
    )


    roi = frame[
        roi_start_y:height,
        :
    ]


    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_RGB2HSV
    )


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


    kernel = np.ones(
        (5, 5),
        np.uint8
    )


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


    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    valid = [

        c for c in contours

        if cv2.contourArea(c)
        >= MIN_CONTOUR_AREA

    ]


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


    # ROI line

    cv2.line(

        output,

        (0, roi_start_y),

        (width, roi_start_y),

        (255, 255, 0),

        2

    )


    if not valid:

        cv2.rectangle(

            output,

            (10, 10),

            (370, 60),

            (0, 0, 0),

            -1

        )


        cv2.putText(

            output,

            "LANE NOT DETECTED",

            (20, 45),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (255, 0, 0),

            2

        )


        return output, lane_data


    largest = max(
        valid,
        key=cv2.contourArea
    )


    x, y, w, h = cv2.boundingRect(
        largest
    )


    lane_left_x = x

    lane_right_x = x + w


    lane_center_x = int(
        (
            lane_left_x
            + lane_right_x
        ) / 2
    )


    position_error = (
        CAR_CENTER_X
        - lane_center_x
    )


    left_distance = max(
        0,
        CAR_CENTER_X
        - lane_left_x
    )


    right_distance = max(
        0,
        lane_right_x
        - CAR_CENTER_X
    )


    points = largest.reshape(
        -1,
        2
    )


    lane_angle = 0.0


    if len(points) >= 2:

        vx, vy, x0, y0 = cv2.fitLine(

            points,

            cv2.DIST_L2,

            0,

            0.01,

            0.01

        )


        vx = float(vx)
        vy = float(vy)


        if abs(vy) > 1e-4:

            lane_angle = float(

                round(

                    np.degrees(
                        np.arctan2(
                            vx,
                            vy
                        )
                    ),

                    2

                )

            )

        else:

            lane_angle = 90.0


    # Shift contour back to original image coordinates

    shifted = (
        largest
        + np.array([
            0,
            roi_start_y
        ])
    )


    cv2.drawContours(

        output,

        [shifted],

        -1,

        (0, 255, 0),

        3

    )


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


    cv2.line(

        output,

        (lane_center_x, roi_start_y),

        (lane_center_x, height),

        (0, 255, 255),

        3

    )


    cv2.line(

        output,

        (CAR_CENTER_X, roi_start_y),

        (CAR_CENTER_X, height),

        (255, 0, 255),

        3

    )


    cv2.circle(

        output,

        (
            CAR_CENTER_X,
            height - 25
        ),

        9,

        (255, 0, 255),

        -1

    )


    cv2.circle(

        output,

        (
            lane_center_x,
            height - 25
        ),

        8,

        (0, 255, 255),

        -1

    )


    cv2.line(

        output,

        (
            CAR_CENTER_X,
            height - 55
        ),

        (
            lane_center_x,
            height - 55
        ),

        (255, 255, 255),

        3

    )


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


    # Information box

    cv2.rectangle(

        output,

        (10, 10),

        (375, 195),

        (0, 0, 0),

        -1

    )


    info = [

        "LANE: DETECTED",

        f"Car4 center: {CAR_CENTER_X}px",

        f"Lane center: {lane_center_x}px",

        f"Left distance: {left_distance}px",

        f"Right distance: {right_distance}px",

        f"Position error: {position_error}px",

        f"Lane angle: {lane_angle:.2f} deg",

        f"Command: {current_command}",

        f"Speed: {current_speed}",

    ]


    y_text = 32


    for text in info:

        cv2.putText(

            output,

            text,

            (20, y_text),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.46,

            (255, 255, 255),

            1,

            cv2.LINE_AA

        )

        y_text += 20


    return output, lane_data


# ============================================================
# DIRECTION LABEL
# ============================================================

def get_direction_label(command):

    return {

        "F": "forward",

        "B": "backward",

        "L": "left",

        "R": "right",

        "STOP": "stop",

    }.get(
        command,
        "stop"
    )


# ============================================================
# DATASET IMAGE CAPTURE
# ============================================================

def capture_dataset_image(
    command,
    speed,
    frame,
    lane_data
):

    global image_counter


    direction = get_direction_label(
        command
    )


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

        IMAGE_ROOT,

        filename

    )


    # Save image

    ok = cv2.imwrite(

        image_path,

        cv2.cvtColor(

            frame,

            cv2.COLOR_RGB2BGR

        ),

        [
            cv2.IMWRITE_JPEG_QUALITY,
            90
        ]

    )


    if not ok:

        print(
            f"[WARNING] "
            f"Failed to save {filename}"
        )

        return


    left_pwm, right_pwm = (
        calculate_motor_values(
            command,
            speed
        )
    )


    timestamp_csv = datetime.now().isoformat(
        timespec="milliseconds"
    )


    row = [

        filename,

        direction,

        command,

        speed,

        int(
            lane_data[
                "lane_detected"
            ]
        ),

        lane_data[
            "car_center_x"
        ],

        lane_data[
            "lane_center_x"
        ],

        lane_data[
            "left_distance"
        ],

        lane_data[
            "right_distance"
        ],

        lane_data[
            "position_error"
        ],

        lane_data[
            "lane_angle"
        ],

        lane_data[
            "lane_left_x"
        ],

        lane_data[
            "lane_right_x"
        ],

        left_pwm,

        right_pwm,

        timestamp_csv,

    ]


    try:

        with data_lock:

            with open(

                DATASET_CSV,

                "a",

                newline=""

            ) as f:

                csv.writer(
                    f
                ).writerow(row)


    except Exception as error:

        print(
            f"[WARNING] CSV error: "
            f"{error}"
        )

        return


    print(

        f"[DATASET] "

        f"{filename} | "

        f"{direction} | "

        f"speed={speed} | "

        f"lane="
        f"{lane_data['lane_detected']}"

    )


# ============================================================
# CAMERA STREAM
# ============================================================

def generate_frames():

    global latest_lane_data
    global last_capture_time


    while running:

        if camera is None:

            time.sleep(0.2)

            continue


        try:

            with camera_lock:

                frame = camera.capture_array()


            overlay, lane_data = (
                detect_black_lane(frame)
            )


            latest_lane_data = lane_data


            now = time.time()


            if (

                now - last_capture_time

                >= CAPTURE_INTERVAL

            ):

                capture_dataset_image(

                    current_command,

                    current_speed,

                    frame,

                    lane_data

                )

                last_capture_time = now


            bgr = cv2.cvtColor(

                overlay,

                cv2.COLOR_RGB2BGR

            )


            success, encoded = cv2.imencode(

                ".jpg",

                bgr,

                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    85
                ]

            )


            if not success:

                continue


            yield (

                b"--frame\r\n"

                b"Content-Type: image/jpeg\r\n\r\n"

                + encoded.tobytes()

                + b"\r\n"

            )


        except Exception as error:

            print(

                f"[WARNING] "
                f"Camera stream error: "
                f"{error}"

            )

            time.sleep(0.2)


# ============================================================
# ZIP HELPER
# ============================================================

def _zip_add_if_exists(
    zf,
    path,
    arcname
):

    if os.path.isfile(path):

        zf.write(
            path,
            arcname
        )


# ============================================================
# CREATE COMPLETE DATASET ZIP
# ============================================================

def create_dataset_zip():

    temp_zip = (
        DATASET_ZIP
        + ".tmp"
    )


    print(
        "[ZIP] Creating complete dataset ZIP..."
    )


    with zip_lock:

        try:

            if os.path.exists(temp_zip):

                os.remove(temp_zip)


            count = 0


            with zipfile.ZipFile(

                temp_zip,

                "w",

                compression=zipfile.ZIP_STORED

            ) as zf:


                # --------------------------------------------
                # Images
                # --------------------------------------------

                if os.path.isdir(
                    IMAGE_ROOT
                ):

                    for filename in sorted(

                        os.listdir(
                            IMAGE_ROOT
                        )

                    ):

                        path = os.path.join(

                            IMAGE_ROOT,

                            filename

                        )


                        if os.path.isfile(
                            path
                        ):

                            zf.write(

                                path,

                                f"images/{filename}"

                            )

                            count += 1


                # --------------------------------------------
                # Labels
                # --------------------------------------------

                _zip_add_if_exists(

                    zf,

                    DATASET_CSV,

                    "labels.csv"

                )


                # --------------------------------------------
                # Motor log
                # --------------------------------------------

                _zip_add_if_exists(

                    zf,

                    MOTOR_LOG_CSV,

                    "motor_log.csv"

                )


            os.replace(

                temp_zip,

                DATASET_ZIP

            )


            size_mb = (

                os.path.getsize(
                    DATASET_ZIP
                )
                /
                (1024 * 1024)

            )


            print(

                f"[ZIP] Dataset ready: "

                f"{count} images, "

                f"{size_mb:.2f} MB"

            )


            return DATASET_ZIP


        except Exception:

            try:

                if os.path.exists(
                    temp_zip
                ):

                    os.remove(
                        temp_zip
                    )

            except Exception:

                pass


            raise


# ============================================================
# CREATE IMAGES-ONLY ZIP
# ============================================================

def create_images_zip():

    temp_zip = (
        IMAGES_ZIP
        + ".tmp"
    )


    print(
        "[ZIP] Creating images-only ZIP..."
    )


    with zip_lock:

        try:

            if os.path.exists(
                temp_zip
            ):

                os.remove(
                    temp_zip
                )


            count = 0


            with zipfile.ZipFile(

                temp_zip,

                "w",

                compression=zipfile.ZIP_STORED

            ) as zf:


                if os.path.isdir(
                    IMAGE_ROOT
                ):

                    for filename in sorted(

                        os.listdir(
                            IMAGE_ROOT
                        )

                    ):

                        path = os.path.join(

                            IMAGE_ROOT,

                            filename

                        )


                        if os.path.isfile(
                            path
                        ):

                            zf.write(

                                path,

                                filename

                            )

                            count += 1


            os.replace(

                temp_zip,

                IMAGES_ZIP

            )


            size_mb = (

                os.path.getsize(
                    IMAGES_ZIP
                )
                /
                (1024 * 1024)

            )


            print(

                f"[ZIP] Images ZIP ready: "

                f"{count} images, "

                f"{size_mb:.2f} MB"

            )


            return IMAGES_ZIP


        except Exception:

            try:

                if os.path.exists(
                    temp_zip
                ):

                    os.remove(
                        temp_zip
                    )

            except Exception:

                pass


            raise


# ============================================================
# CLEAR ALL DATA
# ============================================================

def clear_all_data():

    global image_counter
    global last_capture_time


    # Stop motors first

    send_motor_command(
        "STOP",
        0
    )


    with data_lock:


        # --------------------------------------------
        # Delete all images
        # --------------------------------------------

        if os.path.isdir(
            IMAGE_ROOT
        ):

            for filename in os.listdir(
                IMAGE_ROOT
            ):

                path = os.path.join(

                    IMAGE_ROOT,

                    filename

                )


                try:

                    if os.path.isfile(
                        path
                    ):

                        os.remove(
                            path
                        )

                except Exception as error:

                    print(

                        f"[CLEAR WARNING] "

                        f"{path}: "

                        f"{error}"

                    )


        # --------------------------------------------
        # Delete CSVs and ZIPs
        # --------------------------------------------

        for path in [

            DATASET_CSV,

            MOTOR_LOG_CSV,

            DATASET_ZIP,

            IMAGES_ZIP

        ]:

            try:

                if os.path.isfile(
                    path
                ):

                    os.remove(
                        path
                    )

            except Exception as error:

                print(

                    f"[CLEAR WARNING] "

                    f"{path}: "

                    f"{error}"

                )


        # --------------------------------------------
        # Recreate CSV files
        # --------------------------------------------

        initialize_csv_files()


        # --------------------------------------------
        # Reset counter
        # --------------------------------------------

        image_counter = 0

        last_capture_time = 0.0


    print(
        "[CLEAR] All images, CSVs and ZIPs removed."
    )


# ============================================================
# HTML DASHBOARD
# ============================================================

HTML_PAGE = r"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>Car4 Lane Control</title>


<style>

* {
    box-sizing: border-box;
}


body {

    margin: 0;

    padding: 20px;

    background: #111;

    color: white;

    font-family: Arial, sans-serif;

    text-align: center;

}


.container {

    max-width: 1000px;

    margin: auto;

}


h1 {

    margin-top: 0;

    font-size: 30px;

}


.camera-box,
.status,
.download-panel,
.data-panel {

    background: #222;

    padding: 15px;

    border-radius: 12px;

    margin-bottom: 15px;

}


.camera-box img {

    width: 100%;

    max-width: 640px;

    border-radius: 8px;

    display: block;

    margin: auto;

}


.status-grid {

    display: grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap: 10px;

}


.status-item {

    background: #333;

    padding: 12px;

    border-radius: 8px;

    font-size: 16px;

}


.status-item strong {

    display: block;

    margin-top: 5px;

    font-size: 19px;

}


.control-panel {

    margin: 20px auto;

    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 12px;

    max-width: 500px;

}


button {

    min-height: 70px;

    border: 0;

    border-radius: 12px;

    font-size: 19px;

    font-weight: bold;

    color: white;

    cursor: pointer;

    user-select: none;

    touch-action: none;

}


button:active {

    transform: scale(.95);

}


.forward {

    background: #27ae60;

}


.left,
.right {

    background: #3498db;

}


.backward {

    background: #e67e22;

}


.stop {

    background: #e74c3c;

}


.clear {

    background: #8e44ad;

    min-height: 55px;

}


.speed-panel {

    margin: 20px 0;

}


.speed-panel input {

    width: 80%;

    max-width: 700px;

}


.speed-number {

    font-size: 28px;

    font-weight: bold;

}


.download-panel a {

    display: inline-block;

    margin: 6px;

    padding: 13px 18px;

    background: #555;

    color: white;

    text-decoration: none;

    border-radius: 8px;

    font-size: 15px;

}


.download-panel a:hover {

    background: #777;

}


.warning {

    color: #f1c40f;

    margin: 15px 0;

    font-size: 17px;

}


.dataset-info {

    color: #2ecc71;

    font-size: 16px;

}


@media(max-width:600px) {

    body {

        padding: 10px;

    }


    .status-grid {

        grid-template-columns: 1fr;

    }


    button {

        min-height: 60px;

        font-size: 17px;

    }


    .download-panel a {

        width: 90%;

    }

}

</style>

</head>


<body>


<div class="container">


<h1>
    Car4 Lane Control Dashboard
</h1>


<div class="data-panel">

<h2>
    Dataset
</h2>

<p class="dataset-info">

All images are saved into one
<strong>captured_images</strong>
folder.

</p>

<p>

All labels are saved into
<strong>labels.csv</strong>.

</p>

<p>

All motor commands are saved into
<strong>motor_log.csv</strong>.

</p>

<button
    class="clear"
    id="clear_button"
>
    CLEAR ALL DATA
</button>

</div>


<div class="camera-box">

<img
    src="/video_feed"
    alt="Car4 Camera"
>

</div>


<div class="status">

<h2>
    Lane Information
</h2>


<div class="status-grid">


<div class="status-item">

Lane detected

<strong id="lane_detected">
---
</strong>

</div>


<div class="status-item">

Car4 center

<strong id="car_center_x">
---
</strong>

</div>


<div class="status-item">

Lane center

<strong id="lane_center_x">
---
</strong>

</div>


<div class="status-item">

Left distance

<strong id="left_distance">
---
</strong>

</div>


<div class="status-item">

Right distance

<strong id="right_distance">
---
</strong>

</div>


<div class="status-item">

Position error

<strong id="position_error">
---
</strong>

</div>


<div class="status-item">

Lane angle

<strong id="lane_angle">
---
</strong>

</div>


<div class="status-item">

Command

<strong id="current_command">
STOP
</strong>

</div>


<div class="status-item">

Speed

<strong id="current_speed">
0
</strong>

</div>


</div>

</div>


<div class="speed-panel">

<h2>
    Speed
</h2>


<div class="speed-number">

<span id="speed_value">
100
</span>

</div>


<input
    type="range"
    id="speed_slider"
    min="0"
    max="255"
    value="100"
>


</div>


<div class="control-panel">


<div></div>


<button
    id="forward_button"
    class="forward"
>
    FORWARD
</button>


<div></div>


<button
    id="left_button"
    class="left"
>
    LEFT
</button>


<button
    id="stop_button"
    class="stop"
>
    STOP
</button>


<button
    id="right_button"
    class="right"
>
    RIGHT
</button>


<div></div>


<button
    id="backward_button"
    class="backward"
>
    BACKWARD
</button>


<div></div>


</div>


<div class="warning">

Press and hold a movement button.
Release it to STOP Car4.

</div>


<div class="download-panel">

<h2>
    Dataset Downloads
</h2>


<a href="/download_dataset">

Download Complete Dataset ZIP

</a>


<a href="/download_images">

Download Images ZIP

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

let activeCommand = null;


const slider =
    document.getElementById(
        'speed_slider'
    );


const speedValue =
    document.getElementById(
        'speed_value'
    );


slider.addEventListener(
    'input',
    () => {

        selectedSpeed =
            parseInt(
                slider.value
            );

        speedValue.innerText =
            selectedSpeed;

    }
);


function sendCommand(
    command,
    speed
) {

    fetch(
        '/motor',
        {

            method: 'POST',

            headers: {
                'Content-Type':
                    'application/json'
            },

            body: JSON.stringify({

                command: command,

                speed: speed

            })

        }

    ).catch(
        e => console.log(e)
    );

}


function startMovement(
    command
) {

    if (
        movementActive
        &&
        activeCommand === command
    ) {

        return;

    }


    movementActive = true;

    activeCommand = command;


    sendCommand(
        command,
        selectedSpeed
    );

}


function stopMovement() {

    if (
        !movementActive
        &&
        activeCommand === null
    ) {

        sendCommand(
            'STOP',
            0
        );

        return;

    }


    movementActive = false;

    activeCommand = null;


    sendCommand(
        'STOP',
        0
    );

}


function setupMovementButton(
    id,
    command
) {

    const button =
        document.getElementById(id);


    button.addEventListener(
        'mousedown',
        e => {

            e.preventDefault();

            startMovement(
                command
            );

        }
    );


    button.addEventListener(
        'mouseup',
        e => {

            e.preventDefault();

            stopMovement();

        }
    );


    button.addEventListener(
        'mouseleave',
        () => {

            stopMovement();

        }
    );


    button.addEventListener(
        'touchstart',
        e => {

            e.preventDefault();

            startMovement(
                command
            );

        },
        {
            passive: false
        }
    );


    button.addEventListener(
        'touchend',
        e => {

            e.preventDefault();

            stopMovement();

        },
        {
            passive: false
        }
    );


    button.addEventListener(
        'touchcancel',
        e => {

            e.preventDefault();

            stopMovement();

        },
        {
            passive: false
        }
    );

}


setupMovementButton(
    'forward_button',
    'F'
);


setupMovementButton(
    'left_button',
    'L'
);


setupMovementButton(
    'right_button',
    'R'
);


setupMovementButton(
    'backward_button',
    'B'
);


document
    .getElementById(
        'stop_button'
    )
    .addEventListener(
        'click',
        () => {

            movementActive = false;

            activeCommand = null;

            sendCommand(
                'STOP',
                0
            );

        }
    );


document.addEventListener(
    'mouseup',
    () => {

        if (movementActive) {

            stopMovement();

        }

    }
);


window.addEventListener(
    'blur',
    () => {

        movementActive = false;

        activeCommand = null;

        sendCommand(
            'STOP',
            0
        );

    }
);


window.addEventListener(
    'beforeunload',
    () => {

        navigator.sendBeacon(

            '/motor_stop',

            new Blob(

                [
                    JSON.stringify({

                        command: 'STOP',

                        speed: 0

                    })

                ],

                {
                    type:
                        'application/json'
                }

            )

        );

    }
);


function updateStatus() {

    fetch('/status')

        .then(
            r => r.json()
        )

        .then(
            d => {

                document
                    .getElementById(
                        'lane_detected'
                    )
                    .innerText =
                    d.lane_detected
                    ? 'YES'
                    : 'NO';


                document
                    .getElementById(
                        'car_center_x'
                    )
                    .innerText =
                    d.car_center_x
                    + ' px';


                document
                    .getElementById(
                        'lane_center_x'
                    )
                    .innerText =
                    d.lane_center_x
                    + ' px';


                document
                    .getElementById(
                        'left_distance'
                    )
                    .innerText =
                    d.left_distance
                    + ' px';


                document
                    .getElementById(
                        'right_distance'
                    )
                    .innerText =
                    d.right_distance
                    + ' px';


                document
                    .getElementById(
                        'position_error'
                    )
                    .innerText =
                    d.position_error
                    + ' px';


                document
                    .getElementById(
                        'lane_angle'
                    )
                    .innerText =
                    d.lane_angle
                    + ' deg';


                document
                    .getElementById(
                        'current_command'
                    )
                    .innerText =
                    d.current_command;


                document
                    .getElementById(
                        'current_speed'
                    )
                    .innerText =
                    d.current_speed;

            }
        )

        .catch(
            e => console.log(e)
        );

}


setInterval(
    updateStatus,
    300
);


updateStatus();


document
    .getElementById(
        'clear_button'
    )
    .addEventListener(
        'click',
        () => {

            if (
                !confirm(
                    'Delete ALL images, CSV files and ZIP files? This cannot be undone.'
                )
            ) {

                return;

            }


            fetch(
                '/clear_data',
                {
                    method: 'POST'
                }
            )

            .then(
                r => r.json()
            )

            .then(
                d => {

                    if (d.success) {

                        alert(
                            'All dataset data has been cleared.'
                        );

                    } else {

                        alert(
                            d.error
                            ||
                            'Clear failed.'
                        );

                    }

                }
            )

            .catch(
                e => alert(e)
            );

        }
    );


</script>


</body>

</html>
"""


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template_string(
        HTML_PAGE
    )


# ============================================================
# VIDEO FEED
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(

        generate_frames(),

        mimetype=
        "multipart/x-mixed-replace; boundary=frame"

    )


# ============================================================
# MOTOR CONTROL
# ============================================================

@app.route(
    "/motor",
    methods=["POST"]
)
def motor_control():

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )


    command = str(

        data.get(
            "command",
            "STOP"
        )

    ).upper()


    try:

        speed = int(

            data.get(
                "speed",
                0
            )

        )

    except Exception:

        speed = 0


    if command not in [
        "F",
        "B",
        "L",
        "R",
        "STOP"
    ]:

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

        "serial_command":
            serial_command

    })


# ============================================================
# EMERGENCY STOP
# ============================================================

@app.route(
    "/motor_stop",
    methods=["POST"]
)
def motor_stop():

    send_motor_command(
        "STOP",
        0
    )


    log_motor_command(
        "STOP",
        0
    )


    return jsonify({

        "success": True,

        "command": "STOP"

    })


# ============================================================
# STATUS
# ============================================================

@app.route("/status")
def status():

    data = dict(
        latest_lane_data
    )


    data[
        "current_command"
    ] = current_command


    data[
        "current_speed"
    ] = current_speed


    # Count images

    try:

        image_count = len([

            f

            for f in os.listdir(
                IMAGE_ROOT
            )

            if f.lower().endswith(
                ".jpg"
            )

        ])

    except Exception:

        image_count = 0


    data[
        "image_count"
    ] = image_count


    return jsonify(
        data
    )


# ============================================================
# CLEAR ALL DATA
# ============================================================

@app.route(
    "/clear_data",
    methods=["POST"]
)
def clear_data_route():

    try:

        clear_all_data()


        return jsonify({

            "success": True

        })


    except Exception as error:

        print(
            f"[CLEAR ERROR] "
            f"{error}"
        )


        return jsonify({

            "success": False,

            "error": str(error)

        }), 500


# ============================================================
# DOWNLOAD COMPLETE DATASET
# ============================================================

@app.route(
    "/download_dataset"
)
def download_dataset():

    try:

        zip_path = (
            create_dataset_zip()
        )


        return send_file(

            zip_path,

            mimetype=
            "application/zip",

            as_attachment=True,

            download_name=
            "car4_cnn_dataset.zip",

            max_age=0

        )


    except Exception as error:

        print(
            f"[DOWNLOAD ERROR] "
            f"{error}"
        )


        return (

            "<h2>"
            "Dataset download failed"
            "</h2>"

            f"<p>{error}</p>"

        ), 500


# ============================================================
# DOWNLOAD IMAGES ONLY
# ============================================================

@app.route(
    "/download_images"
)
def download_images():

    try:

        zip_path = (
            create_images_zip()
        )


        return send_file(

            zip_path,

            mimetype=
            "application/zip",

            as_attachment=True,

            download_name=
            "car4_images.zip",

            max_age=0

        )


    except Exception as error:

        print(
            f"[IMAGE DOWNLOAD ERROR] "
            f"{error}"
        )


        return (

            "<h2>"
            "Image download failed"
            "</h2>"

            f"<p>{error}</p>"

        ), 500


# ============================================================
# DOWNLOAD LABELS
# ============================================================

@app.route(
    "/download_labels"
)
def download_labels():

    initialize_csv_files()


    return send_file(

        DATASET_CSV,

        mimetype="text/csv",

        as_attachment=True,

        download_name="labels.csv",

        max_age=0

    )


# ============================================================
# DOWNLOAD MOTOR LOG
# ============================================================

@app.route(
    "/download_motor_log"
)
def download_motor_log():

    initialize_csv_files()


    return send_file(

        MOTOR_LOG_CSV,

        mimetype="text/csv",

        as_attachment=True,

        download_name="motor_log.csv",

        max_age=0

    )


# ============================================================
# SAFE SHUTDOWN
# ============================================================

def shutdown():

    global running


    running = False


    print(
        "[INFO] "
        "Shutting down Car4..."
    )


    # Stop motors

    try:

        send_motor_command(
            "STOP",
            0
        )

    except Exception:

        pass


    # Stop camera

    try:

        if camera is not None:

            camera.stop()

    except Exception:

        pass


    # Close Arduino

    try:

        if (
            arduino is not None
            and arduino.is_open
        ):

            arduino.close()

    except Exception:

        pass


    print(
        "[INFO] "
        "Car4 stopped safely."
    )


atexit.register(
    shutdown
)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":


    print()
    print(
        "=========================================="
    )
    print(
        "        CAR4 LANE CONTROL SYSTEM"
    )
    print(
        "=========================================="
    )


    print(
        f"[INFO] Project: "
        f"{BASE_DIR}"
    )


    print(
        f"[INFO] Camera: "
        f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}"
    )


    print(
        f"[INFO] Arduino: "
        f"{SERIAL_PORT}"
    )


    print(
        f"[INFO] Images: "
        f"{IMAGE_ROOT}"
    )


    print(
        f"[INFO] Labels: "
        f"{DATASET_CSV}"
    )


    print(
        f"[INFO] Motor log: "
        f"{MOTOR_LOG_CSV}"
    )


    print()


    # Start camera

    initialize_camera()


    # Connect Arduino

    connect_arduino()


    print()
    print(
        "[INFO] Dashboard starting..."
    )


    print(
        "[INFO] Open from another device:"
    )


    print(
        "       http://PI_IP_ADDRESS:5000"
    )


    print()


    # Start Flask

    app.run(

        host="0.0.0.0",

        port=5000,

        threaded=True,

        debug=False,

        use_reloader=False

    )
