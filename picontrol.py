from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    Response,
    send_file
)

import serial
import time
import threading
import cv2
import csv
import os

from datetime import datetime
from picamera2 import Picamera2


app = Flask(__name__)

# =====================================================
# CONFIGURATION
# =====================================================

ARDUINO_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

LOG_FOLDER = "logs"
LOG_FILE = os.path.join(LOG_FOLDER, "motor_log.csv")

IMAGE_FOLDER = "captured_images"

os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)


# =====================================================
# ARDUINO SERIAL
# =====================================================

arduino = serial.Serial(
    ARDUINO_PORT,
    BAUD_RATE,
    timeout=1
)

time.sleep(2)

serial_lock = threading.Lock()
log_lock = threading.Lock()


# =====================================================
# CAMERA SETUP
# =====================================================

camera = Picamera2()

camera_config = camera.create_video_configuration(
    main={
        "size": (640, 480),
        "format": "RGB888"
    }
)

camera.configure(camera_config)
camera.start()

time.sleep(2)


# =====================================================
# IMAGE CAPTURE FUNCTION
# PUT YOUR FUNCTION HERE
# =====================================================

def capture_command_image(
    direction,
    speed,
    left_pwm,
    right_pwm
):
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )[:-3]

    filename = (
        f"{timestamp}_"
        f"{direction}_"
        f"speed{speed}_"
        f"L{left_pwm}_"
        f"R{right_pwm}.jpg"
    )

    filepath = os.path.join(
        IMAGE_FOLDER,
        filename
    )

    frame = camera.capture_array()

    success, buffer = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 90]
    )

    if success:
        with open(filepath, "wb") as image_file:
            image_file.write(buffer.tobytes())

        return filename

    return ""


# =====================================================
# CAMERA STREAM
# =====================================================

def generate_frames():
    while True:
        frame = camera.capture_array()

        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# =====================================================
# CSV LOGGING FUNCTION
# =====================================================

def log_command(
    command,
    direction,
    speed_pwm,
    left_pwm,
    right_pwm,
    image_filename,
    response
):
    with log_lock:
        with open(LOG_FILE, "a", newline="") as file:
            writer = csv.writer(file)

            writer.writerow([
                datetime.now().isoformat(
                    timespec="milliseconds"
                ),
                command,
                direction,
                speed_pwm,
                left_pwm,
                right_pwm,
                image_filename,
                response
            ])


# =====================================================
# WEB ROUTES START HERE
# =====================================================

@app.route("/")
def index():
    return render_template("index.html")
