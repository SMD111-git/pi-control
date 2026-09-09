import os
import csv
import time
import atexit
import threading
import zipfile
import serial

from datetime import datetime

from flask import (
    Flask,
    Response,
    jsonify,
    render_template_string,
    request,
    send_file
)

from picamera2 import Picamera2
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Capture at most one image every 0.5 seconds
CAPTURE_INTERVAL = 0.5

# ============================================================
# FILES AND FOLDERS
# ============================================================

IMAGE_FOLDER = os.path.join(
    BASE_DIR,
    "captured_images"
)

LOG_FOLDER = os.path.join(
    BASE_DIR,
    "logs"
)

DATASET_CSV = os.path.join(
    BASE_DIR,
    "labels.csv"
)

MOTOR_LOG_CSV = os.path.join(
    LOG_FOLDER,
    "motor_log.csv"
)

DATASET_ZIP = os.path.join(
    BASE_DIR,
    "car4_cnn_dataset.zip"
)

os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# GLOBAL VARIABLES
# ============================================================

arduino = None
camera = None

camera_lock = threading.Lock()
serial_lock = threading.Lock()
csv_lock = threading.Lock()

last_capture_time = 0
image_counter = 0

last_command = "STOP"
last_speed = 0


# ============================================================
# LABELS
# ============================================================

def get_direction_label(command):
    """
    Convert Arduino command to CNN label.
    """

    labels = {
        "F": "forward",
        "B": "backward",
        "L": "left",
        "R": "right",
        "STOP": "stop"
    }

    return labels.get(command.upper(), "unknown")


# ============================================================
# IMAGE COUNTER
# ============================================================

def initialize_image_counter():
    """
    Continue numbering from existing images.
    """

    global image_counter

    numbers = []

    for filename in os.listdir(IMAGE_FOLDER):

        try:

            number = int(
                filename.split("_")[0]
            )

            numbers.append(number)

        except ValueError:

            continue

    if numbers:

        image_counter = max(numbers)

    else:

        image_counter = 0

    print(
        "Next image number:",
        image_counter + 1
    )


# ============================================================
# CSV INITIALIZATION
# ============================================================

def create_csv_files():
    """
    Create CSV files and headers.
    """

    if not os.path.exists(DATASET_CSV):

        with open(
            DATASET_CSV,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "image_filename",
                "direction",
                "command",
                "speed",
                "left_motor_pwm",
                "right_motor_pwm",
                "timestamp"
            ])

    if not os.path.exists(MOTOR_LOG_CSV):

        with open(
            MOTOR_LOG_CSV,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "timestamp",
                "command",
                "direction",
                "speed",
                "left_motor_pwm",
                "right_motor_pwm",
                "image_filename"
            ])

    print("CSV files ready")


# ============================================================
# ARDUINO CONNECTION
# ============================================================

def connect_arduino():
    """
    Connect Raspberry Pi to Arduino.
    """

    global arduino

    try:

        arduino = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            timeout=1
        )

        time.sleep(2)

        print("-----------------------------------")
        print("Arduino connected")
        print("Port:", SERIAL_PORT)
        print("-----------------------------------")

    except Exception as error:

        arduino = None

        print("-----------------------------------")
        print("Arduino connection failed")
        print("Error:", error)
        print("-----------------------------------")


def send_to_arduino(command):
    """
    Send command to Arduino.

    Examples:
        F,100
        B,100
        L,100
        R,100
        STOP
    """

    if arduino is None:

        print("Arduino is not connected")

        return False

    try:

        with serial_lock:

            arduino.write(
                (command + "\n").encode("utf-8")
            )

            arduino.flush()

        print("Sent:", command)

        return True

    except Exception as error:

        print("Serial error:", error)

        return False


# ============================================================
# CAMERA
# ============================================================

def setup_camera():
    """
    Initialize OV5647 camera.
    """

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

        camera.configure(config)

        camera.start()

        time.sleep(2)

        print("-----------------------------------")
        print("Camera started")
        print(
            f"Resolution: "
            f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}"
        )
        print("-----------------------------------")

    except Exception as error:

        camera = None

        print("Camera error:", error)


# ============================================================
# IMAGE CAPTURE AND CNN LABELING
# ============================================================

def capture_command_image(
    command,
    speed,
    left_motor_pwm,
    right_motor_pwm
):
    """
    Capture an image and save its CNN label.

    Example filename:

    000001_forward_speed_100_20260908_123001.jpg
    """

    global last_capture_time
    global image_counter

    if camera is None:

        print("Camera unavailable")

        return ""

    current_time = time.time()

    # Prevent excessive image capture
    if (
        current_time - last_capture_time
        < CAPTURE_INTERVAL
    ):

        return ""

    try:

        with camera_lock:

            frame = camera.capture_array()

            image_counter += 1

            direction = get_direction_label(
                command
            )

            timestamp = datetime.now()

            timestamp_text = timestamp.strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )

            filename_timestamp = timestamp.strftime(
                "%Y%m%d_%H%M%S_%f"
            )

            filename = (
                f"{image_counter:06d}_"
                f"{direction}_"
                f"speed_{speed}_"
                f"{filename_timestamp}.jpg"
            )

            filepath = os.path.join(
                IMAGE_FOLDER,
                filename
            )

            # Save image
            image = Image.fromarray(frame)

            image.save(
                filepath,
                format="JPEG",
                quality=90
            )

            # Save CNN label information
            with csv_lock:

                with open(
                    DATASET_CSV,
                    "a",
                    newline=""
                ) as file:

                    writer = csv.writer(file)

                    writer.writerow([
                        filename,
                        direction,
                        command,
                        speed,
                        left_motor_pwm,
                        right_motor_pwm,
                        timestamp_text
                    ])

            last_capture_time = current_time

            print(
                f"IMAGE: {filename} | "
                f"LABEL: {direction} | "
                f"SPEED: {speed}"
            )

            return filename

    except Exception as error:

        print("Image capture error:", error)

        return ""


# ============================================================
# LIVE CAMERA STREAM
# ============================================================

def generate_frames():
    """
    Generate live camera stream.
    """

    if camera is None:

        return

    import cv2

    while True:

        try:

            with camera_lock:

                frame = camera.capture_array()

            frame_bgr = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR
            )

            success, encoded = cv2.imencode(
                ".jpg",
                frame_bgr
            )

            if not success:

                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + encoded.tobytes()
                + b"\r\n"
            )

            time.sleep(0.03)

        except Exception as error:

            print("Stream error:", error)

            time.sleep(0.2)


# ============================================================
# MOTOR PWM
# ============================================================

def calculate_motor_values(command, speed):
    """
    Calculate left and right PWM values.
    """

    speed = max(
        0,
        min(255, int(speed))
    )

    if command == "F":

        return speed, speed

    elif command == "B":

        return speed, speed

    elif command == "L":

        return 0, speed

    elif command == "R":

        return speed, 0

    else:

        return 0, 0


# ============================================================
# MOTOR CSV LOGGING
# ============================================================

def log_motor_command(
    command,
    speed,
    left_motor_pwm,
    right_motor_pwm,
    image_filename=""
):
    """
    Save motor command information.
    """

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )

    direction = get_direction_label(
        command
    )

    with csv_lock:

        with open(
            MOTOR_LOG_CSV,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                timestamp,
                command,
                direction,
                speed,
                left_motor_pwm,
                right_motor_pwm,
                image_filename
            ])


# ============================================================
# DASHBOARD HTML
# ============================================================

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <title>Car4 CNN Dataset Collection</title>

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
            margin-bottom: 20px;
        }

        .camera-box {
            width: 100%;
            max-width: 640px;
            margin: auto;
        }

        .camera-box img {
            width: 100%;
            border: 3px solid #444;
            border-radius: 10px;
        }

        .status {
            margin: 15px auto;
            padding: 15px;
            max-width: 640px;
            background: #222;
            border-radius: 8px;
            font-size: 18px;
            line-height: 1.8;
        }

        .speed-container {
            max-width: 640px;
            margin: 20px auto;
        }

        input[type="range"] {
            width: 90%;
        }

        .controls {
            display: grid;
            grid-template-columns: repeat(3, 100px);
            justify-content: center;
            gap: 12px;
            margin-top: 20px;
        }

        button {
            height: 70px;
            font-size: 24px;
            font-weight: bold;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            background: #333;
            color: white;
            user-select: none;
            -webkit-user-select: none;
            touch-action: none;
        }

        button:active {
            background: #777;
        }

        .stop-button {
            background: #b00020;
        }

        .stop-button:active {
            background: #e00030;
        }

        .download-button {
            height: auto;
            margin: 8px;
            padding: 15px 20px;
            background: #087f23;
            font-size: 16px;
        }

        .download-button:hover {
            background: #0ca832;
        }

        .empty {
            visibility: hidden;
        }

        .info {
            margin-top: 25px;
            color: #aaa;
            font-size: 14px;
        }

    </style>

</head>

<body>

    <h1>Car4 CNN Dataset Collection</h1>

    <div class="camera-box">

        <img
            src="/video_feed"
            alt="Live Camera"
        >

    </div>

    <div class="status">

        Direction:
        <strong id="direction">stop</strong>

        <br>

        Command:
        <strong id="command">STOP</strong>

        <br>

        Speed:
        <strong id="speedValue">0</strong>

        <br>

        Last Image:
        <strong id="lastImage">None</strong>

    </div>

    <div class="speed-container">

        <label for="speed">
            Motor Speed
        </label>

        <br><br>

        <input
            type="range"
            id="speed"
            min="0"
            max="255"
            value="100"
            oninput="updateSpeed()"
        >

        <br>

        <span id="sliderValue">100</span>

    </div>

    <div class="controls">

        <button class="empty"></button>

        <button
            id="forwardButton"
            data-command="F">
            ▲
        </button>

        <button class="empty"></button>

        <button
            id="leftButton"
            data-command="L">
            ◀
        </button>

        <button
            id="stopButton"
            class="stop-button">
            ■
        </button>

        <button
            id="rightButton"
            data-command="R">
            ▶
        </button>

        <button class="empty"></button>

        <button
            id="backwardButton"
            data-command="B">
            ▼
        </button>

        <button class="empty"></button>

    </div>

    <br>

    <h2>Download Dataset</h2>

    <button
        class="download-button"
        onclick="downloadDataset()">
        Download Images + Labels ZIP
    </button>

    <button
        class="download-button"
        onclick="downloadImages()">
        Download Images ZIP
    </button>

    <button
        class="download-button"
        onclick="downloadLabels()">
        Download labels.csv
    </button>

    <div class="info">

        Press and hold a movement button.
        Release it to stop the car.

    </div>


    <script>

        let currentCommand = "STOP";
        let commandTimer = null;
        let buttonIsPressed = false;


        // ====================================================
        // SPEED
        // ====================================================

        function updateSpeed() {

            let speed = document.getElementById(
                "speed"
            ).value;

            document.getElementById(
                "sliderValue"
            ).innerText = speed;

            document.getElementById(
                "speedValue"
            ).innerText = speed;

        }


        // ====================================================
        // SEND COMMAND
        // ====================================================

        function sendCommandToServer(
            command,
            speed
        ) {

            fetch(
                "/command?cmd=" +
                encodeURIComponent(command) +
                "&speed=" +
                encodeURIComponent(speed)
            )

            .then(
                response => response.json()
            )

            .then(
                data => {

                    if (data.success) {

                        document.getElementById(
                            "direction"
                        ).innerText =
                            data.direction;

                        document.getElementById(
                            "command"
                        ).innerText =
                            data.command;

                        document.getElementById(
                            "speedValue"
                        ).innerText =
                            data.speed;

                        if (
                            data.image_filename !== ""
                        ) {

                            document.getElementById(
                                "lastImage"
                            ).innerText =
                                data.image_filename;

                        }

                    }

                }
            )

            .catch(
                error => {
                    console.log(
                        "Connection error:",
                        error
                    );
                }
            );

        }


        // ====================================================
        // START MOVEMENT
        // ====================================================

        function startMovement(command) {

            if (buttonIsPressed) {

                return;

            }

            buttonIsPressed = true;

            currentCommand = command;

            let speed = document.getElementById(
                "speed"
            ).value;

            document.getElementById(
                "command"
            ).innerText = command;

            document.getElementById(
                "speedValue"
            ).innerText = speed;

            // Send immediately
            sendCommandToServer(
                command,
                speed
            );

            // Keep sending while button is held
            commandTimer = setInterval(
                function() {

                    if (!buttonIsPressed) {

                        return;

                    }

                    let currentSpeed =
                        document.getElementById(
                            "speed"
                        ).value;

                    sendCommandToServer(
                        currentCommand,
                        currentSpeed
                    );

                },
                200
            );

        }


        // ====================================================
        // STOP MOVEMENT
        // ====================================================

        function stopMovement() {

            buttonIsPressed = false;

            currentCommand = "STOP";

            if (commandTimer !== null) {

                clearInterval(commandTimer);

                commandTimer = null;

            }

            document.getElementById(
                "command"
            ).innerText = "STOP";

            document.getElementById(
                "direction"
            ).innerText = "stop";

            document.getElementById(
                "speedValue"
            ).innerText = "0";

            sendCommandToServer(
                "STOP",
                0
            );

        }


        // ====================================================
        // MOVEMENT BUTTON SETUP
        // ====================================================

        function setupMovementButton(
            buttonId,
            command
        ) {

            const button = document.getElementById(
                buttonId
            );


            // Mouse press
            button.addEventListener(
                "mousedown",
                function(event) {

                    event.preventDefault();

                    startMovement(command);

                }
            );


            // Mouse release
            button.addEventListener(
                "mouseup",
                function(event) {

                    event.preventDefault();

                    stopMovement();

                }
            );


            // Mouse leaves button
            button.addEventListener(
                "mouseleave",
                function() {

                    if (buttonIsPressed) {

                        stopMovement();

                    }

                }
            );


            // Touch press
            button.addEventListener(
                "touchstart",
                function(event) {

                    event.preventDefault();

                    startMovement(command);

                },
                {
                    passive: false
                }
            );


            // Touch release
            button.addEventListener(
                "touchend",
                function(event) {

                    event.preventDefault();

                    stopMovement();

                },
                {
                    passive: false
                }
            );


            // Touch cancelled
            button.addEventListener(
                "touchcancel",
                function(event) {

                    event.preventDefault();

                    stopMovement();

                },
                {
                    passive: false
                }
            );

        }


        // ====================================================
        // INITIALIZE BUTTONS
        // ====================================================

        setupMovementButton(
            "forwardButton",
            "F"
        );

        setupMovementButton(
            "backwardButton",
            "B"
        );

        setupMovementButton(
            "leftButton",
            "L"
        );

        setupMovementButton(
            "rightButton",
            "R"
        );


        // ====================================================
        // STOP BUTTON
        // ====================================================

        document.getElementById(
            "stopButton"
        ).addEventListener(
            "click",
            function() {

                stopMovement();

            }
        );


        // ====================================================
        // SAFETY STOP
        // ====================================================

        document.addEventListener(
            "mouseup",
            function() {

                if (buttonIsPressed) {

                    stopMovement();

                }

            }
        );

        window.addEventListener(
            "blur",
            function() {

                if (buttonIsPressed) {

                    stopMovement();

                }

            }
        );

        window.addEventListener(
            "beforeunload",
            function() {

                sendCommandToServer(
                    "STOP",
                    0
                );

            }
        );


        // ====================================================
        // DOWNLOAD
        // ====================================================

        function downloadDataset() {

            window.location.href =
                "/download_dataset";

        }

        function downloadImages() {

            window.location.href =
                "/download_images";

        }

        function downloadLabels() {

            window.location.href =
                "/download_labels";

        }


        updateSpeed();

    </script>

</body>

</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template_string(
        HTML_PAGE
    )


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        )
    )


@app.route("/command")
def command():

    global last_command
    global last_speed

    cmd = request.args.get(
        "cmd",
        "STOP"
    ).upper()

    try:

        speed = int(
            request.args.get(
                "speed",
                0
            )
        )

    except ValueError:

        speed = 0

    valid_commands = [
        "F",
        "B",
        "L",
        "R",
        "STOP"
    ]

    if cmd not in valid_commands:

        return jsonify({
            "success": False,
            "error": "Invalid command"
        }), 400

    speed = max(
        0,
        min(255, speed)
    )

    if cmd == "STOP":

        speed = 0

    left_pwm, right_pwm = calculate_motor_values(
        cmd,
        speed
    )

    if cmd == "STOP":

        arduino_command = "STOP"

    else:

        arduino_command = f"{cmd},{speed}"

    success = send_to_arduino(
        arduino_command
    )

    image_filename = capture_command_image(
        command=cmd,
        speed=speed,
        left_motor_pwm=left_pwm,
        right_motor_pwm=right_pwm
    )

    log_motor_command(
        command=cmd,
        speed=speed,
        left_motor_pwm=left_pwm,
        right_motor_pwm=right_pwm,
        image_filename=image_filename
    )

    last_command = cmd
    last_speed = speed

    return jsonify({
        "success": success,
        "command": cmd,
        "direction": get_direction_label(cmd),
        "speed": speed,
        "left_motor_pwm": left_pwm,
        "right_motor_pwm": right_pwm,
        "image_filename": image_filename
    })


@app.route("/stop")
def stop():

    success = send_to_arduino(
        "STOP"
    )

    log_motor_command(
        command="STOP",
        speed=0,
        left_motor_pwm=0,
        right_motor_pwm=0,
        image_filename=""
    )

    return jsonify({
        "success": success,
        "command": "STOP",
        "direction": "stop"
    })


@app.route("/status")
def status():

    return jsonify({
        "arduino_connected": arduino is not None,
        "camera_connected": camera is not None,
        "last_command": last_command,
        "last_speed": last_speed,
        "image_folder": IMAGE_FOLDER,
        "labels_csv": DATASET_CSV,
        "motor_log_csv": MOTOR_LOG_CSV
    })


# ============================================================
# DATASET ZIP
# ============================================================

def create_dataset_zip():
    """
    Create ZIP containing:

        images/
        labels.csv
        motor_log.csv
    """

    if os.path.exists(DATASET_ZIP):

        os.remove(DATASET_ZIP)

    with zipfile.ZipFile(
        DATASET_ZIP,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as zip_file:

        # Add images
        for filename in sorted(
            os.listdir(IMAGE_FOLDER)
        ):

            filepath = os.path.join(
                IMAGE_FOLDER,
                filename
            )

            if os.path.isfile(filepath):

                zip_file.write(
                    filepath,
                    arcname=os.path.join(
                        "images",
                        filename
                    )
                )

        # Add labels CSV
        if os.path.exists(DATASET_CSV):

            zip_file.write(
                DATASET_CSV,
                arcname="labels.csv"
            )

        # Add motor log CSV
        if os.path.exists(MOTOR_LOG_CSV):

            zip_file.write(
                MOTOR_LOG_CSV,
                arcname="motor_log.csv"
            )

    print("Dataset ZIP created")

    return DATASET_ZIP


@app.route("/download_dataset")
def download_dataset():

    try:

        zip_path = create_dataset_zip()

        return send_file(
            zip_path,
            as_attachment=True,
            download_name="car4_cnn_dataset.zip",
            mimetype="application/zip"
        )

    except Exception as error:

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


@app.route("/download_images")
def download_images():

    try:

        zip_path = create_dataset_zip()

        return send_file(
            zip_path,
            as_attachment=True,
            download_name="car4_images_and_labels.zip",
            mimetype="application/zip"
        )

    except Exception as error:

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


@app.route("/download_labels")
def download_labels():

    if not os.path.exists(DATASET_CSV):

        return jsonify({
            "success": False,
            "error": "labels.csv not found"
        }), 404

    return send_file(
        DATASET_CSV,
        as_attachment=True,
        download_name="labels.csv",
        mimetype="text/csv"
    )


# ============================================================
# SAFE SHUTDOWN
# ============================================================

def shutdown():

    print("Shutting down Car4...")

    try:

        if arduino is not None:

            arduino.write(
                b"STOP\n"
            )

            arduino.flush()
            arduino.close()

            print("Arduino stopped")

    except Exception as error:

        print("Arduino shutdown error:", error)

    try:

        if camera is not None:

            camera.stop()
            camera.close()

            print("Camera stopped")

    except Exception as error:

        print("Camera shutdown error:", error)


atexit.register(shutdown)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("===================================")
    print("Starting Car4 CNN Data Collection")
    print("===================================")

    initialize_image_counter()

    create_csv_files()

    connect_arduino()

    setup_camera()

    print("===================================")
    print("Open dashboard:")
    print("http://<RASPBERRY_PI_IP>:5000")
    print("===================================")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )
