#!/usr/bin/env python3
"""
Car4 Automatic Motor Test
================================================================
Runs the motors automatically for 2 seconds in each direction.

SEQUENCE:

    FORWARD   -> 2 seconds
    STOP      -> 1 second
    BACKWARD  -> 2 seconds
    STOP      -> 1 second
    RIGHT     -> 2 seconds
    STOP      -> 1 second
    LEFT      -> 2 seconds
    STOP      -> 1 second

Then the test finishes and GPIO is cleaned up.

No Flask
No camera
No CNN
No keyboard commands

Run:
    python3 motor_test.py

IMPORTANT:
    Keep the wheels OFF THE GROUND during testing.
================================================================
"""

import time
import RPi.GPIO as GPIO


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


# ============================================================
# MOTOR CALIBRATION
# ============================================================

LEFT_SCALE = 1.00
RIGHT_SCALE = 0.875

# Right motor wiring is reversed.
INVERT_LEFT = False
INVERT_RIGHT = True

GPIO_PWM_FREQ = 1000


# ============================================================
# TEST SETTINGS
# ============================================================

SPEED = 150

MOVE_TIME = 2.0
STOP_TIME = 1.0


# ============================================================
# GPIO SETUP
# ============================================================

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

for pin in (
    PWMA,
    AIN1,
    AIN2,
    STBY_LEFT,
    PWMB,
    BIN1,
    BIN2,
    STBY_RIGHT,
):
    GPIO.setup(pin, GPIO.OUT)


# ============================================================
# PWM SETUP
# ============================================================

pwm_a = GPIO.PWM(PWMA, GPIO_PWM_FREQ)
pwm_b = GPIO.PWM(PWMB, GPIO_PWM_FREQ)

pwm_a.start(0)
pwm_b.start(0)


# ============================================================
# ENABLE MOTOR DRIVERS
# ============================================================

GPIO.output(STBY_LEFT, GPIO.HIGH)
GPIO.output(STBY_RIGHT, GPIO.HIGH)


print(
    f"[GPIO] Left driver:  "
    f"PWMA={PWMA} "
    f"AIN1={AIN1} "
    f"AIN2={AIN2} "
    f"STBY={STBY_LEFT}"
)

print(
    f"[GPIO] Right driver: "
    f"PWMB={PWMB} "
    f"BIN1={BIN1} "
    f"BIN2={BIN2} "
    f"STBY={STBY_RIGHT}"
)


# ============================================================
# LOW-LEVEL MOTOR CONTROL
# ============================================================

def _set_channel(in1, in2, pwm_channel, duty_percent):
    """
    Set one motor.

    duty_percent:
        +100 = forward
        0    = stop
        -100 = backward
    """

    duty_percent = max(
        -100,
        min(100, duty_percent)
    )

    if duty_percent > 0:

        GPIO.output(in1, GPIO.HIGH)
        GPIO.output(in2, GPIO.LOW)

    elif duty_percent < 0:

        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)

    else:

        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)

    pwm_channel.ChangeDutyCycle(
        abs(duty_percent)
    )


# ============================================================
# LEFT MOTOR
# ============================================================

def _drive_left(duty_percent):

    d = (
        duty_percent
        * LEFT_SCALE
        * (-1 if INVERT_LEFT else 1)
    )

    _set_channel(
        AIN1,
        AIN2,
        pwm_a,
        d
    )


# ============================================================
# RIGHT MOTOR
# ============================================================

def _drive_right(duty_percent):

    d = (
        duty_percent
        * RIGHT_SCALE
        * (-1 if INVERT_RIGHT else 1)
    )

    _set_channel(
        BIN1,
        BIN2,
        pwm_b,
        d
    )


# ============================================================
# DRIVE COMMAND
# ============================================================

def drive(command, speed_0_255):

    """
    Drive the car.

    F = forward
    B = backward
    L = left
    R = right
    S = stop
    """

    duty = (
        max(0, min(255, int(speed_0_255)))
        / 255.0
    ) * 100.0

    command = command.upper()

    if command == "F":

        # Both motors forward
        left = duty
        right = duty

    elif command == "B":

        # Both motors backward
        left = -duty
        right = -duty

    elif command == "L":

        # Turn left
        left = -duty
        right = duty

    elif command == "R":

        # Turn right
        left = duty
        right = -duty

    else:

        left = 0
        right = 0

    # Make sure drivers are enabled
    GPIO.output(
        STBY_LEFT,
        GPIO.HIGH
    )

    GPIO.output(
        STBY_RIGHT,
        GPIO.HIGH
    )

    _drive_left(left)
    _drive_right(right)


# ============================================================
# STOP
# ============================================================

def stop():

    """
    Stop both motors.
    """

    _drive_left(0)
    _drive_right(0)


# ============================================================
# RUN ONE MOVEMENT
# ============================================================

def run_movement(name, command):

    print()
    print("--------------------------------------------")
    print(f"-> {name}")
    print(f"   speed = {SPEED}")
    print(f"   time  = {MOVE_TIME} seconds")
    print("--------------------------------------------")

    drive(
        command,
        SPEED
    )

    time.sleep(MOVE_TIME)

    stop()

    print("-> STOP")

    time.sleep(STOP_TIME)


# ============================================================
# CLEANUP
# ============================================================

def cleanup():

    print()
    print("[INFO] Stopping motors...")

    try:
        stop()

        pwm_a.stop()
        pwm_b.stop()

        GPIO.output(
            STBY_LEFT,
            GPIO.LOW
        )

        GPIO.output(
            STBY_RIGHT,
            GPIO.LOW
        )

    finally:

        GPIO.cleanup()

        print("[INFO] GPIO cleaned up.")
        print("[INFO] Bye!")


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("============================================")
    print(" Car4 Automatic Motor Test")
    print("============================================")
    print()
    print("WARNING:")
    print("WHEELS MUST BE OFF THE GROUND!")
    print()
    print(f"Speed      : {SPEED}/255")
    print(f"Move time  : {MOVE_TIME} seconds")
    print(f"Stop time  : {STOP_TIME} second")
    print()
    print("Sequence:")
    print("  1. FORWARD")
    print("  2. BACKWARD")
    print("  3. RIGHT")
    print("  4. LEFT")
    print()
    print("============================================")
    print()

    try:

        # ----------------------------------------------------
        # Give the user a moment before starting
        # ----------------------------------------------------

        print("Starting in 3 seconds...")

        time.sleep(1)

        print("2...")

        time.sleep(1)

        print("1...")

        time.sleep(1)

        print("GO!")

        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        run_movement(
            "FORWARD",
            "F"
        )

        # ----------------------------------------------------
        # BACKWARD
        # ----------------------------------------------------

        run_movement(
            "BACKWARD",
            "B"
        )

        # ----------------------------------------------------
        # RIGHT
        # ----------------------------------------------------

        run_movement(
            "RIGHT",
            "R"
        )

        # ----------------------------------------------------
        # LEFT
        # ----------------------------------------------------

        run_movement(
            "LEFT",
            "L"
        )

        # ----------------------------------------------------
        # Finished
        # ----------------------------------------------------

        stop()

        print()
        print("============================================")
        print(" TEST COMPLETE")
        print("============================================")

    except KeyboardInterrupt:

        print()
        print("[INFO] Ctrl+C detected.")

    finally:

        cleanup()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
