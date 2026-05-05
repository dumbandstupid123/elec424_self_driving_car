import lgpio   # Linux GPIO library for hardware PWM on Raspberry Pi / SBC
import time

# GPIO pin connected to the servo signal wire (physical board pin varies by platform)
SERVO_PIN = 19
PWM_FREQ  = 50  # Standard servo control frequency: 50 Hz → 20 ms period

def val_to_duty(v):
    """
    Convert a normalized steering value to a PWM duty cycle percentage.

    Servos expect a pulse width between 1.0 ms (full left) and 2.0 ms (full right),
    with 1.5 ms being the neutral/center position — all within a 20 ms period.

    v = -1.0  →  1.0 ms pulse  →  5.0%  duty  (full left)
    v =  0.0  →  1.5 ms pulse  →  7.5%  duty  (center / straight)
    v = +1.0  →  2.0 ms pulse  → 10.0%  duty  (full right)
    """
    return ((1.5 + v * 0.5) / 20.0) * 100.0

# Try the Pi 5 GPIO chip first (gpiochip4)
try:
    h = lgpio.gpiochip_open(4)
except lgpio.error:
    h = lgpio.gpiochip_open(0)

# Claim the servo pin as a push-pull output, initially driven low
lgpio.gpio_claim_output(h, SERVO_PIN, 0)

def move(v, label, hold=1.5):
    """
    Command the servo to a position and hold it for `hold` seconds.
    Prints a diagnostic line so you can verify the duty cycle math live.
    """
    duty = val_to_duty(v)
    print(f"{label:20s}  val={v:+.2f}  duty={duty:.2f}%")
    lgpio.tx_pwm(h, SERVO_PIN, PWM_FREQ, duty)
    time.sleep(hold)  # Hold position long enough for the servo to physically reach it

# --- Sweep test: exercises the full range of steering motion ---
print("Servo test on GPIO19")
print("---------------------")
move( 0.0, "CENTER")      # Confirm neutral before anything moves
move(-1.0, "FULL LEFT")   # Mechanical limit check — left
move( 0.0, "CENTER")
move( 1.0, "FULL RIGHT")  # Mechanical limit check — right
move( 0.0, "CENTER")
move(-0.5, "HALF LEFT")   # Mid-range linearity check
move( 0.0, "CENTER")
move( 0.5, "HALF RIGHT")
move( 0.0, "CENTER")      # Always return to center before releasing the signal

# Stop PWM output and release the GPIO — leaves the servo in a de-energized state
lgpio.tx_pwm(h, SERVO_PIN, 0, 0)
lgpio.gpiochip_close(h)
print("Done.")
