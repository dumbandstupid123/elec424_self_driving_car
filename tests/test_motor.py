import lgpio
import time

# GPIO pin connected to the ESC signal wire
ESC_PIN  = 18
PWM_FREQ = 50  # Standard 50 Hz servo/ESC control signal → 20 ms period

def val_to_duty(v):
    """
    Convert a normalized throttle value to a PWM duty cycle percentage.

    ESCs use the same pulse-width convention as servos:
      v =  0.0  →  1.5 ms pulse  →  7.5%  duty  (neutral / brake)
      v = +1.0  →  2.0 ms pulse  → 10.0%  duty  (full forward)
      v = -1.0  →  1.0 ms pulse  →  5.0%  duty  (full reverse, if ESC supports it)

    Only positive values are used here since this test is forward-only.
    """
    return ((1.5 + v * 0.5) / 20.0) * 100.0

# Pi 5 uses gpiochip4
try:
    h = lgpio.gpiochip_open(4)
except lgpio.error:
    h = lgpio.gpiochip_open(0)

# Claim the ESC signal pin as a digital output, initially low
lgpio.gpio_claim_output(h, ESC_PIN, 0)

def set_esc(v, label, hold=2.0):
    """
    Send a throttle command to the ESC and hold it for `hold` seconds.
    Prints a live diagnostic so you can correlate commanded value with
    actual motor behavior during the test.
    """
    duty = val_to_duty(v)
    print(f"{label:25s}  val={v:+.2f}  duty={duty:.2f}%")
    lgpio.tx_pwm(h, ESC_PIN, PWM_FREQ, duty)
    time.sleep(hold)  # Give the motor time to spin up/respond before the next command

# --- Arming sequence ---
# Most ESCs require seeing a neutral (1.5 ms) pulse for a few seconds on power-up
# before they'll accept throttle commands. Skipping this = ESC stays locked out.
print("ESC test on GPIO18")
print("------------------")
print("Arming ESC (neutral for 3s)...")
lgpio.tx_pwm(h, ESC_PIN, PWM_FREQ, val_to_duty(0.0))
time.sleep(3)

# --- Throttle ramp test: forward-only, increasing speed in steps ---
set_esc(0.10, "SLOW FORWARD",    hold=2)  # ~1.55 ms — gentle start, checks minimum throttle response
set_esc(0.0,  "NEUTRAL",         hold=1)  # Back to neutral between steps to confirm clean stop
set_esc(0.15, "MEDIUM FORWARD",  hold=2)  # ~1.575 ms — mid-range linearity check
set_esc(0.0,  "NEUTRAL",         hold=1)
set_esc(0.20, "FAST FORWARD",    hold=2)  # ~1.6 ms — upper range check (still conservative, not full throttle)
set_esc(0.0,  "NEUTRAL",         hold=2)  # Longer neutral at the end — let motor fully spin down

# Cut the PWM signal entirely and release the GPIO cleanly
# This de-energizes the ESC rather than leaving it holding a neutral pulse
lgpio.tx_pwm(h, ESC_PIN, 0, 0)
lgpio.gpiochip_close(h)
print("Done.")
