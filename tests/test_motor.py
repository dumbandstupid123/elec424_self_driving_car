import lgpio
import time

ESC_PIN  = 18
PWM_FREQ = 50

def val_to_duty(v):
    return ((1.5 + v * 0.5) / 20.0) * 100.0

try:
    h = lgpio.gpiochip_open(4)
except lgpio.error:
    h = lgpio.gpiochip_open(0)

lgpio.gpio_claim_output(h, ESC_PIN, 0)

def set_esc(v, label, hold=2.0):
    duty = val_to_duty(v)
    print(f"{label:25s}  val={v:+.2f}  duty={duty:.2f}%")
    lgpio.tx_pwm(h, ESC_PIN, PWM_FREQ, duty)
    time.sleep(hold)

print("ESC test on GPIO18")
print("------------------")
print("Arming ESC (neutral for 3s)...")
lgpio.tx_pwm(h, ESC_PIN, PWM_FREQ, val_to_duty(0.0))
time.sleep(3)

set_esc(0.10, "SLOW FORWARD",    hold=2)
set_esc(0.0,  "NEUTRAL",         hold=1)
set_esc(0.15, "MEDIUM FORWARD",  hold=2)
set_esc(0.0,  "NEUTRAL",         hold=1)
set_esc(0.20, "FAST FORWARD",    hold=2)
set_esc(0.0,  "NEUTRAL",         hold=2)

lgpio.tx_pwm(h, ESC_PIN, 0, 0)
lgpio.gpiochip_close(h)
print("Done.")
