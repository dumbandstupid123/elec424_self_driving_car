import lgpio
import time

SERVO_PIN = 19
PWM_FREQ  = 50

def val_to_duty(v):
    # v in [-1.0, 1.0]; 0 = 1.5ms = 7.5% = straight
    return ((1.5 + v * 0.5) / 20.0) * 100.0

try:
    h = lgpio.gpiochip_open(4)
except lgpio.error:
    h = lgpio.gpiochip_open(0)

lgpio.gpio_claim_output(h, SERVO_PIN, 0)

def move(v, label, hold=1.5):
    duty = val_to_duty(v)
    print(f"{label:20s}  val={v:+.2f}  duty={duty:.2f}%")
    lgpio.tx_pwm(h, SERVO_PIN, PWM_FREQ, duty)
    time.sleep(hold)

print("Servo test on GPIO19")
print("---------------------")
move( 0.0, "CENTER")
move(-1.0, "FULL LEFT")
move( 0.0, "CENTER")
move( 1.0, "FULL RIGHT")
move( 0.0, "CENTER")
move(-0.5, "HALF LEFT")
move( 0.0, "CENTER")
move( 0.5, "HALF RIGHT")
move( 0.0, "CENTER")

lgpio.tx_pwm(h, SERVO_PIN, 0, 0)
lgpio.gpiochip_close(h)
print("Done.")
