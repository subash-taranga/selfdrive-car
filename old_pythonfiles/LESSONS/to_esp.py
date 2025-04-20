import cv2
import numpy as np
import serial
import time

# === SERIAL SETUP ===
# Update the COM port as per your system (e.g., 'COM3' for Windows or '/dev/ttyUSB0' for Linux)
ser = serial.Serial('COM6', 115200, timeout=1)
time.sleep(2)  # wait for the serial connection to initialize

# === VIDEO SETUP ===
use_camera = False
if use_camera:
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture("track.mp4")

if not camera.isOpened():
    print("Failed to open video source (camera or video file).")
    exit()

# === HSV COLOR RANGES ===
red_lower1 = np.array([0, 70, 70])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([160, 70, 70])
red_upper2 = np.array([179, 255, 255])
blue_lower = np.array([85, 40, 40])
blue_upper = np.array([135, 255, 255])

# === FUNCTIONS ===
def calculate_lane_center_and_offsets(mask_red, mask_blue, frame, y_focus_area):
    red_x = []
    blue_x = []

    red_edges, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_edges, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for edge in red_edges:
        if cv2.contourArea(edge) > 100:
            for point in edge:
                x, y = point[0]
                if y >= y_focus_area:
                    red_x.append(x)

    for edge in blue_edges:
        if cv2.contourArea(edge) > 100:
            for point in edge:
                x, y = point[0]
                if y >= y_focus_area:
                    blue_x.append(x)

    if red_x and blue_x:
        avg_red = int(np.mean(red_x))
        avg_blue = int(np.mean(blue_x))
        center_x = (avg_red + avg_blue) // 2
        car_position_x = frame.shape[1] // 2

        cv2.line(frame, (center_x, 0), (center_x, frame.shape[0]), (0, 255, 255), 2)
        cv2.line(frame, (car_position_x, 0), (car_position_x, frame.shape[0]), (255, 255, 255), 2)

        dist_left = car_position_x - avg_red
        dist_right = avg_blue - car_position_x

        return avg_red, avg_blue, center_x, dist_left, dist_right
    
    return None, None, None, None, None

def highlight_lane_points(mask, frame, color=(0, 255, 0), y_focus_area=0):
    edges, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for edge in edges:
        if cv2.contourArea(edge) > 100:
            for point in edge:
                x, y = point[0]
                if y >= y_focus_area:
                    cv2.circle(frame, (x, y), 2, color, -1)

def analyze_frame_and_get_direction(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)

    height = frame.shape[0]
    y_focus_area = int(height * 0.30)

    highlight_lane_points(mask_red, frame, (0, 0, 255), y_focus_area)
    highlight_lane_points(mask_blue, frame, (255, 0, 0), y_focus_area)

    red_x, blue_x, center_x, dist_left, dist_right = calculate_lane_center_and_offsets(
        mask_red, mask_blue, frame, y_focus_area)

    if center_x is not None:
        print(f"[INFO] Distance to Red: {dist_left}, Distance to Blue: {dist_right}, Center X: {center_x}")
        if abs(dist_left - dist_right) < 10:
            return 'F'
        elif dist_left > dist_right:
            return 'L'
        else:
            return 'R'
    else:
        print("[INFO] Lines not detected")
        return 'F'

# === MAIN LOOP ===
while True:
    ret, frame = camera.read()
    if not ret or frame is None:
        print("Video has ended or failed to read.")
        break

    command = analyze_frame_and_get_direction(frame)

    # ==== SEND COMMAND TO ESP32 ====
    try:
        ser.write(command.encode())  # send command as bytes
    except Exception as e:
        print(f"[ERROR] Serial write failed: {e}")

    # === SHOW ON FRAME ===
    cv2.putText(frame, f"Move To: {command}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.imshow("Lane Detection Output", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# === CLEANUP ===
camera.release()
cv2.destroyAllWindows()
ser.close()
