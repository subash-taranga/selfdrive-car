import cv2
import numpy as np
import serial
import sys

# --- Serial setup ---
try:
    ser = serial.Serial('/dev/ttyUSB0', baudrate=115200, timeout=1)
    print(f"[INFO] Opened serial port {ser.portstr} at {ser.baudrate} baud")
except serial.SerialException as e:
    print(f"[ERROR] Could not open serial port: {e}")
    sys.exit(1)

# --- Video source setup ---
use_camera = False
if use_camera:
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture("/home/team3/SmartHome/LESSONS/track.mp4")

if not camera.isOpened():
    print("[ERROR] Failed to open video source.")
    ser.close()
    sys.exit(1)

# --- HSV thresholds ---
red_lower1 = np.array([0, 70, 70])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([160, 70, 70])
red_upper2 = np.array([179, 255, 255])

blue_lower = np.array([85, 40, 40])
blue_upper = np.array([135, 255, 255])

def compute_center_line_and_distances(mask_red, mask_blue, frame, y_focus_area):
    red_x, blue_x = [], []

    red_edges, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_edges, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for edge in red_edges:
        if cv2.contourArea(edge) > 100:
            for (x, y) in edge.squeeze():
                if y >= y_focus_area:
                    red_x.append(int(x))

    for edge in blue_edges:
        if cv2.contourArea(edge) > 100:
            for (x, y) in edge.squeeze():
                if y >= y_focus_area:
                    blue_x.append(int(x))

    if red_x and blue_x:
        avg_red  = int(np.mean(red_x))
        avg_blue = int(np.mean(blue_x))
        center_x = (avg_red + avg_blue) // 2
        car_x    = frame.shape[1] // 2

        # draw lines
        cv2.line(frame, (center_x, 0), (center_x, frame.shape[0]), (0,255,255), 2)
        cv2.line(frame, (car_x, 0),    (car_x, frame.shape[0]),    (255,255,255),2)

        return avg_red, avg_blue, center_x, (car_x - avg_red), (avg_blue - car_x)
    else:
        return None, None, None, None, None

def draw_dots(mask, frame, color, y_focus_area):
    edges, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for edge in edges:
        if cv2.contourArea(edge) > 100:
            for (x, y) in edge.squeeze():
                if y >= y_focus_area:
                    cv2.circle(frame, (int(x), int(y)), 2, color, -1)

def process_frame(frame):
    hsv       = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_red  = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)

    y_focus   = int(frame.shape[0] * 0.30)
    draw_dots(mask_red,  frame, (0,0,255),   y_focus)
    draw_dots(mask_blue, frame, (255,0,0),   y_focus)

    r, b, cx, dl, dr = compute_center_line_and_distances(mask_red, mask_blue, frame, y_focus)
    if cx is not None:
        print(f"[INFO] DistLeft: {dl}, DistRight: {dr}, Cx: {cx}")
        if abs(dl - dr) < 10:
            return 'F'
        return 'L' if dl > dr else 'R'
    else:
        print("[INFO] Lines not detected")
        return 'F'

# --- Main loop ---
try:
    while True:
        ret, frame = camera.read()
        if not ret or frame is None:
            print("[INFO] Video ended or frame missing.")
            break

        cmd = process_frame(frame)

        # send over serial
        try:
            ser.write(cmd.encode('utf-8'))
        except serial.SerialTimeoutException:
            print("[WARNING] Serial write timeout")

        # overlay and display
        cv2.putText(frame, f"Move To: {cmd}", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
        cv2.imshow("Lane Detection Output", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

finally:
    # cleanup
    camera.release()
    cv2.destroyAllWindows()
    ser.close()
    print("[INFO] Cleaned up and closed serial port.")
