import cv2
import mediapipe as mp
import numpy as np
import math

def run_body_as_brush():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    ret, frame = cap.read()
    if not ret:
        return

    height, width, _ = frame.shape
    canvas = np.zeros_like(frame)

    colors = [(0, 255, 255), (255, 0, 0), (0, 0, 255), (0, 255, 0)]  # yellow, blue, red, green
    color_index = 0
    brush_color = colors[color_index]
    prev_point = None

    # Debounce cooldown to avoid dozens of color changes within one second
    color_cooldown = 0
    clear_cooldown = 0

    print("Body-as-Brush [Gesture Control Edition] Running.")
    print("  Right Wrist: Draw")
    print("  Left Hand Up: Change Color")
    print("  Hands Together: Clear Canvas")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        current_point = None

        # Cooldown tick-down
        if color_cooldown > 0: color_cooldown -= 1
        if clear_cooldown > 0: clear_cooldown -= 1

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]

            # 1. Drawing: right wrist
            if right_wrist.visibility > 0.5:
                x = int(right_wrist.x * width)
                y = int(right_wrist.y * height)
                current_point = (x, y)

            # 2. Color change: left hand raised (left wrist y above left shoulder)
            if left_wrist.visibility > 0.5 and left_shoulder.visibility > 0.5:
                if left_wrist.y < left_shoulder.y and color_cooldown == 0:
                    color_index = (color_index + 1) % len(colors)
                    brush_color = colors[color_index]
                    color_cooldown = 30  # ~1 s at ~30 fps
                    print(f"Color changed! Index: {color_index}")

            # 3. Clear canvas: hands together / crossed (wrists close)
            if right_wrist.visibility > 0.5 and left_wrist.visibility > 0.5:
                # Distance between the two wrists
                dist = math.hypot(right_wrist.x - left_wrist.x, right_wrist.y - left_wrist.y)
                if dist < 0.1 and clear_cooldown == 0:  # threshold in normalized coords
                    canvas = np.zeros_like(canvas)
                    clear_cooldown = 30
                    print("Canvas Cleared!")

        # Draw stroke trail
        if current_point is not None:
            if prev_point is not None:
                cv2.line(canvas, prev_point, current_point, brush_color, thickness=8)
            prev_point = current_point
        else:
            prev_point = None

        # Composite frame and UI overlay
        output = cv2.addWeighted(frame, 0.4, canvas, 0.8, 0)
        cv2.putText(output, "Left Hand Up: Color | Hands Crossed: Clear", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Current color swatch
        cv2.circle(output, (50, 80), 20, brush_color, -1)
        cv2.putText(output, "Current Color", (80, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Body-as-Brush", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_body_as_brush()