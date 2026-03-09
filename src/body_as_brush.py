import cv2
import mediapipe as mp
import numpy as np


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
        print("Error: Cannot read from webcam.")
        cap.release()
        return

    height, width, _ = frame.shape
    canvas = np.zeros_like(frame)

    drawing_enabled = True
    brush_color = (0, 255, 255)  # Yellow-ish
    prev_point = None

    print("Body-as-Brush MVP running.")
    print("Controls:")
    print("  d - toggle drawing on/off")
    print("  c - clear canvas")
    print("  1/2/3 - change brush color")
    print("  q - quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        current_point = None

        if results.pose_landmarks:
            # Draw landmarks lightly on the video feed for debugging / demo
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )

            # Use right wrist as the "brush"
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            if right_wrist.visibility > 0.5:
                x = int(right_wrist.x * width)
                y = int(right_wrist.y * height)
                current_point = (x, y)

        if drawing_enabled and current_point is not None:
            if prev_point is not None:
                cv2.line(canvas, prev_point, current_point, brush_color, thickness=5)
            prev_point = current_point
        else:
            prev_point = None

        # Combine webcam frame and drawing canvas
        output = cv2.addWeighted(frame, 0.3, canvas, 0.7, 0)

        cv2.imshow("Body-as-Brush - Camera + Trails", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            drawing_enabled = not drawing_enabled
            print(f"Drawing enabled: {drawing_enabled}")
        elif key == ord("c"):
            canvas = np.zeros_like(canvas)
            print("Canvas cleared.")
        elif key == ord("1"):
            brush_color = (0, 255, 255)  # Yellow
            print("Brush color: yellow")
        elif key == ord("2"):
            brush_color = (255, 0, 0)  # Blue
            print("Brush color: blue")
        elif key == ord("3"):
            brush_color = (0, 0, 255)  # Red
            print("Brush color: red")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_body_as_brush()

