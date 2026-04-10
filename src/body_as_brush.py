import cv2
import mediapipe as mp
import numpy as np
import math

def run_body_as_brush():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    # --- 1. Dual-model setup ---
    # Core A: Pose — body / shoulders / wrist distances for macro gestures
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Core B: Hands — high-res fingertips and pinch for drawing
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils

    ret, frame = cap.read()
    if not ret: return

    height, width, _ = frame.shape
    canvas = np.zeros_like(frame)

    colors = [(0, 255, 255), (255, 0, 0), (0, 0, 255), (0, 255, 0)]  # yellow, blue, red, green
    color_index = 0
    brush_color = colors[color_index]
    prev_point = None

    color_cooldown = 0
    clear_cooldown = 0

    print("Body-as-Brush [Dual-Model Fusion Edition] Running.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Mirror horizontally (selfie-style)
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- 2. Run both models on the same frame ---
        results_pose = pose.process(rgb)
        results_hands = hands.process(rgb)

        current_point = None
        is_drawing = False

        if color_cooldown > 0: color_cooldown -= 1
        if clear_cooldown > 0: clear_cooldown -= 1

        # --- 3. Macro gestures (Pose) ---
        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark
            
            # Physical left = MediaPipe RIGHT after mirror; physical right wrist = LEFT_WRIST
            phys_left_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            phys_left_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            phys_right_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

            # A. Color: physical left wrist above left shoulder
            if phys_left_wrist.visibility > 0.5 and phys_left_shoulder.visibility > 0.5:
                if phys_left_wrist.y < phys_left_shoulder.y and color_cooldown == 0:
                    color_index = (color_index + 1) % len(colors)
                    brush_color = colors[color_index]
                    color_cooldown = 30
                    print("Color changed!")

            # B. Clear canvas: wrists close (hands crossed)
            if phys_right_wrist.visibility > 0.5 and phys_left_wrist.visibility > 0.5:
                dist_wrists = math.hypot(phys_right_wrist.x - phys_left_wrist.x, phys_right_wrist.y - phys_left_wrist.y)
                if dist_wrists < 0.1 and clear_cooldown == 0:
                    canvas = np.zeros_like(canvas)
                    clear_cooldown = 30
                    print("Canvas Cleared!")

        # --- 4. Drawing (Hands): precise index tip + pinch ---
        if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
            for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
                # After mirror, physical right hand is labeled "Right"
                if handedness.classification[0].label == 'Right':
                    # INDEX_FINGER_TIP (8), THUMB_TIP (4)
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    
                    x = int(index_tip.x * width)
                    y = int(index_tip.y * height)
                    current_point = (x, y)

                    # Pinch distance (tune ~0.03–0.05 for your hand size)
                    pinch_dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
                    
                    if pinch_dist < 0.04:
                        is_drawing = True
                        cv2.circle(frame, current_point, 8, brush_color, -1)  # pinched: solid dot
                    else:
                        cv2.circle(frame, current_point, 8, (255, 255, 255), 2)  # open: crosshair ring

        # --- 5. Composite + UI ---
        if is_drawing and current_point is not None:
            if prev_point is not None:
                cv2.line(canvas, prev_point, current_point, brush_color, thickness=8)
            prev_point = current_point
        else:
            prev_point = None

        output = cv2.addWeighted(frame, 0.4, canvas, 0.8, 0)

        # Skeleton overlay on top of the blend so it stays visible
        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(
                output,
                results_pose.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=3
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(200, 200, 200), thickness=2
                ),
            )

        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    output,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 200, 255), thickness=2, circle_radius=2
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 140, 255), thickness=2
                    ),
                )
        
        cv2.putText(output, "Pinch Right Fingers to Draw | Left Hand Up: Color | Cross Wrists: Clear", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.circle(output, (40, 70), 15, brush_color, -1)
        cv2.putText(output, "Pen Color", (65, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        status_text = "Drawing (Pinched)" if is_drawing else "Hovering"
        status_color = (0, 255, 0) if is_drawing else (0, 0, 255)
        cv2.putText(output, f"Status: {status_text}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        cv2.imshow("Body-as-Brush [Dual-Fusion]", output)

        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_body_as_brush()