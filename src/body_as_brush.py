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
                    
                    if pinch_dist < 0.03:
                        is_drawing = True
                        cv2.circle(frame, current_point, 8, brush_color, -1)  # pinched: solid dot
                    else:
                        cv2.circle(frame, current_point, 8, (255, 255, 255), 2)  # open: crosshair ring

        # --- 5. Rendering (modern UI) ---
        
        # Neon brush: thick glow on live frame, thin solid stroke on persistent canvas
        if is_drawing and current_point is not None:
            if prev_point is not None:
                # Glow layer (thick stroke on camera frame)
                cv2.line(frame, prev_point, current_point, brush_color, thickness=20)
                # Core stroke on canvas
                cv2.line(canvas, prev_point, current_point, brush_color, thickness=6)
            prev_point = current_point
        else:
            prev_point = None

        # Composite camera + canvas
        output = cv2.addWeighted(frame, 0.5, canvas, 0.9, 0)

        # Left dashboard panel with alpha blend
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (320, height), (20, 20, 20), -1)  # panel fill
        cv2.rectangle(overlay, (0, 0), (320, 60), (40, 40, 40), -1)  # header bar
        # Blend overlay at 0.7
        output = cv2.addWeighted(overlay, 0.7, output, 0.3, 0)

        # Labels and indicators
        cv2.putText(output, "BODY AS BRUSH", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        
        # Status LED
        cv2.putText(output, "SYSTEM STATUS", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        if is_drawing:
            cv2.circle(output, (30, 130), 8, (0, 255, 0), -1)  # solid green
            cv2.putText(output, "Pen: Drawing", (50, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.circle(output, (30, 130), 8, (100, 100, 100), 2)  # gray outline
            cv2.putText(output, "Pen: Hovering", (50, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Color swatch
        cv2.putText(output, "CURRENT COLOR", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.circle(output, (40, 230), 20, brush_color, -1)
        cv2.circle(output, (40, 230), 22, (255, 255, 255), 2)  # white ring

        # Help text (bottom)
        cv2.putText(output, "CONTROLS", (20, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(output, "- Pinch: Draw", (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(output, "- L-Arm Up: Color", (20, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(output, "- Cross Wrists: Clear", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        cv2.imshow("Body-as-Brush [Pro UI]", output)

        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_body_as_brush()