"""This file tests simple finger pressure with one camera view."""
import cv2
import numpy as np
import mediapipe as mp

# 黄线位置比例
KEYBOARD_Y_RATIO = 0.44

# 触发按压的高度范围比例（相对于整个画面高度）
PRESS_THRESHOLD_RATIO = 0.05  # 0.05表示±5%的画面高度范围

# 关键点集合（MediaPipe手部关键点索引）
FINGERTIP_IDS = [8, 12, 16, 20]  # 食指、中指、无名指、小拇指指尖

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# This function runs the main program.
def main():
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # 使用比例来计算黄线位置
            keyboard_y = int(h * KEYBOARD_Y_RATIO)
            cv2.line(frame, (0, keyboard_y), (w, keyboard_y), (0, 255, 255), 2)

            # 检测手部关键点
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    for tip_id in FINGERTIP_IDS:
                        lm = hand_landmarks.landmark[tip_id]
                        px = int(lm.x * w)
                        py = int(lm.y * h)

                        # 计算指尖与黄线的相对高度比例
                        tip_y_ratio = py / h
                        distance_ratio = KEYBOARD_Y_RATIO - tip_y_ratio
                        pressed = distance_ratio <= PRESS_THRESHOLD_RATIO and distance_ratio >= 0

                        color = (0, 0, 255) if pressed else (255, 255, 0)
                        cv2.circle(frame, (px, py), 8, color, -1)

                        text = f"PRESSED" if pressed else f"{distance_ratio:.3f}"
                        cv2.putText(frame, text, (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(frame, f"keyboard_y_ratio={KEYBOARD_Y_RATIO}, press_threshold_ratio={PRESS_THRESHOLD_RATIO}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "让指尖向黄线靠拢触发按压", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Paper Piano Simple", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
