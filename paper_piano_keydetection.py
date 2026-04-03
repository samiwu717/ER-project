"""
简化版纸质钢琴键位检测
功能：使用ArUco标记 + Homography变换 + MediaPipe手指检测
      判定手指在哪个按键上（11个白键的纵向堆叠）
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List

# ==================== 配置 ====================
PREFERRED_CAM_INDEX = 2
CAM_WIDTH = 1280
CAM_HEIGHT = 720

# 纸质模板坐标系（纵向）
PAPER_W = 800
PAPER_H = 1100

# 键盘区域（纸质坐标）
KEYBOARD_X0 = 92
KEYBOARD_X1 = 720
KEYBOARD_Y0 = 190
KEYBOARD_Y1 = 1045
NUM_WHITE_KEYS = 11
KEYBOARD_STACK_VERTICAL = True

# ArUco标记
ARUCO_DICT_NAME = cv2.aruco.DICT_5X5_50
ARUCO_CORNER_IDS = {
    "tl": 0,
    "tr": 1,
    "br": 2,
    "bl": 3,
}
ARUCO_CORNER_INDEX = {
    "tl": 0,
    "tr": 1,
    "br": 2,
    "bl": 3,
}

# 手指关键点（索引）
FINGERTIP_IDS = [8, 12, 16, 20]  # 食指、中指、无名指、小拇指指尖

# 音符信息
NOTE_NAMES = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5", "F5"]
NOTE_FREQS = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25, 587.33, 659.25, 698.46]

# 黑键位置（在白键之间）
BLACK_KEY_AFTER_WHITE = [0, 1, 3, 4, 5, 7, 8]


class PaperPianoKeyDetector:
    """纸质钢琴键位检测器"""
    
    def __init__(self):
        self.cap = cv2.VideoCapture(PREFERRED_CAM_INDEX, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index={PREFERRED_CAM_INDEX}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        
        # MediaPipe 手部检测
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # ArUco检测
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Homography变换
        self.last_good_H = None
        self.last_good_H_inv = None
        self.last_fps_time = None
        
        print(f"[OK] Camera ready: index={PREFERRED_CAM_INDEX}, shape=({CAM_HEIGHT}, {CAM_WIDTH})", flush=True)
    
    def detect_aruco_markers(self, frame: np.ndarray) -> dict:
        """检测ArUco标记，返回四个角的位置"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        marker_corners = {}
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                marker_corners[int(marker_id)] = corners[i][0]  # 4x2 数组
        
        return marker_corners
    
    def compute_homography(self, marker_corners: dict) -> Optional[np.ndarray]:
        """计算 Homography 矩阵（从摄像头坐标到纸质坐标）"""
        # 纸质模板的四个角
        paper_corners = np.array([
            [0, 0],                    # tl
            [PAPER_W, 0],              # tr
            [PAPER_W, PAPER_H],        # br
            [0, PAPER_H],              # bl
        ], dtype=np.float32)
        
        # 摄像头中的四个标记角
        cam_corners = []
        for corner_name in ["tl", "tr", "br", "bl"]:
            marker_id = ARUCO_CORNER_IDS[corner_name]
            idx = ARUCO_CORNER_INDEX[corner_name]
            if marker_id in marker_corners:
                cam_corners.append(marker_corners[marker_id][idx])
        
        if len(cam_corners) != 4:
            return None
        
        cam_corners = np.array(cam_corners, dtype=np.float32)
        H, _ = cv2.findHomography(cam_corners, paper_corners)
        return H
    
    def camera_to_paper(self, cam_pt: np.ndarray, H: np.ndarray) -> np.ndarray:
        """将摄像头坐标变换到纸质坐标"""
        cam_pt = np.array([[cam_pt[0], cam_pt[1]]], dtype=np.float32).reshape(-1, 1, 2)
        paper_pt = cv2.perspectiveTransform(cam_pt, H)
        return paper_pt[0][0]
    
    def locate_key(self, paper_pt: np.ndarray) -> Optional[int]:
        """判定手指在哪个按键上（0-10）"""
        x, y = float(paper_pt[0]), float(paper_pt[1])
        x0, y0, x1, y1 = KEYBOARD_X0, KEYBOARD_Y0, KEYBOARD_X1, KEYBOARD_Y1
        
        # 检查是否在键盘区域内
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return None
        
        if KEYBOARD_STACK_VERTICAL:
            # 纵向堆叠：按 y 坐标计算
            key_h = (y1 - y0) / NUM_WHITE_KEYS
            idx = int((y - y0) / key_h)
            idx = max(0, min(NUM_WHITE_KEYS - 1, idx))
            return idx
        else:
            # 横向堆叠：按 x 坐标计算
            key_w = (x1 - x0) / NUM_WHITE_KEYS
            idx = int((x - x0) / key_w)
            idx = max(0, min(NUM_WHITE_KEYS - 1, idx))
            return idx
    
    def draw_keyboard_overlay(self, frame: np.ndarray, H: np.ndarray) -> np.ndarray:
        """在摄像头画面上绘制虚拟键盘"""
        if H is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # 定义纸质坐标中的键盘区域
        x0, y0, x1, y1 = KEYBOARD_X0, KEYBOARD_Y0, KEYBOARD_X1, KEYBOARD_Y1
        
        if KEYBOARD_STACK_VERTICAL:
            key_h = (y1 - y0) / NUM_WHITE_KEYS
            for k in range(NUM_WHITE_KEYS + 1):
                py0 = y0 + k * key_h
                py1 = y0 + ((k + 1) * key_h if k < NUM_WHITE_KEYS else NUM_WHITE_KEYS * key_h)
                
                # 纸质坐标的四个点
                paper_pts = np.array([
                    [x0, py0],
                    [x1, py0],
                    [x1, py1],
                    [x0, py1],
                ], dtype=np.float32)
                
                # 变换到摄像头坐标
                H_inv = cv2.invert(H)[1]
                cam_pts = cv2.perspectiveTransform(paper_pts.reshape(-1, 1, 2), H_inv)
                cam_pts = cam_pts.reshape(-1, 2).astype(int)
                
                # 绘制边界
                cv2.polylines(frame, [cam_pts], True, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        """主循环"""
        frame_count = 0
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            frame_count += 1
            
            # 1. 检测ArUco标记
            marker_corners = self.detect_aruco_markers(frame)
            
            # 2. 计算Homography
            H = self.compute_homography(marker_corners)
            if H is not None:
                self.last_good_H = H
                self.last_good_H_inv = cv2.invert(H)[1]
            
            # 3. 绘制虚拟键盘
            if self.last_good_H is not None:
                frame = self.draw_keyboard_overlay(frame, self.last_good_H)
            
            # 4. 检测手部关键点
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            if results.multi_hand_landmarks and self.last_good_H is not None:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # 检查所有指尖
                    for tip_id in FINGERTIP_IDS:
                        lm = hand_landmarks.landmark[tip_id]
                        cam_x = int(lm.x * w)
                        cam_y = int(lm.y * h)
                        
                        # 变换到纸质坐标
                        cam_pt = np.array([cam_x, cam_y], dtype=np.float32)
                        paper_pt = self.camera_to_paper(cam_pt, self.last_good_H)
                        
                        # 判定按键
                        key_idx = self.locate_key(paper_pt)
                        
                        if key_idx is not None:
                            note = NOTE_NAMES[key_idx]
                            freq = NOTE_FREQS[key_idx]
                            color = (0, 0, 255)  # 红色：在键上
                            cv2.circle(frame, (cam_x, cam_y), 10, color, -1)
                            text = f"Key {key_idx}: {note} ({freq:.0f}Hz)"
                            cv2.putText(frame, text, (cam_x + 15, cam_y - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        else:
                            # 黄色：不在键盘区域
                            color = (0, 255, 255)
                            cv2.circle(frame, (cam_x, cam_y), 8, color, -1)
            
            # 5. 显示状态
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if self.last_good_H is not None:
                cv2.putText(frame, "Homography: OK", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Homography: Searching...", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Paper Piano - Key Detection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


if __name__ == '__main__':
    try:
        detector = PaperPianoKeyDetector()
        detector.run()
    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
