"""This file is the third paper piano version with more geometry."""
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import pygame
except Exception:
    pygame = None

from prediction import draw_landmarks_on_image, get_camera_matrix, predict


# =========================
# User-tunable parameters
# =========================
CAM_INDEX = 0
CAM_WIDTH = 1280
CAM_HEIGHT = 720

# Canonical top-plane coordinate system for keyboard mapping
PAPER_W = 800
PAPER_H = 1100

# Keep only a narrow horizontal strip on the top plane
KEYBOARD_X0 = 80
KEYBOARD_X1 = 720
KEYBOARD_Y0 = 485
KEYBOARD_Y1 = 625
NUM_WHITE_KEYS = 8
KEYBOARD_STACK_VERTICAL = False

FINGERTIP_IDS = [8, 12, 16, 20]

# Trigger behavior
KEY_COOLDOWN_SEC = 0.12
ACTIVE_FLASH_SEC = 0.20
POSE_HOLD_SEC = 0.45

# Finger filtering + downward press detection
TIP_SMOOTH_ALPHA = 0.35
SPEED_EMA_ALPHA = 0.55
MAX_STEP_PX = 110.0
DOWN_ARM_SPEED_PX = 220.0
DOWN_FIRE_SPEED_PX = 120.0
DOWN_MIN_TRAVEL_PX = 18.0
DOWN_MAX_WINDOW_SEC = 0.26
REARM_OUTSIDE_SEC = 0.08

DEBUG_DRAW_RAW_HAND = True
DEBUG_DRAW_PRESS_INFO = True

# ArUco settings
ARUCO_DICT_NAME = cv2.aruco.DICT_5X5_50
ARUCO_USE_MULTI_PASS = True
MIRROR_DISPLAY = True

# Three-layer rig geometry (millimeters)
RIG_W_MM = 297.0
RIG_D_MM = 210.0
RIG_WALL_H_MM = 58.0
MARKER_SIZE_MM = 32.0
MARKER_MARGIN_MM = 24.0

# Marker IDs on folded sheet (default from generated template)
ARUCO_ID_BACK_LEFT = 0
ARUCO_ID_BACK_RIGHT = 1
ARUCO_ID_FRONT_RIGHT = 2
ARUCO_ID_FRONT_LEFT = 3
KNOWN_MARKER_IDS = [
    ARUCO_ID_BACK_LEFT,
    ARUCO_ID_BACK_RIGHT,
    ARUCO_ID_FRONT_RIGHT,
    ARUCO_ID_FRONT_LEFT,
]

NOTE_FREQS = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
NOTE_NAMES = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]


# This function opens the camera.
def open_camera(index: int = CAM_INDEX) -> cv2.VideoCapture:
    backends = [cv2.CAP_ANY]
    for backend in (getattr(cv2, "CAP_DSHOW", None), getattr(cv2, "CAP_MSMF", None)):
        if backend is not None:
            backends.append(backend)

    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            return cap
        cap.release()
    return cv2.VideoCapture(index)


class SimpleSynth:
    # This function sets up the object.
    def __init__(self) -> None:
        self.enabled = False
        self.sounds: List[Optional["pygame.mixer.Sound"]] = []
        if pygame is None:
            return
        try:
            pygame.mixer.pre_init(44100, -16, 2, 512)
            pygame.mixer.init()
            self.sounds = [self._make_tone(f) for f in NOTE_FREQS]
            self.enabled = True
        except Exception as exc:
            print(f"[WARN] Audio disabled: {exc}")
            self.sounds = [None] * len(NOTE_FREQS)

    # This function makes make tone.
    def _make_tone(self, freq: float, duration: float = 0.55, sr: int = 44100, volume: float = 0.45):
        n = int(sr * duration)
        t = np.linspace(0.0, duration, n, endpoint=False)
        wave = (
            1.0 * np.sin(2 * np.pi * freq * t)
            + 0.40 * np.sin(2 * np.pi * 2 * freq * t)
            + 0.20 * np.sin(2 * np.pi * 3 * freq * t)
        )
        wave /= 1.60

        attack = int(sr * 0.01)
        decay = int(sr * 0.08)
        release = int(sr * 0.20)
        sustain = 0.65
        env = np.ones(n, dtype=np.float64) * sustain
        if attack > 0:
            env[:attack] = np.linspace(0.0, 1.0, attack)
        if attack + decay <= n:
            env[attack:attack + decay] = np.linspace(1.0, sustain, decay)
        if release < n:
            env[-release:] = np.linspace(sustain, 0.0, release)

        audio = (wave * env * 32767 * volume).astype(np.int16)
        stereo = np.column_stack([audio, audio])
        return pygame.sndarray.make_sound(stereo)

    # This function plays the note sound.
    def play(self, key_idx: int) -> None:
        if not self.enabled:
            return
        snd = self.sounds[key_idx]
        if snd is not None:
            snd.play()

    # This function closes and cleans up things.
    def close(self) -> None:
        if pygame is not None:
            try:
                if pygame.mixer.get_init():
                    pygame.mixer.quit()
            except Exception:
                pass


class PaperPianoV3:
    # This function sets up the object.
    def __init__(self) -> None:
        self.cap = open_camera()
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera.")

        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV aruco module is required. Please install opencv-contrib-python.")

        self.synth = SimpleSynth()
        self.last_hit_time = [0.0] * NUM_WHITE_KEYS
        self.active_until = [0.0] * NUM_WHITE_KEYS
        self.finger_states: Dict[Tuple[int, int], Dict[str, float]] = {}

        self.frame_w = CAM_WIDTH
        self.frame_h = CAM_HEIGHT
        self.last_fps_time = time.time()
        self.fps = 0.0

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 61
        self.aruco_params.adaptiveThreshWinSizeStep = 6
        self.aruco_params.adaptiveThreshConstant = 7.0
        self.aruco_params.minMarkerPerimeterRate = 0.01
        self.aruco_params.maxMarkerPerimeterRate = 6.0
        self.aruco_params.polygonalApproxAccuracyRate = 0.08
        self.aruco_params.minCornerDistanceRate = 0.01
        self.aruco_params.minDistanceToBorder = 1
        self.aruco_params.minOtsuStdDev = 2.0
        if hasattr(self.aruco_params, "useAruco3Detection"):
            self.aruco_params.useAruco3Detection = True
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.id_to_center_3d = self._build_marker_centers_3d()
        self.top_plane_3d = np.array([
            [-0.5 * RIG_W_MM, 0.0, -0.5 * RIG_D_MM],  # back-left
            [+0.5 * RIG_W_MM, 0.0, -0.5 * RIG_D_MM],  # back-right
            [+0.5 * RIG_W_MM, 0.0, +0.5 * RIG_D_MM],  # front-right
            [-0.5 * RIG_W_MM, 0.0, +0.5 * RIG_D_MM],  # front-left
        ], dtype=np.float32)
        self.back_wall_3d = np.array([
            [-0.5 * RIG_W_MM, 0.0, -0.5 * RIG_D_MM],
            [+0.5 * RIG_W_MM, 0.0, -0.5 * RIG_D_MM],
            [+0.5 * RIG_W_MM, -RIG_WALL_H_MM, -0.5 * RIG_D_MM],
            [-0.5 * RIG_W_MM, -RIG_WALL_H_MM, -0.5 * RIG_D_MM],
        ], dtype=np.float32)

        self.last_good_H: Optional[np.ndarray] = None
        self.last_good_H_inv: Optional[np.ndarray] = None
        self.last_back_wall_poly: Optional[np.ndarray] = None
        self.last_top_quad: Optional[np.ndarray] = None
        self.last_marker_centers_2d: Dict[int, np.ndarray] = {}
        self.last_detected_ids: List[int] = []
        self.last_pose_time = -999.0

    # This function builds build marker centers 3d.
    def _build_marker_centers_3d(self) -> Dict[int, np.ndarray]:
        x_l = -0.5 * RIG_W_MM + MARKER_MARGIN_MM
        x_r = +0.5 * RIG_W_MM - MARKER_MARGIN_MM
        y_m = -0.5 * RIG_WALL_H_MM
        z_b = -0.5 * RIG_D_MM
        z_f = +0.5 * RIG_D_MM
        return {
            ARUCO_ID_BACK_LEFT: np.array([x_l, y_m, z_b], dtype=np.float32),
            ARUCO_ID_BACK_RIGHT: np.array([x_r, y_m, z_b], dtype=np.float32),
            ARUCO_ID_FRONT_RIGHT: np.array([x_r, y_m, z_f], dtype=np.float32),
            ARUCO_ID_FRONT_LEFT: np.array([x_l, y_m, z_f], dtype=np.float32),
        }

    # This function closes and cleans up things.
    def close(self) -> None:
        try:
            if self.cap is not None:
                self.cap.release()
        finally:
            self.synth.close()
        cv2.destroyAllWindows()

    # This function collects collect id to corners.
    def _collect_id_to_corners(self, gray: np.ndarray) -> Dict[int, np.ndarray]:
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        out: Dict[int, np.ndarray] = {}
        if ids is None:
            return out
        for marker_corners, marker_id_arr in zip(corners, ids):
            out[int(marker_id_arr[0])] = marker_corners.reshape(4, 2).astype(np.float32)
        return out

    # This function detects detect markers multipass.
    def _detect_markers_multipass(self, gray: np.ndarray) -> Dict[int, np.ndarray]:
        id_to_corners = self._collect_id_to_corners(gray)
        if ARUCO_USE_MULTI_PASS and len(id_to_corners) < len(KNOWN_MARKER_IDS):
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
            id2 = self._collect_id_to_corners(clahe)
            if len(id2) > len(id_to_corners):
                id_to_corners = id2
            if len(id_to_corners) < len(KNOWN_MARKER_IDS):
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                id3 = self._collect_id_to_corners(blur)
                if len(id3) > len(id_to_corners):
                    id_to_corners = id3
        return id_to_corners

    # This function handles estimate pose from marker centers.
    def _estimate_pose_from_marker_centers(self, id_to_corners: Dict[int, np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[int, np.ndarray]]:
        obj_pts = []
        img_pts = []
        marker_centers_2d: Dict[int, np.ndarray] = {}
        for mid in KNOWN_MARKER_IDS:
            corners = id_to_corners.get(mid)
            if corners is None:
                continue
            c2d = corners.mean(axis=0)
            marker_centers_2d[mid] = c2d
            obj_pts.append(self.id_to_center_3d[mid])
            img_pts.append(c2d)

        if len(obj_pts) < 3:
            return None, None, marker_centers_2d

        obj = np.asarray(obj_pts, dtype=np.float32)
        img = np.asarray(img_pts, dtype=np.float32)
        K = get_camera_matrix(self.frame_w, self.frame_h).astype(np.float32)
        dist = np.zeros((5, 1), dtype=np.float32)

        flag = cv2.SOLVEPNP_SQPNP if len(obj_pts) >= 3 else cv2.SOLVEPNP_EPNP
        success, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=flag)
        if not success and len(obj_pts) >= 4:
            success, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_EPNP)
        if not success:
            return None, None, marker_centers_2d

        if len(obj_pts) >= 4:
            ok_refine, rvec2, tvec2 = cv2.solvePnP(
                obj, img, K, dist, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if ok_refine:
                rvec, tvec = rvec2, tvec2
        return rvec, tvec, marker_centers_2d

    # This function handles project points.
    def _project_points(self, pts3d: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        K = get_camera_matrix(self.frame_w, self.frame_h).astype(np.float32)
        dist = np.zeros((5, 1), dtype=np.float32)
        pts2d, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
        return pts2d.reshape(-1, 2).astype(np.float32)

    # This function updates update tracking.
    def _update_tracking(self, frame: np.ndarray, now: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict[int, np.ndarray]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        id_to_corners = self._detect_markers_multipass(gray)
        self.last_detected_ids = sorted(id_to_corners.keys())

        rvec, tvec, marker_centers = self._estimate_pose_from_marker_centers(id_to_corners)
        if rvec is not None and tvec is not None:
            top_img = self._project_points(self.top_plane_3d, rvec, tvec)
            back_poly = self._project_points(self.back_wall_3d, rvec, tvec)

            dst = np.array([
                [0, 0],
                [PAPER_W, 0],
                [PAPER_W, PAPER_H],
                [0, PAPER_H],
            ], dtype=np.float32)
            H = cv2.getPerspectiveTransform(top_img.astype(np.float32), dst)
            H_inv = cv2.getPerspectiveTransform(dst, top_img.astype(np.float32))

            self.last_good_H = H
            self.last_good_H_inv = H_inv
            self.last_top_quad = top_img
            self.last_back_wall_poly = back_poly
            self.last_marker_centers_2d = marker_centers
            self.last_pose_time = now
            return H, H_inv, top_img, back_poly, marker_centers

        if self.last_good_H is not None and (now - self.last_pose_time) <= POSE_HOLD_SEC:
            return self.last_good_H, self.last_good_H_inv, self.last_top_quad, self.last_back_wall_poly, self.last_marker_centers_2d
        return None, None, None, None, marker_centers

    # This function changes paper points to image points.
    def paper_to_image(self, H_inv: np.ndarray, points: np.ndarray) -> np.ndarray:
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        out = cv2.perspectiveTransform(pts, H_inv)
        return out.reshape(-1, 2)

    # This function changes image points to paper points.
    def image_to_paper(self, H: np.ndarray, point_xy: Tuple[float, float]) -> Optional[np.ndarray]:
        pts = np.array([[point_xy]], dtype=np.float32)
        out = cv2.perspectiveTransform(pts, H)[0, 0]
        if np.any(np.isnan(out)) or np.any(np.isinf(out)):
            return None
        return out

    # This function builds build key rects.
    def build_key_rects(self) -> List[Tuple[float, float, float, float]]:
        rects: List[Tuple[float, float, float, float]] = []
        if KEYBOARD_STACK_VERTICAL:
            total_h = KEYBOARD_Y1 - KEYBOARD_Y0
            key_h = total_h / NUM_WHITE_KEYS
            for k in range(NUM_WHITE_KEYS):
                y0 = KEYBOARD_Y0 + k * key_h
                y1 = KEYBOARD_Y0 + (k + 1) * key_h
                rects.append((KEYBOARD_X0, y0, KEYBOARD_X1, y1))
        else:
            total_w = KEYBOARD_X1 - KEYBOARD_X0
            key_w = total_w / NUM_WHITE_KEYS
            for k in range(NUM_WHITE_KEYS):
                x0 = KEYBOARD_X0 + k * key_w
                x1 = KEYBOARD_X0 + (k + 1) * key_w
                rects.append((x0, KEYBOARD_Y0, x1, KEYBOARD_Y1))
        return rects

    # This function finds locate key.
    def locate_key(self, paper_pt: np.ndarray) -> Optional[int]:
        x, y = float(paper_pt[0]), float(paper_pt[1])
        if not (KEYBOARD_X0 <= x <= KEYBOARD_X1 and KEYBOARD_Y0 <= y <= KEYBOARD_Y1):
            return None
        if KEYBOARD_STACK_VERTICAL:
            key_h = (KEYBOARD_Y1 - KEYBOARD_Y0) / NUM_WHITE_KEYS
            idx = int((y - KEYBOARD_Y0) / key_h)
        else:
            key_w = (KEYBOARD_X1 - KEYBOARD_X0) / NUM_WHITE_KEYS
            idx = int((x - KEYBOARD_X0) / key_w)
        return max(0, min(NUM_WHITE_KEYS - 1, idx))

    # This function handles is occluded by back plane.
    def _is_occluded_by_back_plane(self, point_xy: Tuple[float, float], back_poly: Optional[np.ndarray]) -> bool:
        if back_poly is None:
            return False
        contour = back_poly.reshape(-1, 1, 2).astype(np.float32)
        val = cv2.pointPolygonTest(contour, (float(point_xy[0]), float(point_xy[1])), False)
        return val >= 0.0

    # This function draws draw overlay.
    def draw_overlay(self, frame: np.ndarray, H_inv: Optional[np.ndarray], top_quad: Optional[np.ndarray], back_poly: Optional[np.ndarray], marker_centers: Dict[int, np.ndarray], now: float) -> None:
        for mid, c in marker_centers.items():
            p = tuple(np.int32(c))
            cv2.circle(frame, p, 6, (0, 255, 255), -1)
            cv2.putText(frame, str(mid), (p[0] + 4, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        if back_poly is not None:
            poly = np.int32(back_poly)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly], (10, 80, 180))
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0.0, frame)
            cv2.polylines(frame, [poly], True, (20, 130, 255), 2)

        if top_quad is not None:
            cv2.polylines(frame, [np.int32(top_quad)], True, (255, 255, 0), 2)

        if H_inv is None:
            return

        for idx, (x0, y0, x1, y1) in enumerate(self.build_key_rects()):
            poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
            poly_img = self.paper_to_image(H_inv, poly)
            color = (40, 220, 40) if now < self.active_until[idx] else (255, 255, 255)
            thickness = 3 if now < self.active_until[idx] else 2
            cv2.polylines(frame, [np.int32(poly_img)], True, color, thickness)

    # This function draws draw key labels.
    def draw_key_labels(self, frame: np.ndarray, H_inv: Optional[np.ndarray], now: float, mirrored_display: bool) -> None:
        if H_inv is None:
            return
        h, w = frame.shape[:2]
        for idx, (x0, y0, x1, y1) in enumerate(self.build_key_rects()):
            tx = 0.5 * (x0 + x1)
            ty = 0.5 * (y0 + y1)
            p = self.paper_to_image(H_inv, np.array([[tx, ty]], dtype=np.float32))[0]
            x_img, y_img = float(p[0]), float(p[1])
            if mirrored_display:
                x_img = (w - 1) - x_img
            label = NOTE_NAMES[idx]
            color = (40, 220, 40) if now < self.active_until[idx] else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.72
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
            org = (int(x_img - 0.5 * tw), int(y_img + 0.5 * th))
            cv2.putText(frame, label, org, font, scale, color, thickness)

    # This function draws draw status.
    def draw_status(self, frame: np.ndarray, H_ok: bool) -> None:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (690, 132), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0.0, frame)

        status = "rig locked" if H_ok else "looking for rig markers"
        cv2.putText(frame, f"Status: {status}", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
        cv2.putText(frame, "V3: downward pulse + back-plane occlusion gate", (28, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (28, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 2)
        ids_text = ",".join(str(i) for i in self.last_detected_ids) if self.last_detected_ids else "-"
        cv2.putText(frame, f"Aruco IDs: {ids_text}", (260, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 2)
        cv2.putText(frame, "Press rule: fast downward movement in one key lane.", (w - 560, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)

    # This function handles process hands.
    def process_hands(self, frame: np.ndarray, H: Optional[np.ndarray], back_poly: Optional[np.ndarray], now: float) -> None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection = predict(frame_rgb)
        if DEBUG_DRAW_RAW_HAND and detection is not None:
            draw_landmarks_on_image(frame, detection)
        hands = getattr(detection, "hand_landmarks", None) if detection is not None else None

        current_ids = set()
        if not hands:
            for fid in list(self.finger_states.keys()):
                self.finger_states.pop(fid, None)
            return

        h, w = frame.shape[:2]
        for hand_idx, hand_landmarks in enumerate(hands):
            for tip_id in FINGERTIP_IDS:
                fid = (hand_idx, tip_id)
                current_ids.add(fid)

                lm = hand_landmarks[tip_id]
                raw_x = float(lm.x * w)
                raw_y = float(lm.y * h)

                st = self.finger_states.get(fid)
                if st is None:
                    st = {
                        "sx": raw_x,
                        "sy": raw_y,
                        "prev_t": now,
                        "speed_ema": 0.0,
                        "vy_ema": 0.0,
                        "key_id": -1.0,
                        "armed_key": -1.0,
                        "arm_y": raw_y,
                        "arm_time": now,
                        "pressed_latch": 0.0,
                        "outside_since": -1.0,
                    }
                    self.finger_states[fid] = st

                prev_x = float(st["sx"])
                prev_y = float(st["sy"])
                dt = max(now - float(st["prev_t"]), 1e-4)

                sx = TIP_SMOOTH_ALPHA * raw_x + (1.0 - TIP_SMOOTH_ALPHA) * prev_x
                sy = TIP_SMOOTH_ALPHA * raw_y + (1.0 - TIP_SMOOTH_ALPHA) * prev_y
                step = float(np.hypot(sx - prev_x, sy - prev_y))
                if step > MAX_STEP_PX:
                    sx = prev_x
                    sy = prev_y
                    step = 0.0

                inst_speed = step / dt
                inst_vy = (sy - prev_y) / dt
                speed_ema = SPEED_EMA_ALPHA * inst_speed + (1.0 - SPEED_EMA_ALPHA) * float(st["speed_ema"])
                vy_ema = SPEED_EMA_ALPHA * inst_vy + (1.0 - SPEED_EMA_ALPHA) * float(st["vy_ema"])

                st["sx"] = sx
                st["sy"] = sy
                st["prev_t"] = now
                st["speed_ema"] = speed_ema
                st["vy_ema"] = vy_ema

                px, py = int(round(sx)), int(round(sy))
                cv2.circle(frame, (px, py), 6, (0, 128, 255), -1)
                cv2.circle(frame, (int(round(raw_x)), int(round(raw_y))), 3, (255, 180, 0), -1)

                occluded = self._is_occluded_by_back_plane((sx, sy), back_poly)
                curr_key = None
                if H is not None and not occluded:
                    p = self.image_to_paper(H, (sx, sy))
                    if p is not None:
                        curr_key = self.locate_key(p)
                        if curr_key is not None:
                            cv2.putText(frame, NOTE_NAMES[curr_key], (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)

                if curr_key is None:
                    st["armed_key"] = -1.0
                    st["key_id"] = -1.0
                    if float(st["outside_since"]) < 0.0:
                        st["outside_since"] = now
                    if now - float(st["outside_since"]) >= REARM_OUTSIDE_SEC:
                        st["pressed_latch"] = 0.0
                    if DEBUG_DRAW_PRESS_INFO:
                        cv2.putText(frame, f"vy:{vy_ema:4.0f} a:0 l:{int(st['pressed_latch']>0.5)} o:{int(occluded)}", (px + 8, py + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 220, 120), 1)
                    continue

                st["outside_since"] = -1.0

                prev_key = int(st["key_id"]) if st["key_id"] >= 0.0 else None
                if prev_key != curr_key:
                    st["key_id"] = float(curr_key)
                    st["armed_key"] = -1.0
                    st["pressed_latch"] = 0.0

                latched = bool(st["pressed_latch"] > 0.5)
                armed_key = int(st["armed_key"]) if st["armed_key"] >= 0.0 else None
                if (not latched) and vy_ema >= DOWN_ARM_SPEED_PX and armed_key != curr_key:
                    st["armed_key"] = float(curr_key)
                    st["arm_y"] = sy
                    st["arm_time"] = now

                armed_key = int(st["armed_key"]) if st["armed_key"] >= 0.0 else None
                if armed_key == curr_key:
                    arm_age = now - float(st["arm_time"])
                    travel = sy - float(st["arm_y"])
                    if arm_age > DOWN_MAX_WINDOW_SEC:
                        st["armed_key"] = -1.0
                    elif vy_ema >= DOWN_FIRE_SPEED_PX and travel >= DOWN_MIN_TRAVEL_PX:
                        if now - self.last_hit_time[curr_key] > KEY_COOLDOWN_SEC:
                            self.last_hit_time[curr_key] = now
                            self.active_until[curr_key] = now + ACTIVE_FLASH_SEC
                            self.synth.play(curr_key)
                            print(f"Played {NOTE_NAMES[curr_key]}")
                        st["pressed_latch"] = 1.0
                        st["armed_key"] = -1.0

                if DEBUG_DRAW_PRESS_INFO:
                    a = 1 if st["armed_key"] >= 0.0 else 0
                    l = 1 if st["pressed_latch"] > 0.5 else 0
                    cv2.putText(frame, f"vy:{vy_ema:4.0f} a:{a} l:{l} o:{int(occluded)}", (px + 8, py + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 220, 120), 1)

        stale = [fid for fid in self.finger_states if fid not in current_ids]
        for fid in stale:
            self.finger_states.pop(fid, None)

    # This function runs the full app loop.
    def run(self) -> None:
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    break

                self.frame_h, self.frame_w = frame.shape[:2]
                now = time.time()
                dt = max(now - self.last_fps_time, 1e-6)
                self.fps = 1.0 / dt
                self.last_fps_time = now

                H, H_inv, top_quad, back_poly, marker_centers = self._update_tracking(frame, now)
                self.draw_overlay(frame, H_inv, top_quad, back_poly, marker_centers, now)
                self.process_hands(frame, H, back_poly, now)

                display = cv2.flip(frame, 1) if MIRROR_DISPLAY else frame
                self.draw_key_labels(display, H_inv, now, mirrored_display=MIRROR_DISPLAY)
                self.draw_status(display, H is not None)
                cv2.imshow("Paper Piano V3", display)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break
        finally:
            self.close()


if __name__ == "__main__":
    PaperPianoV3().run()
