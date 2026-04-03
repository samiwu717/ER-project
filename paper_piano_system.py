import time
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Iterator

import cv2
import numpy as np
import mediapipe as mp

try:
    import pygame
except Exception:
    pygame = None

# =========================
# Defaults / configuration
# =========================
# Camera selection (prefer external USB cameras for better stability)
DEFAULT_TOP_CAM_INDEX = 2
DEFAULT_SIDE_CAM_INDEX = 3
DEFAULT_SINGLE_CAM_INDEX = 2
EXTERNAL_MIN_INDEX = 2
CAM_SCAN_MAX_INDEX = 8
CAM_WIDTH = 1280
CAM_HEIGHT = 720

# Keyboard / notes
NUM_KEYS = 11
NUM_WHITE_KEYS = 11
NOTE_NAMES = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5", "F5"]
NOTE_FREQS = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25, 587.33, 659.25, 698.46]

# Dual-cam press trigger thresholds as a ratio of frame height.
# 35 / 720 ~= 0.0486, 70 / 720 ~= 0.0972
PRESS_DIST_RATIO = 35.0 / 720.0
RELEASE_DIST_RATIO = 70.0 / 720.0
MIN_APPROACH_SPEED = 0.8
KEY_COOLDOWN_SEC = 0.08

# Single-cam pressure detection
KEYBOARD_Y_RATIO = 0.44
PRESS_THRESHOLD_RATIO = 5
PRESS_DEPTH_THRESHOLD = 0.46
RELEASE_DEPTH_THRESHOLD = 0.32

# Paper template / homography settings
PAPER_W = 800
PAPER_H = 1100
KEYBOARD_X0 = 92
KEYBOARD_X1 = 720
KEYBOARD_Y0 = 190
KEYBOARD_Y1 = 1045
KEYBOARD_STACK_VERTICAL = True
KEY_EDGE_MARGIN_RATIO = 0.06

# Keyboard outer-rectangle tracking
KEYBOARD_RECT_HOLD_SEC = 0.60
KEYBOARD_RECT_SMOOTH_ALPHA = 0.32
KEYBOARD_SEARCH_MARGIN_X = 0.42
KEYBOARD_SEARCH_MARGIN_Y = 0.30
KEYBOARD_MIN_AREA_RATIO = 0.35
KEYBOARD_MAX_AREA_RATIO = 2.10
KEYBOARD_MIN_SIDE_RATIO = 0.55
KEYBOARD_MAX_SIDE_RATIO = 1.80
KEYBOARD_RECT_UPDATE_EVERY_N_FRAMES = 3

# Optional black-key overlay to better match the printed template
BLACK_KEY_AFTER_WHITE = [0, 1, 3, 4, 5, 7, 8]
BLACK_KEY_X0_RATIO = 0.56
BLACK_KEY_X1_RATIO = 0.92
BLACK_KEY_HEIGHT_RATIO = 0.64

# Marker tracking smoothing
SMOOTH_ALPHA = 0.35
MARKER_HOLD_SEC = 0.45
HOMOGRAPHY_HOLD_SEC = 0.40
MARKER_STABLE_SEC = 0.85
MARKER_STABLE_MOVE_PX = 4.0
MARKER_BIG_MOVE_PX = 55.0
MARKER_BIG_MOVE_CONFIRM_FRAMES = 3
RETRACK_MIN_SEC = 2.2
STATUS_NOTICE_SEC = 2.6

# ArUco marker settings
ARUCO_DICT_NAME = cv2.aruco.DICT_5X5_50
ARUCO_CORNER_IDS = {"tl": 0, "tr": 1, "br": 2, "bl": 3}
ARUCO_CORNER_INDEX = {"tl": 0, "tr": 1, "br": 2, "bl": 3}
ARUCO_USE_MULTI_PASS = True

# Hand tracking
FINGERTIP_IDS = [8, 12, 16, 20]
PRIMARY_TIP_ID = 8  # index fingertip
HAND_SMOOTH_ALPHA = 0.45

# Display settings
MIRROR_TOP = True
MIRROR_SIDE = True
MIRROR_SINGLE = True
MIRROR_DISPLAY = True
MAX_CONSECUTIVE_READ_FAILS = 10


@dataclass
class PressState:
    is_down: bool = False
    last_dist: Optional[float] = None
    last_press_t: float = 0.0


@dataclass
class FingerObservation:
    finger_id: str
    tip_id: int
    hand_label: str
    point: np.ndarray


class SimpleSynth:
    def __init__(self) -> None:
        self.enabled = False
        self.sounds: List[Optional["pygame.mixer.Sound"]] = []
        if pygame is None:
            print("[WARN] pygame not installed, audio disabled")
            return
        try:
            pygame.mixer.pre_init(44100, -16, 2, 512)
            pygame.mixer.init()
            self.sounds = [self._make_tone(f) for f in NOTE_FREQS]
            self.enabled = True
        except Exception as exc:
            print(f"[WARN] Audio disabled: {exc}")
            self.enabled = False
            self.sounds = [None] * len(NOTE_FREQS)

    def _make_tone(self, freq: float, duration: float = 0.45, sr: int = 44100, volume: float = 0.45):
        n = int(sr * duration)
        t = np.linspace(0.0, duration, n, endpoint=False)
        wave = (
            1.00 * np.sin(2 * np.pi * freq * t)
            + 0.40 * np.sin(2 * np.pi * 2 * freq * t)
            + 0.20 * np.sin(2 * np.pi * 3 * freq * t)
        )
        wave /= 1.60

        attack = int(sr * 0.01)
        decay = int(sr * 0.08)
        release = int(sr * 0.12)
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

    def play(self, key_idx: int) -> None:
        if not self.enabled:
            return
        if 0 <= key_idx < len(self.sounds):
            snd = self.sounds[key_idx]
            if snd is not None:
                snd.play()

    def close(self) -> None:
        if pygame is not None:
            try:
                if pygame.mixer.get_init():
                    pygame.mixer.quit()
            except Exception:
                pass


class MediaPipeHandTracker:
    def __init__(self, max_num_hands: int = 2) -> None:
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def process(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)

    def draw(self, frame_bgr: np.ndarray, results) -> None:
        if not results.multi_hand_landmarks:
            return
        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    def get_primary_tip_px(self, frame_bgr: np.ndarray, results) -> Optional[np.ndarray]:
        if not results.multi_hand_landmarks:
            return None

        h, w = frame_bgr.shape[:2]
        best_tip = None
        best_area = -1.0
        for hand in results.multi_hand_landmarks:
            xs = [lm.x for lm in hand.landmark]
            ys = [lm.y for lm in hand.landmark]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            tip = hand.landmark[PRIMARY_TIP_ID]
            px = np.array([tip.x * w, tip.y * h], dtype=np.float32)
            if area > best_area:
                best_area = area
                best_tip = px
        return best_tip

    def iter_fingertips(self, frame_bgr: np.ndarray, results):
        if not results.multi_hand_landmarks:
            return
        h, w = frame_bgr.shape[:2]
        for hand in results.multi_hand_landmarks:
            for tip_id in FINGERTIP_IDS:
                lm = hand.landmark[tip_id]
                yield tip_id, np.array([lm.x * w, lm.y * h], dtype=np.float32)

    def iter_finger_observations(self, frame_bgr: np.ndarray, results) -> Iterator[FingerObservation]:
        if not results.multi_hand_landmarks:
            return

        handedness_list = results.multi_handedness or []
        h, w = frame_bgr.shape[:2]

        for hand_idx, hand in enumerate(results.multi_hand_landmarks):
            hand_label = f"hand{hand_idx}"
            if hand_idx < len(handedness_list):
                try:
                    hand_label = handedness_list[hand_idx].classification[0].label.lower()
                except Exception:
                    pass

            for tip_id in FINGERTIP_IDS:
                lm = hand.landmark[tip_id]
                yield FingerObservation(
                    finger_id=f"{hand_label}_{tip_id}",
                    tip_id=tip_id,
                    hand_label=hand_label,
                    point=np.array([lm.x * w, lm.y * h], dtype=np.float32),
                )

    def close(self) -> None:
        self.hands.close()


class ArucoKeyboardMapper:
    def __init__(self) -> None:
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Tuned detection parameters for better robustness
        if ARUCO_USE_MULTI_PASS:
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
            self.aruco_params.errorCorrectionRate = 0.8
            if hasattr(self.aruco_params, "useAruco3Detection"):
                self.aruco_params.useAruco3Detection = True
        
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.last_good_H: Optional[np.ndarray] = None
        self.last_good_H_inv: Optional[np.ndarray] = None
        self.last_good_time: float = -999.0
        self.keyboard_rect = np.array([KEYBOARD_X0, KEYBOARD_Y0, KEYBOARD_X1, KEYBOARD_Y1], dtype=np.float32)
        self.last_keyboard_rect_time: float = -999.0
        self.last_keyboard_rect_detect_frame: int = -999999
        self.keyboard_rect_locked: bool = False
        self.keyboard_rect_initialized: bool = False
        self.frame_counter: int = 0
        self.tracking_mode: str = "INIT"
        self.marker_stable_since: float = -1.0
        self.prev_markers_for_stability: Optional[Dict[str, np.ndarray]] = None
        self.locked_markers: Optional[Dict[str, np.ndarray]] = None
        self.retrack_start_time: float = -1.0
        self.big_move_frame_count: int = 0
        self.status_notice_text: str = ""
        self.status_notice_until: float = -1.0
        
        # Marker smoothing for stability
        self.smoothed_markers: Dict[str, np.ndarray] = {}
        self.last_seen_marker_time: Dict[str, float] = {}
        self.last_detected_ids: List[int] = []

    def _collect_id_to_corners(self, img: np.ndarray) -> Dict[int, np.ndarray]:
        corners, ids, _ = self.aruco_detector.detectMarkers(img)
        id_to_corners: Dict[int, np.ndarray] = {}
        if ids is None:
            return id_to_corners
        for marker_corners, marker_id_arr in zip(corners, ids):
            marker_id = int(marker_id_arr[0])
            id_to_corners[marker_id] = marker_corners.reshape(4, 2).astype(np.float32)
        return id_to_corners

    def _extract_aruco_corner_points(self, gray: np.ndarray) -> Dict[str, np.ndarray]:
        id_to_corners = self._collect_id_to_corners(gray)
        if ARUCO_USE_MULTI_PASS and len(id_to_corners) < 4:
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
            id_to_corners_clahe = self._collect_id_to_corners(clahe)
            if len(id_to_corners_clahe) > len(id_to_corners):
                id_to_corners = id_to_corners_clahe
            if len(id_to_corners) < 4:
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                id_to_corners_blur = self._collect_id_to_corners(blur)
                if len(id_to_corners_blur) > len(id_to_corners):
                    id_to_corners = id_to_corners_blur

        self.last_detected_ids = sorted(id_to_corners.keys())
        detected: Dict[str, np.ndarray] = {}
        for corner_name, marker_id in ARUCO_CORNER_IDS.items():
            marker_corners = id_to_corners.get(marker_id)
            if marker_corners is None:
                continue
            idx = ARUCO_CORNER_INDEX[corner_name]
            detected[corner_name] = marker_corners[idx].astype(np.float32)
        return detected

    @staticmethod
    def _copy_markers(markers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {name: pt.astype(np.float32).copy() for name, pt in markers.items()}

    @staticmethod
    def _marker_motion(markers_a: Dict[str, np.ndarray], markers_b: Dict[str, np.ndarray]) -> float:
        motions: List[float] = []
        for name in ["tl", "tr", "br", "bl"]:
            pa = markers_a.get(name)
            pb = markers_b.get(name)
            if pa is None or pb is None:
                continue
            motions.append(float(np.linalg.norm(pa - pb)))
        if not motions:
            return float("inf")
        return float(max(motions))

    def _reset_stability_window(self) -> None:
        self.marker_stable_since = -1.0
        self.prev_markers_for_stability = None

    def _update_marker_stability(self, markers: Optional[Dict[str, np.ndarray]], now: float) -> bool:
        if markers is None:
            self._reset_stability_window()
            return False

        if self.prev_markers_for_stability is None:
            self.prev_markers_for_stability = self._copy_markers(markers)
            self.marker_stable_since = now
            return False

        move_px = self._marker_motion(markers, self.prev_markers_for_stability)
        self.prev_markers_for_stability = self._copy_markers(markers)
        if move_px > MARKER_STABLE_MOVE_PX:
            self.marker_stable_since = now

        if self.marker_stable_since < 0.0:
            self.marker_stable_since = now
            return False
        return (now - self.marker_stable_since) >= MARKER_STABLE_SEC

    def _set_status_notice(self, text: str, now: float) -> None:
        self.status_notice_text = text
        self.status_notice_until = now + STATUS_NOTICE_SEC

    def _enter_locked_state(
        self,
        markers: Dict[str, np.ndarray],
        H: np.ndarray,
        H_inv: np.ndarray,
        now: float,
    ) -> None:
        self.tracking_mode = "LOCKED"
        self.locked_markers = self._copy_markers(markers)
        self.last_good_H = H
        self.last_good_H_inv = H_inv
        self.last_good_time = now
        self.big_move_frame_count = 0
        self._reset_stability_window()
        self._set_status_notice("Tracking stabilized: keyboard locked.", now)

    def _start_retrack(self, now: float) -> None:
        self.tracking_mode = "RETRACK"
        self.retrack_start_time = now
        self.big_move_frame_count = 0
        self._reset_stability_window()
        self._set_status_notice("Large marker movement detected. Re-tracking...", now)

    def detect_markers(self, frame: np.ndarray, now: float) -> Optional[Dict[str, np.ndarray]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = self._extract_aruco_corner_points(gray)
        found: Dict[str, np.ndarray] = {}
        for name in ["tl", "tr", "br", "bl"]:
            pt = detected.get(name)
            if pt is not None:
                if name in self.smoothed_markers:
                    pt = (SMOOTH_ALPHA * pt + (1.0 - SMOOTH_ALPHA) * self.smoothed_markers[name]).astype(np.float32)
                self.smoothed_markers[name] = pt
                self.last_seen_marker_time[name] = now
                found[name] = pt
                continue

            cached = self.smoothed_markers.get(name)
            cached_t = self.last_seen_marker_time.get(name, -1e9)
            if cached is not None and (now - cached_t) <= MARKER_HOLD_SEC:
                found[name] = cached
                continue
            return None
        return found

    def update_homography(
        self,
        frame: np.ndarray,
        now: float = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
        if now is None:
            now = time.time()
        self.frame_counter += 1

        markers = self.detect_markers(frame, now)
        if self.tracking_mode == "LOCKED":
            if markers is not None and self.locked_markers is not None:
                big_move_px = self._marker_motion(markers, self.locked_markers)
                if big_move_px >= MARKER_BIG_MOVE_PX:
                    self.big_move_frame_count += 1
                else:
                    self.big_move_frame_count = 0
                if self.big_move_frame_count >= MARKER_BIG_MOVE_CONFIRM_FRAMES:
                    self._start_retrack(now)

            if self.tracking_mode == "LOCKED":
                return self.last_good_H, self.last_good_H_inv, markers

        if markers is None:
            self._update_marker_stability(None, now)
            if self.last_good_H is not None and (now - self.last_good_time) <= HOMOGRAPHY_HOLD_SEC:
                return self.last_good_H, self.last_good_H_inv, None
            return None, None, None

        src = np.array([markers["tl"], markers["tr"], markers["br"], markers["bl"]], dtype=np.float32)
        dst = np.array([[0, 0], [PAPER_W, 0], [PAPER_W, PAPER_H], [0, PAPER_H]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        H_inv = cv2.getPerspectiveTransform(dst, src)
        self.last_good_H = H
        self.last_good_H_inv = H_inv
        self.last_good_time = now

        stable_ready = self._update_marker_stability(markers, now)
        retrack_wait_done = True
        if self.tracking_mode == "RETRACK":
            retrack_wait_done = (now - self.retrack_start_time) >= RETRACK_MIN_SEC

        keyboard_ready_for_lock = (
            self.keyboard_rect_initialized
            and (now - self.last_keyboard_rect_time) <= KEYBOARD_RECT_HOLD_SEC
        )
        if stable_ready and retrack_wait_done and keyboard_ready_for_lock:
            self._enter_locked_state(markers, H, H_inv, now)
            return self.last_good_H, self.last_good_H_inv, markers

        return H, H_inv, markers

    def paper_to_image(self, H_inv: np.ndarray, points: np.ndarray) -> np.ndarray:
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        out = cv2.perspectiveTransform(pts, H_inv)
        return out.reshape(-1, 2)

    def image_to_paper(self, H: np.ndarray, point_xy: Tuple[float, float]) -> Optional[np.ndarray]:
        pts = np.array([[point_xy]], dtype=np.float32)
        out = cv2.perspectiveTransform(pts, H)[0, 0]
        if np.any(np.isnan(out)) or np.any(np.isinf(out)):
            return None
        return out

    def _default_keyboard_rect(self) -> np.ndarray:
        return np.array([KEYBOARD_X0, KEYBOARD_Y0, KEYBOARD_X1, KEYBOARD_Y1], dtype=np.float32)

    def _current_keyboard_rect(self) -> Tuple[float, float, float, float]:
        rect = self.keyboard_rect if self.keyboard_rect is not None else self._default_keyboard_rect()
        x0, y0, x1, y1 = [float(v) for v in rect]
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return x0, y0, x1, y1

    def _build_black_key_rects(self, keyboard_rect: Tuple[float, float, float, float]) -> List[Tuple[float, float, float, float]]:
        x0, y0, x1, y1 = keyboard_rect
        rects: List[Tuple[float, float, float, float]] = []
        total_w = x1 - x0
        total_h = y1 - y0
        key_h = total_h / NUM_WHITE_KEYS

        black_x0 = x0 + total_w * BLACK_KEY_X0_RATIO
        black_x1 = x0 + total_w * BLACK_KEY_X1_RATIO
        black_h = key_h * BLACK_KEY_HEIGHT_RATIO

        for white_idx in BLACK_KEY_AFTER_WHITE:
            boundary_y = y0 + (white_idx + 1) * key_h
            by0 = boundary_y - 0.5 * black_h
            by1 = boundary_y + 0.5 * black_h
            rects.append((black_x0, by0, black_x1, by1))
        return rects

    def _detect_keyboard_rect(self, frame: np.ndarray, H: np.ndarray) -> Optional[np.ndarray]:
        paper = cv2.warpPerspective(frame, H, (PAPER_W, PAPER_H))
        gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)

        exp_x0, exp_y0, exp_x1, exp_y1 = self._current_keyboard_rect()
        exp_w = max(1.0, exp_x1 - exp_x0)
        exp_h = max(1.0, exp_y1 - exp_y0)
        sx0 = int(max(0, np.floor(exp_x0 - exp_w * KEYBOARD_SEARCH_MARGIN_X)))
        sx1 = int(min(PAPER_W, np.ceil(exp_x1 + exp_w * KEYBOARD_SEARCH_MARGIN_X)))
        sy0 = int(max(0, np.floor(exp_y0 - exp_h * KEYBOARD_SEARCH_MARGIN_Y)))
        sy1 = int(min(PAPER_H, np.ceil(exp_y1 + exp_h * KEYBOARD_SEARCH_MARGIN_Y)))
        if sx1 - sx0 < 20 or sy1 - sy0 < 20:
            return None

        roi = gray[sy0:sy1, sx0:sx1]
        roi = cv2.GaussianBlur(roi, (5, 5), 0)
        bw = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours_data = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]
        if not contours:
            return None

        d_x0, d_y0, d_x1, d_y1 = [float(v) for v in self._default_keyboard_rect()]
        d_w = max(1.0, d_x1 - d_x0)
        d_h = max(1.0, d_y1 - d_y0)
        d_area = d_w * d_h
        d_aspect = d_w / d_h
        exp_cx = 0.5 * (exp_x0 + exp_x1)
        exp_cy = 0.5 * (exp_y0 + exp_y1)

        best_rect: Optional[np.ndarray] = None
        best_score = -1e9
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            if peri < 100:
                continue
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx if len(approx) >= 4 else cnt)
            if w < 20 or h < 20:
                continue

            area = float(w * h)
            if area < d_area * KEYBOARD_MIN_AREA_RATIO or area > d_area * KEYBOARD_MAX_AREA_RATIO:
                continue

            aspect = float(w) / max(float(h), 1e-6)
            if aspect < d_aspect * KEYBOARD_MIN_SIDE_RATIO or aspect > d_aspect * KEYBOARD_MAX_SIDE_RATIO:
                continue

            cx = sx0 + x + 0.5 * w
            cy = sy0 + y + 0.5 * h
            center_penalty = abs(cx - exp_cx) / exp_w + abs(cy - exp_cy) / exp_h
            ratio_penalty = abs(np.log(max(aspect, 1e-6) / max(d_aspect, 1e-6)))
            area_score = min(area / d_area, 1.25)
            quad_bonus = 1.0 if len(approx) == 4 and cv2.isContourConvex(approx) else 0.0
            score = 2.0 * quad_bonus + 1.2 * area_score - 1.5 * center_penalty - 1.8 * ratio_penalty

            if score > best_score:
                best_score = score
                best_rect = np.array([sx0 + x, sy0 + y, sx0 + x + w, sy0 + y + h], dtype=np.float32)

        return best_rect

    def update_keyboard_rect(self, frame: np.ndarray, H: Optional[np.ndarray], now: float) -> None:
        if self.tracking_mode == "LOCKED":
            self.keyboard_rect_locked = self.keyboard_rect_initialized
            return

        detected: Optional[np.ndarray] = None
        if H is not None:
            detected = self._detect_keyboard_rect(frame, H)

        if detected is not None:
            if not self.keyboard_rect_initialized:
                self.keyboard_rect = detected.astype(np.float32)
                self.keyboard_rect_initialized = True
            else:
                self.keyboard_rect = (
                    KEYBOARD_RECT_SMOOTH_ALPHA * detected
                    + (1.0 - KEYBOARD_RECT_SMOOTH_ALPHA) * self.keyboard_rect
                ).astype(np.float32)
            self.last_keyboard_rect_time = now
            self.keyboard_rect_locked = True
            return

        if self.keyboard_rect_initialized and (now - self.last_keyboard_rect_time) <= KEYBOARD_RECT_HOLD_SEC:
            self.keyboard_rect_locked = True
            return

        if not self.keyboard_rect_initialized:
            self.keyboard_rect = self._default_keyboard_rect()
        self.keyboard_rect_locked = False

    def build_key_rects(self) -> List[Tuple[float, float, float, float]]:
        rects: List[Tuple[float, float, float, float]] = []
        x0, y0, x1, y1 = self._current_keyboard_rect()
        if KEYBOARD_STACK_VERTICAL:
            key_h = (y1 - y0) / NUM_WHITE_KEYS
            for k in range(NUM_WHITE_KEYS):
                rects.append((x0, y0 + k * key_h, x1, y0 + (k + 1) * key_h))
        else:
            key_w = (x1 - x0) / NUM_WHITE_KEYS
            for k in range(NUM_WHITE_KEYS):
                rects.append((x0 + k * key_w, y0, x0 + (k + 1) * key_w, y1))
        return rects

    def locate_key(self, paper_pt: np.ndarray) -> Optional[int]:
        x, y = float(paper_pt[0]), float(paper_pt[1])
        x0, y0, x1, y1 = self._current_keyboard_rect()
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return None

        if KEYBOARD_STACK_VERTICAL:
            key_h = (y1 - y0) / NUM_WHITE_KEYS
            idx = int((y - y0) / key_h)
        else:
            key_w = (x1 - x0) / NUM_WHITE_KEYS
            idx = int((x - x0) / key_w)
        return int(np.clip(idx, 0, NUM_WHITE_KEYS - 1))

    def locate_key_stable(self, paper_pt: np.ndarray) -> Optional[int]:
        x, y = float(paper_pt[0]), float(paper_pt[1])
        x0, y0, x1, y1 = self._current_keyboard_rect()
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return None

        if KEYBOARD_STACK_VERTICAL:
            key_h = (y1 - y0) / NUM_WHITE_KEYS
            local = y - y0
            idx = int(local / key_h)
            idx = int(np.clip(idx, 0, NUM_WHITE_KEYS - 1))
            inner_pos = (local - idx * key_h) / key_h
        else:
            key_w = (x1 - x0) / NUM_WHITE_KEYS
            local = x - x0
            idx = int(local / key_w)
            idx = int(np.clip(idx, 0, NUM_WHITE_KEYS - 1))
            inner_pos = (local - idx * key_w) / key_w

        if inner_pos < KEY_EDGE_MARGIN_RATIO or inner_pos > (1.0 - KEY_EDGE_MARGIN_RATIO):
            return None
        return idx

    def draw_paper_overlay(
        self,
        frame: np.ndarray,
        H_inv: Optional[np.ndarray],
        markers: Optional[Dict[str, np.ndarray]],
        now: float,
        active_key: Optional[int] = None,
    ) -> None:
        if markers is not None:
            for name, pt in markers.items():
                cv2.circle(frame, tuple(np.int32(pt)), 8, (0, 255, 255), -1)
                cv2.putText(
                    frame,
                    f"{name}:{ARUCO_CORNER_IDS[name]}",
                    tuple(np.int32(pt + np.array([8, -8], dtype=np.float32))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )

        if H_inv is None:
            return

        paper_quad = np.array([[0, 0], [PAPER_W, 0], [PAPER_W, PAPER_H], [0, PAPER_H]], dtype=np.float32)
        paper_img = self.paper_to_image(H_inv, paper_quad)
        cv2.polylines(frame, [np.int32(paper_img)], True, (255, 255, 0), 2)

        kx0, ky0, kx1, ky1 = self._current_keyboard_rect()
        kpoly = np.array([[kx0, ky0], [kx1, ky0], [kx1, ky1], [kx0, ky1]], dtype=np.float32)
        kpoly_img = self.paper_to_image(H_inv, kpoly)
        cv2.polylines(frame, [np.int32(kpoly_img)], True, (255, 255, 0), 2)

        for idx, (x0, y0, x1, y1) in enumerate(self.build_key_rects()):
            poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
            poly_img = self.paper_to_image(H_inv, poly)
            color = (0, 255, 255) if active_key == idx else (255, 255, 255)
            thickness = 3 if active_key == idx else 2
            cv2.polylines(frame, [np.int32(poly_img)], True, color, thickness)
            label_pos = np.mean(poly_img, axis=0).astype(int)

        for x0, y0, x1, y1 in self._build_black_key_rects((kx0, ky0, kx1, ky1)):
            poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
            poly_img = self.paper_to_image(H_inv, poly)
            cv2.fillConvexPoly(frame, np.int32(poly_img), (25, 25, 25))
            cv2.polylines(frame, [np.int32(poly_img)], True, (255, 255, 255), 1)


class ClickLineSelector:
    def __init__(self, window_name: str) -> None:
        self.window_name = window_name
        self.points: List[Tuple[int, int]] = []
        cv2.setMouseCallback(window_name, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(self.points) >= 2:
            self.points = []
        self.points.append((int(x), int(y)))

    def line(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if len(self.points) < 2:
            return None
        p1 = np.array(self.points[0], dtype=np.float32)
        p2 = np.array(self.points[1], dtype=np.float32)
        if np.linalg.norm(p2 - p1) < 1e-6:
            return None
        return p1, p2

    def reset(self) -> None:
        self.points = []


def open_camera(index: int) -> cv2.VideoCapture:
    """Open camera with robust error handling and fallback logic."""
    print(f"[INFO] Opening camera index={index}...", flush=True)
    
    # Try with DSHOW backend first (preferred for Windows)
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    
    # Fallback to default API if DSHOW fails
    if not cap.isOpened():
        print(f"[WARN] CAP_DSHOW failed, trying default backend...", flush=True)
        cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index={index}")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    
    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    except Exception:
        pass
    
    # Test multiple frame reads for robustness
    print("[INFO] Testing first frame read...", flush=True)
    for attempt in range(3):
        ok, frame = cap.read()
        if ok and frame is not None and frame.shape == (CAM_HEIGHT, CAM_WIDTH, 3):
            print(f"[OK] Camera {index} ready: shape={frame.shape}", flush=True)
            return cap
        if attempt < 2:
            time.sleep(0.1)
    
    cap.release()
    raise RuntimeError(
        f"Camera index={index} opened but frame read failed. "
        "Close other camera applications and try again."
    )


def detect_available_cameras(max_index: int = CAM_SCAN_MAX_INDEX) -> List[int]:
    available: List[int] = []
    for idx in range(max_index + 1):
        try:
            # Try with DSHOW first
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            
            # Fallback to default API
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
            
            if not cap.isOpened():
                cap.release()
                continue
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            
            # Try to read a frame
            for attempt in range(2):
                ok, frame = cap.read()
                if ok and frame is not None and frame.shape == (CAM_HEIGHT, CAM_WIDTH, 3):
                    available.append(idx)
                    print(f"[INFO] Detected camera {idx}: {frame.shape}")
                    break
                time.sleep(0.05)
            
            cap.release()
        except Exception as e:
            continue
    
    return available


def choose_two_camera_indices(
    top_idx: Optional[int],
    side_idx: Optional[int],
    scan_max_index: int,
    prefer_external_min_index: int = EXTERNAL_MIN_INDEX,
) -> Tuple[int, int]:
    if top_idx is not None and side_idx is not None and top_idx != side_idx:
        return top_idx, side_idx

    available = detect_available_cameras(scan_max_index)
    if len(available) < 2:
        raise RuntimeError(f"Need at least 2 cameras, found: {available}")

    external_first = [i for i in available if i >= prefer_external_min_index]
    fallback = [i for i in available if i < prefer_external_min_index]
    ordered = external_first + fallback

    if top_idx is None:
        top_idx = ordered[0]
    if side_idx is None:
        for idx in ordered:
            if idx != top_idx:
                side_idx = idx
                break

    if side_idx is None or top_idx == side_idx:
        raise RuntimeError(f"Could not pick two different cameras from: {available}")
    return int(top_idx), int(side_idx)


def signed_distance_point_to_line(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    ap = pt - a
    denom = float(np.linalg.norm(ab))
    if denom < 1e-6:
        return 1e9
    cross_z = float(ab[0] * ap[1] - ab[1] * ap[0])
    return cross_z / denom


def draw_line(frame: np.ndarray, line: Optional[Tuple[np.ndarray, np.ndarray]], color=(50, 220, 255), thickness: int = 2) -> None:
    if line is None:
        return
    cv2.line(frame, tuple(np.int32(line[0])), tuple(np.int32(line[1])), color, thickness)


def format_finger_label(obs: FingerObservation) -> str:
    return f"{obs.hand_label[0].upper()}{obs.tip_id}"


def transform_point_for_flips(
    pt: np.ndarray,
    frame_shape: Tuple[int, int, int],
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> np.ndarray:
    h, w = frame_shape[:2]
    x = float(pt[0])
    y = float(pt[1])
    if flip_horizontal:
        x = (w - 1) - x
    if flip_vertical:
        y = (h - 1) - y
    return np.array([x, y], dtype=np.float32)


def invert_hand_label(label: str) -> str:
    if label == "left":
        return "right"
    if label == "right":
        return "left"
    return label


def run_dual_camera_mode(top_cam: int, side_cam: int) -> None:
    """Dual camera mode:
    - Top camera: ArUco markers -> key detection
    - Bottom camera: hand pressure detection (distance to yellow line)
    """
    tracker_top = MediaPipeHandTracker(max_num_hands=2)
    tracker_side = MediaPipeHandTracker(max_num_hands=2)
    mapper = ArucoKeyboardMapper()
    synth = SimpleSynth()
    
    # Press state per fingertip
    press_states: Dict[str, PressState] = {}
    
    top_cap = open_camera(top_cam)
    side_cap = open_camera(side_cam)

    cv2.namedWindow("top")
    cv2.namedWindow("bottom")
    side_selector = ClickLineSelector("bottom")

    print("\n=== Integrated Paper Piano: Dual Camera Mode ===")
    print("Top camera:    ArUco + virtual keyboard + key detection")
    print("Bottom camera: click 2 points to draw press line, then detect press by fingertip-to-line distance")
    print("Controls: q quit | r reset press line\n")

    consecutive_read_fails = 0
    frame_count = 0
    fps_time = time.time()
    fps = 0.0

    try:
        while True:
            try:
                ok_top, top_frame = top_cap.read()
                ok_bottom, bottom_frame = side_cap.read()
                
                if not ok_top or top_frame is None or not ok_bottom or bottom_frame is None:
                    consecutive_read_fails += 1
                    print(f"[WARN] Camera read failed ({consecutive_read_fails}/{MAX_CONSECUTIVE_READ_FAILS})")
                    if consecutive_read_fails >= MAX_CONSECUTIVE_READ_FAILS:
                        print("[ERROR] Too many consecutive camera read failures. Exiting.")
                        break
                    cv2.waitKey(10)
                    continue

                consecutive_read_fails = 0
                frame_count += 1
                
                # Calculate FPS
                now = time.time()
                dt = max(now - fps_time, 1e-6)
                fps = 1.0 / dt
                fps_time = now

                # =============== TOP CAMERA: Key Detection ===============
                top_proc = top_frame.copy()

                # 1) Use the original top frame for ArUco / homography / keyboard projection.
                H, H_inv, markers = mapper.update_homography(top_proc, now)
                mapper.update_keyboard_rect(top_proc, H, now)

                # 2) Draw the projected keyboard on the original frame first.
                top_display = top_proc.copy()
                
                # Process all fingertips on top camera
                active_keys_top: Dict[str, Optional[int]] = {}
                active_pressed_key: Optional[int] = None
                mapper.draw_paper_overlay(top_display, H_inv, markers, now, active_key=active_pressed_key)

                # 3) Flip the projected result horizontally and vertically, then run MediaPipe on that view.
                top_hand_frame = cv2.flip(top_display, -1)
                res_top = tracker_top.process(top_hand_frame)
                tracker_top.draw(top_hand_frame, res_top)

                if res_top.multi_hand_landmarks and H is not None:
                    for obs in tracker_top.iter_finger_observations(top_hand_frame, res_top):
                        obs.hand_label = invert_hand_label(obs.hand_label)
                        obs.finger_id = f"{obs.hand_label}_{obs.tip_id}"
                        cam_x = int(obs.point[0])
                        cam_y = int(obs.point[1])

                        orig_point = transform_point_for_flips(
                            obs.point,
                            top_hand_frame.shape,
                            flip_horizontal=True,
                            flip_vertical=True,
                        )
                        paper_pt = mapper.image_to_paper(H, (float(orig_point[0]), float(orig_point[1])))
                        key_idx = mapper.locate_key_stable(paper_pt) if paper_pt is not None else None
                        active_keys_top[obs.finger_id] = key_idx

                        state = press_states.setdefault(obs.finger_id, PressState())
                        if state.is_down and key_idx is not None:
                            active_pressed_key = key_idx

                        if key_idx is not None:
                            color = (255, 0, 0) if not state.is_down else (0, 0, 255)
                            cv2.circle(top_hand_frame, (cam_x, cam_y), 10, color, -1)
                            text = f"{format_finger_label(obs)} {NOTE_NAMES[key_idx]}"
                        else:
                            color = (0, 255, 255)
                            cv2.circle(top_hand_frame, (cam_x, cam_y), 8, color, -1)
                            text = f"{format_finger_label(obs)}"

                        cv2.putText(top_hand_frame, text, (cam_x + 15, cam_y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    top_hand_frame = cv2.flip(top_display, -1)
                    if H_inv is not None:
                        for idx, (x0, y0, x1, y1) in enumerate(mapper.build_key_rects()):
                            poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
                            poly_img = mapper.paper_to_image(H_inv, poly)
                            label_pos = np.mean(poly_img, axis=0).astype(int)
                            flipped_x = top_hand_frame.shape[1] - 1 - label_pos[0]
                            flipped_y = top_hand_frame.shape[0] - 1 - label_pos[1]
                            color = (0, 255, 255) if active_pressed_key == idx else (255, 255, 255)
                            cv2.putText(top_hand_frame, NOTE_NAMES[idx], (flipped_x - 18, flipped_y + 6),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                if active_pressed_key is not None:
                    top_display = top_proc.copy()
                    mapper.draw_paper_overlay(top_display, H_inv, markers, now, active_key=active_pressed_key)
                    top_hand_frame = cv2.flip(top_display, -1)
                    tracker_top.draw(top_hand_frame, res_top)
                    if res_top.multi_hand_landmarks:
                        for obs in tracker_top.iter_finger_observations(top_hand_frame, res_top):
                            obs.hand_label = invert_hand_label(obs.hand_label)
                            obs.finger_id = f"{obs.hand_label}_{obs.tip_id}"
                            cam_x = int(obs.point[0])
                            cam_y = int(obs.point[1])
                            key_idx = active_keys_top.get(obs.finger_id)
                            state = press_states.setdefault(obs.finger_id, PressState())
                            if key_idx is not None:
                                color = (255, 0, 0) if not state.is_down else (0, 0, 255)
                                cv2.circle(top_hand_frame, (cam_x, cam_y), 10, color, -1)
                                text = f"{format_finger_label(obs)} {NOTE_NAMES[key_idx]}"
                            else:
                                color = (0, 255, 255)
                                cv2.circle(top_hand_frame, (cam_x, cam_y), 8, color, -1)
                                text = f"{format_finger_label(obs)}"
                            cv2.putText(top_hand_frame, text, (cam_x + 15, cam_y - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if H_inv is not None:
                    for idx, (x0, y0, x1, y1) in enumerate(mapper.build_key_rects()):
                        poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
                        poly_img = mapper.paper_to_image(H_inv, poly)
                        label_pos = np.mean(poly_img, axis=0).astype(int)
                        flipped_x = top_hand_frame.shape[1] - 1 - label_pos[0]
                        flipped_y = top_hand_frame.shape[0] - 1 - label_pos[1]
                        color = (0, 255, 255) if active_pressed_key == idx else (255, 255, 255)
                        cv2.putText(top_hand_frame, NOTE_NAMES[idx], (flipped_x - 18, flipped_y + 6),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                else:
                    if H_inv is not None:
                        for idx, (x0, y0, x1, y1) in enumerate(mapper.build_key_rects()):
                            poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
                            poly_img = mapper.paper_to_image(H_inv, poly)
                            label_pos = np.mean(poly_img, axis=0).astype(int)
                            flipped_x = top_hand_frame.shape[1] - 1 - label_pos[0]
                            flipped_y = top_hand_frame.shape[0] - 1 - label_pos[1]
                            color = (0, 255, 255) if active_pressed_key == idx else (255, 255, 255)
                            cv2.putText(top_hand_frame, NOTE_NAMES[idx], (flipped_x - 18, flipped_y + 6),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                top_display = top_hand_frame

                # =============== BOTTOM CAMERA: Pressure Detection ===============
                if MIRROR_SIDE:
                    bottom_frame = cv2.flip(bottom_frame, 1)
                
                side_h = max(1, bottom_frame.shape[0])
                press_line = side_selector.line()
                draw_line(bottom_frame, press_line, color=(0, 255, 255), thickness=3)
                
                # Detect hands on bottom camera
                res_bottom = tracker_side.process(bottom_frame)
                tracker_side.draw(bottom_frame, res_bottom)
                
                # Process fingertips on bottom camera for pressure
                if res_bottom.multi_hand_landmarks:
                    for obs in tracker_side.iter_finger_observations(bottom_frame, res_bottom):
                        px = int(obs.point[0])
                        py = int(obs.point[1])
                        state = press_states.setdefault(obs.finger_id, PressState())

                        dist_abs = None
                        approach_speed = 0.0
                        is_pressed = False

                        if press_line is not None:
                            signed_d = signed_distance_point_to_line(obs.point, press_line[0], press_line[1])
                            dist_abs = abs(signed_d) / float(side_h)
                            if state.last_dist is not None:
                                approach_speed = state.last_dist - dist_abs
                            state.last_dist = dist_abs
                            is_pressed = dist_abs <= PRESS_DIST_RATIO
                        else:
                            state.last_dist = None

                        key_idx = active_keys_top.get(obs.finger_id)
                        if is_pressed and not state.is_down:
                            state.is_down = True
                            if key_idx is not None and (now - state.last_press_t) >= KEY_COOLDOWN_SEC:
                                state.last_press_t = now
                                synth.play(key_idx)
                                print(f"[PRESS] {obs.finger_id}: Key {key_idx} ({NOTE_NAMES[key_idx]})")
                        elif state.is_down:
                            if dist_abs is None or dist_abs >= RELEASE_DIST_RATIO:
                                state.is_down = False

                        if state.is_down and key_idx is not None:
                            color = (0, 0, 255)
                        elif state.is_down:
                            color = (0, 255, 255)
                        elif key_idx is not None:
                            color = (255, 0, 0)
                        else:
                            color = (255, 255, 0)
                        cv2.circle(bottom_frame, (px, py), 10, color, -1)

                        if dist_abs is None:
                            status = "draw line"
                        elif state.is_down and key_idx is not None:
                            status = f"PRESS+KEY {dist_abs:.3f}"
                        elif state.is_down:
                            status = f"PRESS {dist_abs:.3f}"
                        else:
                            status = f"d={dist_abs:.3f} v={approach_speed:.3f}"
                        note_text = NOTE_NAMES[key_idx] if key_idx is not None else "-"
                        text = f"{format_finger_label(obs)} {note_text} {status}"
                        cv2.putText(bottom_frame, text, (px + 15, py - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # =============== Status Display ===============
                # Top camera status
                cv2.putText(top_display, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                cv2.putText(top_display, f"Homography: {'OK' if H is not None else 'Searching 4 ArUco corners...'}", 
                           (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if H is not None else (0,0,255), 2)
                cv2.putText(top_display, f"Tracking: {mapper.tracking_mode}", 
                           (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,255), 2)
                cv2.putText(top_display, f"Keyboard box: {'locked' if mapper.keyboard_rect_locked else 'acquiring'}", 
                           (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)
                cv2.putText(top_display, f"Aruco IDs: {','.join(map(str, mapper.last_detected_ids)) if mapper.last_detected_ids else '-'}", 
                           (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)
                if now <= mapper.status_notice_until and mapper.status_notice_text:
                    cv2.putText(top_display, mapper.status_notice_text,
                               (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120,255,120), 2)

                # Bottom camera status
                cv2.putText(bottom_frame, f"press<= {PRESS_DIST_RATIO:.3f} | release>= {RELEASE_DIST_RATIO:.3f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(bottom_frame, f"min approach speed= {MIN_APPROACH_SPEED:.2f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(bottom_frame, "BOTTOM: Click 2 points to draw the press line", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,255), 2)

                cv2.imshow("top", top_display)
                cv2.imshow("bottom", bottom_frame)

                k = cv2.waitKey(1) & 0xFF
                if k == 27 or k == ord("q"):
                    print("[INFO] Exit requested by user.")
                    break
                if k == ord("r"):
                    side_selector.reset()
                    for state in press_states.values():
                        state.is_down = False
                        state.last_dist = None
                    print("[INFO] Press line reset.")
            
            except Exception as exc:
                print(f"[ERROR] Exception in dual camera mode: {type(exc).__name__}: {exc}")
                import traceback
                traceback.print_exc()
                break
    
    finally:
        top_cap.release()
        side_cap.release()
        tracker_top.close()
        tracker_side.close()
        synth.close()
        cv2.destroyAllWindows()


def run_single_camera_mode(cam_idx: int) -> None:
    """Single camera mode:
    - Combines ArUco key detection + pressure detection in one frame
    - Top half: virtual keyboard
    - Bottom half: yellow line for pressure threshold
    """
    cap = open_camera(cam_idx)
    tracker = MediaPipeHandTracker(max_num_hands=2)
    mapper = ArucoKeyboardMapper()
    synth = SimpleSynth()
    
    # Press state per fingertip
    press_states: Dict[str, PressState] = {}
    cv2.namedWindow("Paper Piano - Single Camera")
    line_selector = ClickLineSelector("Paper Piano - Single Camera")
    
    consecutive_read_fails = 0
    frame_count = 0
    fps_time = time.time()
    fps = 0.0

    print("\n=== Integrated Paper Piano: Single Camera Mode ===")
    print("- Top: ArUco markers + key detection (all fingertips)")
    print("- Bottom: yellow line for pressure detection")
    print("Controls: q quit\n")

    try:
        while True:
            try:
                ok, frame = cap.read()
                if not ok or frame is None:
                    consecutive_read_fails += 1
                    print(f"[WARN] Camera read failed ({consecutive_read_fails}/{MAX_CONSECUTIVE_READ_FAILS})")
                    if consecutive_read_fails >= MAX_CONSECUTIVE_READ_FAILS:
                        print("[ERROR] Too many consecutive camera read failures. Exiting.")
                        break
                    cv2.waitKey(10)
                    continue

                consecutive_read_fails = 0
                frame_count += 1
                
                # Calculate FPS
                now = time.time()
                dt = max(now - fps_time, 1e-6)
                fps = 1.0 / dt
                fps_time = now

                proc_frame = frame.copy()
                h, w = proc_frame.shape[:2]
                
                # Update homography from ArUco markers on the original frame.
                H, H_inv, markers = mapper.update_homography(proc_frame, now)
                mapper.update_keyboard_rect(proc_frame, H, now)
                
                # Detect hands
                results = tracker.process(proc_frame)
                tracker.draw(proc_frame, results)
                
                # Draw keyboard overlay
                press_line = line_selector.line()
                draw_line(proc_frame, press_line, color=(0, 255, 255), thickness=3)
                
                # Process all fingertips
                active_pressed_key: Optional[int] = None
                proc_h = max(1, proc_frame.shape[0])

                if results.multi_hand_landmarks:
                    for obs in tracker.iter_finger_observations(proc_frame, results):
                        px = int(obs.point[0])
                        py = int(obs.point[1])

                        paper_pt = mapper.image_to_paper(H, (float(obs.point[0]), float(obs.point[1]))) if H is not None else None
                        key_idx = mapper.locate_key_stable(paper_pt) if paper_pt is not None else None

                        state = press_states.setdefault(obs.finger_id, PressState())
                        dist_abs = None
                        approach_speed = 0.0
                        is_press_event = False

                        if press_line is not None:
                            signed_d = signed_distance_point_to_line(obs.point, press_line[0], press_line[1])
                            dist_abs = abs(signed_d) / float(proc_h)
                            if state.last_dist is not None:
                                approach_speed = state.last_dist - dist_abs
                            state.last_dist = dist_abs
                            is_press_event = dist_abs <= PRESS_DIST_RATIO
                        else:
                            state.last_dist = None

                        if is_press_event and not state.is_down:
                            state.is_down = True
                            if key_idx is not None and (now - state.last_press_t) >= KEY_COOLDOWN_SEC:
                                state.last_press_t = now
                                synth.play(key_idx)
                                print(f"[PRESS] {obs.finger_id}: Key {key_idx} ({NOTE_NAMES[key_idx]})")
                        elif state.is_down:
                            if dist_abs is None or dist_abs >= RELEASE_DIST_RATIO:
                                state.is_down = False

                        if state.is_down and key_idx is not None:
                            active_pressed_key = key_idx

                        if state.is_down and key_idx is not None:
                            color = (0, 0, 255)
                        elif state.is_down:
                            color = (0, 255, 255)
                        elif key_idx is not None:
                            color = (255, 0, 0)
                        else:
                            color = (0, 255, 255)

                        cv2.circle(proc_frame, (px, py), 10, color, -1)

                        label = NOTE_NAMES[key_idx] if key_idx is not None else "-"
                        if dist_abs is None:
                            status = "draw line"
                        elif state.is_down and key_idx is not None:
                            status = f"PRESS+KEY {dist_abs:.3f}"
                        elif state.is_down:
                            status = f"PRESS {dist_abs:.3f}"
                        else:
                            status = f"d={dist_abs:.3f} v={approach_speed:.3f}"
                        text = f"{format_finger_label(obs)}:{label} {status}"
                        cv2.putText(proc_frame, text, (px + 15, py - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                mapper.draw_paper_overlay(proc_frame, H_inv, markers, now, active_key=active_pressed_key)

                # Draw status information
                cv2.putText(proc_frame, f"FPS: {fps:.1f} | Mode: Single Camera", 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                cv2.putText(proc_frame, f"press<= {PRESS_DIST_RATIO:.3f} | release>= {RELEASE_DIST_RATIO:.3f}",
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(proc_frame, f"Homography: {'OK' if H is not None else 'Searching 4 ArUco corners...'}", 
                           (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (0,255,0) if H is not None else (0,0,255), 2)
                cv2.putText(proc_frame, f"Tracking: {mapper.tracking_mode}", 
                           (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
                cv2.putText(proc_frame, f"Keyboard box: {'locked' if mapper.keyboard_rect_locked else 'acquiring'}", 
                           (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
                cv2.putText(proc_frame, f"Aruco IDs: {','.join(map(str, mapper.last_detected_ids)) if mapper.last_detected_ids else '-'}", 
                           (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
                if now <= mapper.status_notice_until and mapper.status_notice_text:
                    cv2.putText(proc_frame, mapper.status_notice_text,
                               (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120,255,120), 2)
                cv2.putText(proc_frame, "Red=Pressed | Blue=OnKey | Cyan=OffKey | Yellow=PressLine", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                display_frame = cv2.flip(proc_frame, 1) if MIRROR_SINGLE else proc_frame
                cv2.imshow("Paper Piano - Single Camera", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    print("[INFO] Exit requested by user.")
                    break
                if key == ord('r'):
                    line_selector.reset()
                    for state in press_states.values():
                        state.is_down = False
                        state.last_dist = None
                    print("[INFO] Press line reset.")
            
            except Exception as exc:
                print(f"[ERROR] Exception in single camera mode: {type(exc).__name__}: {exc}")
                import traceback
                traceback.print_exc()
                break
    
    finally:
        cap.release()
        tracker.close()
        synth.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrated paper piano system")
    parser.add_argument("--mode", choices=["dual", "single"], default="dual", help="dual: top+side cameras; single: one-camera fallback")
    parser.add_argument("--top-cam", type=int, default=DEFAULT_TOP_CAM_INDEX, help="Top camera index for dual mode")
    parser.add_argument("--side-cam", type=int, default=DEFAULT_SIDE_CAM_INDEX, help="Side camera index for dual mode")
    parser.add_argument("--cam", type=int, default=DEFAULT_SINGLE_CAM_INDEX, help="Single camera index for single mode")
    parser.add_argument("--scan-max", type=int, default=CAM_SCAN_MAX_INDEX, help="Max camera index for auto scan")
    parser.add_argument("--external-min", type=int, default=EXTERNAL_MIN_INDEX, help="Indices >= this are treated as external")
    args = parser.parse_args()

    if args.mode == "dual":
        top_idx, side_idx = choose_two_camera_indices(
            top_idx=args.top_cam,
            side_idx=args.side_cam,
            scan_max_index=int(max(1, args.scan_max)),
            prefer_external_min_index=args.external_min,
        )
        print(f"[INFO] Using cameras: top={top_idx}, side={side_idx}")
        run_dual_camera_mode(top_idx, side_idx)
    else:
        print(f"[INFO] Using single camera: {args.cam}")
        run_single_camera_mode(args.cam)


if __name__ == "__main__":
    main()
