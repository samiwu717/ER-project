import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import pygame
except Exception:
    pygame = None

from prediction import predict


# =========================
# User-tunable parameters
# =========================
# Camera selection:
# Camera selection for your current Windows setup:
#   index=0 -> laptop camera
#   index=1 -> Camo camera
#   index=2 -> external USB camera
# We force DSHOW in open_camera() to avoid hanging on the USB camera.
PREFERRED_CAM_INDEX = 2
EXTERNAL_CAMERA_FIRST = False
CAM_WIDTH = 1280
CAM_HEIGHT = 720

# Canonical paper coordinate system for the current printed template (landscape).
# IMPORTANT:
# These coordinates are no longer based on an A4 portrait sheet.
# They are based on the marker-corner rectangle of the printed piano template
# shown in the reference photo, so the virtual overlay matches the paper keyboard.
# Back to portrait (vertical) layout
PAPER_W = 800
PAPER_H = 1100

# Keyboard area tuned to the uploaded portrait PDF template.
# Based on the visual layout in keyboard with markers.pdf:
# - 11 white keys stacked top -> bottom
# - labels on the left
# - black keys occupy the right side of the white-key area
KEYBOARD_X0 = 92
KEYBOARD_X1 = 720
KEYBOARD_Y0 = 190
KEYBOARD_Y1 = 1045
NUM_WHITE_KEYS = 11
KEYBOARD_STACK_VERTICAL = True

# Keyboard outer-rectangle tracking (in paper coordinates).
# We first track the printed big box, then place all internal keys by ratio.
KEYBOARD_RECT_HOLD_SEC = 0.60
KEYBOARD_RECT_SMOOTH_ALPHA = 0.32
KEYBOARD_SEARCH_MARGIN_X = 0.42
KEYBOARD_SEARCH_MARGIN_Y = 0.30
KEYBOARD_MIN_AREA_RATIO = 0.35
KEYBOARD_MAX_AREA_RATIO = 2.10
KEYBOARD_MIN_SIDE_RATIO = 0.55
KEYBOARD_MAX_SIDE_RATIO = 1.80

# Use all four fingertips for a simple first version
FINGERTIP_IDS = [8, 12, 16, 20]

# Trigger behavior
KEY_COOLDOWN_SEC = 0.12
ACTIVE_FLASH_SEC = 0.20
HOMOGRAPHY_HOLD_SEC = 0.40
SMOOTH_ALPHA = 0.35
MARKER_HOLD_SEC = 0.45
HAND_SMOOTH_ALPHA = 0.35
KEY_EDGE_MARGIN_RATIO = 0.10
KEY_STABLE_FRAMES = 3
PRESS_DY_THRESHOLD = 6.0
RELEASE_DY_THRESHOLD = -2.0
MAX_CONSECUTIVE_READ_FAILS = 10

# ArUco marker settings (printed A4 template)
ARUCO_DICT_NAME = cv2.aruco.DICT_5X5_50
ARUCO_CORNER_IDS = {
    "tl": 0,
    "tr": 1,
    "br": 2,
    "bl": 3,
}
# For each corner marker, which marker corner corresponds to paper corner.
# Marker corner order: [top-left, top-right, bottom-right, bottom-left]
ARUCO_CORNER_INDEX = {
    "tl": 0,
    "tr": 1,
    "br": 2,
    "bl": 3,
}
ARUCO_USE_MULTI_PASS = True
MIRROR_DISPLAY = True

NOTE_FREQS = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25, 587.33, 659.25, 698.46]
NOTE_NAMES = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5", "F5"]

# Optional: draw black-key overlay so the virtual keyboard visually matches the printout better.
# For the portrait PDF, black keys sit BETWEEN white keys and extend from the right side inward.
# Black keys appear after: C, D, F, G, A, c, d
BLACK_KEY_AFTER_WHITE = [0, 1, 3, 4, 5, 7, 8]

# Portrait-template proportions estimated from the uploaded PDF.
BLACK_KEY_X0_RATIO = 0.56   # black key starts further to the right
BLACK_KEY_X1_RATIO = 0.92   # do not extend fully to the white-key edge
BLACK_KEY_HEIGHT_RATIO = 0.64  # slightly shorter, closer to the printed template


def open_camera(index: Optional[int] = None) -> cv2.VideoCapture:
    """Stable camera open for the current Windows setup.

    index=0 -> laptop camera
    index=1 -> Camo / virtual camera
    index=2 -> external USB camera

    We force DSHOW because CAP_ANY may hang on the USB camera.
    """
    if index is None:
        index = PREFERRED_CAM_INDEX

    print(f"[INFO] Opening camera index={index} with DSHOW backend", flush=True)
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index={index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    except Exception:
        pass

    print("[INFO] Testing first frame read...", flush=True)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(
            f"Camera index={index} opened but first frame read failed. "
            "Close Camera/Camo/Zoom/OBS and try again."
        )

    print(f"[OK] Camera ready: index={index}, shape={frame.shape}", flush=True)
    return cap



class SimpleSynth:
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
            self.enabled = False
            self.sounds = [None] * len(NOTE_FREQS)

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

    def play(self, key_idx: int) -> None:
        if not self.enabled:
            return
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


class PaperPiano:
    def __init__(self) -> None:
        self.cap = open_camera(None)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

        self.synth = SimpleSynth()
        self.prev_key_by_finger: Dict[Tuple[int, int], Optional[int]] = {}
        self.last_hit_time = [0.0] * NUM_WHITE_KEYS
        self.active_until = [0.0] * NUM_WHITE_KEYS
        self.keyboard_rect = np.array([KEYBOARD_X0, KEYBOARD_Y0, KEYBOARD_X1, KEYBOARD_Y1], dtype=np.float32)
        self.last_keyboard_rect_time: float = -999.0
        self.keyboard_rect_locked: bool = False

        # Hand-tracking stabilization state
        self.smoothed_fingertips: Dict[Tuple[int, int], np.ndarray] = {}
        self.candidate_key_by_finger: Dict[Tuple[int, int], Optional[int]] = {}
        self.stable_count_by_finger: Dict[Tuple[int, int], int] = {}

        # Multi-finger press state machine
        self.prev_raw_fingertips: Dict[Tuple[int, int], np.ndarray] = {}
        self.finger_state: Dict[Tuple[int, int], str] = {}
        self.finger_down_key: Dict[Tuple[int, int], Optional[int]] = {}
        self.hand_label_overlays: List[Tuple[int, float, float, Tuple[int, int, int]]] = []

        self.smoothed_markers: Dict[str, np.ndarray] = {}
        self.last_seen_marker_time: Dict[str, float] = {}
        self.last_good_H: Optional[np.ndarray] = None
        self.last_good_H_inv: Optional[np.ndarray] = None
        self.last_good_time: float = -999.0
        self.last_fps_time = time.time()
        self.fps = 0.0
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV aruco module is required. Please install opencv-contrib-python.")
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
        self.aruco_params = cv2.aruco.DetectorParameters()
        # Looser settings for printed-paper scenes under uneven lighting.
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
        self.last_detected_ids: List[int] = []

    @staticmethod
    def transform_display_point(
        x: float,
        y: float,
        w: int,
        h: int,
        mirror_horizontal: bool = False,
        flip_vertical: bool = False,
    ) -> Tuple[float, float]:
        if mirror_horizontal:
            x = (w - 1) - x
        if flip_vertical:
            y = (h - 1) - y
        return x, y

    def close(self) -> None:
        try:
            if self.cap is not None:
                self.cap.release()
        finally:
            self.synth.close()
        cv2.destroyAllWindows()

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

    def detect_markers(self, frame: np.ndarray, now: float) -> Optional[Dict[str, np.ndarray]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = self._extract_aruco_corner_points(gray)

        # Debug: show every detected marker corner point immediately,
        # even before all 4 required markers are found.
        for name, pt in detected.items():
            cv2.circle(frame, tuple(np.int32(pt)), 9, (0, 165, 255), -1)
            cv2.putText(
                frame,
                f"{name}:{ARUCO_CORNER_IDS[name]}",
                tuple(np.int32(pt + np.array([8, -8], dtype=np.float32))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2,
            )
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

    def update_homography(self, frame: np.ndarray, now: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
        markers = self.detect_markers(frame, now)
        if markers is not None:
            src = np.array([
                markers["tl"],
                markers["tr"],
                markers["br"],
                markers["bl"],
            ], dtype=np.float32)
            dst = np.array([
                [0, 0],
                [PAPER_W, 0],
                [PAPER_W, PAPER_H],
                [0, PAPER_H],
            ], dtype=np.float32)
            H = cv2.getPerspectiveTransform(src, dst)
            H_inv = cv2.getPerspectiveTransform(dst, src)
            self.last_good_H = H
            self.last_good_H_inv = H_inv
            self.last_good_time = now
            return H, H_inv, markers

        if self.last_good_H is not None and (now - self.last_good_time) <= HOMOGRAPHY_HOLD_SEC:
            return self.last_good_H, self.last_good_H_inv, None

        return None, None, None

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
        if self.keyboard_rect is None:
            rect = self._default_keyboard_rect()
        else:
            rect = self.keyboard_rect
        x0, y0, x1, y1 = [float(v) for v in rect]
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return x0, y0, x1, y1

    def _build_black_key_rects(self, keyboard_rect: Tuple[float, float, float, float]) -> List[Tuple[float, float, float, float]]:
        x0, y0, x1, y1 = keyboard_rect
        rects = []
        total_w = x1 - x0
        total_h = y1 - y0
        key_h = total_h / NUM_WHITE_KEYS

        black_x0 = x0 + total_w * BLACK_KEY_X0_RATIO
        black_x1 = x0 + total_w * BLACK_KEY_X1_RATIO
        black_h = key_h * BLACK_KEY_HEIGHT_RATIO

        for white_idx in BLACK_KEY_AFTER_WHITE:
            # Center each black key on the boundary between adjacent white keys.
            boundary_y = y0 + (white_idx + 1) * key_h
            by0 = boundary_y - 0.5 * black_h
            by1 = boundary_y + 0.5 * black_h
            rects.append((black_x0, by0, black_x1, by1))
        return rects

    def _detect_keyboard_rect(self, frame: np.ndarray, H: np.ndarray) -> Optional[np.ndarray]:
        """Detect printed keyboard big-box in paper coordinates."""
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
        bw = cv2.adaptiveThreshold(
            roi,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            7,
        )
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
        detected: Optional[np.ndarray] = None
        if H is not None:
            detected = self._detect_keyboard_rect(frame, H)

        if detected is not None:
            self.keyboard_rect = (
                KEYBOARD_RECT_SMOOTH_ALPHA * detected
                + (1.0 - KEYBOARD_RECT_SMOOTH_ALPHA) * self.keyboard_rect
            ).astype(np.float32)
            self.last_keyboard_rect_time = now
            self.keyboard_rect_locked = True
            return

        if (now - self.last_keyboard_rect_time) <= KEYBOARD_RECT_HOLD_SEC:
            self.keyboard_rect_locked = True
            return

        self.keyboard_rect = self._default_keyboard_rect()
        self.keyboard_rect_locked = False

    def build_key_rects(self) -> List[Tuple[float, float, float, float]]:
        rects = []
        x0, y0, x1, y1 = self._current_keyboard_rect()
        if KEYBOARD_STACK_VERTICAL:
            total_h = y1 - y0
            key_h = total_h / NUM_WHITE_KEYS
            for k in range(NUM_WHITE_KEYS):
                ky0 = y0 + k * key_h
                ky1 = y0 + (k + 1) * key_h
                rects.append((x0, ky0, x1, ky1))
        else:
            total_w = x1 - x0
            key_w = total_w / NUM_WHITE_KEYS
            for k in range(NUM_WHITE_KEYS):
                kx0 = x0 + k * key_w
                kx1 = x0 + (k + 1) * key_w
                rects.append((kx0, y0, kx1, y1))
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
        idx = max(0, min(NUM_WHITE_KEYS - 1, idx))
        return idx

    def locate_key_stable(self, paper_pt: np.ndarray) -> Optional[int]:
        """Stable trigger zone: narrower center region to avoid boundary jitter."""
        x, y = float(paper_pt[0]), float(paper_pt[1])
        x0, y0, x1, y1 = self._current_keyboard_rect()
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return None

        if KEYBOARD_STACK_VERTICAL:
            key_h = (y1 - y0) / NUM_WHITE_KEYS
            local = y - y0
            idx = int(local / key_h)
            idx = max(0, min(NUM_WHITE_KEYS - 1, idx))

            inner_pos = (local - idx * key_h) / key_h
            margin = KEY_EDGE_MARGIN_RATIO
            if inner_pos < margin or inner_pos > (1.0 - margin):
                return None
            return idx
        else:
            key_w = (x1 - x0) / NUM_WHITE_KEYS
            local = x - x0
            idx = int(local / key_w)
            idx = max(0, min(NUM_WHITE_KEYS - 1, idx))

            inner_pos = (local - idx * key_w) / key_w
            margin = KEY_EDGE_MARGIN_RATIO
            if inner_pos < margin or inner_pos > (1.0 - margin):
                return None
            return idx

    def locate_key_hold(self, paper_pt: np.ndarray) -> Optional[int]:
        """Hold zone: slightly larger than trigger zone so a pressed finger can stay down stably."""
        x, y = float(paper_pt[0]), float(paper_pt[1])
        x0, y0, x1, y1 = self._current_keyboard_rect()
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return None

        hold_margin = max(0.04, KEY_EDGE_MARGIN_RATIO * 0.55)

        if KEYBOARD_STACK_VERTICAL:
            key_h = (y1 - y0) / NUM_WHITE_KEYS
            local = y - y0
            idx = int(local / key_h)
            idx = max(0, min(NUM_WHITE_KEYS - 1, idx))
            inner_pos = (local - idx * key_h) / key_h
            if inner_pos < hold_margin or inner_pos > (1.0 - hold_margin):
                return None
            return idx
        else:
            key_w = (x1 - x0) / NUM_WHITE_KEYS
            local = x - x0
            idx = int(local / key_w)
            idx = max(0, min(NUM_WHITE_KEYS - 1, idx))
            inner_pos = (local - idx * key_w) / key_w
            if inner_pos < hold_margin or inner_pos > (1.0 - hold_margin):
                return None
            return idx

    def draw_paper_overlay(self, frame: np.ndarray, H_inv: Optional[np.ndarray], markers: Optional[Dict[str, np.ndarray]], now: float) -> None:
        if markers is not None:
            for name, pt in markers.items():
                cv2.circle(frame, tuple(np.int32(pt)), 8, (0, 255, 255), -1)

        if H_inv is None:
            return

        paper_quad = np.array([
            [0, 0],
            [PAPER_W, 0],
            [PAPER_W, PAPER_H],
            [0, PAPER_H],
        ], dtype=np.float32)
        paper_img = self.paper_to_image(H_inv, paper_quad)
        cv2.polylines(frame, [np.int32(paper_img)], True, (255, 255, 0), 2)

        kx0, ky0, kx1, ky1 = self._current_keyboard_rect()
        kpoly = np.array([
            [kx0, ky0],
            [kx1, ky0],
            [kx1, ky1],
            [kx0, ky1],
        ], dtype=np.float32)
        kpoly_img = self.paper_to_image(H_inv, kpoly)
        cv2.polylines(frame, [np.int32(kpoly_img)], True, (255, 255, 0), 2)

        key_rects = self.build_key_rects()
        for idx, (x0, y0, x1, y1) in enumerate(key_rects):
            poly = np.array([
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
            ], dtype=np.float32)
            poly_img = self.paper_to_image(H_inv, poly)

            color = (40, 220, 40) if now < self.active_until[idx] else (255, 255, 255)
            thickness = 3 if now < self.active_until[idx] else 2
            cv2.polylines(frame, [np.int32(poly_img)], True, color, thickness)

        for x0, y0, x1, y1 in self._build_black_key_rects((kx0, ky0, kx1, ky1)):
            poly = np.array([
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
            ], dtype=np.float32)
            poly_img = self.paper_to_image(H_inv, poly)
            cv2.fillConvexPoly(frame, np.int32(poly_img), (25, 25, 25))
            cv2.polylines(frame, [np.int32(poly_img)], True, (255, 255, 255), 1)

    def draw_marker_labels(
        self,
        frame: np.ndarray,
        markers: Optional[Dict[str, np.ndarray]],
        mirrored_display: bool,
        flipped_vertical: bool,
    ) -> None:
        if markers is None:
            return

        h, w = frame.shape[:2]
        for name, pt in markers.items():
            x_img = float(pt[0] + 6.0)
            y_img = float(pt[1] - 6.0)
            x_img, y_img = self.transform_display_point(
                x_img,
                y_img,
                w,
                h,
                mirror_horizontal=mirrored_display,
                flip_vertical=flipped_vertical,
            )
            cv2.putText(
                frame,
                name,
                (int(x_img), int(y_img)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

    def draw_key_labels(
        self,
        frame: np.ndarray,
        H_inv: Optional[np.ndarray],
        now: float,
        mirrored_display: bool,
        flipped_vertical: bool,
    ) -> None:
        if H_inv is None:
            return

        h, w = frame.shape[:2]
        key_rects = self.build_key_rects()
        for idx, (x0, y0, x1, y1) in enumerate(key_rects):
            if KEYBOARD_STACK_VERTICAL:
                tx_paper = 0.5 * (x0 + x1)
                ty_paper = 0.5 * (y0 + y1)
            else:
                tx_paper = 0.5 * (x0 + x1)
                ty_paper = y1 - 40

            text_pos_paper = np.array([[tx_paper, ty_paper]], dtype=np.float32)
            text_pos_img = self.paper_to_image(H_inv, text_pos_paper)[0]
            x_img = float(text_pos_img[0])
            y_img = float(text_pos_img[1])
            x_img, y_img = self.transform_display_point(
                x_img,
                y_img,
                w,
                h,
                mirror_horizontal=mirrored_display,
                flip_vertical=flipped_vertical,
            )

            color = (40, 220, 40) if now < self.active_until[idx] else (255, 255, 255)
            label = NOTE_NAMES[idx]
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.72
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
            org = (int(x_img - 0.5 * tw), int(y_img + 0.5 * th))
            cv2.putText(frame, label, org, font, scale, color, thickness)

    def draw_hand_labels(self, frame: np.ndarray, mirrored_display: bool, flipped_vertical: bool) -> None:
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        for key_idx, x_img, y_img, color in self.hand_label_overlays:
            x_img, y_img = self.transform_display_point(
                x_img,
                y_img,
                w,
                h,
                mirror_horizontal=mirrored_display,
                flip_vertical=flipped_vertical,
            )
            cv2.putText(frame, NOTE_NAMES[key_idx], (int(x_img), int(y_img)), font, scale, color, thickness)

    def draw_status(self, frame: np.ndarray, H_ok: bool) -> None:
        h, w = frame.shape[:2]
        panel_h = 110
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (560, 12 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0.0, frame)

        status = "paper locked" if H_ok else "looking for 4 ArUco corner markers"
        cv2.putText(frame, f"Status: {status}", (28, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        kb_status = "tracked" if self.keyboard_rect_locked else "fallback"
        cv2.putText(frame, f"Keyboard box: {kb_status}", (300, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 255, 180), 2)
        cv2.putText(frame, "Multi-finger piano mode: stabilized press/hold/release", (28, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (28, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
        ids_text = ",".join(str(i) for i in self.last_detected_ids) if self.last_detected_ids else "-"
        cv2.putText(frame, f"Aruco IDs: {ids_text}", (300, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

        tips_text = "Use one fingertip first. Keep all 4 ArUco markers (ID 0/1/2/3) visible."
        cv2.putText(frame, tips_text, (w - 600, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def process_hands(self, frame: np.ndarray, H: Optional[np.ndarray], now: float) -> None:
        current_ids = set()
        self.hand_label_overlays = []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection = predict(frame_rgb)
        hands = getattr(detection, "hand_landmarks", None) if detection is not None else None
        if not hands:
            stale = [fid for fid in list(self.prev_key_by_finger.keys()) if fid not in current_ids]
            for fid in stale:
                self.prev_key_by_finger.pop(fid, None)
                self.smoothed_fingertips.pop(fid, None)
                self.prev_raw_fingertips.pop(fid, None)
                self.candidate_key_by_finger.pop(fid, None)
                self.stable_count_by_finger.pop(fid, None)
                self.finger_state.pop(fid, None)
                self.finger_down_key.pop(fid, None)
            return

        h, w = frame.shape[:2]
        for hand_idx, hand_landmarks in enumerate(hands):
            for tip_id in FINGERTIP_IDS:
                finger_id = (hand_idx, tip_id)
                current_ids.add(finger_id)

                lm = hand_landmarks[tip_id]
                raw_pt = np.array([float(lm.x * w), float(lm.y * h)], dtype=np.float32)

                prev_raw = self.prev_raw_fingertips.get(finger_id)
                dy_img = 0.0 if prev_raw is None else float(raw_pt[1] - prev_raw[1])
                self.prev_raw_fingertips[finger_id] = raw_pt

                prev_pt = self.smoothed_fingertips.get(finger_id)
                if prev_pt is None:
                    smoothed_pt = raw_pt
                else:
                    smoothed_pt = (HAND_SMOOTH_ALPHA * raw_pt + (1.0 - HAND_SMOOTH_ALPHA) * prev_pt).astype(np.float32)
                self.smoothed_fingertips[finger_id] = smoothed_pt

                px = int(smoothed_pt[0])
                py = int(smoothed_pt[1])

                cv2.circle(frame, (px, py), 6, (0, 128, 255), -1)
                cv2.circle(frame, (px, py), 10, (255, 255, 255), 2)

                trigger_key = None
                hold_key = None
                if H is not None:
                    paper_pt = self.image_to_paper(H, (px, py))
                    if paper_pt is not None:
                        trigger_key = self.locate_key_stable(paper_pt)
                        hold_key = self.locate_key_hold(paper_pt)

                prev_candidate = self.candidate_key_by_finger.get(finger_id)
                if trigger_key == prev_candidate:
                    self.stable_count_by_finger[finger_id] = self.stable_count_by_finger.get(finger_id, 0) + 1
                else:
                    self.candidate_key_by_finger[finger_id] = trigger_key
                    self.stable_count_by_finger[finger_id] = 1

                state = self.finger_state.get(finger_id, "UP")
                down_key = self.finger_down_key.get(finger_id)
                stable_count = self.stable_count_by_finger.get(finger_id, 0)

                # Visual key label
                shown_key = down_key if state in ("DOWN", "HELD") and down_key is not None else trigger_key
                if shown_key is not None:
                    color = (0, 255, 0) if state in ("DOWN", "HELD") else (0, 220, 220)
                    self.hand_label_overlays.append((shown_key, float(px + 10), float(py - 10), color))

                # State machine per finger
                if state == "UP":
                    if trigger_key is not None:
                        self.finger_state[finger_id] = "HOVER"
                    else:
                        self.finger_state[finger_id] = "UP"

                elif state == "HOVER":
                    if trigger_key is None:
                        self.finger_state[finger_id] = "UP"
                    else:
                        press_ready = (stable_count >= KEY_STABLE_FRAMES and dy_img >= PRESS_DY_THRESHOLD)
                        if press_ready:
                            if now - self.last_hit_time[trigger_key] > KEY_COOLDOWN_SEC:
                                self.last_hit_time[trigger_key] = now
                                self.active_until[trigger_key] = now + ACTIVE_FLASH_SEC
                                self.synth.play(trigger_key)
                                print(f"Played {NOTE_NAMES[trigger_key]}")
                            self.finger_down_key[finger_id] = trigger_key
                            self.prev_key_by_finger[finger_id] = trigger_key
                            self.finger_state[finger_id] = "DOWN"
                        else:
                            self.finger_state[finger_id] = "HOVER"

                elif state == "DOWN":
                    if down_key is not None and hold_key == down_key:
                        self.finger_state[finger_id] = "HELD"
                    else:
                        self.finger_state[finger_id] = "UP"
                        self.finger_down_key[finger_id] = None
                        self.prev_key_by_finger[finger_id] = None

                elif state == "HELD":
                    if down_key is not None and hold_key == down_key:
                        self.finger_state[finger_id] = "HELD"
                    else:
                        # release when leaving the hold zone or moving upward clearly
                        if dy_img <= RELEASE_DY_THRESHOLD or hold_key != down_key:
                            self.finger_state[finger_id] = "UP"
                            self.finger_down_key[finger_id] = None
                            self.prev_key_by_finger[finger_id] = None

        stale = [fid for fid in list(self.prev_key_by_finger.keys()) if fid not in current_ids]
        for fid in stale:
            self.prev_key_by_finger.pop(fid, None)
            self.smoothed_fingertips.pop(fid, None)
            self.prev_raw_fingertips.pop(fid, None)
            self.candidate_key_by_finger.pop(fid, None)
            self.stable_count_by_finger.pop(fid, None)
            self.finger_state.pop(fid, None)
            self.finger_down_key.pop(fid, None)

    def run(self) -> None:
        consecutive_read_fails = 0
        try:
            while True:
                try:
                    ok, frame = self.cap.read()
                    if not ok or frame is None:
                        consecutive_read_fails += 1
                        print(f"[WARN] Camera read failed ({consecutive_read_fails}/{MAX_CONSECUTIVE_READ_FAILS})")
                        if consecutive_read_fails >= MAX_CONSECUTIVE_READ_FAILS:
                            print("[ERROR] Too many consecutive camera read failures. Exiting.")
                            break
                        cv2.waitKey(10)
                        continue

                    consecutive_read_fails = 0

                    now = time.time()
                    dt = max(now - self.last_fps_time, 1e-6)
                    self.fps = 1.0 / dt
                    self.last_fps_time = now

                    H, H_inv, markers = self.update_homography(frame, now)
                    self.update_keyboard_rect(frame, H, now)
                    self.draw_paper_overlay(frame, H_inv, markers, now)
                    self.process_hands(frame, H, now)
                    display_frame = cv2.flip(frame, 1) if MIRROR_DISPLAY else frame
                    display_frame = cv2.flip(display_frame, 0)
                    self.draw_marker_labels(display_frame, markers, mirrored_display=MIRROR_DISPLAY, flipped_vertical=True)
                    self.draw_key_labels(display_frame, H_inv, now, mirrored_display=MIRROR_DISPLAY, flipped_vertical=True)
                    self.draw_hand_labels(display_frame, mirrored_display=MIRROR_DISPLAY, flipped_vertical=True)
                    self.draw_status(display_frame, H is not None)
                    cv2.imshow("Paper Piano V1", display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord('q'):
                        print("[INFO] Exit requested by user.")
                        break

                except Exception as exc:
                    print(f"[ERROR] Exception inside main loop: {type(exc).__name__}: {exc}")
                    import traceback
                    traceback.print_exc()
                    break
        finally:
            self.close()


if __name__ == "__main__":
    print("[INFO] Starting Paper Piano...")
    print(f"[INFO] Preferred camera index = {PREFERRED_CAM_INDEX}")
    print(f"[INFO] Target resolution = {CAM_WIDTH}x{CAM_HEIGHT}")
    print("[INFO] Expected cameras: 0=laptop, 1=Camo, 2=USB", flush=True)
    PaperPiano().run()

