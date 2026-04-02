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
PREFERRED_CAM_INDEX = 1
EXTERNAL_CAMERA_FIRST = False
CAM_WIDTH = 1280
CAM_HEIGHT = 720

# ── Paper coordinate system (LANDSCAPE, paper rotated 90° CCW from portrait) ──
# A4 landscape: 297mm wide × 210mm tall.
# Outer marker corners span 281mm (W) × 194mm (H) after 8mm margin.
# Scale: 3.914 px/mm (X), 4.124 px/mm (Y)
PAPER_W = 1100   # landscape width  (was 800 in portrait)
PAPER_H = 800    # landscape height (was 1100 in portrait)

# ── Keyboard area (landscape) ─────────────────────────────────────────────────
# Keys run LEFT→RIGHT (C4 at left, F5 at right).
# Black keys are at the TOP of the keyboard area.
# Labels (C D E…) are at the BOTTOM.
#
# Physical measurements (same PDF, now in landscape):
#   Keyboard horizontal span: 19.55 cm (was vertical in portrait)
#     Left gap : 52.5 mm from paper edge → (52.5-8)/281*1100 = 174 px
#     Right end: (297-49  -8)/281*1100  =               940 px
#   Keyboard vertical span  : 13.8  cm (was horizontal in portrait)
#     Top gap : (210-138)/2 = 36 mm  → (36-8)/194*800  = 115 px
#     Bottom  :                          800-115         = 685 px
KEYBOARD_X0 = 174
KEYBOARD_X1 = 940
KEYBOARD_Y0 = 115
KEYBOARD_Y1 = 685
NUM_WHITE_KEYS = 11
KEYBOARD_STACK_VERTICAL = False   # keys vary in X (left→right)

# ── Individual white-key widths (paper-space pixels, X direction) ─────────────
# Same physical proportions as the portrait heights; now they are widths.
# C4→F5: [1.70, 1.50, 1.56, 1.58, 1.68, 1.65, 1.65, 1.55, 1.50, 1.50, 2.70] cm
# Scaled to 766 px total (= KEYBOARD_X1 - KEYBOARD_X0):
KEY_W_PX = [70, 62, 64, 65, 69, 68, 68, 64, 62, 62, 112]   # must sum to 766

def _make_key_x_boundaries():
    b = [KEYBOARD_X0]
    for w in KEY_W_PX:
        b.append(b[-1] + w)
    return b

KEY_X_BOUNDARIES = _make_key_x_boundaries()   # 12 values
KEY_X_START = KEY_X_BOUNDARIES[:NUM_WHITE_KEYS]
KEY_X_END   = KEY_X_BOUNDARIES[1:]

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
MIRROR_DISPLAY = False

# ── ArUco corner mapping for LANDSCAPE paper (90° CCW from portrait) ──────────
# Portrait markers: 0=TL, 1=TR, 2=BR, 3=BL.
# After 90° CCW rotation on the table:
#   portrait TL(0) → landscape BL
#   portrait TR(1) → landscape TL   ← camera top-left
#   portrait BR(2) → landscape TR   ← camera top-right
#   portrait BL(3) → landscape BR   ← camera bottom-right
# ARUCO_CORNER_INDEX: which corner pixel of the marker to use as the paper corner.
#   landscape TL = marker-1's top-right corner  (index 1)
#   landscape TR = marker-2's bottom-right corner (index 2)
#   landscape BR = marker-3's bottom-left corner (index 3)
#   landscape BL = marker-0's top-left corner   (index 0)
ARUCO_CORNER_IDS = {
    "tl": 1,
    "tr": 2,
    "br": 3,
    "bl": 0,
}
ARUCO_CORNER_INDEX = {
    "tl": 1,
    "tr": 2,
    "br": 3,
    "bl": 0,
}

NOTE_FREQS = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25, 587.33, 659.25, 698.46]
NOTE_NAMES = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5", "F5"]

# ── Black-key visual overlay (landscape) ─────────────────────────────────────
# Black keys are centred on X-boundaries between adjacent white keys.
# They occupy the TOP portion of the keyboard height (the "black key" side of the paper).
#
# Portrait: black key X-ratio 0.46→0.97 from label-side (LEFT) to black-key-side (RIGHT).
# After 90° CCW:  portrait LEFT → landscape BOTTOM,  portrait RIGHT → landscape TOP.
# In landscape Y: top-of-black = KEYBOARD_Y0 + 3% of height
#                 bot-of-black = KEYBOARD_Y0 + 54% of height
BLACK_KEY_AFTER_WHITE = [0, 1, 3, 4, 5, 7, 8]

_KBD_H = KEYBOARD_Y1 - KEYBOARD_Y0   # 570 px
BLACK_KEY_Y0 = KEYBOARD_Y0 + int(_KBD_H * 0.03)   # ≈ 132 (just below top edge)
BLACK_KEY_Y1 = KEYBOARD_Y0 + int(_KBD_H * 0.54)   # ≈ 423 (halfway down)

# Black key width (X direction) ≈ 2.1 cm → 82 px, centred on key boundary
_AVG_KEY_W_PX = (KEYBOARD_X1 - KEYBOARD_X0) / NUM_WHITE_KEYS   # ≈ 69.6 px
BLACK_KEY_W_PX = _AVG_KEY_W_PX * 1.18                           # ≈ 82 px


def build_black_key_rects() -> List[Tuple[float, float, float, float]]:
    """Return black-key rectangles in landscape paper-space coordinates.

    Each black key is centred on the X-boundary between two adjacent white keys
    and occupies the top portion of the keyboard height (the black-key side).
    """
    rects = []
    half_w = BLACK_KEY_W_PX * 0.5
    for white_idx in BLACK_KEY_AFTER_WHITE:
        boundary_x = KEY_X_BOUNDARIES[white_idx + 1]
        x0 = boundary_x - half_w
        x1 = boundary_x + half_w
        rects.append((x0, BLACK_KEY_Y0, x1, BLACK_KEY_Y1))
    return rects


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

        # Hand-tracking stabilization state
        self.smoothed_fingertips: Dict[Tuple[int, int], np.ndarray] = {}
        self.candidate_key_by_finger: Dict[Tuple[int, int], Optional[int]] = {}
        self.stable_count_by_finger: Dict[Tuple[int, int], int] = {}

        # Multi-finger press state machine
        self.prev_raw_fingertips: Dict[Tuple[int, int], np.ndarray] = {}
        self.finger_state: Dict[Tuple[int, int], str] = {}
        self.finger_down_key: Dict[Tuple[int, int], Optional[int]] = {}

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

    def build_key_rects(self) -> List[Tuple[float, float, float, float]]:
        """One rectangle per white key; keys run left→right (X direction)."""
        return [
            (KEY_X_START[k], KEYBOARD_Y0, KEY_X_END[k], KEYBOARD_Y1)
            for k in range(NUM_WHITE_KEYS)
        ]

    def locate_key(self, paper_pt: np.ndarray) -> Optional[int]:
        """Return white-key index (0-10) under paper_pt, or None."""
        x, y = float(paper_pt[0]), float(paper_pt[1])
        if not (KEYBOARD_X0 <= x <= KEYBOARD_X1 and KEYBOARD_Y0 <= y <= KEYBOARD_Y1):
            return None
        for k in range(NUM_WHITE_KEYS):
            if KEY_X_START[k] <= x < KEY_X_END[k]:
                return k
        return NUM_WHITE_KEYS - 1

    def locate_key_stable(self, paper_pt: np.ndarray) -> Optional[int]:
        """Stable trigger zone: reject fingertips near left/right key edges."""
        x, y = float(paper_pt[0]), float(paper_pt[1])
        if not (KEYBOARD_X0 <= x <= KEYBOARD_X1 and KEYBOARD_Y0 <= y <= KEYBOARD_Y1):
            return None
        for k in range(NUM_WHITE_KEYS):
            if KEY_X_START[k] <= x < KEY_X_END[k]:
                key_w = KEY_X_END[k] - KEY_X_START[k]
                inner_pos = (x - KEY_X_START[k]) / key_w
                margin = KEY_EDGE_MARGIN_RATIO
                if inner_pos < margin or inner_pos > (1.0 - margin):
                    return None
                return k
        return None

    def locate_key_hold(self, paper_pt: np.ndarray) -> Optional[int]:
        """Hold zone: slightly wider than stable trigger zone."""
        x, y = float(paper_pt[0]), float(paper_pt[1])
        if not (KEYBOARD_X0 <= x <= KEYBOARD_X1 and KEYBOARD_Y0 <= y <= KEYBOARD_Y1):
            return None
        hold_margin = max(0.04, KEY_EDGE_MARGIN_RATIO * 0.55)
        for k in range(NUM_WHITE_KEYS):
            if KEY_X_START[k] <= x < KEY_X_END[k]:
                key_w = KEY_X_END[k] - KEY_X_START[k]
                inner_pos = (x - KEY_X_START[k]) / key_w
                if inner_pos < hold_margin or inner_pos > (1.0 - hold_margin):
                    return None
                return k
        return None

    def draw_paper_overlay(self, frame: np.ndarray, H_inv: Optional[np.ndarray], markers: Optional[Dict[str, np.ndarray]], now: float) -> None:
        if markers is not None:
            for name, pt in markers.items():
                cv2.circle(frame, tuple(np.int32(pt)), 8, (0, 255, 255), -1)
                cv2.putText(frame, name, tuple(np.int32(pt + np.array([6, -6], dtype=np.float32))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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

        for x0, y0, x1, y1 in build_black_key_rects():
            poly = np.array([
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
            ], dtype=np.float32)
            poly_img = self.paper_to_image(H_inv, poly)
            cv2.fillConvexPoly(frame, np.int32(poly_img), (25, 25, 25))
            cv2.polylines(frame, [np.int32(poly_img)], True, (255, 255, 255), 1)

    def draw_key_labels(self, frame: np.ndarray, H_inv: Optional[np.ndarray], now: float, mirrored_display: bool) -> None:
        if H_inv is None:
            return

        h, w = frame.shape[:2]
        key_rects = self.build_key_rects()
        for idx, (x0, y0, x1, y1) in enumerate(key_rects):
            # Landscape: horizontally centred in key, vertically in label area (bottom ~78%)
            tx_paper = 0.5 * (x0 + x1)
            ty_paper = KEYBOARD_Y0 + (KEYBOARD_Y1 - KEYBOARD_Y0) * 0.78

            text_pos_paper = np.array([[tx_paper, ty_paper]], dtype=np.float32)
            text_pos_img = self.paper_to_image(H_inv, text_pos_paper)[0]
            x_img = float(text_pos_img[0])
            y_img = float(text_pos_img[1])
            if mirrored_display:
                x_img = (w - 1) - x_img

            color = (40, 220, 40) if now < self.active_until[idx] else (255, 255, 255)
            label = NOTE_NAMES[idx]
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.72
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
            org = (int(x_img - 0.5 * tw), int(y_img + 0.5 * th))
            cv2.putText(frame, label, org, font, scale, color, thickness)

    def draw_status(self, frame: np.ndarray, H_ok: bool) -> None:
        h, w = frame.shape[:2]
        panel_h = 110
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (560, 12 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0.0, frame)

        status = "paper locked" if H_ok else "looking for 4 ArUco corner markers"
        cv2.putText(frame, f"Status: {status}", (28, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Multi-finger piano mode: stabilized press/hold/release", (28, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (28, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
        ids_text = ",".join(str(i) for i in self.last_detected_ids) if self.last_detected_ids else "-"
        cv2.putText(frame, f"Aruco IDs: {ids_text}", (300, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

        tips_text = "Use one fingertip first. Keep all 4 ArUco markers (ID 0/1/2/3) visible."
        cv2.putText(frame, tips_text, (w - 600, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def process_hands(self, frame: np.ndarray, H: Optional[np.ndarray], now: float) -> None:
        current_ids = set()

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
                    cv2.putText(frame, NOTE_NAMES[shown_key], (px + 10, py - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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
                    self.draw_paper_overlay(frame, H_inv, markers, now)
                    self.process_hands(frame, H, now)
                    display_frame = cv2.flip(frame, 1) if MIRROR_DISPLAY else frame
                    self.draw_key_labels(display_frame, H_inv, now, mirrored_display=MIRROR_DISPLAY)
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
