"""This file tests a paper piano version with contact inference."""
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import pygame
except Exception:
    pygame = None

from prediction import extract_handedness, predict

# parameters
PREFERRED_CAM_INDEX = 2
EXTERNAL_CAMERA_FIRST = False
CAM_WIDTH = 1280
CAM_HEIGHT = 720

PAPER_W = 800
PAPER_H = 1100

KEYBOARD_X0 = 92
KEYBOARD_X1 = 720
KEYBOARD_Y0 = 190
KEYBOARD_Y1 = 1045
NUM_WHITE_KEYS = 11
KEYBOARD_STACK_VERTICAL = True

KEYBOARD_RECT_HOLD_SEC = 0.60
KEYBOARD_RECT_SMOOTH_ALPHA = 0.32
KEYBOARD_SEARCH_MARGIN_X = 0.42
KEYBOARD_SEARCH_MARGIN_Y = 0.30
KEYBOARD_MIN_AREA_RATIO = 0.35
KEYBOARD_MAX_AREA_RATIO = 2.10
KEYBOARD_MIN_SIDE_RATIO = 0.55
KEYBOARD_MAX_SIDE_RATIO = 1.80

FINGERTIP_IDS = [4, 8, 12, 16, 20]

KEY_COOLDOWN_SEC = 0.10
ACTIVE_FLASH_SEC = 0.20
HOMOGRAPHY_HOLD_SEC = 0.40
SMOOTH_ALPHA = 0.35
MARKER_HOLD_SEC = 0.45
HAND_SMOOTH_ALPHA = 0.40

KEY_EDGE_MARGIN_RATIO = 0.02          # change 0.06 → 0.02
KEY_STABLE_FRAMES = 2
MAX_CONSECUTIVE_READ_FAILS = 10
HAND_MIN_HANDEDNESS_SCORE = 0.60
PRESS_DY_SMOOTH_ALPHA = 0.45
HOLD_MISS_FRAMES = 3
RELEASE_STABLE_FRAMES = 2
FINGER_WARMUP_FRAMES = 5
SHOW_PRESS_DEBUG = True

CONTACT_USES_LOWER_V = True


DEPTH_WEIGHT = 0.45
CURL_WEIGHT = 0.25
APPROACH_WEIGHT = 0.20
DWELL_WEIGHT = 0.10

# tuning parameters
PRESS_DEPTH_RATIO = 0.35              # 0.60 → 0.35
RELEASE_DEPTH_RATIO = 0.25            # 0.45 → 0.25
APPROACH_VEL_THRESHOLD = 1.0          # 2.2 → 1.0
CONTACT_SCORE_THRESHOLD = 0.45        # 0.62 → 0.45
APPROACH_SCORE_THRESHOLD = 0.45
RELEASE_SCORE_THRESHOLD = 0.38
CONTACT_LOW_ZONE_FRAMES = 1           # 2 → 1
CONTACT_STABLE_FRAMES = 2
TIP_REF_SEPARATION_NORM = 24.0
TIP_REF_CURL_GATE_NORM = 8.0

ARUCO_DICT_NAME = cv2.aruco.DICT_5X5_50
ARUCO_CORNER_IDS = {"tl": 0, "tr": 1, "br": 2, "bl": 3}
ARUCO_CORNER_INDEX = {"tl": 0, "tr": 1, "br": 2, "bl": 3}
ARUCO_USE_MULTI_PASS = True
MIRROR_DISPLAY = True

NOTE_FREQS = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25, 587.33, 659.25, 698.46]
NOTE_NAMES = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5", "F5"]

BLACK_KEY_AFTER_WHITE = [0, 1, 3, 4, 5, 7, 8]
BLACK_KEY_X0_RATIO = 0.56
BLACK_KEY_X1_RATIO = 0.92
BLACK_KEY_HEIGHT_RATIO = 0.64

HandFingerId = Tuple[str, int]
FINGER_CHAINS = {
    4: (1, 2, 3, 4),
    8: (5, 6, 7, 8),
    12: (9, 10, 11, 12),
    16: (13, 14, 15, 16),
    20: (17, 18, 19, 20),
}


# opens the camera
def open_camera(index: Optional[int] = None) -> cv2.VideoCapture:
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
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"Camera index={index} opened but first frame read failed.")
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

    # produce tone for a given frequency
    def _make_tone(self, freq: float, duration: float = 0.55, sr: int = 44100, volume: float = 0.45):
        n = int(sr * duration)
        t = np.linspace(0.0, duration, n, endpoint=False)
        wave = (1.0 * np.sin(2 * np.pi * freq * t) +
                0.40 * np.sin(2 * np.pi * 2 * freq * t) +
                0.20 * np.sin(2 * np.pi * 3 * freq * t))
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

    # plays the note for a given key index
    def play(self, key_idx: int) -> None:
        if not self.enabled:
            return
        snd = self.sounds[key_idx]
        if snd is not None:
            snd.play()

    # closes and cleans up the audio system
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
        self.prev_key_by_finger: Dict[HandFingerId, Optional[int]] = {}
        self.last_hit_time = [0.0] * NUM_WHITE_KEYS
        self.active_until = [0.0] * NUM_WHITE_KEYS
        self.keyboard_rect = np.array([KEYBOARD_X0, KEYBOARD_Y0, KEYBOARD_X1, KEYBOARD_Y1], dtype=np.float32)
        self.last_keyboard_rect_time: float = -999.0
        self.keyboard_rect_locked: bool = False
        self.smoothed_fingertips: Dict[HandFingerId, np.ndarray] = {}
        self.candidate_key_by_finger: Dict[HandFingerId, Optional[int]] = {}
        self.stable_count_by_finger: Dict[HandFingerId, int] = {}
        self.prev_raw_fingertips: Dict[HandFingerId, np.ndarray] = {}
        self.prev_paper_tip_by_finger: Dict[HandFingerId, np.ndarray] = {}
        self.smoothed_paper_velocity_by_finger: Dict[HandFingerId, float] = {}
        self.finger_seen_frames: Dict[HandFingerId, int] = {}
        self.finger_state: Dict[HandFingerId, str] = {}
        self.finger_contact_key: Dict[HandFingerId, Optional[int]] = {}
        self.low_zone_count_by_finger: Dict[HandFingerId, int] = {}
        self.approach_count_by_finger: Dict[HandFingerId, int] = {}
        self.contact_count_by_finger: Dict[HandFingerId, int] = {}
        self.release_count_by_finger: Dict[HandFingerId, int] = {}
        self.hold_miss_count_by_finger: Dict[HandFingerId, int] = {}
        self.last_hit_time_by_finger_key: Dict[Tuple[HandFingerId, int], float] = {}
        self.hand_label_overlays: List[Tuple[int, float, float, Tuple[int, int, int]]] = []
        self.hand_name_overlays: List[Tuple[str, float, float, Tuple[int, int, int]]] = []
        self.press_debug_overlays: List[Tuple[List[str], float, float, Tuple[int, int, int]]] = []
        self.smoothed_markers: Dict[str, np.ndarray] = {}
        self.last_seen_marker_time: Dict[str, float] = {}
        self.last_good_H: Optional[np.ndarray] = None
        self.last_good_H_inv: Optional[np.ndarray] = None
        self.last_good_time: float = -999.0
        self.last_fps_time = time.time()
        self.fps = 0.0
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV aruco module required. Install opencv-contrib-python.")
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
        self.aruco_params.errorCorrectionRate = 0.8
        if hasattr(self.aruco_params, "useAruco3Detection"):
            self.aruco_params.useAruco3Detection = True
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.last_detected_ids: List[int] = []

    # changes transform display point
    @staticmethod
    def transform_display_point(x: float, y: float, w: int, h: int,
                                 mirror_horizontal: bool = False, flip_vertical: bool = False) -> Tuple[float, float]:
        if mirror_horizontal:
            x = (w - 1) - x
        if flip_vertical:
            y = (h - 1) - y
        return x, y

    # closes and cleans up resources
    def close(self) -> None:
        try:
            if self.cap is not None:
                self.cap.release()
        finally:
            self.synth.close()
        cv2.destroyAllWindows()

    # collects collect id to corners mapping
    def _collect_id_to_corners(self, img: np.ndarray) -> Dict[int, np.ndarray]:
        corners, ids, _ = self.aruco_detector.detectMarkers(img)
        id_to_corners: Dict[int, np.ndarray] = {}
        if ids is None:
            return id_to_corners
        for marker_corners, marker_id_arr in zip(corners, ids):
            marker_id = int(marker_id_arr[0])
            id_to_corners[marker_id] = marker_corners.reshape(4, 2).astype(np.float32)
        return id_to_corners

    # gets extract aruco corner points
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

    # detects detect markers and updates smoothing and hold state
    def detect_markers(self, frame: np.ndarray, now: float) -> Optional[Dict[str, np.ndarray]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = self._extract_aruco_corner_points(gray)
        for name, pt in detected.items():
            cv2.circle(frame, tuple(np.int32(pt)), 9, (0, 165, 255), -1)
            cv2.putText(frame, f"{name}:{ARUCO_CORNER_IDS[name]}",
                        tuple(np.int32(pt + np.array([8, -8], dtype=np.float32))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
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

    # updates update homography
    def update_homography(self, frame: np.ndarray, now: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
        markers = self.detect_markers(frame, now)
        if markers is not None:
            src = np.array([markers["tl"], markers["tr"], markers["br"], markers["bl"]], dtype=np.float32)
            dst = np.array([[0, 0], [PAPER_W, 0], [PAPER_W, PAPER_H], [0, PAPER_H]], dtype=np.float32)
            H = cv2.getPerspectiveTransform(src, dst)
            H_inv = cv2.getPerspectiveTransform(dst, src)
            self.last_good_H = H
            self.last_good_H_inv = H_inv
            self.last_good_time = now
            return H, H_inv, markers
        if self.last_good_H is not None and (now - self.last_good_time) <= HOMOGRAPHY_HOLD_SEC:
            return self.last_good_H, self.last_good_H_inv, None
        return None, None, None

    # changes paper points to image points
    def paper_to_image(self, H_inv: np.ndarray, points: np.ndarray) -> np.ndarray:
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        out = cv2.perspectiveTransform(pts, H_inv)
        return out.reshape(-1, 2)

    # changes image points to paper points
    def image_to_paper(self, H: np.ndarray, point_xy: Tuple[float, float]) -> Optional[np.ndarray]:
        pts = np.array([[point_xy]], dtype=np.float32)
        out = cv2.perspectiveTransform(pts, H)[0, 0]
        if np.any(np.isnan(out)) or np.any(np.isinf(out)):
            return None
        return out

    # the default default keyboard rect
    def _default_keyboard_rect(self) -> np.ndarray:
        return np.array([KEYBOARD_X0, KEYBOARD_Y0, KEYBOARD_X1, KEYBOARD_Y1], dtype=np.float32)

    # function gets the tracked finger ids
    def _tracked_finger_ids(self) -> List[HandFingerId]:
        tracked = set()
        for state_map in (self.prev_key_by_finger, self.smoothed_fingertips, self.prev_raw_fingertips,
                          self.prev_paper_tip_by_finger, self.smoothed_paper_velocity_by_finger,
                          self.finger_seen_frames, self.candidate_key_by_finger, self.stable_count_by_finger,
                          self.finger_state, self.finger_contact_key, self.low_zone_count_by_finger,
                          self.approach_count_by_finger, self.contact_count_by_finger, self.release_count_by_finger,
                          self.hold_miss_count_by_finger):
            tracked.update(state_map.keys())
        return list(tracked)

    # This function clears clear finger tracking state.
    @staticmethod
    def _clear_finger_tracking_state(finger_id: HandFingerId, *state_maps: Dict) -> None:
        for state_map in state_maps:
            state_map.pop(finger_id, None)

    # This function gets landmark values.
    @staticmethod
    def _landmark_xy(hand_landmarks, landmark_id: int, frame_w: int, frame_h: int) -> np.ndarray:
        lm = hand_landmarks[landmark_id]
        return np.array([float(lm.x * frame_w), float(lm.y * frame_h)], dtype=np.float32)

    # This function gets landmark values.
    @staticmethod
    def _landmark_xyz(hand_landmarks, landmark_id: int) -> np.ndarray:
        lm = hand_landmarks[landmark_id]
        return np.array([float(lm.x), float(lm.y), float(lm.z)], dtype=np.float32)

    # This function gets the joint angle.
    @staticmethod
    def _joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba = a - b
        bc = c - b
        ba_norm = np.linalg.norm(ba)
        bc_norm = np.linalg.norm(bc)
        if ba_norm < 1e-6 or bc_norm < 1e-6:
            return 180.0
        cos_theta = float(np.dot(ba, bc) / (ba_norm * bc_norm))
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return float(np.degrees(np.arccos(cos_theta)))

    # This function handles finger contact features.
    def _finger_contact_features(self, hand_landmarks, world_landmarks, tip_id: int,
                                 frame_w: int, frame_h: int, paper_pt: Optional[np.ndarray],
                                 finger_id: HandFingerId) -> Dict[str, float]:
        ref_id = self._ref_joint_id(tip_id)
        tip_2d = self._landmark_xy(hand_landmarks, tip_id, frame_w, frame_h)
        ref_2d = self._landmark_xy(hand_landmarks, ref_id, frame_w, frame_h)
        paper_axis_value = self._paper_axis_value(paper_pt)
        prev_paper_pt = self.prev_paper_tip_by_finger.get(finger_id)
        prev_axis_value = self._paper_axis_value(prev_paper_pt)
        approach_velocity = self._approach_velocity(prev_axis_value, paper_axis_value, finger_id)
        if paper_pt is not None:
            self.prev_paper_tip_by_finger[finger_id] = np.array(paper_pt, dtype=np.float32)
        tip_below_ref = float(tip_2d[1] - ref_2d[1])
        if world_landmarks is not None:
            tip_3d = self._landmark_xyz(world_landmarks, tip_id)
            ref_3d = self._landmark_xyz(world_landmarks, ref_id)
            if tip_id == 4:
                pip_angle = 180.0
                dip_angle = self._joint_angle_deg(self._landmark_xyz(world_landmarks, 2),
                                                  self._landmark_xyz(world_landmarks, 3), tip_3d)
            else:
                mcp_id, pip_id, dip_id, _ = FINGER_CHAINS[tip_id]
                pip_angle = self._joint_angle_deg(self._landmark_xyz(world_landmarks, mcp_id),
                                                  self._landmark_xyz(world_landmarks, pip_id),
                                                  self._landmark_xyz(world_landmarks, dip_id))
                dip_angle = self._joint_angle_deg(self._landmark_xyz(world_landmarks, pip_id),
                                                  self._landmark_xyz(world_landmarks, dip_id), tip_3d)
        else:
            tip_3d = self._landmark_xyz(hand_landmarks, tip_id)
            ref_3d = self._landmark_xyz(hand_landmarks, ref_id)
            if tip_id == 4:
                pip_angle = 180.0
                dip_angle = self._joint_angle_deg(self._landmark_xyz(hand_landmarks, 2),
                                                  self._landmark_xyz(hand_landmarks, 3), tip_3d)
            else:
                mcp_id, pip_id, dip_id, _ = FINGER_CHAINS[tip_id]
                pip_angle = self._joint_angle_deg(self._landmark_xyz(hand_landmarks, mcp_id),
                                                  self._landmark_xyz(hand_landmarks, pip_id),
                                                  self._landmark_xyz(hand_landmarks, dip_id))
                dip_angle = self._joint_angle_deg(self._landmark_xyz(hand_landmarks, pip_id),
                                                  self._landmark_xyz(hand_landmarks, dip_id), tip_3d)
        tip_depth_delta = float(ref_3d[2] - tip_3d[2])
        return {
            "paper_axis": 0.0 if paper_axis_value is None else float(paper_axis_value),
            "approach_velocity": float(approach_velocity),
            "tip_below_ref": float(tip_below_ref),
            "tip_depth_delta": float(tip_depth_delta),
            "pip_angle": float(pip_angle),
            "dip_angle": float(dip_angle),
        }

    # This function handles paper axis value.
    def _paper_axis_value(self, paper_pt: Optional[np.ndarray]) -> Optional[float]:
        if paper_pt is None:
            return None
        if KEYBOARD_STACK_VERTICAL:
            return float(paper_pt[1])
        return float(paper_pt[0])

    # This function handles key axis bounds.
    def _key_axis_bounds(self, key_idx: int) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.build_key_rects()[key_idx]
        if KEYBOARD_STACK_VERTICAL:
            return float(min(y0, y1)), float(max(y0, y1))
        return float(min(x0, x1)), float(max(x0, x1))

    # This function handles normalized depth in key.
    def _normalized_depth_in_key(self, key_idx: Optional[int], paper_axis_value: Optional[float]) -> float:
        if key_idx is None or paper_axis_value is None:
            return 0.0
        axis_lo, axis_hi = self._key_axis_bounds(key_idx)
        denom = max(axis_hi - axis_lo, 1e-6)
        if CONTACT_USES_LOWER_V:
            depth = (axis_hi - paper_axis_value) / denom
        else:
            depth = (paper_axis_value - axis_lo) / denom
        return float(np.clip(depth, 0.0, 1.2))

    # This function keeps the value in range.
    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    # This function handles approach velocity.
    def _approach_velocity(self, prev_axis: Optional[float], curr_axis: Optional[float], finger_id: HandFingerId) -> float:
        if prev_axis is None or curr_axis is None:
            vel = 0.0
        else:
            raw = (prev_axis - curr_axis) if CONTACT_USES_LOWER_V else (curr_axis - prev_axis)
            prev_smooth = self.smoothed_paper_velocity_by_finger.get(finger_id)
            vel = raw if prev_smooth is None else PRESS_DY_SMOOTH_ALPHA * raw + (1.0 - PRESS_DY_SMOOTH_ALPHA) * prev_smooth
        self.smoothed_paper_velocity_by_finger[finger_id] = float(vel)
        return float(vel)

    # This function handles ref joint id.
    def _ref_joint_id(self, tip_id: int) -> int:
        if tip_id == 4:
            return 3
        return FINGER_CHAINS[tip_id][1]

    # This function gets the current current keyboard rect.
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

    # This function builds build black key rects.
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
            boundary_y = y0 + (white_idx + 1) * key_h
            by0 = boundary_y - 0.5 * black_h
            by1 = boundary_y + 0.5 * black_h
            rects.append((black_x0, by0, black_x1, by1))
        return rects

    # This function detects detect keyboard rect.
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

    # This function updates update keyboard rect.
    def update_keyboard_rect(self, frame: np.ndarray, H: Optional[np.ndarray], now: float) -> None:
        detected: Optional[np.ndarray] = None
        if H is not None:
            detected = self._detect_keyboard_rect(frame, H)
        if detected is not None:
            self.keyboard_rect = (KEYBOARD_RECT_SMOOTH_ALPHA * detected +
                                  (1.0 - KEYBOARD_RECT_SMOOTH_ALPHA) * self.keyboard_rect).astype(np.float32)
            self.last_keyboard_rect_time = now
            self.keyboard_rect_locked = True
            return
        if (now - self.last_keyboard_rect_time) <= KEYBOARD_RECT_HOLD_SEC:
            self.keyboard_rect_locked = True
            return
        self.keyboard_rect = self._default_keyboard_rect()
        self.keyboard_rect_locked = False

    # This function builds build key rects.
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

    # This function finds locate key.
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

    # This function finds locate key stable.
    def locate_key_stable(self, paper_pt: np.ndarray) -> Optional[int]:
        """稳定触发区：边缘剔除比例缩小，按压更容易进入"""
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
            if inner_pos < margin or inner_pos > (1.0 - margin):
                return None
            return idx

    # This function finds locate key hold.
    def locate_key_hold(self, paper_pt: np.ndarray) -> Optional[int]:
        """保持区：比触发区稍宽，防止释放抖动"""
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

    # This function draws draw paper overlay.
    def draw_paper_overlay(self, frame: np.ndarray, H_inv: Optional[np.ndarray],
                           markers: Optional[Dict[str, np.ndarray]], now: float) -> None:
        if markers is not None:
            for name, pt in markers.items():
                cv2.circle(frame, tuple(np.int32(pt)), 8, (0, 255, 255), -1)
        if H_inv is None:
            return
        paper_quad = np.array([[0, 0], [PAPER_W, 0], [PAPER_W, PAPER_H], [0, PAPER_H]], dtype=np.float32)
        paper_img = self.paper_to_image(H_inv, paper_quad)
        cv2.polylines(frame, [np.int32(paper_img)], True, (255, 255, 0), 2)
        kx0, ky0, kx1, ky1 = self._current_keyboard_rect()
        kpoly = np.array([[kx0, ky0], [kx1, ky0], [kx1, ky1], [kx0, ky1]], dtype=np.float32)
        kpoly_img = self.paper_to_image(H_inv, kpoly)
        cv2.polylines(frame, [np.int32(kpoly_img)], True, (255, 255, 0), 2)
        key_rects = self.build_key_rects()
        for idx, (x0, y0, x1, y1) in enumerate(key_rects):
            poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
            poly_img = self.paper_to_image(H_inv, poly)
            color = (40, 220, 40) if now < self.active_until[idx] else (255, 255, 255)
            thickness = 3 if now < self.active_until[idx] else 2
            cv2.polylines(frame, [np.int32(poly_img)], True, color, thickness)
        for x0, y0, x1, y1 in self._build_black_key_rects((kx0, ky0, kx1, ky1)):
            poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
            poly_img = self.paper_to_image(H_inv, poly)
            cv2.fillConvexPoly(frame, np.int32(poly_img), (25, 25, 25))
            cv2.polylines(frame, [np.int32(poly_img)], True, (255, 255, 255), 1)

    # This function draws draw marker labels.
    def draw_marker_labels(self, frame: np.ndarray, markers: Optional[Dict[str, np.ndarray]],
                           mirrored_display: bool, flipped_vertical: bool) -> None:
        if markers is None:
            return
        h, w = frame.shape[:2]
        for name, pt in markers.items():
            x_img = float(pt[0] + 6.0)
            y_img = float(pt[1] - 6.0)
            x_img, y_img = self.transform_display_point(x_img, y_img, w, h,
                                                        mirror_horizontal=mirrored_display,
                                                        flip_vertical=flipped_vertical)
            cv2.putText(frame, name, (int(x_img), int(y_img)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # This function draws draw key labels.
    def draw_key_labels(self, frame: np.ndarray, H_inv: Optional[np.ndarray], now: float,
                        mirrored_display: bool, flipped_vertical: bool) -> None:
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
            x_img, y_img = self.transform_display_point(x_img, y_img, w, h,
                                                        mirror_horizontal=mirrored_display,
                                                        flip_vertical=flipped_vertical)
            color = (40, 220, 40) if now < self.active_until[idx] else (255, 255, 255)
            label = NOTE_NAMES[idx]
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.72
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
            org = (int(x_img - 0.5 * tw), int(y_img + 0.5 * th))
            cv2.putText(frame, label, org, font, scale, color, thickness)

    # This function draws draw hand labels.
    def draw_hand_labels(self, frame: np.ndarray, mirrored_display: bool, flipped_vertical: bool) -> None:
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        for key_idx, x_img, y_img, color in self.hand_label_overlays:
            x_img, y_img = self.transform_display_point(x_img, y_img, w, h,
                                                        mirror_horizontal=mirrored_display,
                                                        flip_vertical=flipped_vertical)
            cv2.putText(frame, NOTE_NAMES[key_idx], (int(x_img), int(y_img)), font, scale, color, thickness)
        for label, x_img, y_img, color in self.hand_name_overlays:
            x_img, y_img = self.transform_display_point(x_img, y_img, w, h,
                                                        mirror_horizontal=mirrored_display,
                                                        flip_vertical=flipped_vertical)
            cv2.putText(frame, label, (int(x_img), int(y_img)), font, 0.62, color, 2)
        if SHOW_PRESS_DEBUG:
            small_scale = 0.48
            line_gap = 16
            for lines, x_img, y_img, color in self.press_debug_overlays:
                x_img, y_img = self.transform_display_point(x_img, y_img, w, h,
                                                            mirror_horizontal=mirrored_display,
                                                            flip_vertical=flipped_vertical)
                base_x = int(x_img)
                base_y = int(y_img)
                for idx, line in enumerate(lines):
                    cv2.putText(frame, line, (base_x, base_y + idx * line_gap), font, small_scale, color, 1)

    # This function builds build hand id.
    def _build_hand_id(self, hand_landmarks, handedness_info: Dict[str, float],
                       occurrence_index: int) -> Tuple[str, Tuple[int, int, int]]:
        label = str(handedness_info.get("label", "Unknown"))
        score = float(handedness_info.get("score", 0.0))
        if label in ("Left", "Right") and score >= HAND_MIN_HANDEDNESS_SCORE:
            hand_name = label if occurrence_index == 0 else f"{label}{occurrence_index + 1}"
            color = (80, 255, 80) if label == "Right" else (80, 180, 255)
            return hand_name, color
        wrist_x = float(hand_landmarks[0].x)
        fallback_side = "LeftSide" if wrist_x < 0.5 else "RightSide"
        hand_name = fallback_side if occurrence_index == 0 else f"{fallback_side}{occurrence_index + 1}"
        return hand_name, (180, 180, 180)

    # This function draws draw status.
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
        cv2.putText(frame, "Multi-finger piano mode: contact inference (optimized)", (28, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (28, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
        ids_text = ",".join(str(i) for i in self.last_detected_ids) if self.last_detected_ids else "-"
        cv2.putText(frame, f"Aruco IDs: {ids_text}", (300, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
        tips_text = "Use fingertips. Keep all 4 ArUco markers (ID 0/1/2/3) visible."
        cv2.putText(frame, tips_text, (w - 600, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # This function handles process hands.
    def process_hands(self, frame: np.ndarray, H: Optional[np.ndarray], now: float) -> None:
        current_ids = set()
        self.hand_label_overlays = []
        self.hand_name_overlays = []
        self.press_debug_overlays = []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            detection = predict(frame_rgb, timestamp_ms=int(now * 1000.0))
        except Exception as exc:
            print(f"[WARN] Hand tracking failed: {exc}")
            detection = None
        hands = getattr(detection, "hand_landmarks", None) if detection is not None else None
        world_hands = getattr(detection, "hand_world_landmarks", None) if detection is not None else None
        handedness_list = extract_handedness(detection)
        if not hands:
            stale = [fid for fid in self._tracked_finger_ids() if fid not in current_ids]
            for fid in stale:
                self._clear_finger_tracking_state(
                    fid, self.prev_key_by_finger, self.smoothed_fingertips, self.prev_raw_fingertips,
                    self.prev_paper_tip_by_finger, self.smoothed_paper_velocity_by_finger,
                    self.finger_seen_frames, self.candidate_key_by_finger, self.stable_count_by_finger,
                    self.finger_state, self.finger_contact_key, self.low_zone_count_by_finger,
                    self.approach_count_by_finger, self.contact_count_by_finger, self.release_count_by_finger,
                    self.hold_miss_count_by_finger)
            return

        h, w = frame.shape[:2]
        hand_occurrences: Counter = Counter()
        for hand_idx, hand_landmarks in enumerate(hands):
            world_landmarks = world_hands[hand_idx] if world_hands is not None and hand_idx < len(world_hands) else None
            handedness_info = handedness_list[hand_idx] if hand_idx < len(handedness_list) else {"label": "Unknown", "score": 0.0}
            occurrence_index = hand_occurrences[handedness_info.get("label", "Unknown")]
            hand_occurrences[handedness_info.get("label", "Unknown")] += 1
            hand_name, hand_color = self._build_hand_id(hand_landmarks, handedness_info, occurrence_index)
            wrist = hand_landmarks[0]
            self.hand_name_overlays.append((f"{hand_name}", float(wrist.x * w + 10.0), float(wrist.y * h - 12.0), hand_color))

            for tip_id in FINGERTIP_IDS:
                finger_id = (hand_name, tip_id)
                current_ids.add(finger_id)
                self.finger_seen_frames[finger_id] = self.finger_seen_frames.get(finger_id, 0) + 1
                lm = hand_landmarks[tip_id]
                raw_pt = np.array([float(lm.x * w), float(lm.y * h)], dtype=np.float32)
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
                paper_pt = None
                if H is not None:
                    paper_pt = self.image_to_paper(H, (float(smoothed_pt[0]), float(smoothed_pt[1])))
                    if paper_pt is not None:
                        trigger_key = self.locate_key_stable(paper_pt)
                        hold_key = self.locate_key_hold(paper_pt)

                features = self._finger_contact_features(hand_landmarks, world_landmarks, tip_id,
                                                         w, h, paper_pt, finger_id)
                prev_candidate = self.candidate_key_by_finger.get(finger_id)
                if trigger_key == prev_candidate:
                    self.stable_count_by_finger[finger_id] = self.stable_count_by_finger.get(finger_id, 0) + 1
                else:
                    self.candidate_key_by_finger[finger_id] = trigger_key
                    self.stable_count_by_finger[finger_id] = 1

                stable_count = self.stable_count_by_finger.get(finger_id, 0)
                warmup_ready = self.finger_seen_frames.get(finger_id, 0) > FINGER_WARMUP_FRAMES
                stable_key = trigger_key if stable_count >= KEY_STABLE_FRAMES else None
                contact_key = self.finger_contact_key.get(finger_id)
                active_key = contact_key if contact_key is not None else stable_key

                paper_axis = None if paper_pt is None else self._paper_axis_value(paper_pt)
                depth_norm = self._normalized_depth_in_key(active_key, paper_axis)
                low_zone = depth_norm >= PRESS_DEPTH_RATIO
                if stable_key is not None and low_zone:
                    self.low_zone_count_by_finger[finger_id] = self.low_zone_count_by_finger.get(finger_id, 0) + 1
                else:
                    self.low_zone_count_by_finger[finger_id] = 0
                dwell_score = self._clip01(self.low_zone_count_by_finger.get(finger_id, 0) / max(CONTACT_LOW_ZONE_FRAMES, 1))
                tip_ref_sep = features["tip_below_ref"]
                curl_score = self._clip01((tip_ref_sep - TIP_REF_CURL_GATE_NORM) / max(TIP_REF_SEPARATION_NORM, 1e-6))
                approach_score = self._clip01(features["approach_velocity"] / max(APPROACH_VEL_THRESHOLD * 2.0, 1e-6))
                contact_score = (DEPTH_WEIGHT * self._clip01(depth_norm) +
                                 CURL_WEIGHT * curl_score +
                                 APPROACH_WEIGHT * approach_score +
                                 DWELL_WEIGHT * dwell_score)

                if stable_key is not None and features["approach_velocity"] >= APPROACH_VEL_THRESHOLD and depth_norm >= RELEASE_DEPTH_RATIO:
                    self.approach_count_by_finger[finger_id] = self.approach_count_by_finger.get(finger_id, 0) + 1
                else:
                    self.approach_count_by_finger[finger_id] = 0

                state = self.finger_state.get(finger_id, "FREE")
                shown_key = contact_key if contact_key is not None else stable_key
                if shown_key is not None:
                    color = (0, 255, 0) if state == "CONTACT" else (0, 220, 220)
                    self.hand_label_overlays.append((shown_key, float(px + 10), float(py - 10), color))

                if SHOW_PRESS_DEBUG:
                    key_name = NOTE_NAMES[shown_key] if shown_key is not None else "-"
                    debug_lines = [
                        f"{hand_name}:{tip_id} {state} key={key_name}",
                        f"st={stable_count} low={self.low_zone_count_by_finger.get(finger_id, 0)} app={self.approach_count_by_finger.get(finger_id, 0)}",
                        f"depth={depth_norm:.2f} curl={curl_score:.2f} vel={features['approach_velocity']:.2f}",
                        f"score={contact_score:.2f} sep={tip_ref_sep:.1f}",
                    ]
                    debug_color = (0, 255, 0) if state == "CONTACT" else (255, 255, 0)
                    self.press_debug_overlays.append((debug_lines, float(px + 14), float(py + 14), debug_color))

                if not warmup_ready or stable_key is None:
                    if state == "CONTACT":
                        miss = self.hold_miss_count_by_finger.get(finger_id, 0) + 1
                        self.hold_miss_count_by_finger[finger_id] = miss
                        if miss >= HOLD_MISS_FRAMES:
                            self.finger_state[finger_id] = "FREE"
                            self.finger_contact_key[finger_id] = None
                            self.contact_count_by_finger[finger_id] = 0
                            self.release_count_by_finger[finger_id] = 0
                    else:
                        self.finger_state[finger_id] = "FREE"
                        self.finger_contact_key[finger_id] = None
                        self.contact_count_by_finger[finger_id] = 0
                        self.release_count_by_finger[finger_id] = 0
                    continue

                if state == "FREE":
                    self.finger_state[finger_id] = "APPROACHING"
                    self.contact_count_by_finger[finger_id] = 0
                    self.release_count_by_finger[finger_id] = 0
                    self.hold_miss_count_by_finger[finger_id] = 0

                elif state == "APPROACHING":
                    # 优化：放宽触发条件，允许慢速按压（仅凭低区累积和接触分数触发）
                    ready_contact = (
                        stable_key is not None
                        and self.low_zone_count_by_finger.get(finger_id, 0) >= CONTACT_LOW_ZONE_FRAMES
                        and (contact_score >= CONTACT_SCORE_THRESHOLD
                             or self.approach_count_by_finger.get(finger_id, 0) >= 1)
                    )
                    if ready_contact:
                        c = self.contact_count_by_finger.get(finger_id, 0) + 1
                        self.contact_count_by_finger[finger_id] = c
                        if c >= CONTACT_STABLE_FRAMES:
                            hit_key = stable_key
                            last = self.last_hit_time_by_finger_key.get((finger_id, hit_key), 0.0)
                            if now - last > KEY_COOLDOWN_SEC:
                                self.last_hit_time_by_finger_key[(finger_id, hit_key)] = now
                                self.last_hit_time[hit_key] = now
                                self.active_until[hit_key] = now + ACTIVE_FLASH_SEC
                                self.synth.play(hit_key)
                                print(f"Played {NOTE_NAMES[hit_key]}")
                            self.finger_contact_key[finger_id] = hit_key
                            self.finger_state[finger_id] = "CONTACT"
                            self.contact_count_by_finger[finger_id] = 0
                            self.release_count_by_finger[finger_id] = 0
                    else:
                        self.contact_count_by_finger[finger_id] = 0

                elif state == "CONTACT":
                    contact_key = self.finger_contact_key.get(finger_id)
                    still_same_key = contact_key is not None and hold_key == contact_key
                    release_ready = (
                        (contact_score <= RELEASE_SCORE_THRESHOLD)
                        or (depth_norm <= RELEASE_DEPTH_RATIO)
                        or (not still_same_key)
                    )
                    if still_same_key and not release_ready:
                        self.hold_miss_count_by_finger[finger_id] = 0
                        self.release_count_by_finger[finger_id] = 0
                    else:
                        if not still_same_key:
                            self.hold_miss_count_by_finger[finger_id] = self.hold_miss_count_by_finger.get(finger_id, 0) + 1
                        r = self.release_count_by_finger.get(finger_id, 0) + 1
                        self.release_count_by_finger[finger_id] = r
                        if r >= RELEASE_STABLE_FRAMES or self.hold_miss_count_by_finger.get(finger_id, 0) >= HOLD_MISS_FRAMES:
                            self.finger_state[finger_id] = "RELEASING"
                            self.release_count_by_finger[finger_id] = 0

                elif state == "RELEASING":
                    self.finger_contact_key[finger_id] = None
                    if stable_key is None or depth_norm <= RELEASE_DEPTH_RATIO:
                        self.finger_state[finger_id] = "FREE"
                    else:
                        self.finger_state[finger_id] = "APPROACHING"

        stale = [fid for fid in self._tracked_finger_ids() if fid not in current_ids]
        for fid in stale:
            self._clear_finger_tracking_state(
                fid, self.prev_key_by_finger, self.smoothed_fingertips, self.prev_raw_fingertips,
                self.prev_paper_tip_by_finger, self.smoothed_paper_velocity_by_finger,
                self.finger_seen_frames, self.candidate_key_by_finger, self.stable_count_by_finger,
                self.finger_state, self.finger_contact_key, self.low_zone_count_by_finger,
                self.approach_count_by_finger, self.contact_count_by_finger, self.release_count_by_finger,
                self.hold_miss_count_by_finger)

    # This function runs the full app loop.
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
                    cv2.imshow("Paper Piano Contact Inference", display_frame)
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
    print("[INFO] Starting Paper Piano (optimized for easier contact inference)...")
    print(f"[INFO] Preferred camera index = {PREFERRED_CAM_INDEX}")
    print(f"[INFO] Target resolution = {CAM_WIDTH}x{CAM_HEIGHT}")
    print("[INFO] Expected cameras: 0=laptop, 1=Camo, 2=USB", flush=True)
    PaperPiano().run()
