"""This file is the fourth paper piano version with better press logic."""
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import pygame
except Exception:
    pygame = None

from prediction import draw_landmarks_on_image, predict


# =========================
# User-tunable parameters
# =========================
CAM_INDEX = 0
CAM_WIDTH = 1280
CAM_HEIGHT = 720

NUM_WHITE_KEYS = 8
FINGERTIP_IDS = [8, 12, 16, 20]

KEY_COOLDOWN_SEC = 0.12
ACTIVE_FLASH_SEC = 0.20
GEOM_HOLD_SEC = 0.45

TIP_SMOOTH_ALPHA = 0.35
SPEED_EMA_ALPHA = 0.55
MAX_STEP_PX = 75.0

# Downward press pulse from multi-frame vertical displacement
DOWN_FIRE_SPEED_PX = 55.0
DOWN_MIN_TRAVEL_PX = 10.0
DOWN_MAX_WINDOW_SEC = 0.26
REARM_OUTSIDE_SEC = 0.08
RELEASE_LIFT_PX = 10.0
RELEASE_UP_SPEED_PX = 90.0
KEY_STABLE_MIN_FRAMES = 3
STROKE_MAX_X_TRAVEL_PX = 30.0
FINGER_REFRACTORY_SEC = 0.18

# Final-press height gate: fingertip cannot be too high above keyline
PRESS_MAX_ABOVE_SCALE = 0.6
PRESS_MAX_ABOVE_MIN_PX = 8.0

# Keyboard strip shape relative to marker pixel size
KEY_STRIP_OFFSET_SCALE = 1.0  # lower than previous version
KEY_STRIP_HEIGHT_SCALE = 0.35  # just a thin visual strip

ARUCO_DICT_NAME = cv2.aruco.DICT_5X5_50
ARUCO_USE_MULTI_PASS = True
MIRROR_DISPLAY = True

DEBUG_DRAW_RAW_HAND = True
DEBUG_DRAW_PRESS_INFO = True

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


class PaperPianoV4:
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

        self.last_detected_ids: List[int] = []
        self.geom_smooth: Optional[Dict[str, object]] = None
        self.last_geom_time = -999.0

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
        for c, id_arr in zip(corners, ids):
            out[int(id_arr[0])] = c.reshape(4, 2).astype(np.float32)
        return out

    # This function detects detect markers multipass.
    def _detect_markers_multipass(self, gray: np.ndarray) -> Dict[int, np.ndarray]:
        id_to_corners = self._collect_id_to_corners(gray)
        if ARUCO_USE_MULTI_PASS and len(id_to_corners) < 2:
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
            id2 = self._collect_id_to_corners(clahe)
            if len(id2) > len(id_to_corners):
                id_to_corners = id2
            if len(id_to_corners) < 2:
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                id3 = self._collect_id_to_corners(blur)
                if len(id3) > len(id_to_corners):
                    id_to_corners = id3
        return id_to_corners

    # This function handles marker feature.
    @staticmethod
    def _marker_feature(corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        center = corners.mean(axis=0)
        idx_top2 = np.argsort(corners[:, 1])[:2]
        top_mid = corners[idx_top2].mean(axis=0)
        side = float(0.5 * (np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[2] - corners[1])))
        return center.astype(np.float32), top_mid.astype(np.float32), side

    # This function handles smooth point.
    def _smooth_point(self, key: str, p: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        assert self.geom_smooth is not None
        prev = self.geom_smooth.get(key)
        if prev is None:
            self.geom_smooth[key] = p.astype(np.float32)
            return p.astype(np.float32)
        sp = (alpha * p + (1.0 - alpha) * prev).astype(np.float32)
        self.geom_smooth[key] = sp
        return sp

    # This function handles estimate geometry.
    def _estimate_geometry(self, id_to_corners: Dict[int, np.ndarray], now: float) -> Optional[Dict[str, object]]:
        self.last_detected_ids = sorted(id_to_corners.keys())
        if len(id_to_corners) < 2:
            if self.geom_smooth is not None and (now - self.last_geom_time) <= GEOM_HOLD_SEC:
                return self.geom_smooth
            return None

        feats = []
        for mid, corners in id_to_corners.items():
            c, t, side = self._marker_feature(corners)
            feats.append((mid, c, t, side))

        # Use the two highest markers (smallest y) as anchors.
        feats = sorted(feats, key=lambda x: float(x[1][1]))
        pair = feats[:2]
        pair = sorted(pair, key=lambda x: float(x[1][0]))  # left-right
        (id_l, c_l, t_l, s_l), (id_r, c_r, t_r, s_r) = pair

        v = c_r - c_l
        length = float(np.linalg.norm(v))
        if length < 40.0:
            if self.geom_smooth is not None and (now - self.last_geom_time) <= GEOM_HOLD_SEC:
                return self.geom_smooth
            return None
        u = v / max(length, 1e-6)

        n1 = np.array([-u[1], u[0]], dtype=np.float32)
        n2 = np.array([u[1], -u[0]], dtype=np.float32)
        n_up = n1 if n1[1] < n2[1] else n2  # choose the more upward normal (smaller y)
        n_down = -n_up

        marker_size = float(0.5 * (s_l + s_r))
        strip_h = max(14.0, KEY_STRIP_HEIGHT_SCALE * marker_size)
        strip_offset = KEY_STRIP_OFFSET_SCALE * marker_size

        a0 = c_l + n_up * strip_offset
        b0 = c_r + n_up * strip_offset

        if self.geom_smooth is None:
            self.geom_smooth = {}
        c_l = self._smooth_point("c_l", c_l)
        c_r = self._smooth_point("c_r", c_r)
        t_l = self._smooth_point("t_l", t_l)
        t_r = self._smooth_point("t_r", t_r)
        a0 = self._smooth_point("a0", a0)
        b0 = self._smooth_point("b0", b0)

        geom: Dict[str, object] = {
            "pair_ids": (id_l, id_r),
            "c_l": c_l,
            "c_r": c_r,
            "t_l": t_l,
            "t_r": t_r,
            "u": u.astype(np.float32),
            "n_up": n_up.astype(np.float32),
            "n_down": n_down.astype(np.float32),
            "length": length,
            "a0": a0,
            "b0": b0,
            "strip_h": strip_h,
        }

        geom["press_max_above"] = max(PRESS_MAX_ABOVE_MIN_PX, PRESS_MAX_ABOVE_SCALE * marker_size)

        self.geom_smooth.update(geom)
        self.last_geom_time = now
        return self.geom_smooth

    # This function handles local point.
    @staticmethod
    def _local_point(geom: Dict[str, object], t: float, v: float) -> np.ndarray:
        a0 = geom["a0"]
        u = geom["u"]
        n_up = geom["n_up"]
        length = float(geom["length"])
        return (a0 + u * (t * length) + n_up * v).astype(np.float32)

    # This function handles locate key.
    def _locate_key(self, p: np.ndarray, geom: Dict[str, object]) -> Optional[Tuple[int, float, float, float]]:
        a0 = geom["a0"]
        u = geom["u"]
        length = float(geom["length"])

        rel = p - a0
        t = float(np.dot(rel, u) / max(length, 1e-6))
        if t < 0.0 or t > 1.0:
            return None

        key_idx = int(t * NUM_WHITE_KEYS)
        key_idx = max(0, min(NUM_WHITE_KEYS - 1, key_idx))
        return key_idx, t, 0.0, 0.0

    # This function handles draw keyboard overlay.
    def _draw_keyboard_overlay(self, frame: np.ndarray, geom: Optional[Dict[str, object]], now: float) -> None:
        if geom is None:
            return

        c_l = np.int32(geom["c_l"])
        c_r = np.int32(geom["c_r"])
        t_l = np.int32(geom["t_l"])
        t_r = np.int32(geom["t_r"])
        id_l, id_r = geom["pair_ids"]

        cv2.circle(frame, tuple(c_l), 7, (0, 255, 255), -1)
        cv2.circle(frame, tuple(c_r), 7, (0, 255, 255), -1)
        cv2.putText(frame, f"id{id_l}", (int(c_l[0]) + 4, int(c_l[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, f"id{id_r}", (int(c_r[0]) + 4, int(c_r[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.line(frame, tuple(t_l), tuple(t_r), (80, 200, 255), 2)

        h2 = 0.5 * float(geom["strip_h"])
        p0 = self._local_point(geom, 0.0, -h2)
        p1 = self._local_point(geom, 1.0, -h2)
        p2 = self._local_point(geom, 1.0, +h2)
        p3 = self._local_point(geom, 0.0, +h2)
        strip_poly = np.int32(np.array([p0, p1, p2, p3], dtype=np.float32))
        cv2.polylines(frame, [strip_poly], True, (255, 255, 0), 2)

        for k in range(NUM_WHITE_KEYS):
            t0 = k / float(NUM_WHITE_KEYS)
            t1 = (k + 1) / float(NUM_WHITE_KEYS)
            q0 = self._local_point(geom, t0, -h2)
            q1 = self._local_point(geom, t1, -h2)
            q2 = self._local_point(geom, t1, +h2)
            q3 = self._local_point(geom, t0, +h2)
            poly = np.int32(np.array([q0, q1, q2, q3], dtype=np.float32))
            color = (40, 220, 40) if now < self.active_until[k] else (255, 255, 255)
            thick = 3 if now < self.active_until[k] else 2
            cv2.polylines(frame, [poly], True, color, thick)

    # This function handles draw key labels.
    def _draw_key_labels(self, frame: np.ndarray, geom: Optional[Dict[str, object]], now: float, mirrored_display: bool) -> None:
        if geom is None:
            return
        h, w = frame.shape[:2]
        for k in range(NUM_WHITE_KEYS):
            t = (k + 0.5) / float(NUM_WHITE_KEYS)
            p = self._local_point(geom, t, 0.0)
            x, y = float(p[0]), float(p[1])
            if mirrored_display:
                x = (w - 1) - x
            label = NOTE_NAMES[k]
            color = (40, 220, 40) if now < self.active_until[k] else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.65
            thick = 2
            (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
            org = (int(x - 0.5 * tw), int(y + 0.5 * th))
            cv2.putText(frame, label, org, font, scale, color, thick)

    # This function handles draw status.
    def _draw_status(self, frame: np.ndarray, geom_ok: bool) -> None:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (740, 132), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0.0, frame)

        status = "2-marker lock" if geom_ok else "looking for 2 top markers"
        cv2.putText(frame, f"Status: {status}", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
        cv2.putText(frame, "V4: horizontal key lanes + vertical (image-y) press pulse", (28, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (28, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 2)
        ids_text = ",".join(str(i) for i in self.last_detected_ids) if self.last_detected_ids else "-"
        cv2.putText(frame, f"Aruco IDs: {ids_text}", (250, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 2)
        cv2.putText(frame, "Gate: horizontal lane + press motion + final height-near-key gate.", (w - 655, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 255), 2)

    # This function handles process hands.
    def _process_hands(self, frame: np.ndarray, geom: Optional[Dict[str, object]], now: float) -> None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection = predict(frame_rgb)
        if DEBUG_DRAW_RAW_HAND and detection is not None:
            draw_landmarks_on_image(frame, detection)
        hands = getattr(detection, "hand_landmarks", None) if detection is not None else None

        current_ids = set()
        if not hands:
            self.finger_states.clear()
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
                        "down_v_ema": 0.0,
                        "last_key": -1.0,
                        "key_hold_frames": 0.0,
                        "stroke_y0": raw_y,
                        "stroke_x0": raw_x,
                        "stroke_t0": now,
                        "last_press_y": raw_y,
                        "last_press_t": -999.0,
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
                speed_ema = SPEED_EMA_ALPHA * inst_speed + (1.0 - SPEED_EMA_ALPHA) * float(st["speed_ema"])
                st["speed_ema"] = speed_ema
                st["sx"] = sx
                st["sy"] = sy
                st["prev_t"] = now

                px, py = int(round(sx)), int(round(sy))
                cv2.circle(frame, (px, py), 6, (0, 128, 255), -1)
                cv2.circle(frame, (int(round(raw_x)), int(round(raw_y))), 3, (255, 180, 0), -1)

                key_idx = None
                height_above = 0.0
                max_above = 9999.0
                if geom is not None:
                    located = self._locate_key(np.array([sx, sy], dtype=np.float32), geom)
                    if located is not None:
                        key_idx, key_t, _, _ = located
                        p2 = np.array([sx, sy], dtype=np.float32)
                        keyline_p = geom["a0"] + geom["u"] * (float(key_t) * float(geom["length"]))
                        # Only gate if the finger is above the keyline; below/on line is allowed.
                        height_above = max(0.0, float(np.dot(keyline_p - p2, geom["n_down"])))
                        max_above = float(geom["press_max_above"])
                        cv2.putText(frame, NOTE_NAMES[key_idx], (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)

                if key_idx is None:
                    st["last_key"] = -1.0
                    st["key_hold_frames"] = 0.0
                    st["stroke_y0"] = sy
                    st["stroke_x0"] = sx
                    st["stroke_t0"] = now
                    if float(st["outside_since"]) < 0.0:
                        st["outside_since"] = now
                    if now - float(st["outside_since"]) >= REARM_OUTSIDE_SEC:
                        st["pressed_latch"] = 0.0
                    if DEBUG_DRAW_PRESS_INFO:
                        cv2.putText(frame, f"vy:{float(st['down_v_ema']):4.0f} d:0 l:{int(st['pressed_latch']>0.5)}", (px + 8, py + 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 220, 120), 1)
                    continue
                st["outside_since"] = -1.0

                # Vertical press = positive image-y motion (downward in image).
                down_v = float((sy - prev_y) / dt)
                down_v_ema = SPEED_EMA_ALPHA * down_v + (1.0 - SPEED_EMA_ALPHA) * float(st["down_v_ema"])
                st["down_v_ema"] = down_v_ema

                last_key = int(st["last_key"]) if float(st["last_key"]) >= 0.0 else None
                if last_key != key_idx:
                    st["last_key"] = float(key_idx)
                    st["key_hold_frames"] = 1.0
                    st["stroke_y0"] = sy
                    st["stroke_x0"] = sx
                    st["stroke_t0"] = now
                    st["pressed_latch"] = 0.0
                else:
                    st["key_hold_frames"] = float(st["key_hold_frames"]) + 1.0

                # Track multi-frame downward travel from the recent local top point.
                if sy < float(st["stroke_y0"]):
                    st["stroke_y0"] = sy
                    st["stroke_x0"] = sx
                    st["stroke_t0"] = now
                if (now - float(st["stroke_t0"])) > DOWN_MAX_WINDOW_SEC:
                    st["stroke_y0"] = sy
                    st["stroke_x0"] = sx
                    st["stroke_t0"] = now
                travel_down = float(sy - float(st["stroke_y0"]))
                travel_x = float(abs(sx - float(st["stroke_x0"])))

                latched = bool(st["pressed_latch"] > 0.5)
                if latched:
                    lifted = float(st["last_press_y"]) - sy
                    if lifted >= RELEASE_LIFT_PX or down_v_ema <= -RELEASE_UP_SPEED_PX:
                        st["pressed_latch"] = 0.0
                        st["stroke_y0"] = sy
                        st["stroke_x0"] = sx
                        st["stroke_t0"] = now
                else:
                    key_stable = float(st["key_hold_frames"]) >= float(KEY_STABLE_MIN_FRAMES)
                    finger_ready = (now - float(st["last_press_t"])) >= FINGER_REFRACTORY_SEC
                    pressed = (
                        key_stable
                        and finger_ready
                        and height_above <= max_above
                        and down_v_ema >= DOWN_FIRE_SPEED_PX
                        and travel_down >= DOWN_MIN_TRAVEL_PX
                        and travel_x <= STROKE_MAX_X_TRAVEL_PX
                    )
                    if pressed:
                        if now - self.last_hit_time[key_idx] > KEY_COOLDOWN_SEC:
                            self.last_hit_time[key_idx] = now
                            self.active_until[key_idx] = now + ACTIVE_FLASH_SEC
                            self.synth.play(key_idx)
                            print(f"Played {NOTE_NAMES[key_idx]}")
                        st["pressed_latch"] = 1.0
                        st["last_press_y"] = sy
                        st["last_press_t"] = now
                        st["stroke_y0"] = sy
                        st["stroke_x0"] = sx
                        st["stroke_t0"] = now

                if DEBUG_DRAW_PRESS_INFO:
                    latch_flag = 1 if st["pressed_latch"] > 0.5 else 0
                    cv2.putText(frame, f"vy:{down_v_ema:4.0f} d:{travel_down:3.0f} dx:{travel_x:3.0f} h:{height_above:3.0f}/{max_above:3.0f} l:{latch_flag}", (px + 8, py + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 220, 120), 1)

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

                now = time.time()
                dt = max(now - self.last_fps_time, 1e-6)
                self.fps = 1.0 / dt
                self.last_fps_time = now

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                id_to_corners = self._detect_markers_multipass(gray)
                geom = self._estimate_geometry(id_to_corners, now)

                self._draw_keyboard_overlay(frame, geom, now)
                self._process_hands(frame, geom, now)

                display = cv2.flip(frame, 1) if MIRROR_DISPLAY else frame
                self._draw_key_labels(display, geom, now, mirrored_display=MIRROR_DISPLAY)
                self._draw_status(display, geom is not None)
                cv2.imshow("Paper Piano V4", display)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break
        finally:
            self.close()


if __name__ == "__main__":
    PaperPianoV4().run()
