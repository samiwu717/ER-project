"""This file runs an older working version of the paper piano."""
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
PREFERRED_CAM_INDEX = 0
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
KEYBOARD_X0 = 48
KEYBOARD_X1 = 690
KEYBOARD_Y0 = 185
KEYBOARD_Y1 = 1000
NUM_WHITE_KEYS = 11
KEYBOARD_STACK_VERTICAL = True

# Use all four fingertips for a simple first version
FINGERTIP_IDS = [8, 12, 16, 20]

# Trigger behavior
KEY_COOLDOWN_SEC = 0.12
ACTIVE_FLASH_SEC = 0.20
HOMOGRAPHY_HOLD_SEC = 0.40
SMOOTH_ALPHA = 0.35
MARKER_HOLD_SEC = 0.45

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
BLACK_KEY_X1_RATIO = 0.93   # do not extend fully to the white-key edge
BLACK_KEY_HEIGHT_RATIO = 0.64  # slightly shorter, closer to the printed template


# This function builds build black key rects.
def build_black_key_rects() -> List[Tuple[float, float, float, float]]:
    rects = []
    total_w = KEYBOARD_X1 - KEYBOARD_X0
    total_h = KEYBOARD_Y1 - KEYBOARD_Y0
    key_h = total_h / NUM_WHITE_KEYS

    black_x0 = KEYBOARD_X0 + total_w * BLACK_KEY_X0_RATIO
    black_x1 = KEYBOARD_X0 + total_w * BLACK_KEY_X1_RATIO
    black_h = key_h * BLACK_KEY_HEIGHT_RATIO

    for white_idx in BLACK_KEY_AFTER_WHITE:
        # Center each black key on the boundary between adjacent white keys.
        boundary_y = KEYBOARD_Y0 + (white_idx + 1) * key_h
        y0 = boundary_y - 0.5 * black_h
        y1 = boundary_y + 0.5 * black_h
        rects.append((black_x0, y0, black_x1, y1))
    return rects


# This function opens the camera.
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
            self.enabled = False
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


class PaperPiano:
    # This function sets up the object.
    def __init__(self) -> None:
        self.cap = open_camera(None)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

        self.synth = SimpleSynth()
        self.prev_key_by_finger: Dict[Tuple[int, int], Optional[int]] = {}
        self.last_hit_time = [0.0] * NUM_WHITE_KEYS
        self.active_until = [0.0] * NUM_WHITE_KEYS

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

    # This function closes and cleans up things.
    def close(self) -> None:
        try:
            if self.cap is not None:
                self.cap.release()
        finally:
            self.synth.close()
        cv2.destroyAllWindows()

    # This function collects collect id to corners.
    def _collect_id_to_corners(self, img: np.ndarray) -> Dict[int, np.ndarray]:
        corners, ids, _ = self.aruco_detector.detectMarkers(img)
        id_to_corners: Dict[int, np.ndarray] = {}
        if ids is None:
            return id_to_corners
        for marker_corners, marker_id_arr in zip(corners, ids):
            marker_id = int(marker_id_arr[0])
            id_to_corners[marker_id] = marker_corners.reshape(4, 2).astype(np.float32)
        return id_to_corners

    # This function gets extract aruco corner points.
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

    # This function detects detect markers.
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

    # This function updates update homography.
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
        rects = []
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
        idx = max(0, min(NUM_WHITE_KEYS - 1, idx))
        return idx

    # This function draws draw paper overlay.
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

    # This function draws draw key labels.
    def draw_key_labels(self, frame: np.ndarray, H_inv: Optional[np.ndarray], now: float, mirrored_display: bool) -> None:
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

    # This function draws draw status.
    def draw_status(self, frame: np.ndarray, H_ok: bool) -> None:
        h, w = frame.shape[:2]
        panel_h = 110
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (560, 12 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0.0, frame)

        status = "paper locked" if H_ok else "looking for 4 ArUco corner markers"
        cv2.putText(frame, f"Status: {status}", (28, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "USB camera final build: portrait PDF keyboard aligned", (28, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (28, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
        ids_text = ",".join(str(i) for i in self.last_detected_ids) if self.last_detected_ids else "-"
        cv2.putText(frame, f"Aruco IDs: {ids_text}", (300, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

        tips_text = "Use one fingertip first. Keep all 4 ArUco markers (ID 0/1/2/3) visible."
        cv2.putText(frame, tips_text, (w - 600, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # This function handles process hands.
    def process_hands(self, frame: np.ndarray, H: Optional[np.ndarray], now: float) -> None:
        current_ids = set()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection = predict(frame_rgb)
        hands = getattr(detection, "hand_landmarks", None) if detection is not None else None
        if not hands:
            stale = [fid for fid in self.prev_key_by_finger if fid not in current_ids]
            for fid in stale:
                self.prev_key_by_finger.pop(fid, None)
            return

        h, w = frame.shape[:2]
        for hand_idx, hand_landmarks in enumerate(hands):
            for tip_id in FINGERTIP_IDS:
                finger_id = (hand_idx, tip_id)
                current_ids.add(finger_id)

                lm = hand_landmarks[tip_id]
                px = int(lm.x * w)
                py = int(lm.y * h)
                cv2.circle(frame, (px, py), 6, (0, 128, 255), -1)

                curr_key = None
                if H is not None:
                    paper_pt = self.image_to_paper(H, (px, py))
                    if paper_pt is not None:
                        curr_key = self.locate_key(paper_pt)
                        if curr_key is not None:
                            cv2.putText(frame, NOTE_NAMES[curr_key], (px + 8, py - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                prev_key = self.prev_key_by_finger.get(finger_id)
                if curr_key is not None and prev_key != curr_key:
                    if now - self.last_hit_time[curr_key] > KEY_COOLDOWN_SEC:
                        self.last_hit_time[curr_key] = now
                        self.active_until[curr_key] = now + ACTIVE_FLASH_SEC
                        self.synth.play(curr_key)
                        print(f"Played {NOTE_NAMES[curr_key]}")

                self.prev_key_by_finger[finger_id] = curr_key

        stale = [fid for fid in self.prev_key_by_finger if fid not in current_ids]
        for fid in stale:
            self.prev_key_by_finger.pop(fid, None)

    # This function runs the full app loop.
    def run(self) -> None:
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    print("[ERROR] Camera read failed during main loop. USB camera disconnected or busy.")
                    break

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
                    break
        finally:
            self.close()


if __name__ == "__main__":
    print("[INFO] Starting Paper Piano...")
    print(f"[INFO] Preferred camera index = {PREFERRED_CAM_INDEX}")
    print(f"[INFO] Target resolution = {CAM_WIDTH}x{CAM_HEIGHT}")
    print("[INFO] Expected cameras: 0=laptop, 1=Camo, 2=USB", flush=True)
    PaperPiano().run()
