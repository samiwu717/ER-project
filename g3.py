import os
from typing import Dict, List, Optional, Tuple

import cv2
import moderngl
import moderngl_window as mglw
import numpy as np
import pygame
from pyrr import Matrix44

# Prefer prediction.py if present; fall back to p.py for older setups
from prediction import predict, get_camera_matrix, get_fov_y, solvepnp


# ── Piano configuration ────────────────────────────────────────────────────────
NUM_WHITE = 8
NUM_BLACK = 5
NUM_KEYS = NUM_WHITE + NUM_BLACK

KEY_W = 3.5
KEY_H = 1.5
KEY_D = 9.0
KEY_Z = -28.0
KEY_Y = -4.0

BK_W = 2.1
BK_H = 2.2
BK_D = 5.5

PRESS_THR = 3.5
COOLDOWN = 0.12

BLACK_OFFSETS = [0, 1, 3, 4, 5]

NOTE_FREQ = [
    261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25,
    277.18, 311.13, 369.99, 415.30, 466.16
]

NOTE_NAMES = [
    "C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5",
    "C#4", "D#4", "F#4", "G#4", "A#4"
]

FINGER_TIPS = [8, 12, 16, 20]

WHITE_KEY = (0.95, 0.95, 0.95)
BLACK_KEY = (0.08, 0.08, 0.08)
PRESSED = (0.30, 0.75, 1.00)

TARGET_WHITE = (1.00, 0.90, 0.25)
TARGET_BLACK = (0.85, 0.70, 0.10)
CORRECT_COLOR = (0.20, 0.95, 0.30)
WRONG_COLOR = (0.95, 0.20, 0.20)
DONE_WHITE = (0.70, 1.00, 0.70)
DONE_BLACK = (0.25, 0.55, 0.25)

MIN_PRESS_MOTION = 0.10
RELEASE_Y_MARGIN = 0.35

# ── ArUco configuration ───────────────────────────────────────────────────────
ARUCO_DICT_NAME = cv2.aruco.DICT_4X4_50
ARUCO_TARGET_ID = 0
ARUCO_MARKER_SIZE_M = 0.05
PIANO_OFFSET_MARKER_LOCAL = np.array([0.0, -0.03, 0.12], dtype=np.float32)

# ── Tutor timing ──────────────────────────────────────────────────────────────
FLASH_TIME = 0.30
BPM = 72.0
BEAT_SEC = 60.0 / BPM


def _make_piano_tone(
    freq: float,
    duration: float = 1.2,
    sample_rate: int = 44100,
    volume: float = 0.5,
) -> pygame.mixer.Sound:
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    harmonics = [1.0, 0.5, 0.25, 0.15, 0.10, 0.06, 0.04]
    wave = np.zeros(n, dtype=np.float64)
    for i, amp in enumerate(harmonics):
        wave += amp * np.sin(2 * np.pi * freq * (i + 1) * t)
    wave /= sum(harmonics)

    a = int(sample_rate * 0.008)
    d = int(sample_rate * 0.120)
    r = int(sample_rate * 0.350)
    s = 0.55

    env = np.ones(n, dtype=np.float64) * s
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a)
    if a + d <= n:
        env[a:a + d] = np.linspace(1.0, s, d)
    if n > r:
        env[n - r:] = np.linspace(s, 0.0, r)

    wave = (wave * env * 32767 * volume).astype(np.int16)
    stereo = np.column_stack([wave, wave])
    return pygame.sndarray.make_sound(stereo)


class PianoAR(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "AR Piano Tutor - Combined Version"
    resource_dir = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.capture: Optional[cv2.VideoCapture] = None
        self.last_frame: Optional[np.ndarray] = None       # OpenGL-ready RGB frame
        self.last_frame_bgr: Optional[np.ndarray] = None   # Original BGR frame
        self.detection_result = None
        self.frame_width = 1280
        self.frame_height = 720
        self.aspect_ratio = 16 / 9

        # smoothing
        self.landmark_history: List[List[np.ndarray]] = []
        self.SMOOTH_N = 3

        # piano state
        self.key_pressed = [False] * NUM_KEYS
        self.press_offset = [0.0] * NUM_KEYS
        self.last_hit = [0.0] * NUM_KEYS

        # audio
        self.audio_enabled = False
        self.sounds = []

        # finger state
        self.prev_tips: Dict[Tuple[int, int], np.ndarray] = {}
        self.finger_holding_key: Dict[Tuple[int, int], Optional[int]] = {}

        # anchor-based piano transform
        self.piano_anchor_pos = np.array([0.0, KEY_Y, KEY_Z], dtype="f4")
        self.piano_anchor_rot = np.eye(3, dtype="f4")
        self.local_key_positions: List[np.ndarray] = []
        self.key_positions: List[np.ndarray] = []

        # aruco state
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.marker_detected = False
        self.marker_rvec: Optional[np.ndarray] = None
        self.marker_tvec: Optional[np.ndarray] = None

        # tutor song
        self.song_notes = [0, 0, 4, 4, 5, 5, 4, 3, 3, 2, 2, 1, 1, 0]
        self.song_index = 0
        self.song_finished = False

        self.correct_hits = 0
        self.wrong_hits = 0
        self.total_presses = 0

        self.correct_flash_until = [0.0] * NUM_KEYS
        self.wrong_flash_until = [0.0] * NUM_KEYS

        # session state / HUD from g3
        self.session_state = "waiting"   # waiting / playing / finished
        self.session_start_time: Optional[float] = None
        self.last_status_text = "Show open palm to start"
        self.last_status_until = 0.0

        self.last_start_gesture_time = -999.0
        self.last_reset_gesture_time = -999.0
        self.gesture_cooldown = 0.8

        pygame.font.init()
        self.hud_font = pygame.font.SysFont("Arial", 28)
        self.hud_big_font = pygame.font.SysFont("Arial", 40, bold=True)
        self.hud_tex = None
        self.hud_tex_size = (0, 0)

        self._init_audio()
        self._init_3d_shader()
        self._init_background_resources()
        self._init_geometry()
        self._init_piano_layout()
        self._init_camera()

    def _init_audio(self) -> None:
        try:
            pygame.mixer.pre_init(44100, -16, 2, 512)
            pygame.mixer.init()
            self.sounds = [_make_piano_tone(f) for f in NOTE_FREQ]
            self.audio_enabled = True
        except Exception as e:
            print(f"[WARN] Audio init failed: {e}")
            self.audio_enabled = False
            self.sounds = [None] * NUM_KEYS

    def _init_3d_shader(self) -> None:
        self.prog3d = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 Mvp;
                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_texcoord_0;
                out vec3 v_vert;
                out vec3 v_norm;
                void main() {
                    gl_Position = Mvp * vec4(in_position, 1.0);
                    v_vert = in_position;
                    v_norm = in_normal;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform vec3 Color;
                uniform vec3 Light;
                in vec3 v_vert;
                in vec3 v_norm;
                out vec4 f_color;
                void main() {
                    float lum = clamp(dot(normalize(Light - v_vert), normalize(v_norm)),
                                      0.0, 1.0) * 0.8 + 0.2;
                    f_color = vec4(Color * lum, 1.0);
                }
            ''',
        )
        self.mvp_u = self.prog3d["Mvp"]
        self.light_u = self.prog3d["Light"]
        self.color_u = self.prog3d["Color"]

    def _init_background_resources(self) -> None:
        self.prog_bg = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_position;
                in vec2 in_texcoord;
                out vec2 v_texcoord;
                void main() {
                    gl_Position = vec4(in_position, 0.0, 1.0);
                    v_texcoord = in_texcoord;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D bg_texture;
                in vec2 v_texcoord;
                out vec4 f_color;
                void main() {
                    f_color = texture(bg_texture, v_texcoord);
                }
            ''',
        )

        # Use g3 orientation, which matches its flipud upload path
        quad_verts = np.array([
            -1, -1, 0, 0,
             1, -1, 1, 0,
            -1,  1, 0, 1,

             1, -1, 1, 0,
             1,  1, 1, 1,
            -1,  1, 0, 1,
        ], dtype="f4")

        vbo = self.ctx.buffer(quad_verts.tobytes())
        self.vao_quad = self.ctx.vertex_array(
            self.prog_bg,
            [(vbo, "2f 2f", "in_position", "in_texcoord")]
        )

        self.bg_tex = self.ctx.texture((1, 1), 3)
        self.bg_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.bg_tex_size = (0, 0)

    def _find_asset(self, filename: str) -> str:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, filename),
            os.path.join(here, "data", filename),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Cannot find asset: {filename}")

    def _init_geometry(self) -> None:
        crate_path = self._find_asset("crate.obj")
        marker_path = self._find_asset("marker.obj")

        self.scene_key = self.load_scene(crate_path)
        self.scene_marker = self.load_scene(marker_path)

        self.vao_key = self.scene_key.root_nodes[0].mesh.vao.instance(self.prog3d)
        self.vao_marker = self.scene_marker.root_nodes[0].mesh.vao.instance(self.prog3d)

    def build_local_key_layout(self) -> List[np.ndarray]:
        local_positions: List[np.ndarray] = []
        total_w = NUM_WHITE * KEY_W
        start_x = -total_w / 2.0 + KEY_W / 2.0

        for i in range(NUM_WHITE):
            local_positions.append(np.array([start_x + i * KEY_W, 0.0, 0.0], dtype="f4"))

        bk_y = 0.0
        bk_z = 2.0
        for i in BLACK_OFFSETS:
            bk_x = start_x + i * KEY_W + KEY_W / 2.0
            local_positions.append(np.array([bk_x, bk_y, bk_z], dtype="f4"))

        return local_positions

    def compute_world_key_positions(self) -> List[np.ndarray]:
        world_positions = []
        for local_pos in self.local_key_positions:
            world_pos = self.piano_anchor_rot @ local_pos + self.piano_anchor_pos
            world_positions.append(world_pos.astype("f4"))
        return world_positions

    def refresh_piano_world_positions(self) -> None:
        self.key_positions = self.compute_world_key_positions()

    def _init_piano_layout(self) -> None:
        self.local_key_positions = self.build_local_key_layout()
        self.refresh_piano_world_positions()

    def _init_camera(self) -> None:
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.capture = cv2.VideoCapture(0)

        if not self.capture.isOpened():
            raise RuntimeError("Failed to open camera.")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        ret, frame = self.capture.read()
        if not ret or frame is None:
            raise RuntimeError("Camera opened, but failed to read the first frame.")

        self.frame_height, self.frame_width = frame.shape[:2]
        self.aspect_ratio = float(self.frame_width) / float(self.frame_height)

    def update_camera_frame(self) -> None:
        if self.capture is None:
            return

        ret, frame = self.capture.read()
        if not ret or frame is None:
            print("[WARN] camera read failed, keep previous frame")
            return

        frame = cv2.flip(frame, 1)
        self.last_frame_bgr = frame.copy()

        win_w, win_h = self.wnd.size
        if win_w <= 0 or win_h <= 0:
            return

        frame_resized = cv2.resize(frame, (win_w, win_h))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_rgb = np.ascontiguousarray(np.flipud(frame_rgb))

        if self.bg_tex is None or self.bg_tex_size != (win_w, win_h):
            try:
                if self.bg_tex is not None:
                    self.bg_tex.release()
            except Exception:
                pass

            self.bg_tex = self.ctx.texture((win_w, win_h), 3)
            self.bg_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.bg_tex.repeat_x = False
            self.bg_tex.repeat_y = False
            self.bg_tex_size = (win_w, win_h)

        try:
            self.bg_tex.write(frame_rgb.tobytes())
            self.last_frame = frame_rgb
            self.frame_height, self.frame_width = frame_resized.shape[:2]
            self.aspect_ratio = float(self.frame_width) / float(self.frame_height)
        except Exception as e:
            print(f"[WARN] bg texture write failed: {e}")

    def update_hand_tracking(self) -> List[np.ndarray]:
        gl_landmarks_list: List[np.ndarray] = []

        if self.last_frame_bgr is None:
            self.detection_result = None
            return gl_landmarks_list

        frame_rgb_for_predict = cv2.cvtColor(self.last_frame_bgr, cv2.COLOR_BGR2RGB)
        self.detection_result = predict(frame_rgb_for_predict)
        if not self.detection_result:
            return gl_landmarks_list

        camera_matrix = get_camera_matrix(self.frame_width, self.frame_height)
        world_list = solvepnp(
            self.detection_result.hand_world_landmarks,
            self.detection_result.hand_landmarks,
            camera_matrix,
            self.frame_width,
            self.frame_height,
        )

        for wl in world_list:
            gl_pts = wl * 100.0
            gl_pts[:, 1] *= -1
            gl_pts[:, 2] *= -1
            gl_landmarks_list.append(gl_pts.astype("f4"))

        return gl_landmarks_list

    def smooth_landmarks(self, gl_landmarks_list: List[np.ndarray]) -> List[np.ndarray]:
        self.landmark_history.append(gl_landmarks_list)
        if len(self.landmark_history) > self.SMOOTH_N:
            self.landmark_history.pop(0)

        num_hands = len(gl_landmarks_list)
        if num_hands == 0:
            return gl_landmarks_list

        smoothed = []
        for hi in range(num_hands):
            frames_with_hand = [h[hi] for h in self.landmark_history if len(h) > hi]
            if len(frames_with_hand) == len(self.landmark_history):
                avg = np.mean(frames_with_hand, axis=0)
            else:
                avg = gl_landmarks_list[hi]
            smoothed.append(avg.astype("f4"))
        return smoothed

    def update_marker_pose(self) -> None:
        self.marker_detected = False
        self.marker_rvec = None
        self.marker_tvec = None

        if self.last_frame_bgr is None:
            return

        gray = cv2.cvtColor(self.last_frame_bgr, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return

        ids = ids.flatten()
        if ARUCO_TARGET_ID not in ids:
            return

        target_idx = int(np.where(ids == ARUCO_TARGET_ID)[0][0])
        target_corners = [corners[target_idx]]

        camera_matrix = get_camera_matrix(self.frame_width, self.frame_height)
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            target_corners,
            ARUCO_MARKER_SIZE_M,
            camera_matrix,
            dist_coeffs
        )

        if rvecs is None or tvecs is None:
            return

        self.marker_rvec = rvecs[0].reshape(3)
        self.marker_tvec = tvecs[0].reshape(3)
        self.marker_detected = True

    def update_piano_anchor_from_marker(self) -> None:
        if not self.marker_detected or self.marker_rvec is None or self.marker_tvec is None:
            return

        R_cv, _ = cv2.Rodrigues(self.marker_rvec)
        t_cv = self.marker_tvec.reshape(3, 1)
        anchor_cv = R_cv @ PIANO_OFFSET_MARKER_LOCAL.reshape(3, 1) + t_cv

        anchor_gl = anchor_cv.copy()
        anchor_gl[1, 0] *= -1.0
        anchor_gl[2, 0] *= -1.0
        self.piano_anchor_pos = (anchor_gl[:, 0] * 100.0).astype("f4")

        flip = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
        R_gl = flip @ R_cv @ flip

        theta = np.deg2rad(180.0)
        Ry = np.array([
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ], dtype=np.float32)

        self.piano_anchor_rot = (R_gl @ Ry).astype("f4")
        self.refresh_piano_world_positions()

    def debug_update_anchor(self, time: float) -> None:
        self.update_marker_pose()
        self.update_piano_anchor_from_marker()
        if not self.marker_detected:
            self.refresh_piano_world_positions()

    def current_target_key(self) -> Optional[int]:
        if self.session_state == "finished":
            return None
        if self.song_index >= len(self.song_notes):
            return None
        return self.song_notes[self.song_index]

    def accuracy(self) -> float:
        if self.total_presses == 0:
            return 0.0
        return 100.0 * self.correct_hits / self.total_presses

    def set_status(self, text: str, time: float, duration: float = 1.2) -> None:
        self.last_status_text = text
        self.last_status_until = time + duration

    def start_session(self, time: float) -> None:
        self.session_state = "playing"
        self.session_start_time = time
        self.song_index = 0
        self.song_finished = False
        self.correct_hits = 0
        self.wrong_hits = 0
        self.total_presses = 0
        self.correct_flash_until = [0.0] * NUM_KEYS
        self.wrong_flash_until = [0.0] * NUM_KEYS
        self.set_status("Started", time, 1.0)

    def reset_tutor(self, time: float) -> None:
        self.session_state = "waiting"
        self.session_start_time = None
        self.song_index = 0
        self.song_finished = False
        self.correct_hits = 0
        self.wrong_hits = 0
        self.total_presses = 0
        self.correct_flash_until = [0.0] * NUM_KEYS
        self.wrong_flash_until = [0.0] * NUM_KEYS
        self.set_status("Reset", time, 1.0)

    def register_tutor_press(self, key_idx: int, time: float) -> None:
        if self.session_state != "playing":
            return

        target = self.current_target_key()
        if target is None:
            return

        self.total_presses += 1

        if key_idx == target:
            self.correct_hits += 1
            self.correct_flash_until[key_idx] = time + FLASH_TIME
            self.song_index += 1
            self.set_status(f"Correct: {NOTE_NAMES[key_idx]}", time, 0.7)

            if self.song_index >= len(self.song_notes):
                self.song_finished = True
                self.session_state = "finished"
                self.set_status("Finished!", time, 2.0)
        else:
            self.wrong_hits += 1
            self.wrong_flash_until[key_idx] = time + FLASH_TIME
            self.set_status(f"Wrong: {NOTE_NAMES[key_idx]}", time, 0.7)

    def is_open_palm(self, hand_pts: np.ndarray) -> bool:
        wrist = hand_pts[0]
        tips = [hand_pts[i] for i in [8, 12, 16, 20]]
        return all(abs(t[1] - wrist[1]) > 3.0 for t in tips)

    def is_pinch(self, hand_pts: np.ndarray) -> bool:
        thumb = hand_pts[4]
        index_tip = hand_pts[8]
        return np.linalg.norm(thumb - index_tip) < 2.5

    def update_session_gestures(self, gl_landmarks_list: List[np.ndarray], time: float) -> None:
        for hand_pts in gl_landmarks_list:
            if self.session_state == "waiting":
                if time - self.last_start_gesture_time > self.gesture_cooldown and self.is_open_palm(hand_pts):
                    self.last_start_gesture_time = time
                    self.start_session(time)
                    return
            elif self.session_state == "finished":
                if time - self.last_reset_gesture_time > self.gesture_cooldown and self.is_pinch(hand_pts):
                    self.last_reset_gesture_time = time
                    self.reset_tutor(time)
                    return

    def locate_key_under_tip(self, tip: np.ndarray) -> Optional[int]:
        for k in range(NUM_WHITE, NUM_KEYS):
            kpos = self.key_positions[k]
            in_x = abs(tip[0] - kpos[0]) < BK_W * 0.55
            in_z = abs(tip[2] - kpos[2]) < PRESS_THR
            in_y = (kpos[1] - BK_H) < tip[1] < (kpos[1] + BK_H * 2.5)
            if in_x and in_z and in_y:
                return k

        for k in range(NUM_WHITE):
            kpos = self.key_positions[k]
            in_x = abs(tip[0] - kpos[0]) < KEY_W * 0.55
            in_z = abs(tip[2] - kpos[2]) < PRESS_THR
            in_y = (kpos[1] - KEY_H) < tip[1] < (kpos[1] + KEY_H * 2.5)
            if in_x and in_z and in_y:
                return k

        return None

    def tip_is_pressing(self, prev_tip: np.ndarray, tip: np.ndarray, key_idx: int) -> bool:
        dy = float(tip[1] - prev_tip[1])
        return abs(dy) > MIN_PRESS_MOTION

    def update_release_state(self, finger_id: Tuple[int, int], tip: np.ndarray) -> None:
        held_key = self.finger_holding_key.get(finger_id)
        if held_key is None:
            return

        held_pos = self.key_positions[held_key]
        key_still_under_tip = (self.locate_key_under_tip(tip) == held_key)
        too_far_in_y = abs(float(tip[1] - held_pos[1])) > (KEY_H * 2.0 + RELEASE_Y_MARGIN)

        if (not key_still_under_tip) or too_far_in_y:
            self.finger_holding_key[finger_id] = None

    def trigger_key_press(self, key_idx: int, time: float) -> None:
        self.key_pressed[key_idx] = True
        if time - self.last_hit[key_idx] > COOLDOWN:
            if self.audio_enabled and self.sounds[key_idx] is not None:
                self.sounds[key_idx].play()
            self.last_hit[key_idx] = time
            self.register_tutor_press(key_idx, time)

    def update_piano_interaction(self, gl_landmarks_list: List[np.ndarray], time: float) -> None:
        self.key_pressed = [False] * NUM_KEYS
        current_finger_ids = set()

        for hand_id, gl_pts in enumerate(gl_landmarks_list):
            for tip_idx in FINGER_TIPS:
                finger_id = (hand_id, tip_idx)
                current_finger_ids.add(finger_id)

                tip = gl_pts[tip_idx]
                prev_tip = self.prev_tips.get(finger_id)
                current_key = self.locate_key_under_tip(tip)

                self.update_release_state(finger_id, tip)

                held_key = self.finger_holding_key.get(finger_id)
                if held_key is not None:
                    self.key_pressed[held_key] = True

                if prev_tip is not None and current_key is not None:
                    already_holding_this = (self.finger_holding_key.get(finger_id) == current_key)
                    not_holding_anything = (self.finger_holding_key.get(finger_id) is None)

                    if not already_holding_this and not_holding_anything:
                        if self.tip_is_pressing(prev_tip, tip, current_key):
                            self.finger_holding_key[finger_id] = current_key
                            self.trigger_key_press(current_key, time)

                self.prev_tips[finger_id] = tip.copy()

        stale_ids = [fid for fid in self.prev_tips.keys() if fid not in current_finger_ids]
        for fid in stale_ids:
            self.prev_tips.pop(fid, None)
            self.finger_holding_key.pop(fid, None)

        for held_key in self.finger_holding_key.values():
            if held_key is not None:
                self.key_pressed[held_key] = True

        for k in range(NUM_KEYS):
            target = -1.2 if self.key_pressed[k] else 0.0
            self.press_offset[k] += (target - self.press_offset[k]) * 0.35

    def base_key_color(self, key_idx: int) -> Tuple[float, float, float]:
        if key_idx < NUM_WHITE:
            shade = 0.88 + (key_idx % 2) * 0.07
            return (shade, shade, shade)
        return BLACK_KEY

    def get_key_color(self, key_idx: int, time: float) -> Tuple[float, float, float]:
        if self.session_state == "finished":
            if self.correct_flash_until[key_idx] > time:
                return CORRECT_COLOR
            return DONE_WHITE if key_idx < NUM_WHITE else DONE_BLACK

        if self.correct_flash_until[key_idx] > time:
            return CORRECT_COLOR
        if self.wrong_flash_until[key_idx] > time:
            return WRONG_COLOR

        target = self.current_target_key()
        if self.session_state == "playing" and target is not None and key_idx == target:
            return TARGET_WHITE if key_idx < NUM_WHITE else TARGET_BLACK
        if self.key_pressed[key_idx]:
            return PRESSED
        return self.base_key_color(key_idx)

    def beat_progress(self, time: float) -> float:
        if self.session_state != "playing" or self.session_start_time is None:
            return 0.0
        elapsed = time - self.session_start_time
        return (elapsed % BEAT_SEC) / BEAT_SEC

    def build_hud_surface(self, time: float) -> pygame.Surface:
        width, height = self.wnd.size
        surf = pygame.Surface((width, height), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))

        panel = pygame.Surface((520, 170), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 120))
        surf.blit(panel, (20, 20))

        if self.session_state == "waiting":
            title = self.hud_big_font.render("AR Piano Tutor", True, (255, 255, 255))
            line1 = self.hud_font.render("Show open palm to start", True, (240, 240, 240))
            line2 = self.hud_font.render("Pinch after finish to reset", True, (220, 220, 220))
            surf.blit(title, (40, 35))
            surf.blit(line1, (40, 85))
            surf.blit(line2, (40, 115))
        elif self.session_state == "playing":
            target = self.current_target_key()
            target_name = NOTE_NAMES[target] if target is not None else "-"
            title = self.hud_big_font.render(f"Target: {target_name}", True, (255, 255, 255))
            score = self.hud_font.render(
                f"Correct {self.correct_hits}   Wrong {self.wrong_hits}   Acc {self.accuracy():.1f}%",
                True, (240, 240, 240)
            )
            progress = self.hud_font.render(
                f"Progress {self.song_index}/{len(self.song_notes)}",
                True, (230, 230, 230)
            )
            surf.blit(title, (40, 35))
            surf.blit(score, (40, 85))
            surf.blit(progress, (40, 115))

            bar_x, bar_y, bar_w, bar_h = 40, 155, 300, 16
            pygame.draw.rect(surf, (90, 90, 90), (bar_x, bar_y, bar_w, bar_h), border_radius=6)
            fill_w = int(bar_w * self.beat_progress(time))
            pygame.draw.rect(surf, (255, 220, 80), (bar_x, bar_y, fill_w, bar_h), border_radius=6)
        else:
            title = self.hud_big_font.render("Finished!", True, (255, 255, 255))
            line1 = self.hud_font.render(
                f"Accuracy {self.accuracy():.1f}%   Correct {self.correct_hits}/{len(self.song_notes)}",
                True, (240, 240, 240)
            )
            line2 = self.hud_font.render("Pinch to reset", True, (220, 220, 220))
            surf.blit(title, (40, 35))
            surf.blit(line1, (40, 85))
            surf.blit(line2, (40, 115))

        if time < self.last_status_until:
            msg = self.hud_font.render(self.last_status_text, True, (255, 255, 255))
            msg_bg = pygame.Surface((msg.get_width() + 24, msg.get_height() + 14), pygame.SRCALPHA)
            msg_bg.fill((0, 0, 0, 150))
            x = width - msg_bg.get_width() - 20
            y = 20
            surf.blit(msg_bg, (x, y))
            surf.blit(msg, (x + 12, y + 7))

        return surf

    def render_background(self) -> None:
        if self.bg_tex is None or self.last_frame is None:
            return

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.BLEND)

        try:
            self.bg_tex.use(location=0)
            self.prog_bg["bg_texture"].value = 0
            self.vao_quad.render(moderngl.TRIANGLES)
        except Exception as e:
            print(f"[WARN] render_background failed: {e}")

        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

    def render_hud(self, time: float) -> None:
        surf = self.build_hud_surface(time)
        surf = pygame.transform.flip(surf, False, True)

        hud_rgba = pygame.image.tostring(surf, "RGBA", False)
        size = surf.get_size()

        if self.hud_tex is None or self.hud_tex_size != size:
            if self.hud_tex is not None:
                self.hud_tex.release()
            self.hud_tex = self.ctx.texture(size, 4)
            self.hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.hud_tex_size = size

        self.hud_tex.write(hud_rgba)

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (
            moderngl.SRC_ALPHA,
            moderngl.ONE_MINUS_SRC_ALPHA,
            moderngl.ONE,
            moderngl.ONE_MINUS_SRC_ALPHA,
        )

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)

        self.hud_tex.use(location=0)
        self.prog_bg["bg_texture"].value = 0
        self.vao_quad.render(moderngl.TRIANGLES)

        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

    def build_projection_matrix(self) -> Matrix44:
        camera_matrix = get_camera_matrix(self.frame_width, self.frame_height)
        fov_y = get_fov_y(camera_matrix, self.frame_height)
        return Matrix44.perspective_projection(fov_y, self.aspect_ratio, 0.1, 1000.0)

    def render_piano(self, proj: Matrix44, time: float) -> None:
        self.light_u.value = (0.0, 20.0, 10.0)

        for k in range(NUM_WHITE):
            pos = self.key_positions[k].copy()
            pos[1] += self.press_offset[k]
            T = Matrix44.from_translation(pos)
            S = Matrix44.from_scale((KEY_W * 0.48, KEY_H * 0.5, KEY_D * 0.5))
            mvp = proj * T * S
            self.color_u.value = self.get_key_color(k, time)
            self.mvp_u.write(mvp.astype("f4"))
            self.vao_key.render()

        for k in range(NUM_WHITE, NUM_KEYS):
            pos = self.key_positions[k].copy()
            pos[1] += self.press_offset[k]
            T = Matrix44.from_translation(pos)
            S = Matrix44.from_scale((BK_W * 0.48, BK_H * 0.5, BK_D * 0.5))
            mvp = proj * T * S
            self.color_u.value = self.get_key_color(k, time)
            self.mvp_u.write(mvp.astype("f4"))
            self.vao_key.render()

    def render_landmarks(self, proj: Matrix44, gl_landmarks_list: List[np.ndarray]) -> None:
        for gl_pts in gl_landmarks_list:
            for i, pt in enumerate(gl_pts):
                if i in FINGER_TIPS:
                    self.color_u.value = (1.0, 0.4, 0.1)
                    sz = 0.45
                else:
                    self.color_u.value = (0.2, 0.9, 0.3)
                    sz = 0.25

                T = Matrix44.from_translation(pt.astype("f4"))
                S = Matrix44.from_scale((sz, sz, sz))
                mvp = proj * T * S
                self.mvp_u.write(mvp.astype("f4"))
                self.vao_marker.render()

    def on_render(self, time: float, frame_time: float) -> None:
        self.ctx.clear(0.1, 0.1, 0.1)

        self.update_camera_frame()
        self.render_background()

        gl_landmarks_list = self.update_hand_tracking()
        gl_landmarks_list = self.smooth_landmarks(gl_landmarks_list)

        self.debug_update_anchor(time)
        self.update_session_gestures(gl_landmarks_list, time)
        self.update_piano_interaction(gl_landmarks_list, time)

        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        proj = self.build_projection_matrix()
        self.render_piano(proj, time)
        self.render_landmarks(proj, gl_landmarks_list)
        self.render_hud(time)

    def _release_resources(self) -> None:
        try:
            if self.capture is not None:
                self.capture.release()
                self.capture = None
        except Exception:
            pass

        try:
            if hasattr(self, "bg_tex") and self.bg_tex is not None:
                self.bg_tex.release()
                self.bg_tex = None
        except Exception:
            pass

        try:
            if self.hud_tex is not None:
                self.hud_tex.release()
                self.hud_tex = None
        except Exception:
            pass

        try:
            if pygame.mixer.get_init():
                pygame.mixer.quit()
        except Exception:
            pass

    def close(self) -> None:
        self._release_resources()

    def __del__(self):
        self._release_resources()


if __name__ == "__main__":
    mglw.run_window_config(PianoAR)