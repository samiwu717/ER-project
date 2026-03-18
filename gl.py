import moderngl
import moderngl_window as mglw
from pyrr import Matrix44

import cv2
import numpy as np
import os
import pygame

from prediction import predict, get_camera_matrix, get_fov_y, solvepnp


# ── Piano configuration ────────────────────────────────────────────────────────
NUM_KEYS   = 8
KEY_W      = 3.5    # cm  (width of one key)
KEY_H      = 1.5    # cm  (thickness / height)
KEY_D      = 9.0    # cm  (depth, towards camera)
KEY_Z      = -28.0  # cm  (distance in front of camera)
KEY_Y      = -4.0   # cm  (vertical position, slightly below centre)
PRESS_THR  = 3.5    # cm  (how close a fingertip must get in Z to trigger)
COOLDOWN   = 0.25   # s   (minimum seconds between two presses of the same key)

# White-key notes: C4 D4 E4 F4 G4 A4 B4 C5
NOTE_FREQ  = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
NOTE_NAMES = ["C4",   "D4",   "E4",   "F4",   "G4",   "A4",   "B4",   "C5"]

# Finger-tip landmark indices (MediaPipe): index / middle / ring / pinky
FINGER_TIPS = [8, 12, 16, 20]

# Colour palette
WHITE_KEY  = (0.95, 0.95, 0.95)
PRESSED    = (0.30, 0.75, 1.00)   # light blue when pressed
LABEL_CLR  = (0.20, 0.20, 0.80)   # marker colour


def _make_sine(freq, duration=0.35, sample_rate=44100, volume=0.5):
    """Return a 16-bit stereo pygame Sound for a pure sine tone."""
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    wave = (np.sin(2 * np.pi * freq * t) * 32767 * volume).astype(np.int16)
    # Fade out to avoid click
    fade = int(sample_rate * 0.05)
    wave[-fade:] = (wave[-fade:] * np.linspace(1, 0, fade)).astype(np.int16)
    stereo = np.column_stack([wave, wave])
    return pygame.sndarray.make_sound(stereo)


class PianoAR(mglw.WindowConfig):
    gl_version  = (3, 3)
    title       = "AR Air Piano"
    resource_dir = os.path.normpath(os.path.join(__file__, '../data'))

    # ── init ──────────────────────────────────────────────────────────────────
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ── pygame audio ──────────────────────────────────────────────────────
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
        self.sounds   = [_make_sine(f) for f in NOTE_FREQ]
        self.last_hit = [0.0] * NUM_KEYS   # cooldown timers

        # ── 3-D shader (same as original) ─────────────────────────────────────
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
                    float lum = clamp(dot(normalize(Light - v_vert),
                                         normalize(v_norm)), 0.0, 1.0) * 0.8 + 0.2;
                    f_color = vec4(Color * lum, 1.0);
                }
            ''',
        )
        self.mvp_u   = self.prog3d['Mvp']
        self.light_u = self.prog3d['Light']
        self.color_u = self.prog3d['Color']

        # Re-use the cube .obj as a piano key shape (we scale it per-key)
        self.scene_key    = self.load_scene('crate.obj')
        self.scene_marker = self.load_scene('marker.obj')
        self.vao_key      = self.scene_key.root_nodes[0].mesh.vao.instance(self.prog3d)
        self.vao_marker   = self.scene_marker.root_nodes[0].mesh.vao.instance(self.prog3d)

        # ── Background shader (identical to original) ─────────────────────────
        self.prog_bg = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_position;
                in vec2 in_texcoord;
                out vec2 v_texcoord;
                void main() {
                    gl_Position = vec4(in_position, 0.0, 1.0);
                    v_texcoord  = in_texcoord;
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
        quad_verts = np.array([
            -1,-1, 0,1,   1,-1, 1,1,  -1, 1, 0,0,
             1,-1, 1,1,   1, 1, 1,0,  -1, 1, 0,0,
        ], dtype='f4')
        vbo = self.ctx.buffer(quad_verts.tobytes())
        self.vao_quad = self.ctx.vertex_array(
            self.prog_bg, [(vbo, '2f 2f', 'in_position', 'in_texcoord')])
        self.bg_tex      = self.ctx.texture((1, 1), 3)
        self.bg_tex_size = (0, 0)

        # ── Piano key state ───────────────────────────────────────────────────
        # Spread keys symmetrically along X axis
        total_w = NUM_KEYS * KEY_W
        start_x = -total_w / 2.0 + KEY_W / 2.0
        self.key_positions = [
            np.array([start_x + i * KEY_W, KEY_Y, KEY_Z], dtype='f4')
            for i in range(NUM_KEYS)
        ]
        self.key_pressed  = [False] * NUM_KEYS
        self.press_offset = [0.0]  * NUM_KEYS   # visual dip when pressed

        # ── OpenCV camera ─────────────────────────────────────────────────────
        self.capture = cv2.VideoCapture(0)
        ret, frame   = self.capture.read()
        self.aspect_ratio  = float(frame.shape[1]) / frame.shape[0]
        self.window_size   = (int(720.0 * self.aspect_ratio), 720)
        self.frame_width   = frame.shape[1]
        self.frame_height  = frame.shape[0]
        self.last_frame    = None
        self.detection_result = None

    # ── render loop ───────────────────────────────────────────────────────────
    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(0.1, 0.1, 0.1)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        # ── 1. Grab camera frame & draw as background ─────────────────────────
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.flip(frame, 1)
            win_w, win_h = self.wnd.size
            frame = cv2.resize(frame, (win_w, win_h))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.bg_tex_size != (win_w, win_h):
                self.bg_tex.release()
                self.bg_tex = self.ctx.texture((win_w, win_h), 3)
                self.bg_tex_size = (win_w, win_h)

            self.bg_tex.write(frame_rgb.tobytes())
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.disable(moderngl.CULL_FACE)
            self.bg_tex.use(location=0)
            self.prog_bg['bg_texture'].value = 0
            self.vao_quad.render(moderngl.TRIANGLES)
            self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

            self.last_frame   = frame_rgb
            self.frame_height, self.frame_width = frame_rgb.shape[:2]

        # ── 2. Hand tracking ──────────────────────────────────────────────────
        gl_landmarks_list = []
        if self.last_frame is not None:
            self.detection_result = predict(self.last_frame)

        if self.detection_result:
            camera_matrix = get_camera_matrix(self.frame_width, self.frame_height)
            world_list = solvepnp(
                self.detection_result.hand_world_landmarks,
                self.detection_result.hand_landmarks,
                camera_matrix, self.frame_width, self.frame_height,
            )
            for wl in world_list:
                gl_pts = wl * 100.0
                gl_pts[:, 1] *= -1
                gl_pts[:, 2] *= -1
                gl_landmarks_list.append(gl_pts)

        # ── 3. Piano key hit detection ────────────────────────────────────────
        # Reset pressed state each frame
        self.key_pressed = [False] * NUM_KEYS

        for gl_pts in gl_landmarks_list:
            for tip_idx in FINGER_TIPS:
                tip = gl_pts[tip_idx]
                for k, kpos in enumerate(self.key_positions):
                    in_x = abs(tip[0] - kpos[0]) < KEY_W * 0.55
                    in_z = abs(tip[2] - kpos[2]) < PRESS_THR
                    # Fingertip must be at or below the key top surface
                    in_y = (kpos[1] - KEY_H) < tip[1] < (kpos[1] + KEY_H * 2)

                    if in_x and in_z and in_y:
                        self.key_pressed[k] = True
                        # Trigger sound with cooldown
                        if time - self.last_hit[k] > COOLDOWN:
                            self.sounds[k].play()
                            self.last_hit[k] = time

        # Animate press offset (smooth dip)
        for k in range(NUM_KEYS):
            target = -1.2 if self.key_pressed[k] else 0.0
            self.press_offset[k] += (target - self.press_offset[k]) * 0.35

        # ── 4. Build projection matrix ────────────────────────────────────────
        camera_matrix = get_camera_matrix(self.frame_width, self.frame_height)
        fov_y = get_fov_y(camera_matrix, self.frame_height)
        proj  = Matrix44.perspective_projection(fov_y, self.aspect_ratio, 0.1, 1000.0)

        self.light_u.value = (0.0, 20.0, 10.0)

        # ── 5. Render piano keys ──────────────────────────────────────────────
        for k, kpos in enumerate(self.key_positions):
            pos = kpos.copy()
            pos[1] += self.press_offset[k]

            T = Matrix44.from_translation(pos)
            S = Matrix44.from_scale((KEY_W * 0.48, KEY_H * 0.5, KEY_D * 0.5))
            mvp = proj * T * S

            if self.key_pressed[k]:
                self.color_u.value = PRESSED
            else:
                # Slight colour variation so adjacent keys are distinguishable
                shade = 0.88 + (k % 2) * 0.07
                self.color_u.value = (shade, shade, shade)

            self.mvp_u.write(mvp.astype('f4'))
            self.vao_key.render()

        # ── 6. Render hand landmark markers ───────────────────────────────────
        for gl_pts in gl_landmarks_list:
            for i, pt in enumerate(gl_pts):
                # Highlight finger tips in a different colour
                if i in FINGER_TIPS:
                    self.color_u.value = (1.0, 0.4, 0.1)
                    sz = 0.45
                else:
                    self.color_u.value = (0.2, 0.9, 0.3)
                    sz = 0.25

                T   = Matrix44.from_translation(pt.astype('f4'))
                S   = Matrix44.from_scale((sz, sz, sz))
                mvp = proj * T * S
                self.mvp_u.write(mvp.astype('f4'))
                self.vao_marker.render()


if __name__ == '__main__':
    PianoAR.run()
