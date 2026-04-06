"""
How to use:
    python tutorial.py                           # single-camera, default index
    python tutorial.py --cam 2                   # pick camera index
    python tutorial.py --mode dual --top-cam 2 --side-cam 3

In-window controls:
    T          — toggle Tutorial / Free-play mode
    R          — restart song (resets step & error count)
    Q / Esc    — quit
    Click x2   — draw press threshold line (same as original)
"""

import argparse
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import paper_piano_system as _sys
from paper_piano_system import (
    ArucoKeyboardMapper, ClickLineSelector, MediaPipeHandTracker,
    PressState, SimpleSynth,
    NOTE_NAMES, NUM_WHITE_KEYS, FINGERTIP_IDS,
    KEY_COOLDOWN_SEC, PRESS_DIST_RATIO, RELEASE_DIST_RATIO,
    MIRROR_SINGLE, MIRROR_TOP,
    MAX_CONSECUTIVE_READ_FAILS,
    DEFAULT_SINGLE_CAM_INDEX, DEFAULT_TOP_CAM_INDEX, DEFAULT_SIDE_CAM_INDEX,
    CAM_SCAN_MAX_INDEX, EXTERNAL_MIN_INDEX, ARUCO_CORNER_IDS,
    PAPER_W, PAPER_H,
    open_camera, choose_two_camera_indices,
    signed_distance_point_to_line, draw_line, format_finger_label,
)

# ── Tutorial config ───────────────────────────────────────────────────────────
TUTORIAL_MODE_DEFAULT = True

SONG_NAME = "Twinkle Twinkle Little Star"
# C4=0 D4=1 E4=2 F4=3 G4=4 A4=5 B4=6 C5=7
SONG_SEQUENCE = [
    0, 0, 4, 4, 5, 5, 4,
    3, 3, 2, 2, 1, 1, 0,
    4, 4, 3, 3, 2, 2, 1,
    4, 4, 3, 3, 2, 2, 1,
    0, 0, 4, 4, 5, 5, 4,
    3, 3, 2, 2, 1, 1, 0,
]

COLOR_TARGET  = (255, 200,  0)
COLOR_CORRECT = ( 40, 220, 40)
COLOR_WRONG   = (  0,   0, 220)
FLASH_SEC     = 0.35


# ── Tutorial state ────────────────────────────────────────────────────────────
@dataclass
class TutorialState:
    mode: bool = TUTORIAL_MODE_DEFAULT
    step: int = 0
    errors: int = 0
    completed: bool = False
    correct_flash_until: List[float] = field(default_factory=lambda: [0.0]*NUM_WHITE_KEYS)
    wrong_flash_until:   List[float] = field(default_factory=lambda: [0.0]*NUM_WHITE_KEYS)

    def reset(self):
        self.step, self.errors, self.completed = 0, 0, False
        self.correct_flash_until = [0.0] * NUM_WHITE_KEYS
        self.wrong_flash_until   = [0.0] * NUM_WHITE_KEYS

    @property
    def target_key(self) -> Optional[int]:
        if self.mode and not self.completed and self.step < len(SONG_SEQUENCE):
            return SONG_SEQUENCE[self.step]
        return None

    def register_press(self, key_idx: int, synth: SimpleSynth, now: float):
        if not self.mode or self.completed:
            return
        target = self.target_key
        if target is None:
            return
        synth.play(key_idx)
        if key_idx == target:
            self.correct_flash_until[key_idx] = now + FLASH_SEC
            self.step += 1
            if self.step >= len(SONG_SEQUENCE):
                self.completed = True
                print(f"[Tutorial] Song complete!  Total errors: {self.errors}")
            else:
                print(f"[Tutorial] CORRECT {NOTE_NAMES[key_idx]}  "
                      f"{self.step}/{len(SONG_SEQUENCE)}  errors={self.errors}")
        else:
            self.wrong_flash_until[key_idx] = now + FLASH_SEC
            self.errors += 1
            print(f"[Tutorial] WRONG {NOTE_NAMES[key_idx]} "
                  f"(expected {NOTE_NAMES[target]})  errors={self.errors}")


# ── Drawing ───────────────────────────────────────────────────────────────────
def _fill(frame, poly_img, color, alpha):
    ov = frame.copy()
    cv2.fillConvexPoly(ov, np.int32(poly_img), color)
    cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)


def draw_tutorial_keyboard(frame, mapper: ArucoKeyboardMapper,
                           H_inv, markers, ts: TutorialState, now):
    if markers is not None:
        for name, pt in markers.items():
            cv2.circle(frame, tuple(np.int32(pt)), 8, (0,255,255), -1)
            cv2.putText(frame, f"{name}:{ARUCO_CORNER_IDS[name]}",
                        tuple(np.int32(pt+np.array([8,-8],np.float32))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
    if H_inv is None:
        return

    paper_quad = np.array([[0,0],[PAPER_W,0],[PAPER_W,PAPER_H],[0,PAPER_H]], np.float32)
    cv2.polylines(frame, [np.int32(mapper.paper_to_image(H_inv, paper_quad))], True, (255,255,0), 2)

    kx0, ky0, kx1, ky1 = mapper._current_keyboard_rect()
    kpoly = np.array([[kx0,ky0],[kx1,ky0],[kx1,ky1],[kx0,ky1]], np.float32)
    cv2.polylines(frame, [np.int32(mapper.paper_to_image(H_inv, kpoly))], True, (255,255,0), 2)

    target = ts.target_key
    for idx, (x0,y0,x1,y1) in enumerate(mapper.build_key_rects()):
        poly = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], np.float32)
        poly_img = mapper.paper_to_image(H_inv, poly)

        if now < ts.correct_flash_until[idx]:
            color, thick = COLOR_CORRECT, 4;  _fill(frame, poly_img, color, 0.35)
        elif ts.mode and now < ts.wrong_flash_until[idx]:
            color, thick = COLOR_WRONG, 4;    _fill(frame, poly_img, color, 0.40)
        elif ts.mode and idx == target:
            color, thick = COLOR_TARGET, 4;   _fill(frame, poly_img, color, 0.28)
        else:
            color, thick = (255,255,255), 2

        cv2.polylines(frame, [np.int32(poly_img)], True, color, thick)
        cx = int(np.mean(poly_img[:,0])); cy = int(np.mean(poly_img[:,1]))
        lbl = NOTE_NAMES[idx]
        fs, sc, lt = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        (tw,th),_ = cv2.getTextSize(lbl, fs, sc, lt)
        cv2.putText(frame, lbl, (cx-tw//2, cy+th//2), fs, sc, color, lt)

    for x0,y0,x1,y1 in mapper._build_black_key_rects((kx0,ky0,kx1,ky1)):
        poly = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], np.float32)
        pi = mapper.paper_to_image(H_inv, poly)
        cv2.fillConvexPoly(frame, np.int32(pi), (25,25,25))
        cv2.polylines(frame, [np.int32(pi)], True, (255,255,255), 1)


def draw_tutorial_hud(frame, ts: TutorialState):
    if not ts.mode:
        return
    h, w = frame.shape[:2]
    hx, hy, hw, hh = w-422, 14, 408, 162
    ov = frame.copy()
    cv2.rectangle(ov, (hx-6,hy), (hx+hw,hy+hh), (20,20,20), -1)
    cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)
    cv2.rectangle(frame, (hx-6,hy), (hx+hw,hy+hh), (100,100,100), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if ts.completed:
        cv2.putText(frame, "Song Complete!", (hx, hy+38), font, 1.0, COLOR_CORRECT, 2)
        cv2.putText(frame, f"Errors: {ts.errors}", (hx, hy+72), font, 0.75, (220,220,220), 2)
        cv2.putText(frame, "R=restart   T=free-play", (hx, hy+110), font, 0.62, (160,160,160), 1)
        return

    total, step = len(SONG_SEQUENCE), ts.step
    cv2.putText(frame, SONG_NAME, (hx, hy+28), font, 0.62, (220,200,80), 2)
    if step < total:
        cv2.putText(frame, "Press:", (hx, hy+62), font, 0.65, (200,200,200), 1)
        cv2.putText(frame, NOTE_NAMES[SONG_SEQUENCE[step]], (hx+82, hy+62), font, 1.0, COLOR_TARGET, 3)
        upcoming = [NOTE_NAMES[SONG_SEQUENCE[s]] for s in range(step+1, min(step+5,total))]
        cv2.putText(frame, "  ->  ".join(upcoming), (hx, hy+92), font, 0.55, (130,130,130), 1)

    bx, by, bw, bth = hx, hy+108, hw-12, 14
    cv2.rectangle(frame, (bx,by), (bx+bw,by+bth), (60,60,60), -1)
    cv2.rectangle(frame, (bx,by), (bx+int(bw*step/max(total,1)),by+bth), (40,190,40), -1)
    cv2.putText(frame, f"{step}/{total}", (bx+bw+4,by+12), font, 0.5, (170,170,170), 1)
    ec = (60,60,220) if ts.errors > 0 else (130,130,130)
    cv2.putText(frame, f"Errors: {ts.errors}", (hx,hy+148), font, 0.65, ec, 2)
    cv2.putText(frame, "T=free-play  R=restart", (hx+190,hy+148), font, 0.50, (90,90,90), 1)


# ── Single-camera tutorial loop ───────────────────────────────────────────────
def run_tutorial_single(cam_idx: int):
    cap     = open_camera(cam_idx)
    tracker = MediaPipeHandTracker(max_num_hands=2)
    mapper  = ArucoKeyboardMapper()
    synth   = SimpleSynth()
    ts      = TutorialState()
    press_states: Dict[str, PressState] = {}

    cv2.namedWindow("Paper Piano - Tutorial")
    line_sel = ClickLineSelector("Paper Piano - Tutorial")

    fps_time = time.time(); fps = 0.0; fails = 0

    print(f"\n=== Tutorial: {SONG_NAME} ({len(SONG_SEQUENCE)} notes) ===")
    print("1. Click 2 points to draw the press threshold line.")
    print("   Finger pressing through this line = key press.")
    print("Controls: T=tutorial/free-play  R=restart  Q=quit\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if fails >= MAX_CONSECUTIVE_READ_FAILS:
                    print("[ERROR] Camera failed. Exiting.")
                    break
                cv2.waitKey(10); continue
            fails = 0
            now = time.time()
            fps = 1.0 / max(now-fps_time, 1e-6)
            fps_time = now

            H, H_inv, markers = mapper.update_homography(frame, now)
            mapper.update_keyboard_rect(frame, H, now)
            results  = tracker.process(frame)
            press_line = line_sel.line()
            draw_line(frame, press_line, color=(0,255,255), thickness=3)
            proc_h = max(1, frame.shape[0])

            if results.multi_hand_landmarks:
                for obs in tracker.iter_finger_observations(frame, results):
                    paper_pt = (mapper.image_to_paper(H, (float(obs.point[0]), float(obs.point[1])))
                                if H is not None else None)
                    key_idx = mapper.locate_key_stable(paper_pt) if paper_pt is not None else None
                    state   = press_states.setdefault(obs.finger_id, PressState())
                    dist_abs = None; is_press = False

                    if press_line is not None:
                        sd = signed_distance_point_to_line(obs.point, press_line[0], press_line[1])
                        dist_abs = abs(sd) / float(proc_h)
                        state.last_dist = dist_abs
                        is_press = dist_abs <= PRESS_DIST_RATIO
                    else:
                        state.last_dist = None

                    if is_press and not state.is_down:
                        state.is_down = True
                        if key_idx is not None and (now-state.last_press_t) >= KEY_COOLDOWN_SEC:
                            state.last_press_t = now
                            if ts.mode and not ts.completed:
                                ts.register_press(key_idx, synth, now)
                            else:
                                synth.play(key_idx)
                                ts.correct_flash_until[key_idx] = now + FLASH_SEC
                    elif state.is_down:
                        if dist_abs is None or dist_abs >= RELEASE_DIST_RATIO:
                            state.is_down = False

                    px, py = int(obs.point[0]), int(obs.point[1])
                    cv2.circle(frame, (px,py), 10,
                               (0,0,255) if state.is_down else (255,100,0), -1)
                    if key_idx is not None:
                        cv2.putText(frame, NOTE_NAMES[key_idx], (px+12,py-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

            draw_tutorial_keyboard(frame, mapper, H_inv, markers, ts, now)
            display = cv2.flip(frame, 1) if MIRROR_SINGLE else frame
            draw_tutorial_hud(display, ts)

            cv2.putText(display, f"FPS:{fps:.1f}  Tutorial:{'ON' if ts.mode else 'OFF (free-play)'}",
                        (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
            cv2.putText(display,
                        f"Homography: {'OK' if H is not None else 'Searching ArUco markers...'}",
                        (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0,255,0) if H is not None else (0,0,255), 2)
            cv2.putText(display,
                        f"Press line: {'set' if press_line else 'click 2 points on screen'}",
                        (10,85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow("Paper Piano - Tutorial", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key in (ord('t'), ord('T')):
                ts.mode = not ts.mode
                print(f"[INFO] Tutorial {'ON' if ts.mode else 'OFF (free-play)'}")
            elif key in (ord('r'), ord('R')):
                ts.reset(); press_states.clear(); line_sel.reset()
                print("[INFO] Tutorial restarted")
    finally:
        cap.release(); tracker.close(); synth.close(); cv2.destroyAllWindows()


# ── Dual-camera tutorial loop ─────────────────────────────────────────────────
def run_tutorial_dual(top_cam: int, side_cam: int):
    tracker_top  = MediaPipeHandTracker(max_num_hands=2)
    tracker_side = MediaPipeHandTracker(max_num_hands=2)
    mapper  = ArucoKeyboardMapper()
    synth   = SimpleSynth()
    ts      = TutorialState()
    press_states: Dict[str, PressState] = {}

    top_cap  = open_camera(top_cam)
    side_cap = open_camera(side_cam)

    cv2.namedWindow("Paper Piano - Tutorial (Side)")
    line_sel = ClickLineSelector("Paper Piano - Tutorial (Side)")
    fps_time = time.time(); fps = 0.0; fails = 0

    print(f"\n=== Tutorial (Dual Camera): {SONG_NAME} ===")
    print("Top camera: key detection. Side window: draw press line.")
    print("Controls: T=tutorial/free-play  R=restart  Q=quit\n")

    try:
        while True:
            ok_t, frame_top  = top_cap.read()
            ok_s, frame_side = side_cap.read()
            if not ok_t or frame_top is None or not ok_s or frame_side is None:
                fails += 1
                if fails >= MAX_CONSECUTIVE_READ_FAILS:
                    print("[ERROR] Camera read failure. Exiting."); break
                cv2.waitKey(10); continue
            fails = 0
            now = time.time(); fps = 1.0/max(now-fps_time,1e-6); fps_time = now

            H, H_inv, markers = mapper.update_homography(frame_top, now)
            mapper.update_keyboard_rect(frame_top, H, now)

            results_top = tracker_top.process(frame_top)
            active_keys: Dict[str, Optional[int]] = {}
            if results_top.multi_hand_landmarks:
                for obs in tracker_top.iter_finger_observations(frame_top, results_top):
                    pp = (mapper.image_to_paper(H,(float(obs.point[0]),float(obs.point[1])))
                          if H is not None else None)
                    ki = mapper.locate_key_stable(pp) if pp is not None else None
                    active_keys[obs.finger_id] = ki
                    px,py = int(obs.point[0]),int(obs.point[1])
                    cv2.circle(frame_top,(px,py),8,(255,100,0),-1)
                    if ki is not None:
                        cv2.putText(frame_top,NOTE_NAMES[ki],(px+10,py-8),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),2)

            draw_tutorial_keyboard(frame_top, mapper, H_inv, markers, ts, now)

            press_line = line_sel.line()
            draw_line(frame_side, press_line, color=(0,255,255), thickness=3)
            proc_h = max(1, frame_side.shape[0])
            results_side = tracker_side.process(frame_side)

            if results_side.multi_hand_landmarks:
                for obs in tracker_side.iter_finger_observations(frame_side, results_side):
                    ki    = active_keys.get(obs.finger_id)
                    state = press_states.setdefault(obs.finger_id, PressState())
                    dist_abs = None; is_press = False
                    if press_line is not None:
                        sd = signed_distance_point_to_line(obs.point,press_line[0],press_line[1])
                        dist_abs = abs(sd)/float(proc_h); state.last_dist = dist_abs
                        is_press = dist_abs <= PRESS_DIST_RATIO
                    else:
                        state.last_dist = None
                    if is_press and not state.is_down:
                        state.is_down = True
                        if ki is not None and (now-state.last_press_t) >= KEY_COOLDOWN_SEC:
                            state.last_press_t = now
                            if ts.mode and not ts.completed:
                                ts.register_press(ki, synth, now)
                            else:
                                synth.play(ki); ts.correct_flash_until[ki] = now+FLASH_SEC
                    elif state.is_down:
                        if dist_abs is None or dist_abs >= RELEASE_DIST_RATIO:
                            state.is_down = False
                    px,py=int(obs.point[0]),int(obs.point[1])
                    cv2.circle(frame_side,(px,py),10,(0,0,255) if state.is_down else (255,100,0),-1)

            disp_top  = cv2.flip(frame_top,1)  if MIRROR_TOP  else frame_top
            disp_side = cv2.flip(frame_side,1) if _sys.MIRROR_SIDE else frame_side
            draw_tutorial_hud(disp_top, ts)
            cv2.putText(disp_top, f"FPS:{fps:.1f} | Tutorial:{'ON' if ts.mode else 'OFF'}",
                        (10,25), cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),2)
            cv2.putText(disp_top,
                        f"Homography: {'OK' if H is not None else 'Searching...'}",
                        (10,55), cv2.FONT_HERSHEY_SIMPLEX,0.65,
                        (0,255,0) if H is not None else (0,0,255),2)
            cv2.imshow("Paper Piano - Tutorial (Top)",  disp_top)
            cv2.imshow("Paper Piano - Tutorial (Side)", disp_side)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')): break
            elif key in (ord('t'),ord('T')):
                ts.mode = not ts.mode
                print(f"[INFO] Tutorial {'ON' if ts.mode else 'OFF (free-play)'}")
            elif key in (ord('r'),ord('R')):
                ts.reset(); press_states.clear(); line_sel.reset()
                print("[INFO] Tutorial restarted")
    finally:
        top_cap.release(); side_cap.release()
        tracker_top.close(); tracker_side.close()
        synth.close(); cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Paper Piano Tutorial")
    parser.add_argument("--mode", choices=["single","dual"], default="single")
    parser.add_argument("--cam",      type=int, default=DEFAULT_SINGLE_CAM_INDEX)
    parser.add_argument("--top-cam",  type=int, default=DEFAULT_TOP_CAM_INDEX)
    parser.add_argument("--side-cam", type=int, default=DEFAULT_SIDE_CAM_INDEX)
    parser.add_argument("--scan-max", type=int, default=CAM_SCAN_MAX_INDEX)
    parser.add_argument("--external-min", type=int, default=EXTERNAL_MIN_INDEX)
    args = parser.parse_args()

    if args.mode == "dual":
        top_idx, side_idx = choose_two_camera_indices(
            top_idx=args.top_cam, side_idx=args.side_cam,
            scan_max_index=int(max(1,args.scan_max)),
            prefer_external_min_index=args.external_min,
        )
        print(f"[INFO] Dual mode — top={top_idx}, side={side_idx}")
        run_tutorial_dual(top_idx, side_idx)
    else:
        print(f"[INFO] Single mode — camera={args.cam}")
        run_tutorial_single(args.cam)


if __name__ == "__main__":
    main()
