"""
tutorial.py
-----------


按键：
  T  — 切换 Tutorial 模式 / 自由弹奏
  R  — 重新开始当前歌曲，清零错误
  Q / Esc — 退出
"""

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# 从原文件导入所有常量和类
from paper_piano_final_working import (
    PaperPiano,
    NOTE_NAMES,
    NUM_WHITE_KEYS,
    KEYBOARD_X0, KEYBOARD_X1, KEYBOARD_Y0, KEYBOARD_Y1,
    KEYBOARD_STACK_VERTICAL,
    FINGERTIP_IDS,
    KEY_COOLDOWN_SEC,
    ACTIVE_FLASH_SEC,
    MIRROR_DISPLAY,
    build_black_key_rects,
)
from prediction import predict

# ──────────────────────────────────────────
# Tutorial 配置
# ──────────────────────────────────────────
TUTORIAL_MODE_DEFAULT = True

TUTORIAL_SONG_NAME = "Twinkle Twinkle Little Star"
# C4=0, D4=1, E4=2, F4=3, G4=4, A4=5, B4=6, C5=7
TUTORIAL_SEQUENCE = [
    0, 0, 4, 4, 5, 5, 4,   # Twin-kle twin-kle lit-tle star
    3, 3, 2, 2, 1, 1, 0,   # How I won-der what you are
    4, 4, 3, 3, 2, 2, 1,   # Up a-bove the world so high
    4, 4, 3, 3, 2, 2, 1,   # Like a dia-mond in the sky
    0, 0, 4, 4, 5, 5, 4,   # Twin-kle twin-kle lit-tle star
    3, 3, 2, 2, 1, 1, 0,   # How I won-der what you are
]

# 颜色
COLOR_TARGET  = (255, 200,  0)   # 目标键：黄色高亮
COLOR_CORRECT = ( 40, 220, 40)   # 按对：绿色
COLOR_WRONG   = (  0,  0, 220)   # 按错：红色
FLASH_SEC     = 0.35             # 闪烁持续时间


class TutorialPiano(PaperPiano):

    def __init__(self) -> None:
        super().__init__()
        self._init_tutorial()

    def _init_tutorial(self) -> None:
        self.tutorial_mode      = TUTORIAL_MODE_DEFAULT
        self.tutorial_step      = 0
        self.tutorial_errors    = 0
        self.tutorial_completed = False
        self.wrong_flash_until  = [0.0] * NUM_WHITE_KEYS

    # ── 工具：当前目标键索引 ───────────────────
    def _target_key(self) -> Optional[int]:
        if self.tutorial_mode and not self.tutorial_completed \
                and self.tutorial_step < len(TUTORIAL_SEQUENCE):
            return TUTORIAL_SEQUENCE[self.tutorial_step]
        return None

    # ── 覆盖：键盘绘制（高亮目标/正确/错误） ────
    def draw_paper_overlay(self, frame, H_inv, markers, now):
        # 先调用父类画纸张轮廓 + ArUco 标记点
        if markers is not None:
            for name, pt in markers.items():
                cv2.circle(frame, tuple(np.int32(pt)), 8, (0, 255, 255), -1)
                cv2.putText(frame, name,
                            tuple(np.int32(pt + np.array([6, -6], np.float32))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if H_inv is None:
            return

        # 纸张边框
        paper_quad = np.array([[0, 0], [800, 0], [800, 1100], [0, 1100]], np.float32)
        cv2.polylines(frame, [np.int32(self.paper_to_image(H_inv, paper_quad))],
                      True, (255, 255, 0), 2)

        target = self._target_key()
        key_rects = self.build_key_rects()

        for idx, (x0, y0, x1, y1) in enumerate(key_rects):
            poly = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], np.float32)
            poly_img = self.paper_to_image(H_inv, poly)

            if now < self.active_until[idx]:
                # 按对了 / 普通弹奏 → 绿色
                color, thick = COLOR_CORRECT, 4
                _fill(frame, poly_img, color, 0.35)
            elif self.tutorial_mode and now < self.wrong_flash_until[idx]:
                # 按错了 → 红色
                color, thick = COLOR_WRONG, 4
                _fill(frame, poly_img, color, 0.40)
            elif self.tutorial_mode and idx == target:
                # 目标键 → 黄色
                color, thick = COLOR_TARGET, 4
                _fill(frame, poly_img, color, 0.28)
            else:
                color, thick = (255, 255, 255), 2

            cv2.polylines(frame, [np.int32(poly_img)], True, color, thick)

        # 黑键
        for x0, y0, x1, y1 in build_black_key_rects():
            poly = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], np.float32)
            poly_img = self.paper_to_image(H_inv, poly)
            cv2.fillConvexPoly(frame, np.int32(poly_img), (25, 25, 25))
            cv2.polylines(frame, [np.int32(poly_img)], True, (255, 255, 255), 1)

    # ── 覆盖：键标签颜色跟随 tutorial 状态 ────
    def draw_key_labels(self, frame, H_inv, now, mirrored_display):
        if H_inv is None:
            return
        h, w = frame.shape[:2]
        target = self._target_key()
        key_rects = self.build_key_rects()

        for idx, (x0, y0, x1, y1) in enumerate(key_rects):
            if KEYBOARD_STACK_VERTICAL:
                tx, ty = 0.5*(x0+x1), 0.5*(y0+y1)
            else:
                tx, ty = 0.5*(x0+x1), y1 - 40

            pt = self.paper_to_image(H_inv, np.array([[tx, ty]], np.float32))[0]
            xi, yi = float(pt[0]), float(pt[1])
            if mirrored_display:
                xi = (w - 1) - xi

            if now < self.active_until[idx]:
                color = COLOR_CORRECT
            elif self.tutorial_mode and now < self.wrong_flash_until[idx]:
                color = COLOR_WRONG
            elif self.tutorial_mode and idx == target:
                color = COLOR_TARGET
            else:
                color = (255, 255, 255)

            label = NOTE_NAMES[idx]
            font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2
            (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
            cv2.putText(frame, label, (int(xi - tw/2), int(yi + th/2)), font, scale, color, thick)

    # ── 覆盖：手部处理（加入 tutorial 逻辑） ────
    def process_hands(self, frame, H, now):
        current_ids = set()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection = predict(frame_rgb)
        hands = getattr(detection, "hand_landmarks", None) if detection else None

        if not hands:
            for fid in list(self.prev_key_by_finger):
                if fid not in current_ids:
                    self.prev_key_by_finger.pop(fid, None)
            return

        h, w = frame.shape[:2]
        target = self._target_key()

        for hand_idx, hand_landmarks in enumerate(hands):
            for tip_id in FINGERTIP_IDS:
                finger_id = (hand_idx, tip_id)
                current_ids.add(finger_id)

                lm = hand_landmarks[tip_id]
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (px, py), 6, (0, 128, 255), -1)

                curr_key = None
                if H is not None:
                    paper_pt = self.image_to_paper(H, (px, py))
                    if paper_pt is not None:
                        curr_key = self.locate_key(paper_pt)
                        if curr_key is not None:
                            cv2.putText(frame, NOTE_NAMES[curr_key], (px+8, py-8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)

                prev_key = self.prev_key_by_finger.get(finger_id)

                if curr_key is not None and prev_key != curr_key:
                    if now - self.last_hit_time[curr_key] > KEY_COOLDOWN_SEC:
                        self.last_hit_time[curr_key] = now

                        if self.tutorial_mode and not self.tutorial_completed:
                            if curr_key == target:
                                # ✓ 按对
                                self.active_until[curr_key] = now + FLASH_SEC
                                self.synth.play(curr_key)
                                self.tutorial_step += 1
                                if self.tutorial_step >= len(TUTORIAL_SEQUENCE):
                                    self.tutorial_completed = True
                                    print(f"[Tutorial] 🎉 Complete! Total Error: {self.tutorial_errors}")
                                else:
                                    target = self._target_key()
                                print(f"[Tutorial] ✓ {NOTE_NAMES[curr_key]}  "
                                      f"{self.tutorial_step}/{len(TUTORIAL_SEQUENCE)}  "
                                      f"errors={self.tutorial_errors}")
                            else:
                                # ✗ 按错
                                self.wrong_flash_until[curr_key] = now + FLASH_SEC
                                self.tutorial_errors += 1
                                self.synth.play(curr_key)
                                print(f"[Tutorial] ✗ {NOTE_NAMES[curr_key]} "
                                      f"(press {NOTE_NAMES[target]})  "
                                      f"errors={self.tutorial_errors}")
                        else:
                            # 自由弹奏
                            self.active_until[curr_key] = now + ACTIVE_FLASH_SEC
                            self.synth.play(curr_key)

                self.prev_key_by_finger[finger_id] = curr_key

        for fid in list(self.prev_key_by_finger):
            if fid not in current_ids:
                self.prev_key_by_finger.pop(fid, None)

    # ── 新增：Tutorial HUD 面板 ──────────────
    def draw_tutorial_hud(self, frame) -> None:
        if not self.tutorial_mode:
            return

        h, w = frame.shape[:2]
        hx, hy, hw, hh = w - 422, 14, 408, 162

        overlay = frame.copy()
        cv2.rectangle(overlay, (hx-6, hy), (hx+hw, hy+hh), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)
        cv2.rectangle(frame, (hx-6, hy), (hx+hw, hy+hh), (100, 100, 100), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.tutorial_completed:
            cv2.putText(frame, "Song Complete!", (hx, hy+38), font, 1.0, COLOR_CORRECT, 2)
            cv2.putText(frame, f"Errors: {self.tutorial_errors}", (hx, hy+72), font, 0.75, (220,220,220), 2)
            cv2.putText(frame, "R = restart   T = free-play", (hx, hy+110), font, 0.62, (160,160,160), 1)
            return

        total, step = len(TUTORIAL_SEQUENCE), self.tutorial_step
        cv2.putText(frame, TUTORIAL_SONG_NAME, (hx, hy+28), font, 0.62, (220,200,80), 2)

        if step < total:
            tgt_name = NOTE_NAMES[TUTORIAL_SEQUENCE[step]]
            cv2.putText(frame, "Press:", (hx, hy+62), font, 0.65, (200,200,200), 1)
            cv2.putText(frame, tgt_name, (hx+82, hy+62), font, 1.0, COLOR_TARGET, 3)

            upcoming = [NOTE_NAMES[TUTORIAL_SEQUENCE[s]]
                        for s in range(step+1, min(step+5, total))]
            cv2.putText(frame, "  ->  ".join(upcoming), (hx, hy+92),
                        font, 0.55, (130,130,130), 1)

        # 进度条
        bx, by, bw, bth = hx, hy+108, hw-12, 14
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bth), (60,60,60), -1)
        cv2.rectangle(frame, (bx, by),
                      (bx + int(bw*step/max(total,1)), by+bth), (40,190,40), -1)
        cv2.putText(frame, f"{step}/{total}", (bx+bw+4, by+12), font, 0.5, (170,170,170), 1)

        err_col = (60,60,220) if self.tutorial_errors > 0 else (130,130,130)
        cv2.putText(frame, f"Errors: {self.tutorial_errors}", (hx, hy+148), font, 0.65, err_col, 2)
        cv2.putText(frame, "T=free  R=restart", (hx+220, hy+148), font, 0.50, (90,90,90), 1)

    # ── 覆盖：主循环（加 T/R 快捷键 + HUD） ──
    def run(self) -> None:
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    print("[ERROR] Camera read failed.")
                    break

                now = time.time()
                dt = max(now - self.last_fps_time, 1e-6)
                self.fps = 1.0 / dt
                self.last_fps_time = now

                H, H_inv, markers = self.update_homography(frame, now)
                self.draw_paper_overlay(frame, H_inv, markers, now)
                self.process_hands(frame, H, now)

                display = cv2.flip(frame, 1) if MIRROR_DISPLAY else frame
                self.draw_key_labels(display, H_inv, now, mirrored_display=MIRROR_DISPLAY)
                self.draw_tutorial_hud(display)
                self.draw_status(display, H is not None)

                cv2.imshow("Paper Piano – Tutorial", display)
                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord('q')):
                    break
                elif key in (ord('t'), ord('T')):
                    self.tutorial_mode = not self.tutorial_mode
                    print(f"[INFO] Tutorial {'ON' if self.tutorial_mode else 'OFF (free-play)'}")
                elif key in (ord('r'), ord('R')):
                    self._init_tutorial()
                    self.active_until = [0.0] * NUM_WHITE_KEYS
                    print("[INFO] Tutorial restarted")
        finally:
            self.close()


# ── 工具函数：半透明填充 ─────────────────────
def _fill(frame, poly_img, color, alpha):
    overlay = frame.copy()
    cv2.fillConvexPoly(overlay, np.int32(poly_img), color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


if __name__ == "__main__":
    print("[INFO] Starting Tutorial Piano...")
    TutorialPiano().run()
