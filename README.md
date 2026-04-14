# ER Project

This repository contains a camera based paper piano tutorial system project.  
It detects fingertips on a printed keyboard (using ArUco markers), maps them to keys, and plays notes in real time.


- Main script: `paper_piano_tutorial_system_final.py`
  - Final integrated version with:
    - Dual-camera mode (top + bottom)
    - Single-camera fallback mode
    - Tutorial mode (Twinkle Twinkle Little Star) and free-play toggle

## What Each Code File Does

### Core System Files

- `paper_piano_tutorial_system_final.py`
  - Final integrated version of the project.
  - Includes tutorial state machine (target note, progress, error counting, guided highlighting).

- `paper_piano_system.py`
  - Core integrated paper piano engine (non-tutorial baseline).

- `tutorial_system.py`
  - Early incompleted verstion, useful for understanding tutorial logic as a separate extension layer.

- `prediction.py`
  - Shared hand tracking and geometry utilities.

### Other Iteration Files

- `paper_piano_v1.py`
  - First version: simple region-entry trigger (finger enters key region -> play note).

- `paper_piano_v2.py`
  - Second version: improved trigger logic (tap / hold-then-press) to reduce hover false positives.

- `paper_piano_final_working.py`
  - Earlier stable working baseline, especially for USB-camera-based setup.

- `paper_piano_multifinger.py`
  - Multi-finger state-machine version with stronger smoothing and optional 3D (z-axis) gating.

- `paper_piano_contact_inference_final.py`
  - Contact-inference experiment:
    - Uses combined features (depth, approach velocity) to infer true key contact.

- `paper_piano_pressure.py`
  - Minimal single-camera pressure threshold prototype (yellow line threshold test).

### Utility Scripts

- `generate_aruco_a4.py`
  - Generates a printable A4 ArUco marker sheet (`DICT_5X5_50`, corner IDs 0/1/2/3).

- `check_cam.py`
  - Quick camera-open check script.

## Resource Files

- `hand_landmarker.task`
  - MediaPipe Hand Landmarker model asset used by `prediction.py`.

- `data/marker.obj`, `data/crate.obj`, `data/crate.png`
  - 3D/texture assets kept in the repository.

