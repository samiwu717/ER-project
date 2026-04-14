"""This file makes an A4 paper with ArUco markers for tracking."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

A4_W_MM = 210.0
A4_H_MM = 297.0
DICT_NAME = cv2.aruco.DICT_5X5_50
CORNER_IDS: Dict[str, int] = {
    "tl": 0,
    "tr": 1,
    "br": 2,
    "bl": 3,
}


#changes millimeters to pixels
def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm * dpi / 25.4))


#saves the image with DPI info
def save_png_with_dpi(gray: np.ndarray, out_path: Path, dpi: int) -> None:
    try:
        from PIL import Image

        Image.fromarray(gray).save(out_path, dpi=(dpi, dpi))
        return
    except Exception:
        pass

    cv2.imwrite(str(out_path), gray)


#makes the A4 marker template
def generate_a4_template(
    out_path: Path,
    dpi: int,
    marker_size_mm: float,
    margin_mm: float,
) -> Tuple[int, int]:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV aruco module is required. Please install opencv-contrib-python.")

    page_w = mm_to_px(A4_W_MM, dpi)
    page_h = mm_to_px(A4_H_MM, dpi)
    marker_size = mm_to_px(marker_size_mm, dpi)
    margin = mm_to_px(margin_mm, dpi)
    if marker_size <= 16:
        raise ValueError("Marker size is too small. Please increase --marker-size-mm.")
    if margin < 0:
        raise ValueError("Margin must be >= 0.")
    if 2 * (margin + marker_size) >= min(page_w, page_h):
        raise ValueError("Marker size + margin is too large for A4.")

    page = np.full((page_h, page_w), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(DICT_NAME)

    placements = {
        "tl": (margin, margin),
        "tr": (page_w - margin - marker_size, margin),
        "br": (page_w - margin - marker_size, page_h - margin - marker_size),
        "bl": (margin, page_h - margin - marker_size),
    }

    for name, marker_id in CORNER_IDS.items():
        x0, y0 = placements[name]
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
        page[y0:y0 + marker_size, x0:x0 + marker_size] = marker

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_png_with_dpi(page, out_path, dpi)
    return page_w, page_h


#reads the command line options
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a printable A4 ArUco template (DICT_5X5_50) with 4 corner markers."
    )
    parser.add_argument("--output", type=Path, default=Path("aruco_a4_5x5_50.png"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--marker-size-mm", type=float, default=32.0)
    parser.add_argument("--margin-mm", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    page_w, page_h = generate_a4_template(
        out_path=args.output,
        dpi=args.dpi,
        marker_size_mm=args.marker_size_mm,
        margin_mm=args.margin_mm,
    )
    print(f"Saved: {args.output}")
    print(f"A4 pixels: {page_w} x {page_h} @ {args.dpi} DPI")
    print("Corner marker IDs: tl=0, tr=1, br=2, bl=3 (DICT_5X5_50)")
    print("Print with 100% scale / Actual size (do not fit to page).")


if __name__ == "__main__":
    main()
