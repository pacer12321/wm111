"""Create a vertical contact sheet from selected video frame indices."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("frames", nargs="+", type=int)
    args = parser.parse_args()
    cap = cv2.VideoCapture(str(args.video))
    selected = []
    wanted = set(args.frames)
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index in wanted:
            selected.append((index, frame))
        index += 1
    cap.release()
    if len(selected) != len(wanted):
        raise ValueError(f"requested {sorted(wanted)}, found {[i for i, _ in selected]}")
    panels = []
    for index, frame in selected:
        frame = cv2.resize(frame, (1512, 288), interpolation=cv2.INTER_AREA)
        cv2.putText(frame, f"frame {index}", (10, 278), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        panels.append(frame)
    if not cv2.imwrite(str(args.output), np.concatenate(panels, axis=0)):
        raise RuntimeError("failed to write contact sheet")


if __name__ == "__main__":
    main()
