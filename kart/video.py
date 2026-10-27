"""Read recordings as unique frames with timestamps (recordings duplicate ~20% of frames)."""
import subprocess

import cv2
import numpy as np


def frame_times(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "frame=pts_time", "-of", "csv=p=0", str(path)], text=True)
    return np.array([float(x.split(",")[0]) for x in out.split() if x.strip()])


def unique_frames(path, dup_thresh=0.5):
    """Yield (index, pts_seconds, frame_bgr) skipping frames identical to the previous one."""
    times = frame_times(path)
    cap = cv2.VideoCapture(str(path))
    prev = None
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        small = cv2.resize(f, (90, 200)).astype(np.int16)
        if prev is None or np.abs(small - prev).mean() >= dup_thresh:
            yield i, times[i] if i < len(times) else times[-1], f
        prev = small
        i += 1
