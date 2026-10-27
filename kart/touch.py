"""Detect the system screen-recording touch marker (user tapped bottom-right on purpose)."""
import cv2
import numpy as np

X0, Y0 = 380, 1150  # in 720x1600 recording pixels


def detect_touch(frame):
    """Return (x, y) of the touch marker in recording pixels, or None."""
    r = frame[Y0:, X0:]
    m = ((r.min(2) > 140) & (np.ptp(r, axis=2) < 75)).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    best = None
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if 250 < a < 900 and 18 < w < 36 and 18 < h < 36 and a > 0.6 * w * h:
            best = (cen[k][0] + X0, cen[k][1] + Y0)
    return best


def clean(pressed, max_gap=2):
    """Fill dropouts of up to max_gap frames inside a press."""
    p = np.asarray(pressed, bool).copy()
    i = 0
    while i < len(p):
        if not p[i] and i > 0 and p[i - 1]:
            j = i
            while j < len(p) and not p[j]:
                j += 1
            if j < len(p) and j - i <= max_gap:
                p[i:j] = True
            i = j
        else:
            i += 1
    return p
