"""Road mask by difference from the camera-fixed sky background.

Works on 360x800 frames (half of the 720x1600 recordings; the phone's 1440x3200
screen has the same aspect). The sky is identical across levels; anything that
differs strongly from it below the HUD is road (top surface plus side walls).
"""
from pathlib import Path

import cv2
import numpy as np

W, H = 360, 800
_SKY = cv2.imread(str(Path(__file__).resolve().parent.parent / "artifacts" / "sky.png")).astype(np.int16)

IGNORE = np.zeros((H, W), bool)
IGNORE[:150] = True              # status bar, timer
IGNORE[:, :25] = True            # progress bar
IGNORE[280:450, 150:300] = True  # car, close smoke


def small(frame):
    return cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)


def hud_ignore(frame_small):
    """Mask only the coloured GO/banner pixels, preserving road around the text."""
    mask = np.zeros((H, W), np.uint8)
    patch = frame_small[160:275, 75:285].astype(np.int16)
    b, g, r = patch[..., 0], patch[..., 1], patch[..., 2]
    yellow = (r > 180) & (g > 130) & (b < 130)
    green = (g > 100) & (g > b+50) & (g > r+20)
    letters = (yellow | green).astype(np.uint8)
    if letters.sum() > 200:
        mask[160:275, 75:285] = letters
        mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    return mask.astype(bool)


def sky_colour(frame_small):
    # the pale wave arc in the sky moves with the scene, so the fixed sky image misses it;
    # its colour band (HSV, measured on all recordings) is never road
    h = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
    return ((h[..., 0] >= 90) & (h[..., 0] <= 100) & (h[..., 1] >= 110) & (h[..., 1] <= 160) &
            (h[..., 2] >= 205) & (h[..., 2] <= 225))


def road_mask(frame_small, thresh=60):
    d = np.abs(frame_small.astype(np.int16) - _SKY).sum(2)
    return (d > thresh) & ~sky_colour(frame_small)


def road_surface(frame_small):
    """Offline top-surface evidence; dark walls are excluded, tyres remain unknown.

    This does not change the deployed perception pipeline. A local palette is
    estimated only from saturated road pixels outside the car/HUD exclusions.
    """
    hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
    road = road_mask(frame_small)
    samples = road & ~IGNORE & (hsv[..., 1] > 180)
    if samples.sum() < 100:
        return np.zeros_like(road), np.zeros_like(road)
    hist = np.bincount(hsv[..., 2][samples], minlength=256)
    hist = np.convolve(hist, np.ones(7), mode='same')
    peak = int(np.argmax(hist[50:220])+50)
    top = road & (hsv[..., 1] > 180) & (hsv[..., 2] >= .9*peak)
    dark = (hsv[..., 2] < 70) & road
    valid = ~IGNORE & ~hud_ignore(frame_small) & ~dark
    return top, valid
