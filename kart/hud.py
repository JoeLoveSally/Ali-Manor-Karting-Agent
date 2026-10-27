"""HUD cues: the yellow "完美漂移" (perfect drift) banner."""
import numpy as np


def go_score(frame):
    """Green GO letters after countdown, in 720x1600 recording coordinates.

    The yellow countdown and perfect-drift banner do not meet this colour gate.
    Require the white race timer as well, so a green menu button cannot start driving.
    """
    r = frame[380:540, 240:480].astype(np.int16)
    b, g, red = r[..., 0], r[..., 1], r[..., 2]
    timer = frame[200:250, 200:520]
    white = timer.min(axis=2) > 220
    # Require digits across the clock, not just a white patch: the menu's
    # animated gift has white pixels in the centre and green pixels below it.
    if min(part.sum() for part in np.array_split(white, 4, axis=1)) < 150:
        return 0
    return int(((g > 160) & (g > b + 65) & (g > red + 35)).sum())


def perfect_drift_score(frame):
    """Count of banner-yellow pixels in the banner area (720x1600 recording pixels)."""
    r = frame[320:450, 150:570]
    b, g, rr = r[..., 0].astype(int), r[..., 1].astype(int), r[..., 2].astype(int)
    return int(((rr > 200) & (g > 170) & (b < 120)).sum())
