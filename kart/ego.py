"""Ground-plane ego-motion from consecutive frames (camera follows the car).

Track road corners with pyramidal LK in screen space. The camera follows the car
in the ground plane and also in height (slopes, jumps), so screen motion of a
road point g is modelled as  p1 = P(g - d) + (0, dz):  d = car ground
displacement, dz = vertical screen shift from height change. Linearised at p0
and solved with iteratively reweighted least squares.
"""
import cv2
import numpy as np

from .geometry import MINV, to_ground, to_screen
from .road import hud_ignore, small

_MASK = np.zeros((1600, 720), np.uint8)
_MASK[330:1560, 60:715] = 255
_MASK[560:900, 300:600] = 0  # car and smoke
_MASK[1150:, 380:] = 0  # recording touch marker (absent on the live phone)


def _gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _jacobian(p):
    """d(screen)/d(ground) at screen points p (N,2), numerically."""
    g = to_ground(p)
    e = 1.0
    ju = (to_screen(g + [e, 0]) - to_screen(g - [e, 0])) / (2 * e)
    jv = (to_screen(g + [0, e]) - to_screen(g - [0, e])) / (2 * e)
    return np.stack([ju, jv], 2)  # (N, 2 screen, 2 ground)


def track(prev_frame, frame, road=None, max_pts=400):
    g0, g1 = _gray(prev_frame), _gray(frame)
    mask = _MASK & cv2.resize((~hud_ignore(small(prev_frame))).astype(np.uint8)*255,
                              (720, 1600), interpolation=cv2.INTER_NEAREST)
    if road is not None:
        r = cv2.resize(cv2.dilate(road.astype(np.uint8), np.ones((5, 5), np.uint8)), (720, 1600),
                       interpolation=cv2.INTER_NEAREST)
        mask = mask & (r * 255)
    p0 = cv2.goodFeaturesToTrack(g0, max_pts, 0.002, 8, mask=mask, blockSize=7)
    if p0 is None or len(p0) < 6:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(31, 31), maxLevel=4)
    p0b, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, winSize=(31, 31), maxLevel=4)
    ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(p0 - p0b, axis=2)[:, 0] < 1.0)
    # A translated straight edge is the same line in the image. Its raster stair
    # steps and straight tyre trails produce false zero-flow "corners". When
    # enough textured features establish motion, zero-flow points are uninformative.
    moving = np.linalg.norm(p1 - p0, axis=2)[:, 0] > 1.5
    if (ok & moving).sum() >= 8:
        ok &= moving
    if ok.sum() < 6:
        return None
    return p0[ok, 0], p1[ok, 0]


def displacement(prev_frame, frame, road=None):
    """Return (du, dv, dz, n_inliers, rms_px) of the car between two frames, or None."""
    t = track(prev_frame, frame, road)
    if t is None:
        return None
    p0, p1 = t
    J = _jacobian(p0)
    # p1 - p0 = -J d + (0, dz)
    A = np.zeros((2 * len(p0), 3))
    A[0::2, :2] = -J[:, 0, :]
    A[1::2, :2] = -J[:, 1, :]
    A[1::2, 2] = 1.0
    b = (p1 - p0).reshape(-1)
    w = np.ones(len(p0))
    x = np.zeros(3)
    for _ in range(6):
        W = np.repeat(w, 2)
        x = np.linalg.lstsq(A * W[:, None], b * W, rcond=None)[0]
        res = np.linalg.norm((A @ x - b).reshape(-1, 2), axis=1)
        w = 1.0 / np.maximum(1.0, res / 1.5)  # Huber-like
    inl = res < 3.0
    if inl.sum() < 6:
        return None
    return x[0], x[1], x[2], int(inl.sum()), float(np.sqrt(np.mean(res[inl] ** 2)))
