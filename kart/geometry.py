"""Screen <-> ground-plane mapping.

Road edges form two line families whose vanishing points (fitted on all
recordings, 720x1600 pixels) share a horizontal horizon. Ground coordinates
(u, v), origin at the car anchor: u points towards VP_A and is the direction the
car moves while the screen is pressed (B); v points towards VP_B and is the
released direction (A). (VP_A / VP_B are named after the line families, not the
A / B inputs.)

U_SCALE: with equal per-pixel scales at the car, straight-line speed along u
was 14% lower than along v and road widths across u 16% narrower than across v
(tools: speed / width comparison over all recordings). The game is symmetric,
so u units are rescaled by 1.15 to make both axes the same length.
"""
import cv2
import numpy as np

VP_A = np.array([1783.4, -585.6, 1.0])
VP_B = np.array([-2335.2, -584.1, 1.0])
CAR = np.array([420.0, 760.0, 1.0])  # approximate car ground anchor, recording pixels


def ground_to_screen_matrix(scale_a=1.0, scale_b=1.0, car=CAR):
    """3x3 M with screen ~ M @ (u, v, 1). Scales set ground units per screen pixel at the car."""
    def unit(vp):  # screen displacement per unit along vp direction at the car
        d = vp[:2] - car[:2]
        return d / np.linalg.norm(d)
    # Columns: vanishing points (directions at infinity) and the origin.
    # Scale columns so that d(screen)/du at the car has length 1/scale_a pixels.
    M = np.column_stack([VP_A, VP_B, car])
    for k, (vp, s) in enumerate([(VP_A, scale_a), (VP_B, scale_b)]):
        # derivative of projection at u=0: (vp_xy - car_xy) * c / 1  (car w=1, vp w=1)
        c = 1.0 / (np.linalg.norm(vp[:2] - car[:2]) * s)
        M[:, k] = vp * c
    return M


U_SCALE = 1.15
M = ground_to_screen_matrix(scale_a=U_SCALE)
MINV = np.linalg.inv(M)


def to_ground(pts, Minv=MINV):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ Minv.T
    return p[:, :2] / p[:, 2:]


def to_screen(pts, M_=M):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ M_.T
    return p[:, :2] / p[:, 2:]


def birdseye_matrix(size=400, res=2.0, M_=M, center=(0.0, 0.0)):
    """Screen->output matrix for a size x size top-down view, res ground units per pixel.

    Output column grows with u (A), row grows with v (B); `center` is the ground point
    placed at the image centre (default: the car).
    """
    T = np.array([[res, 0, center[0] - res * size / 2],
                  [0, res, center[1] - res * size / 2],
                  [0, 0, 1.0]])
    return np.linalg.inv(M_ @ T)


def birdseye(frame, size=400, res=2.0, M_=M, center=(0.0, 0.0), border=(0, 0, 0)):
    return cv2.warpPerspective(frame, birdseye_matrix(size, res, M_, center), (size, size),
                               borderValue=border)
