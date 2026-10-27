"""Lift the calibrated ground homography into a metric 2.5D camera.

Focal length follows from the orthogonal road vanishing directions. Height is
positive upward. All lengths use the existing ground-unit convention.
"""
import numpy as np

from .geometry import M, VP_A, VP_B


PRINCIPAL = np.array([360., 800.])
FOCAL = np.sqrt(-np.dot(VP_A[:2]-PRINCIPAL, VP_B[:2]-PRINCIPAL))
K = np.array([[FOCAL, 0., PRINCIPAL[0]], [0., FOCAL, PRINCIPAL[1]], [0., 0., 1.]])
_r = np.linalg.inv(K) @ M
_scales = np.linalg.norm(_r[:, :2], axis=0)
_normal = np.cross(_r[:, 0]/_scales[0], _r[:, 1]/_scales[1])
HEIGHT_COLUMN = K @ _normal * _scales.mean()
CAMERA = np.column_stack([M[:, :2], HEIGHT_COLUMN, M[:, 2]])


def project(points, camera=(0., 0., 0.)):
    points = np.asarray(points, float)-np.asarray(camera)
    h = np.c_[points, np.ones(len(points))] @ CAMERA.T
    return h[:, :2]/h[:, 2:]


def unproject_plane(pixels, camera=(0., 0., 0.), height=0.):
    cam = np.asarray(camera)
    plane = np.column_stack([M[:, :2], M[:, 2]-CAMERA[:, :3]@cam+HEIGHT_COLUMN*height])
    h = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(plane).T
    return h[:, :2]/h[:, 2:]
