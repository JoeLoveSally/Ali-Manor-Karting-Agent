"""Accumulate per-frame road masks into a world-frame occupancy map."""
import cv2
import numpy as np

from .geometry import M
from .road import IGNORE, H as SH, W as SW

# 360x800 screen -> ground: scale screen by 2 to recording pixels first
_S2 = np.diag([2.0, 2.0, 1.0])


class WorldMap:
    def __init__(self, extent=(-4000, -4000, 60000, 60000), res=8.0):
        self.x0, self.y0, x1, y1 = extent
        self.res = res
        self.w, self.h = int((x1 - self.x0) / res), int((y1 - self.y0) / res)
        self.road = np.zeros((self.h, self.w), np.float32)
        self.seen = np.zeros((self.h, self.w), np.float32)

    def world_to_px(self, pts):
        p = np.asarray(pts, float)
        return np.c_[(p[:, 0] - self.x0) / self.res, (p[:, 1] - self.y0) / self.res]

    def add(self, mask_small, car_xy, radius=900, valid=None):
        """Splat a 360x800 road mask observed with the car at world position car_xy."""
        # screen(360) -> ground(car frame) -> world px
        T = np.array([[1 / self.res, 0, (car_xy[0] - self.x0) / self.res],
                      [0, 1 / self.res, (car_xy[1] - self.y0) / self.res],
                      [0, 0, 1]])
        A = T @ np.linalg.inv(M) @ _S2
        cx, cy = (car_xy[0] - self.x0) / self.res, (car_xy[1] - self.y0) / self.res
        r = int(radius / self.res)
        x0, y0 = max(0, int(cx - r)), max(0, int(cy - r))
        x1, y1 = min(self.w, int(cx + r)), min(self.h, int(cy + r))
        if x1 <= x0 or y1 <= y0:
            return
        Sh = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]])
        size = (x1 - x0, y1 - y0)
        ok = (~IGNORE if valid is None else valid).astype(np.uint8)
        ok[:165] = 0  # far field near the horizon is unreliable
        seen = cv2.warpPerspective(ok, Sh @ A, size, flags=cv2.INTER_NEAREST)
        road = cv2.warpPerspective((mask_small & ok.astype(bool)).astype(np.uint8), Sh @ A, size,
                                   flags=cv2.INTER_NEAREST)
        self.seen[y0:y1, x0:x1] += seen
        self.road[y0:y1, x0:x1] += road

    def prob(self):
        return np.where(self.seen > 0, self.road / np.maximum(self.seen, 1), np.nan)
