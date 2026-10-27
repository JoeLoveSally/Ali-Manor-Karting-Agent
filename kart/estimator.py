"""Online state estimator: frames -> the same (grid, state) observation as kart/vecenv.py.

Per frame (720x1600 BGR, host time, the input we are currently sending):
  * car ground displacement from road feature tracking (kart/ego.py); when that
    fails, the last velocity is extrapolated
  * heading / speed from the displacement over the last few frames
  * a world-frame road map updated from the current road mask; cells hidden by
    the car and its smoke keep what was seen before; no update while airborne
    (large vertical screen shift)
  * boost from the "完美漂移" banner, effective input from the commanded input
    delayed by the fitted latencies
"""
from collections import deque

import cv2
import numpy as np
import torch

from .ego import displacement
from .geometry import M
from .hud import perfect_drift_score
from .road import IGNORE, road_mask, small, hud_ignore
from .vecenv import CELL, G, visibility_mask

MAP_RES = 16.0
MAP_SIZE = 640          # cells per side of the rolling map window (10240 units)
_S2 = np.diag([2.0, 2.0, 1.0])  # 360x800 mask -> recording pixels


class Estimator:
    def __init__(self, params, perfect_thresh=2000):
        self.p = params
        self.perfect_thresh = perfect_thresh
        self.vis = visibility_mask().numpy()
        c = (np.arange(G) - G / 2 + 0.5) * CELL
        V, U = np.meshgrid(c, c, indexing="ij")
        self.grid_off = np.stack([U, V], -1)  # (G, G, 2) ground offsets of the ego cells
        self.reset()

    def reset(self):
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self.hist = deque(maxlen=6)     # (t, pos)
        self.prev = self.prev_road = None
        self.t = None
        self.map = np.full((MAP_SIZE, MAP_SIZE), 0.5, np.float32)   # P(road), 0.5 unknown
        self.map_org = np.array([-MAP_SIZE * MAP_RES / 2, -MAP_SIZE * MAP_RES / 2])
        self.cmd = False
        self.cmd_hist = deque(maxlen=64)  # (t, pressed)
        self.since = 1.0
        self.boost_until = -1.0
        self.banner_on = False
        self.airborne = False
        self.ok_frames = 0
        self.last_d = None

    # ---- map --------------------------------------------------------------
    def _recentre(self):
        """Keep the car inside the middle of the rolling map window."""
        c = ((self.pos - self.map_org) / MAP_RES).astype(int)
        shift = np.zeros(2, int)
        for k in range(2):
            if c[k] < MAP_SIZE * 0.3 or c[k] > MAP_SIZE * 0.7:
                shift[k] = c[k] - MAP_SIZE // 2
        if shift.any():
            new = np.full_like(self.map, 0.5)
            sx, sy = shift
            src = self.map[max(sy, 0):MAP_SIZE + min(sy, 0), max(sx, 0):MAP_SIZE + min(sx, 0)]
            new[max(-sy, 0):max(-sy, 0) + src.shape[0], max(-sx, 0):max(-sx, 0) + src.shape[1]] = src
            self.map = new
            self.map_org = self.map_org + shift * MAP_RES

    def _update_map(self, mask_small, excluded=None, alpha=0.5):
        # screen (360x800) -> ground relative to car -> world -> map cell
        T = np.array([[1 / MAP_RES, 0, (self.pos[0] - self.map_org[0]) / MAP_RES],
                      [0, 1 / MAP_RES, (self.pos[1] - self.map_org[1]) / MAP_RES],
                      [0, 0, 1.0]])
        A = T @ np.linalg.inv(M) @ _S2
        ok = (~IGNORE).astype(np.uint8)
        if excluded is not None:
            ok[excluded] = 0
        ok[:165] = 0  # far field near the horizon
        seen = cv2.warpPerspective(ok, A, (MAP_SIZE, MAP_SIZE), flags=cv2.INTER_NEAREST).astype(bool)
        road = cv2.warpPerspective(mask_small.astype(np.uint8), A, (MAP_SIZE, MAP_SIZE),
                                   flags=cv2.INTER_NEAREST).astype(np.float32)
        self.map[seen] = (1 - alpha) * self.map[seen] + alpha * road[seen]

    # ---- per frame ---------------------------------------------------------
    def update(self, frame, t, commanded):
        """frame: 720x1600 BGR; t: seconds; commanded: input currently being sent."""
        if commanded != self.cmd:
            self.since = 0.0
            self.cmd = commanded
        self.cmd_hist.append((t, commanded))
        dt = 0.0 if self.t is None else t - self.t
        if self.t is not None:
            self.since += dt
        frame_small = small(frame)
        road = road_mask(frame_small)
        d = displacement(self.prev, frame, self.prev_road) if self.prev is not None else None
        self.last_d = d
        if d is not None:
            du, dv, dz, _, _ = d
            # dz is a per-frame displacement: threshold must scale with frame interval.
            # Subpixel fitting noise on a flat road must not freeze the occupancy map.
            self.airborne = abs(dz) > max(2.5, 70.0 * dt)
            step = np.array([du, dv])
            self.ok_frames += 1
        else:
            step = self.vel * dt
        self.pos = self.pos + step
        self.hist.append((t, self.pos.copy()))
        if len(self.hist) >= 3 and self.hist[-1][0] > self.hist[0][0]:
            self.vel = (self.hist[-1][1] - self.hist[0][1]) / (self.hist[-1][0] - self.hist[0][0])
        self._recentre()
        if not self.airborne:
            self._update_map(road, hud_ignore(frame_small))
        on = perfect_drift_score(frame) > self.perfect_thresh
        if on and not self.banner_on:
            self.boost_until = t + self.p.boost_time
        self.banner_on = on
        self.prev, self.prev_road, self.t = frame, road, t

    def effective_input(self):
        """Input the game is applying now, from the commanded history and latencies."""
        eff = False
        for tc, pr in self.cmd_hist:
            lat = self.p.loop_delay + (self.p.latency_press if pr else self.p.latency_release)
            if tc + lat <= self.t:
                eff = pr
        return eff

    def heading(self):
        th = np.arctan2(self.vel[0], self.vel[1])
        return float(np.clip(th, 0.0, np.pi / 2))

    def speed(self):
        return float(np.linalg.norm(self.vel))

    def grid(self):
        """(G, G) occupancy around the car: 1 road, 0 void, 0.5 unknown."""
        q = (self.pos + self.grid_off - self.map_org) / MAP_RES
        c, r = q[..., 0].astype(int), q[..., 1].astype(int)
        inside = (r >= 0) & (r < MAP_SIZE) & (c >= 0) & (c < MAP_SIZE)
        g = np.full((G, G), 0.5, np.float32)
        g[inside] = self.map[r[inside], c[inside]]
        g = np.where(g > 0.6, 1.0, np.where(g < 0.4, 0.0, 0.5)).astype(np.float32)
        g[~self.vis] = 0.5
        return g

    def observe(self):
        th = self.heading()
        boost_left = max(0.0, self.boost_until - self.t) if self.t is not None else 0.0
        speed = self.speed() if self.ok_frames > 3 else self.p.speed
        state = np.array([np.sin(th), np.cos(th), float(self.cmd), float(self.effective_input()),
                          speed / 1000.0, boost_left / 0.5, min(self.since, 2.0), th / (np.pi / 2)],
                         np.float32)
        return torch.tensor(self.grid())[None, None], torch.tensor(state)[None]
