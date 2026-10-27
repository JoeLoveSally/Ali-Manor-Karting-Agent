"""Raster geometry adapter to VecKart; no separate replay dynamics.

Unknown contact is unscorable, not a fall or success. Explicit ramps and jumps
use the shared height/flight kernel. Camera observations are validated separately.
"""
import numpy as np
import torch

from .sim import Params
from .vecenv import VecKart, MAX_JUMPS, MAX_RELIEFS, G


class RasterKart(VecKart):
    def __init__(self, track, n=1, device='cpu', params=Params(), start=(0., 0.),
                 theta=0., reference=None, finish=None, startup_elapsed=None, **kwargs):
        self.track = track
        # Map sections normally start mid-race. Full-race scenes opt in with 0.
        if startup_elapsed is not None and (not np.isfinite(startup_elapsed) or startup_elapsed < 0):
            raise ValueError('Invalid startup elapsed time')
        self.startup_elapsed = startup_elapsed
        self.start, self.start_theta = np.asarray(start, float), theta
        self.reference, self.finish = reference, finish
        super().__init__(n, device=device, params=params, n_segments=2,
                         **({'randomize': 0., 'loop_delay': (0., 0.)} | kwargs))
        self.road_grid = torch.as_tensor(track.road, device=self.dev)
        self.known_grid = torch.as_tensor(track.known, device=self.dev)
        if reference is not None:
            self.ref = torch.as_tensor(reference, dtype=torch.float32, device=self.dev)
            self.ref_dist = torch.cat([torch.zeros(1, device=self.dev),
                                      (self.ref[1:] - self.ref[:-1]).norm(dim=1).cumsum(0)])
            self.prog = self._progress()

    def reset_envs(self, idx):
        super().reset_envs(idx)
        if max(len(self.track.jumps), len(self.track.ramps)) > MAX_JUMPS:
            raise ValueError('Too many explicit jump or ramp regions')
        self.jumps[idx] = -1.
        self.ramps[idx] = -1.
        self.reliefs[idx] = -1.
        if len(self.track.reliefs) > MAX_RELIEFS:
            raise ValueError('Too many full-width relief regions')
        if len(self.track.reliefs):
            self.reliefs[idx, :len(self.track.reliefs)] = torch.as_tensor(
                self.track.reliefs, dtype=torch.float32, device=self.dev)
        if len(self.track.jumps):
            self.jumps[idx, :len(self.track.jumps)] = torch.as_tensor(
                self.track.jumps, dtype=torch.float32, device=self.dev)
        if len(self.track.ramps):
            self.ramps[idx, :len(self.track.ramps)] = torch.as_tensor(
                self.track.ramps, dtype=torch.float32, device=self.dev)
        self.x[idx], self.y[idx], self.th[idx] = *self.start, self.start_theta
        self.eff[idx] = self.pressed[idx] = float(self.start_theta > np.pi/4)
        self.hist[idx] = self.pressed[idx, None]
        if self.startup_elapsed is None:
            self.startup_age[idx] = self.base.startup_duration
        else:
            self.startup_age[idx] += self.startup_elapsed
        fraction = 1. if self.base.startup_duration <= 0 else (self.startup_age[idx]/self.base.startup_duration).clamp(0, 1)
        self.speed_now[idx] = self.P['speed'][idx]*fraction
        self.velocity_history[idx, :, 0] = self.speed_now[idx, None]*np.sin(self.start_theta)
        self.velocity_history[idx, :, 1] = self.speed_now[idx, None]*np.cos(self.start_theta)
        self.map_last_xy[idx] = torch.as_tensor(self.start, dtype=torch.float32, device=self.dev)
        self.distract[idx] = 1e9
        if hasattr(self, 'ref'):
            self.prog[idx] = self._progress()[idx]

    def sample(self, px, py):
        c = ((px-self.track.x0)/self.track.res).floor().long()
        r = ((py-self.track.y0)/self.track.res).floor().long()
        h, w = self.road_grid.shape
        inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        rr, cc = r.clamp(0, h-1), c.clamp(0, w-1)
        known = torch.where(inside, self.known_grid[rr, cc], self.track.outside_known)
        road = inside & known & self.road_grid[rr, cc]
        return road, known

    def on_road(self, px, py, env=None):
        return self.sample(px, py)[0]

    def observation_known(self, px, py):
        return self.sample(px, py)[1]

    def ray(self, dx, dy, max_dist=800., step=4.):
        s = torch.arange(0, max_dist+step, step, device=self.dev)
        road, known = self.sample(self.x[:, None]+dx[:, None]*s,
                                  self.y[:, None]+dy[:, None]*s)
        off = ~road
        first = off.float().argmax(1)
        edge_known = known.gather(1, first[:, None])[:, 0]
        return torch.where(off.any(1), torch.where(edge_known, s[first], torch.inf), max_dist)

    def _progress(self):
        if self.reference is None:
            return torch.zeros_like(self.x)
        q = torch.stack([self.x, self.y], 1)
        # The reference path defines reward/progress only, never road geometry.
        closest = (q[:, None]-self.ref[None]).square().sum(2).argmin(1)
        return self.ref_dist[closest]

    def _finished(self):
        if self.finish is None:
            return torch.zeros(self.n, dtype=torch.bool, device=self.dev)
        axis, coordinate = self.finish
        return (self.x if axis == 0 else self.y) >= coordinate

    def step(self, action):
        reward, done, info = super().step(action)
        m = self.P['margin'].clamp(min=0)
        zero = torch.zeros_like(m)
        dx = torch.stack([zero, m, -m, zero, zero], 1)
        dy = torch.stack([zero, zero, zero, m, -m], 1)
        road, known = self.sample(self.x[:, None]+dx, self.y[:, None]+dy)
        unknown = ~self.air & ~road.any(1) & ~known.all(1)
        known_unsafe = torch.zeros_like(unknown)
        if self.base.ground_inset > 0:
            ground_road, ground_known = self.sample(*self.ground_points())
            known_unsafe = ~self.air & (ground_known & ~ground_road).any(1)
            unknown = (~self.air & ~ground_known.all(1)
                       & ~(ground_known & ~ground_road).any(1))
        if self.base.landing_inset > 0 and info['land'].any():
            landing_road, landing_known = self.sample(*self.landing_points())
            # Missing evidence is not a measured collision. A known off-road
            # footprint point, however, suffices to reject the landing.
            unknown |= (info['land'] & ~landing_known.all(1)
                        & ~(landing_known & ~landing_road).any(1))
            known_unsafe |= info['land'] & (landing_known & ~landing_road).any(1)
        unknown &= ~known_unsafe
        reward = torch.where(unknown, torch.zeros_like(reward), reward)
        info['fell'] &= ~unknown
        info['finished'] &= ~unknown
        info['unknown'] = unknown
        return reward, done | unknown, info

    def observe(self, clean=False):
        grid, state = super().observe(clean)
        _, known = self.sample(*self.observation_points(clean))
        grid = torch.where(known.reshape(self.n, 1, G, G), grid, .5)
        return grid, state
