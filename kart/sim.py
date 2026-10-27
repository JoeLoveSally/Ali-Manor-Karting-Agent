"""Kinematic kart simulator on a ground-plane occupancy grid.

Ground frame: u along B (pressed), v along A (released). Heading theta is the
angle of the velocity from +v towards +u: 0 = A, pi/2 = B. Parameters are the
values fitted from recordings; every one can be randomised for training.
Sim is a compatibility adapter to the shared VecKart kernel.
"""
from dataclasses import dataclass, replace

import numpy as np


@dataclass
class Params:
    speed: float = 1350.0        # ground units / s on a straight
    turn_rate: float = np.radians(210.0)  # rad / s, independent of speed
    drift_slow: float = 0.88     # speed factor while the heading is changing
    latency_press: float = 0.05    # s, touch down -> heading starts to change
    latency_release: float = 0.0   # s, touch up -> heading starts to change
    boost_gain: float = 0.45     # extra speed fraction at the boost peak
    boost_time: float = 0.45     # s, boost duration
    perfect_dist: float = 120.0  # candidate inner-edge threshold during/after an uncredited drift
    margin: float = 0.0          # car anchor may leave the road by this much before falling
    margin_jitter: float = 10.0  # legacy outward tolerance randomization; zero makes margin exact
    ground_inset: float = 0.0    # conservative nine-point footprint on every grounded step
    landing_inset: float = 0.0   # training safety margin: landing footprint must stay inside the road
    constrain_jump_geometry: bool = False  # keep jumps feasible independently of collision inset
    perfect_bonus: float = 0.5   # learning reward; zero leaves physical boosts enabled
    switch_penalty: float = 0.02  # legacy action-change cost; pure PPO uses zero
    loop_delay: float = 0.0      # s, live loop on top of the game: capture/encode/decode/processing + control send
    gravity: float = 5000.0
    ramp_power: float = 2.0
    ramp_length: float = 260.0
    ramp_height: float = 122.0
    relief_height: float = 0.0  # full-width rise/dip amplitude; selected from image-fit evidence
    relief_length: float = 513.0
    perfect_min_angle: float = np.pi / 2
    perfect_start_tolerance: float = np.pi / 2
    boost_rise: float = 0.0     # both zero preserve the historical sinusoidal pulse
    boost_fall: float = 0.0
    speed_tau: float = 0.0
    startup_duration: float = 0.0  # linear acceleration duration; zero preserves full-speed starts
    startup_phase: float = 0.0     # acceleration age at GO detection
    startup_phase_jitter: float = 0.0
    startup_speed_profiles: tuple = ()  # empirical ratios for observed speed, not physical speed
    startup_profile_dt: float = 1/60
    startup_profile_shift: float = 0.0
    startup_profile_noise: float = 0.0
    observation_window: float = 0.0  # velocity averaging interval, seconds
    observation_window_jitter: float = 0.0
    banner_delay: float = 0.0
    banner_duration: float = 0.45  # the deployed estimator's timer, not physical boost duration
    observation_memory: bool = False
    observation_height: bool = False
    camera_height_gain: float = 1.0  # camera/car height ratio; measured ramp candidate uses zero
    air_turn_continues: bool = False  # candidate: finish pre-launch turn, ignore new airborne targets
    air_input_response_probability: float = 0.0  # uncertainty model: fraction of episodes accepting airborne inputs
    observed_press_delay: float = 0.1191369715213808
    observed_release_delay: float = 0.14662272042724387
    dt: float = 1 / 60

    def randomized(self, rng, scale=1.0):
        j = lambda x, r: x * (1 + scale * r * rng.uniform(-1, 1))
        return replace(self, speed=j(self.speed, 0.12), turn_rate=j(self.turn_rate, 0.06),
                       drift_slow=min(1.0, j(self.drift_slow, 0.05)),
                       latency_press=max(0.0, self.latency_press + scale * rng.uniform(-0.02, 0.05)),
                       latency_release=max(0.0, self.latency_release + scale * rng.uniform(-0.02, 0.05)),
                       boost_gain=j(self.boost_gain, 0.3), boost_time=j(self.boost_time, 0.3),
                       perfect_dist=j(self.perfect_dist, 0.3),
                       margin=max(0., self.margin + scale * rng.uniform(-self.margin_jitter, self.margin_jitter)))


class Track:
    """Boolean road grid. cell (row, col) covers v in [y0+row*res, ...), u in [x0+col*res, ...)."""

    def __init__(self, road, origin, res=8.0, known=None, jumps=(), ramps=(), reliefs=()):
        self.road = np.asarray(road, bool)
        self.outside_known = known is None
        self.known = np.ones_like(self.road) if known is None else np.asarray(known, bool)
        if self.known.shape != self.road.shape or res <= 0:
            raise ValueError('Invalid map shape or resolution')
        self.jumps = np.asarray(jumps, float).reshape(-1, 6)
        self.ramps = np.asarray(ramps, float).reshape(-1, 6)
        # [axis, start, end, lateral_min, lateral_max, signed_height].
        # These continuous surfaces retain road support, including below z=0.
        self.reliefs = np.asarray(reliefs, float).reshape(-1, 6)
        if len(self.reliefs) and (not np.isfinite(self.reliefs).all() or
                not np.isin(self.reliefs[:, 0], [0, 1]).all() or
                np.any(self.reliefs[:, 2] <= self.reliefs[:, 1]) or
                np.any(self.reliefs[:, 4] <= self.reliefs[:, 3])):
            raise ValueError('Invalid full-width relief geometry')
        self.x0, self.y0 = float(origin[0]), float(origin[1])
        self.res = float(res)

    def on_road(self, p):
        c, r = int((p[0] - self.x0) // self.res), int((p[1] - self.y0) // self.res)
        return (0 <= r < self.road.shape[0] and 0 <= c < self.road.shape[1]
                and self.known[r, c] and self.road[r, c])

    def is_known(self, p):
        c, r = int((p[0] - self.x0) // self.res), int((p[1] - self.y0) // self.res)
        if 0 <= r < self.road.shape[0] and 0 <= c < self.road.shape[1]:
            return bool(self.known[r, c])
        return self.outside_known

    def ray(self, p, d, max_dist=800.0, step=4.0):
        """Distance from p along unit d until leaving the road (0 if already off)."""
        s = 0.0
        while s < max_dist:
            if not self.is_known(p + d * s):
                return np.inf
            if not self.on_road(p + d * s):
                return s
            s += step
        return max_dist


@dataclass
class State:
    pos: np.ndarray
    theta: float = 0.0
    t: float = 0.0
    pressed: bool = False
    effective: bool = False                       # input the game is currently applying
    boost_left: float = 0.0
    speed: float = 0.0          # actual speed in the last step (what the camera measures)
    drifting: bool = False
    drift_from: float = 0.0
    perfect_count: int = 0
    alive: bool = True
    unknown: bool = False       # unsupported map region, not a measured fall
    airborne: bool = False


class Sim:
    def __init__(self, track, params=Params(), start=(0.0, 0.0), theta=0.0):
        self.track, self.p = track, params
        self.s = State(pos=np.array(start, float), theta=theta)
        # Replay and PPO execute the identical kernel.
        from .rasterenv import RasterKart
        self.env = RasterKart(track, n=1, params=params, start=start, theta=theta,
                              loop_delay=(params.loop_delay, params.loop_delay))
        self._started = False

    def target(self):
        return np.pi / 2 if self.s.effective else 0.0

    def step(self, pressed):
        import torch
        s, e = self.s, self.env
        if not self._started:
            e.pressed.fill_(float(s.pressed))
            e.eff.fill_(float(s.effective))
            e.hist.fill_(float(s.effective))
            e.boost.fill_(s.boost_left)
            e.drifting.fill_(float(s.drifting))
            e.drift_pending.fill_(bool(s.drifting))
            e.drift_from.fill_(s.drift_from)
            self._started = True
        _, _, info = e.step(torch.tensor([int(pressed)]))
        s.pos = np.array([e.x.item(), e.y.item()])
        s.theta, s.t = e.th.item(), e.t.item()
        s.pressed, s.effective = bool(e.pressed.item()), bool(e.eff.item())
        s.speed, s.boost_left = e.speed_now.item(), e.boost.item()
        s.drifting, s.drift_from = bool(e.drifting.item()), e.drift_from.item()
        s.perfect_count += int(info['perfect'].item())
        s.airborne = bool(e.air.item())
        s.alive = s.alive and not bool(info['fell'].item())
        s.unknown = s.unknown or bool(info['unknown'].item())
        return s.alive and not s.unknown

    def supported(self):
        if self.s.airborne:
            return True
        if self.track.on_road(self.s.pos):
            return True
        if self.p.margin <= 0:
            return False
        m = self.p.margin
        return any(self.track.on_road(self.s.pos + np.array(o)) for o in
                   [(m, 0), (-m, 0), (0, m), (0, -m)])

    def inner_distance(self):
        """Distance to the road edge on the side the car came from (lateral to the new heading)."""
        lateral = np.array([0.0, -1.0]) if self.s.effective else np.array([-1.0, 0.0])
        return self.track.ray(self.s.pos, lateral)
