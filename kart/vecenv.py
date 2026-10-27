"""Batched kart environment in torch for PPO (MonoRace-style: policy sees state, not pixels).

Tracks are unions of axis-aligned rectangles minus explicit jump gaps, so
on-road tests for thousands of points across all envs are a few tensor ops.
This is the shared dynamics kernel; kart/sim.py wraps it for scalar replay.

Observation per env:
  grid  (1, G, G) ego-centric occupancy at CELL units/cell: 1 road, 0 void,
        0.5 unknown (outside the camera's view of the ground)
  state (S,)      sin/cos heading, pressed, effective input, speed/1000,
                  boost remaining, time since last switch, turn progress
Action: 0 release, 1 press. Reward: progress along the track, perfect-drift
bonus, -1 on falling; small switch penalty to discourage chatter.
"""
import numpy as np
import torch

from .geometry import M
from .sim import Params

G, CELL = 48, 32.0          # 48x48 cells, 32 units each -> +-768 around the car
MEM_G, MEM_CELL = 104, 16.0  # world-aligned evidence cells, matching the deployed map resolution
STATE_DIM = 8
FALL_PENALTY = 1.0


def visibility_mask():
    """Which ego grid cells are on screen (the camera follows the car, axes fixed)."""
    c = (torch.arange(G) - G / 2 + 0.5) * CELL
    V, U = torch.meshgrid(c, c, indexing="ij")
    pts = np.c_[U.reshape(-1).numpy(), V.reshape(-1).numpy(), np.ones(G * G)]
    s = pts @ M.T
    x, y = s[:, 0] / s[:, 2], s[:, 1] / s[:, 2]
    vis = (s[:, 2] > 0) & (x >= 50) & (x < 720) & (y >= 330) & (y < 1600)
    return torch.tensor(vis.reshape(G, G))


MAX_HOLES, MAX_JUMPS, MAX_DISTRACT, MAX_RELIEFS = 16, 8, 6, 16


def jump_distance_bounds(params):
    """Aligned straight-flight envelope, including any phase of a boost pulse."""
    speed = np.array([params.speed, params.speed*(1+params.boost_gain)])
    vz = params.ramp_power*params.ramp_height/params.ramp_length*speed
    duration = (vz+np.sqrt(vz*vz+2*params.gravity*params.ramp_height))/params.gravity
    distance = speed*duration
    # Landing is detected on the first integration step after height crosses zero.
    return float(distance[0]), float(distance[1]+speed[1]*params.dt)


def random_rects(rng, n_segments=40, jump_prob=0.3, groove_prob=0.15, n_distract=(0, 5),
                 jump_bounds=None, landing_inset=0., relief_height=83., relief_length=513.):
    """Random track. Returns dict of numpy arrays:

    pos      [x0, y0, x1, y1] road rectangles (segments and the start pad)
    neg      jump gaps (full width). Continuous depressions NEVER remove support.
    reliefs  full-width smooth rises/dips [axis, start, end, lat0, lat1, signed_height]
    jumps    [axis, launch, land, lat0, lat1, overshoot]: axis 0 = along u (B), 1 = along v (A);
             a car crossing `launch` inside [lat0, lat1] flies straight to land + overshoot
             (the game has an up-ramp before every gap; the landing is at the pre-ramp height)
    distract road-looking rectangles visible only in the observation (other-height road
             that the flat-ground camera model projects to the wrong place)
    corners  centerline corner points
    """
    from .tracks import sample_length
    w = rng.uniform(400, 500)
    h = w / 2
    p = np.zeros(2)
    d = 1
    pos, neg, jumps, reliefs, corners = [[-h, -400, h, 0]], [], [], [], [p.copy()]
    for i in range(n_segments):
        L = rng.uniform(800, 1400) if i == 0 else sample_length(rng)
        make_jump = (0 < i < n_segments-1 and L > 1100 and rng.random() < jump_prob
                     and len(jumps) < MAX_JUMPS)
        if jump_bounds is not None and jump_bounds[0]-landing_inset-16 <= 150:
            make_jump = False
        if make_jump and jump_bounds is not None:
            # Leave an entire landing straight at the episode's fastest speed.
            # Harder contact rules must not create unavoidable boosted overshoots.
            L = max(L, 2*h+450+jump_bounds[1]+landing_inset+100)
        q = p + (np.array([0.0, L]) if d == 1 else np.array([L, 0.0]))
        pos.append([min(p[0], q[0]) - h, min(p[1], q[1]) - h, max(p[0], q[0]) + h, max(p[1], q[1]) + h])
        if make_jump:
            gap_max = 550 if jump_bounds is None else min(550, jump_bounds[0]-landing_inset-16)
            gl = rng.uniform(150, gap_max)
            clearance = 300+gl if jump_bounds is None else max(300+gl, jump_bounds[1]+landing_inset+100)
            s0 = rng.uniform(h + 450, max(h + 451, L - h - clearance))
            if s0 + gl < L - h - 200:
                if d == 1:
                    neg.append([p[0] - h - 8, p[1] + s0, p[0] + h + 8, p[1] + s0 + gl])
                    jumps.append([1, p[1] + s0, p[1] + s0 + gl, p[0] - h, p[0] + h, rng.uniform(40, 200)])
                else:
                    neg.append([p[0] + s0, p[1] - h - 8, p[0] + s0 + gl, p[1] + h + 8])
                    jumps.append([0, p[0] + s0, p[0] + s0 + gl, p[1] - h, p[1] + h, rng.uniform(40, 200)])
        elif (0 < i < n_segments-1 and L > 900 and rng.random() < groove_prob
              and len(reliefs) < MAX_RELIEFS and relief_height > 0):
            # Camera fit supplies the nominal shape; spread and sign are explicit
            # domain randomisation, not claims of measured depression physics.
            length = min(rng.uniform(.7, 1.3)*relief_length, L-2*h-100)
            start = rng.uniform(h+50, L-h-length)
            height = rng.choice([-1., 1.])*rng.uniform(.5, 1.5)*relief_height
            reliefs.append([d, p[d]+start, p[d]+start+length,
                            p[1-d]-h, p[1-d]+h, height])
        p = q
        corners.append(p.copy())
        d = 1 - d
    pos = np.array(pos)
    distract = []
    for _ in range(rng.integers(n_distract[0], n_distract[1] + 1)):
        for _try in range(20):
            c = corners[rng.integers(len(corners))]
            side = rng.choice([-1, 1])
            off = rng.uniform(700, 1600)
            cen = c + side * off * np.array([1.0, -1.0]) / np.sqrt(2) + rng.uniform(-300, 300, 2)
            a, b = rng.uniform(150, 600, 2)
            r = [cen[0] - a / 2, cen[1] - b / 2, cen[0] + a / 2, cen[1] + b / 2]
            clear = ((pos[:, 0] - 150 > r[2]) | (pos[:, 2] + 150 < r[0]) |
                     (pos[:, 1] - 150 > r[3]) | (pos[:, 3] + 150 < r[1])).all()
            if clear:
                distract.append(r)
                break
    return {"pos": pos, "neg": np.array(neg).reshape(-1, 4), "jumps": np.array(jumps).reshape(-1, 6),
            "reliefs": np.array(reliefs).reshape(-1, 6),
            "distract": np.array(distract).reshape(-1, 4), "corners": np.array(corners)}


class VecKart:
    def __init__(self, n, device="cpu", seed=0, params=Params(), randomize=1.0, n_segments=40,
                 max_time=60.0, loop_delay=(0.05, 0.18), case=None, case_probability=0., balanced_scenes=False):
        self.n, self.dev = n, torch.device(device)
        if not 0 <= case_probability <= 1 or (case_probability and case is None):
            raise ValueError('Invalid case mixture')
        self.case, self.case_probability = case, case_probability
        self.balanced_scenes = balanced_scenes
        if balanced_scenes and (n_segments < 4 or case_probability > .4):
            raise ValueError('Balanced scenes require four segments and at most 40% reconstructed cases')
        if (not np.isfinite(params.air_input_response_probability)
                or not 0 <= params.air_input_response_probability <= 1
                or (params.air_input_response_probability and not params.air_turn_continues)):
            raise ValueError('Invalid airborne response uncertainty')
        if (not np.isfinite([params.startup_duration, params.startup_phase, params.startup_phase_jitter]).all()
                or params.startup_duration < 0 or params.startup_phase_jitter < 0):
            raise ValueError('Invalid startup acceleration parameters')
        if (not np.isfinite([params.ground_inset, params.landing_inset, params.perfect_bonus,
                             params.relief_height, params.relief_length,
                             params.switch_penalty, params.margin_jitter]).all()
                or params.ground_inset < 0 or params.landing_inset < 0 or params.perfect_bonus < 0
                or params.switch_penalty < 0 or params.margin_jitter < 0
                or params.relief_height < 0 or params.relief_length <= 0):
            raise ValueError('Invalid landing inset or perfect bonus')
        self.rng = np.random.default_rng(seed)
        self.base, self.randomize = params, randomize
        self.n_segments, self.max_time = n_segments, max_time
        self.dt = params.dt
        self.loop_delay = loop_delay
        self.vis = visibility_mask().to(self.dev)
        c = (torch.arange(G, device=self.dev) - G / 2 + 0.5) * CELL
        V, U = torch.meshgrid(c, c, indexing="ij")
        self.grid_off = torch.stack([U.reshape(-1), V.reshape(-1)], 1)  # (G*G, 2)
        from .projection3d import HEIGHT_COLUMN
        self.height_flat = torch.as_tensor(np.linalg.solve(M, HEIGHT_COLUMN), dtype=torch.float32, device=self.dev)
        # Camera cells hidden by the kart and close smoke retain prior map evidence.
        projected = np.c_[self.grid_off.cpu().numpy(), np.ones(G*G)] @ M.T
        screen = projected[:, :2]/projected[:, 2:]
        hidden = ((screen[:, 0] >= 300) & (screen[:, 0] < 600) &
                  (screen[:, 1] >= 560) & (screen[:, 1] < 900))
        self.update_vis = self.vis & ~torch.as_tensor(hidden.reshape(G, G), device=self.dev)
        self.map_memory = torch.full((n, 1, MEM_G, MEM_G), .5, device=self.dev)
        mc = (torch.arange(MEM_G, device=self.dev)-MEM_G//2)*MEM_CELL
        mv, mu = torch.meshgrid(mc, mc, indexing='ij')
        self.memory_off = torch.stack([mu.flatten(), mv.flatten()], 1)
        self.map_last_xy = torch.zeros(n, 2, device=self.dev)
        self.map_last_t = torch.full((n,), -1., device=self.dev)
        R = n_segments + 1
        self.pos_r = torch.zeros(n, R, 4, device=self.dev)
        self.neg_r = torch.full((n, MAX_HOLES, 4), 1e9, device=self.dev)
        self.jumps = torch.full((n, MAX_JUMPS, 6), -1.0, device=self.dev)
        # [axis, start, lip, lateral_min, lateral_max, height]; z follows a power profile.
        self.ramps = torch.full((n, MAX_JUMPS, 6), -1.0, device=self.dev)
        self.reliefs = torch.full((n, MAX_RELIEFS, 6), -1.0, device=self.dev)
        self.distract = torch.full((n, MAX_DISTRACT, 4), 1e9, device=self.dev)
        self.corners = torch.zeros(n, n_segments + 1, 2, device=self.dev)
        self.case_active = torch.zeros(n, dtype=torch.bool, device=self.dev)
        self.case_origin = torch.zeros(n, 2, device=self.dev)
        self.case_finish = torch.zeros(n, 2, device=self.dev)
        f = lambda: torch.zeros(n, device=self.dev)
        self.x, self.y, self.th, self.t = f(), f(), f(), f()
        self.pressed, self.eff, self.boost, self.since = f(), f(), f(), f()
        self.drifting, self.prog, self.seg = f(), f(), torch.zeros(n, dtype=torch.long, device=self.dev)
        self.drift_from = f()
        self.drift_pending = torch.zeros(n, dtype=torch.bool, device=self.dev)
        self.drift_credited = torch.zeros(n, dtype=torch.bool, device=self.dev)
        self.z, self.vz = f(), f()
        # commands issued 0..H-1 steps ago; a command acts once its own latency has passed
        physical_delay = max(params.latency_press, params.latency_release)+.05+loop_delay[1]
        observed_delay = max(params.observed_press_delay, params.observed_release_delay)
        self.H = int(np.ceil(max(physical_delay, observed_delay)/self.dt))+3
        self.hist = torch.zeros(n, self.H, device=self.dev)
        self.lag_p = torch.zeros(n, dtype=torch.long, device=self.dev)
        self.lag_r = torch.zeros(n, dtype=torch.long, device=self.dev)
        self.speed_now = f()
        self.obs_window = f()
        self.obs_banner_at = torch.full((n,), torch.inf, device=self.dev)
        self.obs_boost = f()
        self.velocity_history = torch.zeros(n, max(1, int(np.ceil(
            (params.observation_window+params.observation_window_jitter)/self.dt))), 2, device=self.dev)
        # airborne: flying straight along air_axis until that coordinate reaches air_end
        self.air = torch.zeros(n, dtype=torch.bool, device=self.dev)
        self.air_axis, self.air_end, self.air_until = f(), f(), f()
        self.air_target = f()
        self.startup_age = f()
        profiles = np.asarray(params.startup_speed_profiles, float)
        if profiles.size and (profiles.ndim != 2 or profiles.shape[1] < 2 or
                not np.isfinite(profiles).all() or profiles.min() < 0 or
                params.startup_duration <= 0 or
                not np.isfinite([params.startup_profile_dt, params.startup_profile_shift,
                                 params.startup_profile_noise]).all() or
                params.startup_profile_dt <= 0 or params.startup_profile_shift < 0 or params.startup_profile_noise < 0):
            raise ValueError('Invalid startup speed observation profiles')
        self.startup_profiles = torch.as_tensor(profiles, dtype=torch.float32, device=self.dev)
        self.startup_profile_id = torch.zeros(n, dtype=torch.long, device=self.dev)
        self.startup_profile_offset, self.startup_profile_bias = f(), f()
        self.P = {k: f() for k in ["speed", "turn_rate", "drift_slow", "lat_p", "lat_r", "boost_gain",
                                   "boost_time", "perfect_dist", "margin", "gravity", "ramp_power",
                                   "perfect_min_angle", "perfect_start_tolerance", "boost_rise", "boost_fall", "speed_tau"]}
        # observation randomisation: ramp/kerb walls seen on the camera side, map scale, noise
        self.O = {k: f() for k in ["wall", "gscale", "noise"]}
        self.reset_envs(torch.arange(n, device=self.dev))

    # ---- track geometry -------------------------------------------------
    @staticmethod
    def _in_rects(px, py, rects):
        X, Y = px[..., None], py[..., None]
        return ((X >= rects[:, None, :, 0]) & (X < rects[:, None, :, 2]) &
                (Y >= rects[:, None, :, 1]) & (Y < rects[:, None, :, 3])).any(-1)

    def on_road(self, px, py, env=None):
        """px, py: (n, K) -> bool (n, K)."""
        pr = self.pos_r if env is None else self.pos_r[env]
        nr = self.neg_r if env is None else self.neg_r[env]
        return self._in_rects(px, py, pr) & ~self._in_rects(px, py, nr)

    def ray(self, dx, dy, max_dist=400.0, step=8.0):
        s = torch.arange(0, max_dist + step, step, device=self.dev)
        px = self.x[:, None] + dx[:, None] * s
        py = self.y[:, None] + dy[:, None] * s
        off = ~self.on_road(px, py)
        first = torch.where(off.any(1), off.float().argmax(1), torch.full_like(self.x, len(s) - 1).long())
        return s[first]

    def edge_distance(self):
        # The inner side of the new road, not an arbitrary nearby outer edge.
        # Nearest-boundary credit caused an unobserved boost in the bend failure replay.
        return self.ray(torch.where(self.eff > 0, 0., -1.),
                        torch.where(self.eff > 0, -1., 0.))

    # ---- episode management ----------------------------------------------
    def height_at(self, px, py):
        r = self.ramps
        along = torch.where(r[:, None, :, 0] == 0, px[..., None], py[..., None])
        lateral = torch.where(r[:, None, :, 0] == 0, py[..., None], px[..., None])
        active = ((r[:, None, :, 0] >= 0) & (along >= r[:, None, :, 1]) &
                  (along <= r[:, None, :, 2]) & (lateral >= r[:, None, :, 3]) &
                  (lateral <= r[:, None, :, 4]))
        fraction = ((along-r[:, None, :, 1])/(r[:, None, :, 2]-r[:, None, :, 1]).clamp(min=1)).clamp(0, 1)
        h = r[:, None, :, 5].clamp(min=0)*fraction.pow(self.P['ramp_power'][:, None, None])
        ramp_height = torch.where(active, h, 0.).amax(2)
        r = self.reliefs
        along = torch.where(r[:, None, :, 0] == 0, px[..., None], py[..., None])
        lateral = torch.where(r[:, None, :, 0] == 0, py[..., None], px[..., None])
        active = ((r[:, None, :, 0] >= 0) & (along >= r[:, None, :, 1]) &
                  (along <= r[:, None, :, 2]) & (lateral >= r[:, None, :, 3]) &
                  (lateral <= r[:, None, :, 4]))
        phase = ((along-r[:, None, :, 1])/(r[:, None, :, 2]-r[:, None, :, 1]).clamp(min=1)).clamp(0, 1)
        relief = torch.where(active, r[:, None, :, 5]*torch.sin(np.pi*phase).square(), 0.).sum(2)
        return ramp_height+relief

    def landing_points(self):
        # A conservative square footprint, used only at ground contact after flight.
        return self.footprint_points(self.base.landing_inset)

    def footprint_points(self, d):
        offsets = torch.tensor([-d, 0., d], device=self.dev)
        dy, dx = torch.meshgrid(offsets, offsets, indexing='ij')
        return self.x[:, None]+dx.flatten(), self.y[:, None]+dy.flatten()

    def ground_points(self):
        px, py = self.footprint_points(self.base.ground_inset)
        # A conservative front probe must not hit the jump void before the
        # center reaches the physical launch lip. Clamp only forward probes
        # while the center is actually on an explicit ramp; lateral clearance
        # and the original launch/landing dynamics remain enforced.
        r = self.ramps
        along = torch.where(r[..., 0] == 0, self.x[:, None], self.y[:, None])
        lateral = torch.where(r[..., 0] == 0, self.y[:, None], self.x[:, None])
        active = ((r[..., 0] >= 0) & (along >= r[..., 1]) & (along < r[..., 2]) &
                  (lateral >= r[..., 3]) & (lateral <= r[..., 4]))
        for axis in (0, 1):
            limit = torch.where(active & (r[..., 0] == axis), r[..., 2]-.01, torch.inf).amin(1)
            if axis == 0: px = torch.minimum(px, limit[:, None])
            else: py = torch.minimum(py, limit[:, None])
        return px, py

    def state_dict(self):
        tensors = {k: v.clone() for k, v in vars(self).items() if isinstance(v, torch.Tensor)}
        return {'tensors': tensors, 'P': {k: v.clone() for k, v in self.P.items()},
                'O': {k: v.clone() for k, v in self.O.items()}, 'rng': self.rng.bit_generator.state}

    def load_state_dict(self, state):
        for k, v in state['tensors'].items():
            current = getattr(self, k)
            if current.shape != v.shape:
                raise ValueError(f'Environment shape changed: {k}')
            setattr(self, k, v.to(self.dev).clone())
        for name in ['P', 'O']:
            setattr(self, name, {k: v.to(self.dev).clone() for k, v in state[name].items()})
        self.rng.bit_generator.state = state['rng']

    def reset_envs(self, idx):
        self.case_active[idx] = False
        if not hasattr(self, 'episode_kind'):
            self.episode_kind = torch.zeros(self.n, dtype=torch.long, device=self.dev)
            self.air_responds = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
        self.episode_kind[idx] = 0
        if self.base.air_input_response_probability:
            self.air_responds[idx] = torch.as_tensor(
                self.rng.random(len(idx)) < self.base.air_input_response_probability, device=self.dev)
        else:
            self.air_responds[idx] = False
        # Preserve the exact per-episode RNG order while staging each field on
        # the CPU. A single transfer per field avoids hundreds of scalar CUDA
        # writes and synchronizations whenever short episodes finish together.
        count = len(idx)
        fields = [('pos_r', 'pos', 0., self.n_segments+1),
                  ('corners', 'corners', 0., self.n_segments+1),
                  ('neg_r', 'neg', 1e9, MAX_HOLES), ('jumps', 'jumps', -1., MAX_JUMPS),
                  ('distract', 'distract', 1e9, MAX_DISTRACT),
                  ('reliefs', 'reliefs', -1., MAX_RELIEFS)]
        staged = {name: np.full((count, *getattr(self, name).shape[1:]), fill, np.float32)
                  for name, _, fill, _ in fields}
        staged['ramps'] = np.full((count, MAX_JUMPS, 6), -1., np.float32)
        attributes = {k: k for k in self.P}
        attributes.update(lat_p='latency_press', lat_r='latency_release')
        parameters = {k: np.empty(count, np.float32) for k in self.P}
        observations = {k: np.empty(count, np.float32) for k in self.O}
        lag_p, lag_r = np.empty(count, np.int64), np.empty(count, np.int64)
        windows = np.empty(count, np.float32)
        for j in range(count):
            p = self.base.randomized(self.rng, self.randomize)
            bounds = jump_distance_bounds(p) if p.landing_inset > 0 or p.constrain_jump_geometry else None
            tr = random_rects(self.rng, self.n_segments, jump_bounds=bounds, landing_inset=p.landing_inset,
                              relief_height=p.relief_height, relief_length=p.relief_length)
            for name, key, _, cap in fields:
                a = tr[key][:cap]
                if len(a):
                    staged[name][j, :len(a)] = a
            for k, jump in enumerate(tr['jumps'][:MAX_JUMPS]):
                axis, lip, _, l0, l1, _ = jump
                staged['ramps'][j, k] = [axis, lip-p.ramp_length, lip, l0, l1, p.ramp_height]
            for k, attr in attributes.items():
                parameters[k][j] = getattr(p, attr)
            r = self.randomize
            ld = self.rng.uniform(*self.loop_delay)
            lag_p[j] = min(self.H - 1, int(np.ceil((p.latency_press + ld) / self.dt - 1e-9)))
            lag_r[j] = min(self.H - 1, int(np.ceil((p.latency_release + ld) / self.dt - 1e-9)))
            observations['wall'][j] = self.rng.uniform(0, 48)*r
            observations['gscale'][j] = 1+self.rng.uniform(-.08, .08)*r
            observations['noise'][j] = self.rng.uniform(0, .03)*r
            windows[j] = max(0., p.observation_window+self.rng.uniform(-1, 1)*r*p.observation_window_jitter)
        for name, values in staged.items():
            getattr(self, name)[idx] = torch.as_tensor(values, device=self.dev)
        for k, values in parameters.items():
            self.P[k][idx] = torch.as_tensor(values, device=self.dev)
        for k, values in observations.items():
            self.O[k][idx] = torch.as_tensor(values, device=self.dev)
        self.lag_p[idx] = torch.as_tensor(lag_p, device=self.dev)
        self.lag_r[idx] = torch.as_tensor(lag_r, device=self.dev)
        self.obs_window[idx] = torch.as_tensor(windows, device=self.dev)
        for a in [self.x, self.y, self.th, self.t, self.pressed, self.eff, self.boost, self.drifting, self.drift_from,
                  self.prog, self.air_axis, self.air_end, self.air_until, self.air_target]:
            a[idx] = 0
        self.z[idx] = self.vz[idx] = 0
        self.drift_credited[idx] = False
        self.drift_pending[idx] = False
        self.speed_now[idx] = self.P["speed"][idx]
        self.startup_age[idx] = self.base.startup_phase
        if self.base.startup_phase_jitter > 0:
            self.startup_age[idx] += (self.randomize*self.base.startup_phase_jitter
                                     *torch.as_tensor(self.rng.uniform(-1, 1, len(idx)), device=self.dev))
        if self.base.startup_duration > 0:
            self.speed_now[idx] *= (self.startup_age[idx]/self.base.startup_duration).clamp(0, 1)
        self.startup_profile_id[idx] = 0
        self.startup_profile_offset[idx] = self.startup_profile_bias[idx] = 0
        if self.startup_profiles.numel() and self.randomize > 0:
            self.startup_profile_id[idx] = torch.as_tensor(
                self.rng.integers(len(self.startup_profiles), size=len(idx)), device=self.dev)
            self.startup_profile_offset[idx] = torch.as_tensor(self.rng.uniform(
                -self.base.startup_profile_shift, self.base.startup_profile_shift, len(idx))*self.randomize,
                dtype=torch.float32, device=self.dev)
            self.startup_profile_bias[idx] = torch.as_tensor(self.rng.uniform(
                -self.base.startup_profile_noise, self.base.startup_profile_noise, len(idx))*self.randomize,
                dtype=torch.float32, device=self.dev)
        self.air[idx] = False
        self.since[idx] = 1.0
        self.seg[idx] = 0
        self.hist[idx] = 0
        self.velocity_history[idx] = 0
        self.velocity_history[idx, :, 1] = self.speed_now[idx, None]
        self.obs_banner_at[idx] = torch.inf
        self.obs_boost[idx] = 0
        self.map_memory[idx] = .5
        self.map_last_xy[idx] = 0
        self.map_last_t[idx] = -1
        if self.case is not None and self.case_probability:
            from .cases import apply_case
            selected = idx[torch.as_tensor(self.rng.random(len(idx)) < self.case_probability, device=self.dev)]
            apply_case(self, selected, self.case)
            self.episode_kind[selected] = 1
        if self.balanced_scenes:
            from .curriculum import apply_training_mix
            apply_training_mix(self, idx)

    # ---- dynamics -----------------------------------------------------------
    def step(self, action):
        a = action.float()
        P, dt = self.P, self.dt
        changed = a != self.pressed
        self.pressed = a
        self.since = torch.where(changed, torch.zeros_like(self.since), self.since + dt)
        # effective input: the newest command whose latency (press / release differ) has passed
        self.hist = torch.cat([a[:, None], self.hist[:, :-1]], 1)
        k = torch.arange(self.H, device=self.dev)[None]
        lag = torch.where(self.hist > 0, self.lag_p[:, None], self.lag_r[:, None])
        valid = k >= lag
        first = valid.float().argmax(1)
        eff_new = self.hist.gather(1, first[:, None])[:, 0]
        prev_eff = self.eff
        self.eff = torch.where(valid.any(1), eff_new, self.eff)

        # New airborne commands do not replace the target captured at take-off.
        # Whether the existing turn continues is an independently audited mode.
        ground = ~self.air
        tgt = self.eff * (np.pi / 2)
        can_turn = ground
        if self.base.air_turn_continues:
            tgt = torch.where(self.air & ~self.air_responds, self.air_target, tgt)
            can_turn = torch.ones_like(ground)
        err = tgt - self.th
        turning = (err.abs() > 1e-6) & can_turn
        starting = turning & ground & ((self.drifting == 0) | (self.eff != prev_eff))
        self.drift_from = torch.where(starting, self.th, self.drift_from)
        self.drift_pending |= starting
        self.drift_credited &= ~starting
        self.drifting = torch.where(turning, torch.ones_like(self.drifting), self.drifting)
        self.th = torch.where(can_turn, self.th + torch.clamp(err, -P["turn_rate"] * dt, P["turn_rate"] * dt), self.th)
        # Finishing the heading response aloft does not settle the road drift:
        # retain its speed/credit state until ground contact is restored.
        done_turn = ground & (self.drifting > 0) & ((tgt - self.th).abs() <= 1e-6)
        # Require sufficient angular travel and at most one credit per drift.
        enough_turn = (self.th-self.drift_from).abs()+1e-6 >= P['perfect_min_angle']
        full_start = torch.minimum(self.drift_from.abs(), (np.pi/2-self.drift_from).abs()) <= P['perfect_start_tolerance']
        # Trigger geometry is a calibration candidate until the held-out event gate passes.
        perfect = torch.zeros_like(self.x, dtype=torch.bool)
        # Completing the heading response ends drift slowdown, but the same
        # drift may still earn its first edge reward until another turn starts.
        eligible = ground & self.drift_pending & enough_turn & full_start & ~self.drift_credited
        if eligible.any():
            inner = self.edge_distance()
            perfect = eligible & (inner <= P["perfect_dist"]) & self.on_road(self.x[:, None], self.y[:, None])[:, 0]
        self.drift_credited |= perfect
        self.drift_pending &= ~perfect
        self.drifting = torch.where(done_turn, torch.zeros_like(self.drifting), self.drifting)
        self.boost = torch.where(perfect, P["boost_time"], self.boost)

        # An unfinished drift keeps its low speed during flight, as in the recordings.
        slowing = turning | (self.air & (self.drifting > 0))
        speed = P["speed"] * torch.where(slowing, P["drift_slow"], torch.ones_like(self.x))
        pulse = torch.sin(np.pi*self.boost/P['boost_time'])
        shaped = (P['boost_rise'] > 0) | (P['boost_fall'] > 0)
        plateau = torch.minimum(((P['boost_time']-self.boost)/P['boost_rise'].clamp(min=1e-6)).clamp(0, 1),
                                (self.boost/P['boost_fall'].clamp(min=1e-6)).clamp(0, 1))
        pulse = torch.where(shaped, plateau, pulse)
        bfac = torch.where(self.boost > 0,
                           1 + P["boost_gain"] * pulse,
                           torch.ones_like(self.x))
        speed = speed * bfac
        if self.base.startup_duration > 0:
            duration = self.base.startup_duration
            # Average the ramp over the integration interval, including crossing
            # its endpoint. This preserves distance when the step size changes.
            def integral(age):
                return age.clamp(0, duration).square()/(2*duration)+(age-duration).clamp(min=0)
            age = self.startup_age
            ramp = ((integral(age+dt)-integral(age))/dt).clamp(0, 1)
            speed = speed*ramp
            self.startup_age = (age+dt).clamp(max=duration)
        alpha = torch.where(P['speed_tau'] > 0, 1-torch.exp(-dt/P['speed_tau'].clamp(min=1e-6)), 1.)
        speed = self.speed_now+alpha*(speed-self.speed_now)
        self.speed_now = speed
        self.boost = torch.clamp(self.boost - dt, min=0)
        x0, y0 = self.x, self.y
        self.x = self.x + speed * torch.sin(self.th) * dt
        self.y = self.y + speed * torch.cos(self.th) * dt
        self.t = self.t + dt
        velocity = torch.stack([speed*torch.sin(self.th), speed*torch.cos(self.th)], 1)
        self.velocity_history = torch.cat([velocity[:, None], self.velocity_history[:, :-1]], 1)
        self.obs_banner_at = torch.where(perfect, self.t+self.base.banner_delay, self.obs_banner_at)
        banner = self.t+1e-6 >= self.obs_banner_at
        self.obs_boost = torch.where(banner, self.base.banner_duration,
                                     (self.obs_boost-dt).clamp(min=0))
        self.obs_banner_at = torch.where(banner, torch.inf, self.obs_banner_at)

        # Ballistic height evolves independently of ground support and control.
        z_next = self.z+self.vz*dt-.5*P['gravity']*dt*dt
        vz_next = self.vz-P['gravity']*dt
        self.z = torch.where(self.air, z_next, self.height_at(self.x[:, None], self.y[:, None])[:, 0])
        self.vz = torch.where(self.air, vz_next, torch.zeros_like(self.vz))
        # take-off: crossing a launch line forwards, inside its lateral extent
        J = self.jumps
        ax = J[..., 0]
        along0 = torch.where(ax == 0, x0[:, None], y0[:, None])
        along1 = torch.where(ax == 0, self.x[:, None], self.y[:, None])
        lat0 = torch.where(ax == 0, y0[:, None], x0[:, None])
        lat1 = torch.where(ax == 0, self.y[:, None], self.x[:, None])
        crossing = ((J[..., 1]-along0)/(along1-along0).clamp(min=1e-8)).clamp(0, 1)
        lat_cross = lat0+crossing*(lat1-lat0)
        hit = ((ax >= 0) & (along0 < J[..., 1]) & (along1 >= J[..., 1]) &
               (lat_cross >= J[..., 3]) & (lat_cross <= J[..., 4]))
        launch = ground & hit.any(1)
        if launch.any():
            j = hit.float().argmax(1)
            Jj = J[torch.arange(self.n, device=self.dev), j]
            self.air = self.air | launch
            self.air_target = torch.where(launch, tgt, self.air_target)
            self.air_axis = torch.where(launch, Jj[:, 0], self.air_axis)
            self.air_end = torch.where(launch, Jj[:, 2] + Jj[:, 5], self.air_end)
            ramp = self.ramps[torch.arange(self.n, device=self.dev), j]
            has_ramp = (ramp[:, 0] == Jj[:, 0]) & ((ramp[:, 2]-Jj[:, 1]).abs() < 1e-3)
            along_speed = torch.where(Jj[:, 0] == 0, speed*torch.sin(self.th), speed*torch.cos(self.th))
            slope = P['ramp_power']*ramp[:, 5]/(ramp[:, 2]-ramp[:, 1]).clamp(min=1)
            # Legacy explicitly annotated gaps retain their length specification,
            # but execute the same ballistic integrator rather than forced landing.
            flight_time = (Jj[:, 2]+Jj[:, 5]-Jj[:, 1])/along_speed.clamp(min=1)
            launch_vz = torch.where(has_ramp, slope*along_speed, .5*P['gravity']*flight_time)
            launch_z = torch.where(has_ramp, ramp[:, 5], torch.zeros_like(self.z))
            # Take-off occurs within this integration step. Advancing the car
            # past the lip while leaving height at t=0 delays the whole flight.
            remaining = dt*(1-crossing.gather(1, j[:, None])[:, 0])
            launch_z = launch_z+launch_vz*remaining-.5*P['gravity']*remaining.square()
            launch_vz = launch_vz-P['gravity']*remaining
            self.z = torch.where(launch, launch_z, self.z)
            self.vz = torch.where(launch, launch_vz, self.vz)
        terrain = self.height_at(self.x[:, None], self.y[:, None])[:, 0]
        land = self.air & (self.vz <= 0) & (self.z <= terrain+1e-4)
        self.air = self.air & ~land
        self.z = torch.where(land, terrain, self.z)
        self.vz = torch.where(land, torch.zeros_like(self.vz), self.vz)

        m = P["margin"]
        offs = torch.stack([torch.zeros_like(m), m, -m, torch.zeros_like(m), torch.zeros_like(m)], 1)
        offs_y = torch.stack([torch.zeros_like(m), torch.zeros_like(m), torch.zeros_like(m), m, -m], 1)
        alive = self.air | self.on_road(self.x[:, None] + offs, self.y[:, None] + offs_y).any(1)
        unsafe_ground = torch.zeros_like(land)
        if self.base.ground_inset > 0:
            safe = self.on_road(*self.ground_points()).all(1)
            unsafe_ground = ~self.air & ~safe
            alive = self.air | safe
        unsafe_landing = torch.zeros_like(land)
        if self.base.landing_inset > 0 and land.any():
            unsafe_landing = land & ~self.on_road(*self.landing_points()).all(1)
            alive &= ~unsafe_landing

        # progress: distance along the corner polyline (advance seg when past the next corner)
        prog_new = self._progress()
        reward = (prog_new - self.prog) / 1000.0
        self.prog = prog_new
        reward = reward + self.base.perfect_bonus * perfect.float() - self.base.switch_penalty * changed.float()
        finished = self._finished() & alive
        timeout = self.t >= self.max_time
        reward = reward - FALL_PENALTY * (~alive).float()
        done = (~alive) | finished | timeout
        info = {"perfect": perfect, "fell": ~alive, "finished": finished, "timeout": timeout,
                "progress": self.prog.clone(), "seg": self.seg.clone(), "air": self.air.clone(),
                "launch": launch, "land": land, "unsafe_landing": unsafe_landing, "height": self.z.clone(),
                "unsafe_ground": unsafe_ground}
        return reward, done, info

    def _finished(self):
        # Reaching the last segment is not reaching its endpoint.
        length = (self.corners[:, 1:] - self.corners[:, :-1]).abs().sum((1, 2))
        finished = (self.seg == self.n_segments - 1) & (self.prog >= length)
        coordinate = torch.where(self.case_finish[:, 0] == 0, self.x, self.y)
        return torch.where(self.case_active, coordinate >= self.case_finish[:, 1], finished)

    def _seg_dist(self, k):
        """Distance from the car to centerline segment k (n,) ."""
        idx = torch.arange(self.n, device=self.dev)
        k = torch.clamp(k, max=self.n_segments - 1)
        a, b = self.corners[idx, k], self.corners[idx, k + 1]
        lo, hi = torch.minimum(a, b), torch.maximum(a, b)
        q = torch.stack([self.x, self.y], 1)
        return (q - torch.clamp(q, lo, hi)).norm(dim=1)

    def _progress(self):
        # The car cuts corners inside the road width (short segments can be shorter than
        # the road is wide), so it never has to reach a corner point: take the nearest of
        # the current and next two centerline segments, never going backwards.
        idx = torch.arange(self.n, device=self.dev)
        d = torch.stack([self._seg_dist(self.seg + j) for j in range(3)], 1)
        step = d.argmin(1)
        self.seg = torch.clamp(self.seg + step, max=self.n_segments - 1)
        a = self.corners[idx, self.seg]
        along_v = (self.seg % 2 == 0)
        within = torch.where(along_v, self.y - a[:, 1], self.x - a[:, 0])
        seglen = (self.corners[:, 1:] - self.corners[:, :-1]).abs().sum(-1)
        done_len = torch.cumsum(seglen, 1)
        base = torch.where(self.seg > 0, done_len[idx, torch.clamp(self.seg - 1, min=0)],
                           torch.zeros_like(self.x))
        regular = base + within.clamp(min=0)
        local = ((self.x-self.case_origin[:, 0])+(self.y-self.case_origin[:, 1])).clamp(min=0)
        straight = (self.episode_kind == 2) | (self.episode_kind == 4)
        axis = self.case_finish[:, 0] == 0
        along = torch.where(axis, self.x-self.case_origin[:, 0], self.y-self.case_origin[:, 1])
        local = torch.where(straight, along.clamp(min=0), local)
        return torch.where(self.case_active, local, regular)

    # ---- observation -------------------------------------------------------
    def observation_known(self, px, py):
        return torch.ones_like(px, dtype=torch.bool)

    def observation_points(self, clean=False, offsets=None):
        scale = torch.ones_like(self.x) if clean else self.O['gscale']
        offsets = self.grid_off[None] if offsets is None else offsets
        off = offsets*scale[:, None, None]
        px, py = self.x[:, None]+off[..., 0], self.y[:, None]+off[..., 1]
        if not clean and self.base.observation_height:
            # Intersect each complete surface patch, including rays whose flat
            # projection lies OUTSIDE that patch. Sampling height only at the
            # initial flat point silently clipped the visible sides of rises.
            b = self.height_flat
            coefficient = off*b[2]-b[:2]
            camera_z = self.base.camera_height_gain*self.z[:, None]
            flat_x = px-coefficient[..., 0]*camera_z
            flat_y = py-coefficient[..., 1]*camera_z
            r = torch.cat([self.ramps, self.reliefs], 1)
            # Exact broad-phase culling: intersect only patches reachable by a
            # visible ray over their entire signed height range. Distant terrain
            # must not multiply the cost of every 16-unit evidence cell.
            magnitude = r[..., 5].abs().amax(1)
            reach = coefficient.abs().amax(1)*magnitude[:, None]
            lo = torch.stack([flat_x.amin(1), flat_y.amin(1)], 1)-reach
            hi = torch.stack([flat_x.amax(1), flat_y.amax(1)], 1)+reach
            rx0 = torch.where(r[..., 0] == 0, r[..., 1], r[..., 3])
            rx1 = torch.where(r[..., 0] == 0, r[..., 2], r[..., 4])
            ry0 = torch.where(r[..., 0] == 0, r[..., 3], r[..., 1])
            ry1 = torch.where(r[..., 0] == 0, r[..., 4], r[..., 2])
            possible = ((r[..., 0] >= 0) & (rx1 >= lo[:, 0, None]) & (rx0 <= hi[:, 0, None])
                        & (ry1 >= lo[:, 1, None]) & (ry0 <= hi[:, 1, None]))
            count = int(possible.sum(1).max().item())
            if count == 0:
                return flat_x, flat_y
            ids = possible.long().argsort(1, descending=True)[:, :count]
            r = r.gather(1, ids[..., None].expand(-1, -1, 6)).clone()
            r[..., 0] = torch.where(possible.gather(1, ids), r[..., 0], -1.)
            axis = r[:, None, :, 0]
            start, end, height = r[:, None, :, 1], r[:, None, :, 2], r[:, None, :, 5]
            a0 = torch.where(axis == 0, flat_x[..., None], flat_y[..., None])
            ca = torch.where(axis == 0, coefficient[..., 0, None], coefficient[..., 1, None])
            is_relief = ids[:, None] >= MAX_JUMPS
            low = torch.minimum(height, torch.zeros_like(height)).expand_as(a0)
            high = torch.maximum(height, torch.zeros_like(height)).expand_as(a0)
            for _ in range(10):
                mid = (low+high)*.5
                phase = ((a0+ca*mid-start)/(end-start).clamp(min=1)).clamp(0, 1)
                surface = height*torch.where(is_relief, torch.sin(np.pi*phase).square(),
                                            phase.pow(self.P['ramp_power'][:, None, None]))
                below = mid < surface
                low, high = torch.where(below, mid, low), torch.where(below, high, mid)
            hit_z = (low+high)*.5
            along = a0+ca*hit_z
            lateral = torch.where(axis == 0, flat_y[..., None]+coefficient[..., 1, None]*hit_z,
                                  flat_x[..., None]+coefficient[..., 0, None]*hit_z)
            valid = ((axis >= 0) & (along >= start) & (along <= end) &
                     (lateral >= r[:, None, :, 3]) & (lateral <= r[:, None, :, 4]))
            nearest = torch.where(valid, hit_z, -torch.inf).amax(2)
            flat_height = self.height_at(flat_x, flat_y)
            w = self.O['wall'][:, None]
            flat_road = (self.on_road(flat_x, flat_y) | self.on_road(flat_x+w, flat_y+w)
                         | self._in_rects(flat_x, flat_y, self.distract))
            # Empty space at z=0 must not hide a lower road surface.
            flat_valid = (flat_height.abs() < 1e-5) & flat_road
            nearest = torch.maximum(nearest, torch.where(flat_valid, 0., -torch.inf))
            hit = torch.isfinite(nearest)
            dz = torch.where(hit, nearest, 0.)-camera_z
            background = ~hit & (flat_height.abs() < 1e-5)
            px = torch.where(hit | background, px+coefficient[..., 0]*dz, 1e9)
            py = torch.where(hit | background, py+coefficient[..., 1]*dz, 1e9)
        return px, py

    def _camera_occupancy(self, px, py, flip=None):
        w = self.O['wall'][:, None]
        occ = self.on_road(px, py) | self.on_road(px+w, py+w) | self._in_rects(px, py, self.distract)
        if flip is None:
            flip = torch.rand_like(px) < self.O['noise'][:, None]
        return torch.where(flip, ~occ, occ).float()

    def _remembered_grid(self):
        """Retain evidence on world cells; do not blur history at every camera shift."""
        import torch.nn.functional as F
        xy = torch.stack([self.x, self.y], 1)
        center = torch.floor(xy/MEM_CELL)*MEM_CELL
        previous = torch.floor(self.map_last_xy/MEM_CELL)*MEM_CELL
        shift = (center-previous)/MEM_CELL
        indices = self.memory_off.reshape(MEM_G, MEM_G, 2)/MEM_CELL
        coords = (indices[None]+shift[:, None, None]+.5)*2/MEM_G
        old = F.grid_sample(self.map_memory-.5, coords, mode='nearest',
                            padding_mode='zeros', align_corners=False)[:, 0]+.5
        off = center[:, None]+self.memory_off[None]-xy[:, None]
        matrix = torch.as_tensor(M, dtype=torch.float32, device=self.dev)
        screen = torch.cat([off, torch.ones_like(off[..., :1])], -1) @ matrix.T
        sx, sy = screen[..., 0]/screen[..., 2], screen[..., 1]/screen[..., 2]
        visible = ((screen[..., 2] > 0) & (sx >= 50) & (sx < 720) & (sy >= 330) & (sy < 1600)
                   & ~((sx >= 300) & (sx < 600) & (sy >= 560) & (sy < 900)))
        visible &= ~self.air[:, None]
        # Query surfaces only for columns that can update a camera-visible
        # world cell in this batch. The complete memory and its shift stay
        # intact. Draw the original full noise field to preserve RNG streams.
        ids = visible.any(0).nonzero().squeeze(1)
        flip = torch.rand_like(off[..., 0]) < self.O['noise'][:, None]
        occ = torch.zeros((self.n, MEM_G*MEM_G), device=self.dev)
        if ids.numel():
            px, py = self.observation_points(offsets=off[:, ids])
            occ[:, ids] = self._camera_occupancy(px, py, flip[:, ids])
            visible[:, ids] &= self.observation_known(px, py)
        occ = occ.reshape(self.n, MEM_G, MEM_G)
        updated = torch.where(visible.reshape(self.n, MEM_G, MEM_G), .5*old+.5*occ, old)
        fresh = self.t != self.map_last_t
        self.map_memory = torch.where(fresh[:, None, None, None], updated[:, None], self.map_memory)
        self.map_last_t = self.t.clone()
        self.map_last_xy = xy.clone()
        # Runtime Estimator samples by floor, not interpolation of thresholded cells.
        cell = torch.floor((xy[:, None]+self.grid_off[None]-center[:, None])/MEM_CELL).long()+MEM_G//2
        col, row = cell[..., 0], cell[..., 1]
        inside = (col >= 0) & (col < MEM_G) & (row >= 0) & (row < MEM_G)
        sample = self.map_memory[:, 0][torch.arange(self.n, device=self.dev)[:, None],
                                       row.clamp(0, MEM_G-1), col.clamp(0, MEM_G-1)]
        sample = torch.where(inside, sample, .5)
        return torch.where(sample > .6, 1., torch.where(sample < .4, 0., .5)).reshape(self.n, G, G)

    def observe(self, clean=False):
        """clean=True gives the true occupancy (no walls, distractors, scale error or noise)."""
        if not clean and self.base.observation_memory:
            occ = self._remembered_grid()
        else:
            px, py = self.observation_points(clean)
            occ = self.on_road(px, py).float() if clean else self._camera_occupancy(px, py)
            occ = occ.view(self.n, G, G)
        occ = torch.where(self.vis[None], occ, torch.full_like(occ, 0.5))
        th, speed, boost, effective = self.th, self.speed_now, self.boost, self.eff
        if not clean and self.base.observation_window > 0:
            # Fractional box filter avoids rounding measured 0.18--0.24 s windows.
            age = torch.arange(self.velocity_history.shape[1], device=self.dev)*self.dt
            window = self.obs_window
            if self.base.startup_duration > 0 and getattr(self, 'startup_elapsed', 0.) is not None:
                # The deployed estimator has only post-GO history at race start.
                # Mid-race sections retain their prefilled steady history.
                window = torch.where(self.case_active, window, torch.minimum(window, self.t.clamp(min=self.dt)))
            weights = ((window[:, None]-age)/self.dt).clamp(0, 1)
            velocity = (self.velocity_history*weights[..., None]).sum(1)/weights.sum(1).clamp(min=1e-6)[:, None]
            th = torch.atan2(velocity[:, 0], velocity[:, 1]).clamp(0, np.pi/2)
            speed = velocity.norm(dim=1)
            boost = self.obs_boost
            age = torch.arange(self.H, device=self.dev)*self.dt
            delay = torch.where(self.hist > 0, self.base.observed_press_delay, self.base.observed_release_delay)
            eligible = age[None] >= delay
            first = eligible.float().argmax(1)
            effective = self.hist.gather(1, first[:, None])[:, 0]
        if (not clean and self.startup_profiles.numel()
                and getattr(self, 'startup_elapsed', 0.) is not None):
            end = (self.startup_profiles.shape[1]-1)*self.base.startup_profile_dt
            index = ((self.t+self.startup_profile_offset)/self.base.startup_profile_dt).clamp(
                0, self.startup_profiles.shape[1]-1)
            lo = index.floor().long()
            hi = (lo+1).clamp(max=self.startup_profiles.shape[1]-1)
            a = self.startup_profiles[self.startup_profile_id, lo]
            b = self.startup_profiles[self.startup_profile_id, hi]
            ratio = (a+(b-a)*(index-lo)+self.startup_profile_bias).clamp(min=0)
            speed = speed*torch.where((self.t < end) & ~self.case_active, ratio, 1.)
        turn_prog = th / (np.pi / 2)
        state = torch.stack([torch.sin(th), torch.cos(th), self.pressed, effective,
                             speed / 1000.0, boost / 0.5, torch.clamp(self.since, max=2.0),
                             turn_prog], 1)
        return occ[:, None], state
