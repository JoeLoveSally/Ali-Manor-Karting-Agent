"""Independent simulator starting-state coverage; no action labels or demonstrations."""
from dataclasses import replace

import numpy as np
import torch

KINDS = ('random', 'reconstructed', 'ramp', 'bends', 'relief')


def local_straight(env, indices, kind, rng=None, stratified=False):
    """Recoverable A/B approaches, including both sides and unfinished heading response.

    Uses the existing shared physics and a local finish. Terrain support is never
    inferred from a driving trajectory. Stratification belongs to validation only.
    """
    rng = env.rng if rng is None else rng
    ids = indices.tolist()
    count = len(ids)
    if not count:
        return
    from .vecenv import jump_distance_bounds
    fields = {name: np.full((count, *getattr(env, name).shape[1:]), fill, np.float32)
              for name, fill in [('pos_r', 1e9), ('corners', 0), ('neg_r', 1e9),
                                 ('jumps', -1), ('ramps', -1), ('reliefs', -1), ('distract', 1e9)]}
    params = {k: v[indices].cpu().numpy() for k, v in env.P.items()}
    starts = np.zeros((count, 2), np.float32)
    angles, commands, finishes = np.empty(count, np.float32), np.empty(count, np.float32), np.empty((count, 2), np.float32)
    responses = env.air_responds[indices].cpu().numpy().copy()
    for j in range(count):
        axis = j % 2 if stratified else int(rng.integers(2))
        half = rng.uniform(200, 250)
        angle = rng.uniform(0, .6)
        lateral = rng.uniform(-.85, .85)*half
        speed, rate = params['speed'][j], params['turn_rate'][j]
        # With the current target already aligned, this much lateral distance
        # suffices to finish the initial heading response without leaving road.
        shift = speed*(1-np.cos(angle))/rate
        lateral = min(lateral, half-shift-12)
        starts[j, 1-axis] = lateral
        angles[j] = np.pi/2-angle if axis == 0 else angle
        commands[j] = float(axis == 0)
        if stratified:
            responses[j] = bool((j//2) % 2)
        if kind == 'ramp':
            sample = replace(env.base, speed=float(speed), boost_gain=float(params['boost_gain'][j]))
            minimum, maximum = jump_distance_bounds(sample)
            if minimum <= 170:
                raise ValueError('Ramp curriculum has no feasible jump distance')
            lip = rng.uniform(750, 1250)
            gap = rng.uniform(150, min(350, minimum-16-env.base.landing_inset))
            end = lip+maximum+500
            hole = [-half-8, lip, half+8, lip+gap]
            fields['neg_r'][j, 0] = hole if axis == 1 else np.asarray(hole)[[1, 0, 3, 2]]
            fields['jumps'][j, 0] = [axis, lip, lip+gap, -half, half, 0]
            fields['ramps'][j, 0] = [axis, lip-env.base.ramp_length, lip, -half, half, env.base.ramp_height]
        else:
            start = rng.uniform(450, 850)
            length = rng.uniform(.7, 1.3)*env.base.relief_length
            height = rng.choice([-1, 1])*rng.uniform(.5, 1.5)*env.base.relief_height
            end = start+length+600
            fields['reliefs'][j, 0] = [axis, start, start+length, -half, half, height]
        rect = [-half, -400, half, end+half]
        fields['pos_r'][j, 0] = rect if axis == 1 else np.asarray(rect)[[1, 0, 3, 2]]
        endpoint = np.zeros(2, np.float32);endpoint[axis] = end
        fields['corners'][j] = endpoint
        fields['corners'][j, 0] = 0
        finishes[j] = [axis, end]
    tensor = lambda v: torch.as_tensor(v, device=env.dev)
    for name, values in fields.items():
        getattr(env, name)[indices] = tensor(values)
    env.x[indices], env.y[indices] = tensor(starts[:, 0]), tensor(starts[:, 1])
    env.th[indices] = tensor(angles)
    env.case_active[indices] = True
    env.case_origin[indices] = tensor(starts)
    env.case_finish[indices] = tensor(finishes)
    env.pressed[indices] = env.eff[indices] = tensor(commands)
    env.hist[indices] = tensor(commands)[:, None]
    env.since[indices] = 2
    env.startup_age[indices] = env.base.startup_duration
    env.speed_now[indices] = env.P['speed'][indices]
    env.velocity_history[indices, :, 0] = (env.speed_now[indices]*tensor(np.sin(angles)))[:, None]
    env.velocity_history[indices, :, 1] = (env.speed_now[indices]*tensor(np.cos(angles)))[:, None]
    env.map_memory[indices] = .5
    env.map_last_xy[indices] = tensor(starts)
    env.map_last_t[indices] = -1
    env.episode_kind[indices] = KINDS.index(kind)
    if stratified:
        env.air_responds[indices] = tensor(responses)


def tight_bends(env, indices, rng):
    """Full routes with short/intermediate connectors, using ordinary route bookkeeping."""
    for i in indices.tolist():
        half = rng.uniform(200, 250)
        lengths = np.r_[rng.uniform(800, 1400), rng.uniform(180, 1200, env.n_segments-1)]
        points, rects = [np.zeros(2)], [[-half, -400, half, 0]]
        for k, length in enumerate(lengths):
            a = points[-1]
            b = a + ([0., length] if k % 2 == 0 else [length, 0.])
            rects.append([*np.minimum(a, b)-half, *np.maximum(a, b)+half])
            points.append(b)
        env.pos_r[i] = torch.as_tensor(np.asarray(rects), device=env.dev, dtype=torch.float32)
        env.corners[i] = torch.as_tensor(np.asarray(points), device=env.dev, dtype=torch.float32)
        for name, value in [('neg_r', 1e9), ('distract', 1e9), ('jumps', -1), ('ramps', -1), ('reliefs', -1)]:
            getattr(env, name)[i] = value
    env.episode_kind[indices] = KINDS.index('bends')


def apply_training_mix(env, indices):
    available = indices[~env.case_active[indices]]
    # Absolute reset probabilities: 30% ramps, 20% bends, 10% relief;
    # reconstructed-case probability is configured independently, rest random.
    draws = env.rng.random(len(available))*(1-env.case_probability)
    local_straight(env, available[torch.as_tensor(draws < .30, device=env.dev)], 'ramp')
    tight_bends(env, available[torch.as_tensor((draws >= .30) & (draws < .50), device=env.dev)], env.rng)
    local_straight(env, available[torch.as_tensor((draws >= .50) & (draws < .60), device=env.dev)], 'relief')
