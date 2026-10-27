"""Explicit, reconstruction-based planar cases mixed into random-track PPO."""
import json
from pathlib import Path

import numpy as np
import torch


def load_case(path):
    case = json.loads(Path(path).read_text())
    if case.get('schema') != 1 or case.get('approved_for_training') is not True:
        raise ValueError('Case requires explicit local-geometry training approval')
    rects, corners = np.asarray(case['rectangles'], float), np.asarray(case['corners'], float)
    start, finish = np.asarray(case['start'], float), np.asarray(case['finish'], float)
    if (rects.ndim != 2 or rects.shape[1] != 4 or corners.ndim != 2 or corners.shape[1] != 2
            or start.shape != (2,) or finish.shape != (2,) or finish[0] not in (0, 1)
            or not all(np.isfinite(x).all() for x in [rects, corners, start, finish])
            or np.any(rects[:, 2:] <= rects[:, :2]) or not 0 <= case['theta'] <= np.pi/2):
        raise ValueError('Invalid case geometry')
    return case


def apply_case(env, indices, case, perturb=True):
    """Keep visible geometry; collision clearance is applied separately by step()."""
    rects = np.asarray(case['rectangles'], float)
    corners = np.asarray(case['corners'], float)
    if len(rects) > env.pos_r.shape[1] or len(corners) > env.corners.shape[1]:
        raise ValueError('Too few segments for the supplied case')
    count = len(indices)
    if not count:
        return
    roads = np.full((count, *env.pos_r.shape[1:]), 1e9, np.float32)
    paths = np.empty((count, *env.corners.shape[1:]), np.float32)
    starts, finishes = np.empty((count, 2), np.float32), np.empty((count, 2), np.float32)
    angles, commands = np.empty(count, np.float32), np.empty(count, np.float32)
    sine, cosine = np.empty(count, np.float32), np.empty(count, np.float32)
    for i in range(count):
        mirror = perturb and env.randomize > 0 and env.rng.random() < .5
        r, c = rects.copy(), corners.copy()
        start = np.asarray(case['start'], float).copy()
        axis, goal = case['finish']
        theta = case['theta']
        if perturb:
            start += env.rng.uniform(-16, 16, 2)*min(env.randomize, 1.)
        if mirror:
            r, c, start = r[:, [1, 0, 3, 2]], c[:, ::-1].copy(), start[::-1].copy()
            theta, axis = np.pi/2-theta, 1-axis
        roads[i, :len(r)] = r
        paths[i] = c[-1]; paths[i, :len(c)] = c
        starts[i], finishes[i] = start, [axis, goal]
        angles[i], commands[i] = theta, float(theta > np.pi/4)
        sine[i], cosine[i] = np.sin(theta), np.cos(theta)
    tensor = lambda v: torch.as_tensor(v, device=env.dev)
    env.pos_r[indices], env.corners[indices] = tensor(roads), tensor(paths)
    for name, value in [('neg_r', 1e9), ('distract', 1e9), ('jumps', -1), ('ramps', -1), ('reliefs', -1)]:
        getattr(env, name)[indices] = value
    env.x[indices], env.y[indices], env.th[indices] = tensor(starts[:, 0]), tensor(starts[:, 1]), tensor(angles)
    env.case_active[indices] = True
    env.case_origin[indices], env.case_finish[indices] = tensor(starts), tensor(finishes)
    command = tensor(commands)
    env.pressed[indices] = env.eff[indices] = command
    env.hist[indices] = command[:, None]
    env.since[indices] = 2.
    env.startup_age[indices] = env.base.startup_duration
    env.speed_now[indices] = env.P['speed'][indices]
    env.velocity_history[indices, :, 0] = (env.speed_now[indices]*tensor(sine))[:, None]
    env.velocity_history[indices, :, 1] = (env.speed_now[indices]*tensor(cosine))[:, None]
    env.map_last_xy[indices] = env.case_origin[indices]
