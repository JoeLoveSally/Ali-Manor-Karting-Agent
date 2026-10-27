"""Replay recorded touch sequences in the simulator on the reconstructed maps.

For many start times, initialise the sim at the reconstructed position/heading,
feed the recorded touches for `horizon` seconds and compare with the
reconstructed trajectory. Also reports whether the sim keeps the car on the road.
"""
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kart.sim import Params, Sim, Track  # noqa: E402

RELEASE_SHIFT = 0.125  # marker fade; actual drive logs need no marker correction


def load(name):
    m = np.load(name)
    P = np.where(m["seen"] > 0, m["road"] / np.maximum(m["seen"], 1), 0)
    road = P > 0.5
    pos, t, touch = m["pos"], m["t"], m["touch"].astype(bool)
    org, res = m["origin"], float(m["res"])
    known = (m['seen'] >= 3) & ((P <= .2) | (P >= .8))
    return Track(road, org, res, known=known), pos, t, touch


def touch_at(t, touch, tq):
    k = np.searchsorted(t, tq, side="right") - 1
    k2 = np.searchsorted(t, tq + RELEASE_SHIFT, side="right") - 1
    # a press counts from its marker; a release from RELEASE_SHIFT before the marker drops
    return bool(touch[max(k, 0)] and touch[max(k2, 0)])


def smooth_heading(pos, t, k, w=0.05):
    a, b = np.searchsorted(t, t[k] - w), np.searchsorted(t, t[k] + w)
    d = pos[min(b, len(pos) - 1)] - pos[max(a, 0)]
    return np.arctan2(d[0], d[1])


def evaluate(params, horizon=1.5, files=None, every=0.5):
    errs, alive = [], []
    for f in files or sorted(glob.glob("data/maps/*.npz")):
        if "173101" in f:
            continue
        track, pos, t, touch = load(f)
        t0s = np.arange(t[0] + 3.5, t[-1] - horizon - 0.5, every)
        for t0 in t0s:
            k = np.searchsorted(t, t0)
            th = smooth_heading(pos, t, k)
            if not (abs(th) < 0.05 or abs(th - np.pi / 2) < 0.05):
                continue  # start on a straight so the initial state is unambiguous
            sim = Sim(track, params, start=pos[k], theta=0.0 if abs(th) < 0.05 else np.pi / 2)
            sim.s.pressed = sim.s.effective = th > 0.7
            n = int(horizon / params.dt)
            ok = True
            for j in range(n):
                ok = sim.step(touch_at(t, touch, t[k] + j * params.dt)) and ok
            ke = np.searchsorted(t, t[k] + n * params.dt)
            if ke < len(pos):
                errs.append(np.linalg.norm(sim.s.pos - pos[ke]))
                alive.append(np.nan if sim.s.unknown else ok)
    return np.array(errs), np.array(alive)


if __name__ == "__main__":
    e, a = evaluate(Params())
    print(f"segments {len(e)}  pos err pct(5,25,50,75,95) {np.percentile(e, [5, 25, 50, 75, 95]).round(0)}"
          f"  known-map survival {np.nanmean(a):.3f}; unknown {np.isnan(a).mean():.3f}")
