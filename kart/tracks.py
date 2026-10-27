"""Random tracks: axis-aligned rectangles alternating A (+v) and B (+u) directions.

Segment lengths include short zigzags, intermediate connectors and long straights;
the reconstructed level645 connector is about 829 units. Widths ~350-500. Grooves
lower the entire road width and retain support. Small rises use the same
continuous height representation with the opposite sign.
"""
import numpy as np

from .sim import Track

RES = 8.0


def sample_length(rng):
    draw = rng.random()
    if draw < 0.45:
        return rng.uniform(180, 620)
    if draw < 0.70:
        return rng.uniform(620, 1200)
    return rng.uniform(1200, 2600)


def random_track(rng, n_segments=40, width=None, groove_prob=0.15):
    """Return (Track, centerline points). The car starts at (0, 0) heading A."""
    pts = [np.zeros(2)]
    d = 1  # 1: A (+v), 0: B (+u)
    w = width if width is not None else rng.uniform(340, 500)
    segs = []
    for i in range(n_segments):
        L = rng.uniform(800, 1400) if i == 0 else sample_length(rng)
        step = np.array([0.0, L]) if d == 1 else np.array([L, 0.0])
        segs.append((pts[-1].copy(), pts[-1] + step, d))
        pts.append(pts[-1] + step)
        d = 1 - d
    P = np.array(pts)
    lo, hi = P.min(0) - 2 * w, P.max(0) + 2 * w
    nx, ny = int((hi[0] - lo[0]) / RES) + 1, int((hi[1] - lo[1]) / RES) + 1
    road = np.zeros((ny, nx), bool)

    def fill(x0, y0, x1, y1, val=True):
        c0, c1 = int((min(x0, x1) - lo[0]) / RES), int(np.ceil((max(x0, x1) - lo[0]) / RES))
        r0, r1 = int((min(y0, y1) - lo[1]) / RES), int(np.ceil((max(y0, y1) - lo[1]) / RES))
        road[max(r0, 0):r1, max(c0, 0):c1] = val

    h = w / 2
    for a, b, _ in segs:
        fill(min(a[0], b[0]) - h, min(a[1], b[1]) - h, max(a[0], b[0]) + h, max(a[1], b[1]) + h)
    # start pad behind the car
    fill(-h, -400, h, 0)
    reliefs = []
    for a, b, dd in segs[1:-1]:
        L = np.linalg.norm(b - a)
        if L < 900 or rng.random() > groove_prob:
            continue
        length = min(rng.uniform(.7, 1.3)*513, L-2*h-100)
        start = rng.uniform(h+50, L-h-length)
        reliefs.append([dd, a[dd]+start, a[dd]+start+length,
                        a[1-dd]-h, a[1-dd]+h, -rng.uniform(.5, 1.5)*83])
    return Track(road, lo, RES, reliefs=reliefs), P
