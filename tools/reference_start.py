"""Image-based start alignment for system-identification action replays only."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kart.road import road_surface, small
from kart.scenes import sha256
from tools.track_ramp import frames


def features(frame):
    top, valid = road_surface(small(frame))
    return (cv2.resize(top.astype(np.float32), (90, 200), interpolation=cv2.INTER_AREA),
            cv2.resize(valid.astype(np.float32), (90, 200), interpolation=cv2.INTER_AREA) > .95)


class ReferenceStart:
    def __init__(self, path):
        with np.load(path) as src:
            self.t, self.top, self.valid = src['t'], src['top'], src['valid']
            self.switch_time = float(src['switch_time'])
        if (self.top.shape != self.valid.shape or self.top.shape != (len(self.t), 200, 90)
                or not np.isfinite(self.top).all() or not np.isfinite(self.t).all()
                or np.any(np.diff(self.t) <= 0) or not self.t[0] < self.switch_time < self.t[-1]):
            raise ValueError('Invalid reference-start atlas')

    def match(self, frame):
        top, valid = features(frame)
        known = self.valid & valid[None]
        counts = known.sum((1, 2))
        costs = (np.abs(self.top-top)*known).sum((1, 2))/np.maximum(counts, 1)
        costs[counts < 1000] = np.inf
        index = int(costs.argmin())
        return {'phase': float(self.t[index]), 'cost': float(costs[index]),
                'ready': bool(self.t[index]+1e-4 >= self.switch_time and costs[index] <= .025)}


def build(run, out):
    path = Path(run)
    log = json.loads((path/'log.json').read_text())
    switch = next(r['elapsed'] for r in log['rows'] if r['changed'])
    data = [(t, *features(f)) for t, f in frames(run, switch-.3, switch+.3)]
    np.savez_compressed(out, t=[r[0] for r in data], top=[r[1] for r in data],
                        valid=[r[2] for r in data], switch_time=switch)
    inputs = [path/f for f in ['capture.h264', 'frames.jsonl', 'log.json']]
    inputs += [Path(__file__), Path('kart/road.py'), Path('artifacts/sky.png')]
    Path(out).with_suffix('.json').write_text(json.dumps({
        'reference_run': run, 'switch_time': switch, 'data_sha256': sha256(out),
        'source_sha256': {str(p): sha256(p) for p in inputs},
        'scope': 'Same-level first straight; calibration alignment only'}, indent=2)+'\n')


def audit(atlas, runs, out):
    reference = ReferenceStart(atlas)
    results = []
    for run in runs:
        log = json.loads((Path(run)/'log.json').read_text())
        decisions = {round(r['pts']-log['meta']['start_pts'], 6) for r in log['rows']}
        found = None
        for t, frame in frames(run, .6, 1.6):
            if round(t, 6) not in decisions:
                continue
            match = reference.match(frame)
            if match['ready']:
                found = {'time': t, **match}
                break
        results.append({'run': run, 'alignment': found})
    Path(out).write_text(json.dumps({'results': results, 'atlas_sha256': sha256(atlas),
        'code_sha256': sha256(__file__),
        'scope': 'Offline first-switch registration; does not validate the rest of the action replay'}, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    b = sub.add_parser('build'); b.add_argument('run'); b.add_argument('out')
    a = sub.add_parser('audit'); a.add_argument('atlas'); a.add_argument('out'); a.add_argument('runs', nargs='+')
    args = p.parse_args()
    if args.mode == 'build':
        build(args.run, args.out)
    else:
        audit(args.atlas, args.runs, args.out)
