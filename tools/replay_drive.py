"""Replay exactly the frames consumed by a live run, at their recorded video PTS.

Uses logged commands as input. Policy outputs are counterfactual after the first
disagreement and must never be reported as a new closed-loop driving result.
"""
import argparse
import hashlib
import io
import json
import sys
from dataclasses import replace
from pathlib import Path

import av
import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kart.estimator import Estimator
from kart.ppo import ActorCritic, load_params
from tools.replay_estimator import grid_image


def replay(run, output, snapshots=False):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    log = json.loads((run/'log.json').read_text())
    if log['meta'].get('clock') != 'device_pts':
        raise ValueError('Exact replay requires a run recorded with device_pts timing')
    times = [json.loads(s) for s in (run/'frames.jsonl').read_text().splitlines()]
    consumed = {f[1]: f for f in log['frames']}
    decisions = {r['pts']: r for r in log['rows']}
    net = ActorCritic()
    checkpoint = run/'policy.pt'
    if not checkpoint.exists():
        checkpoint = Path(log['meta']['run'])/log['meta'].get('checkpoint', 'best.pt')
    model_bytes = checkpoint.read_bytes()
    model_sha = hashlib.sha256(model_bytes).hexdigest()
    if model_sha != log['meta']['model_sha256']:
        raise ValueError('Checkpoint changed since this run; use the recorded policy.pt')
    net.load_state_dict(torch.load(io.BytesIO(model_bytes), map_location='cpu', weights_only=True))
    net.eval()
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    est = Estimator(replace(load_params(), loop_delay=log['meta']['loop_delay']))
    rows = []
    with av.open(str(run/'capture.h264')) as recording:
        for index, decoded in enumerate(recording.decode(video=0)):
            pts = times[index]['pts']
            if pts not in consumed:
                continue
            frame = decoded.to_ndarray(format='bgr24')
            est.update(frame, pts, bool(consumed[pts][-1]))
            if pts not in decisions:
                continue
            before = decisions[pts]
            grid, state = est.observe()
            with torch.no_grad():
                p = float(net(grid, state)[0].softmax(1)[0, 1])
            row = dict(elapsed=before['elapsed'], speed=est.speed(), heading=float(np.degrees(est.heading())),
                       airborne=bool(est.airborne), p=p, recorded_cmd=before['cmd'],
                       old_speed=before['speed'], old_heading=before['heading'], old_p=before['p'],
                       d=[float(v) for v in est.last_d] if est.last_d else None)
            rows.append(row)
            if snapshots and len(rows) % 8 == 1:
                im = np.full((800, 720, 3), 245, np.uint8)
                im[:, :360] = cv2.resize(frame, (360, 800))
                im[30:390, 360:] = grid_image(grid[0, 0].numpy())
                for y, s in enumerate([f't={row["elapsed"]:.2f} p={p:.2f}',
                                       f'heading={row["heading"]:.0f} v={row["speed"]:.0f}',
                                       f'air={est.airborne} recorded={before["cmd"]}']):
                    cv2.putText(im, s, (370, 430+35*y), 0, .55, (0, 0, 0), 1)
                cv2.imwrite(str(output/f'{len(rows):04}.png'), im)
    if len(rows) != len(decisions):
        raise ValueError(f'Missing decision frames: replayed {len(rows)} of {len(decisions)}')
    summary = {'source': str(run), 'decisions': len(rows), 'model_sha256': model_sha,
               'airborne_fraction': float(np.mean([r['airborne'] for r in rows])),
               'mean_speed_change': float(np.mean([abs(r['speed']-r['old_speed']) for r in rows])),
               'max_heading_change': float(max(abs(r['heading']-r['old_heading']) for r in rows)),
               'command_disagreements': sum((r['p']>.5) != bool(r['recorded_cmd']) for r in rows),
               'limitation': 'Recorded commands, not a new closed-loop race'}
    (output/'replay.json').write_text(json.dumps({'summary': summary, 'rows': rows}, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    p.add_argument('--out', required=True)
    p.add_argument('--snapshots', action='store_true')
    a = p.parse_args()
    replay(a.run, a.out, a.snapshots)
