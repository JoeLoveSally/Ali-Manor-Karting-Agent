"""Drive one race from a verified start dialog. Save frames, timing and actions.

The green GO cue starts estimation afresh; intro/countdown motion never enters
its map. Device video PTS is the estimator clock; host monotonic time measures
processing and command sends. Leaves the race's result page untouched; --prepare
can claim an existing ordinary reward to enter a new race before starting.

Usage: python tools/drive.py models --serial DEVICE_SERIAL [--prepare] [--dry]
"""
import argparse
import hashlib
import io
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kart.device import Scrcpy
from kart.estimator import Estimator
from kart.hud import go_score
from kart.ppo import ActorCritic, load_params
from kart.ui import outcome, preparation_action

START_XY = (0.5, 0.694)
CUT = 40.0


def thumb(frame):
    return cv2.resize(frame, (90, 200), interpolation=cv2.INTER_AREA).astype(np.float32)


def start_page(frame):
    h, w = frame.shape[:2]
    b = frame[int(0.677*h):int(0.71*h), int(0.32*w):int(0.68*w)].astype(int)
    green = ((b[..., 1] > b[..., 2]+40) & (b[..., 1] > b[..., 0]+40)).mean()
    t = frame[int(0.25*h):int(0.28*h), int(0.15*w):int(0.3*w)].astype(int)
    blue = (t[..., 0] > t[..., 2]+60).mean()
    return green > 0.6 and blue > 0.6, {"green": round(float(green), 2), "blue": round(float(blue), 2)}


def drive(a):
    out = Path(a.out) if a.out else Path('runs') / time.strftime('drive-%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    params = replace(load_params(), loop_delay=a.loop_delay)
    checkpoint = Path(a.run) / a.checkpoint
    model_bytes = checkpoint.read_bytes()
    net = ActorCritic()
    net.load_state_dict(torch.load(io.BytesIO(model_bytes), map_location='cpu', weights_only=True))
    net.eval()
    (out/'policy.pt').write_bytes(model_bytes)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    est = Estimator(params)
    rows, frames_log, grids = [], [], []
    meta = {"run": a.run, "checkpoint": a.checkpoint, "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
            "loop_delay": a.loop_delay, "clock": "device_pts", "start_trigger": "green_go",
            "stop_reason": "not_started", "outcome": "unknown", "controller": "actor_critic"}
    action_script = None
    script_after = 0.
    reference_clock = None
    script_shift = 0.
    if getattr(a, 'action_script', None):
        script_path = Path(a.action_script)
        script = json.loads(script_path.read_text())
        action_script = np.asarray(script['events'], dtype=float)
        if (script.get('clock') != 'elapsed_from_green_go' or action_script.ndim != 2 or
                action_script.shape[1] != 2 or not len(action_script) or
                not np.isfinite(action_script).all() or action_script[0, 0] != 0 or
                np.any(np.diff(action_script[:, 0]) <= 0) or
                not np.isin(action_script[:, 1], [0, 1]).all()):
            raise ValueError('Invalid calibration action script')
        meta.update(controller='calibration_action_script', action_script=script,
                    action_script_sha256=hashlib.sha256(script_path.read_bytes()).hexdigest())
        if 'first_switch_v' in script:
            raise ValueError('Odometry-aligned calibration is unsupported: startup flow can miss motion. '
                             'Use a timed script and validate its alignment from recorded road images.')
        script_after = float(script.get('policy_until', 0.))
        if not np.isfinite(script_after) or script_after < 0:
            raise ValueError('Invalid calibration intervention start time')
        if 'first_switch_reference' in script:
            from tools.reference_start import ReferenceStart
            reference_path = Path(script['first_switch_reference'])
            reference_clock = ReferenceStart(reference_path)
            if (len(action_script) < 2 or script_after != 0 or
                    abs(reference_clock.switch_time-action_script[1, 0]) > 1e-3):
                raise ValueError('Reference first switch does not match calibration script')
            (out/'reference-start.npz').write_bytes(reference_path.read_bytes())
            meta['reference_start_sha256'] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    source_root = Path(__file__).resolve().parent.parent
    source_files = list((source_root/'kart').glob('*.py'))+[Path(__file__).resolve()]
    if reference_clock is not None:
        source_files.append(Path(__file__).with_name('reference_start.py').resolve())
    meta['source_sha256'] = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(source_files)}
    assets = [source_root/'artifacts/sim_params.json', source_root/'artifacts/sky.png',
              *sorted((source_root/'artifacts/ui').glob('*.png'))]
    meta['asset_sha256'] = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in assets}
    f = None
    try:
        with Scrcpy(a.serial, out) as cam:
            for _ in range(20):
                f, _, _ = cam.read()
            if f.shape[:2] != (1600, 720):
                raise ValueError(f'Expected 720x1600 frames, got {f.shape}')
            if a.prepare and not a.dry:
                meta['navigation'] = []
                deadline, last_tap = time.monotonic()+10, -1e9
                while time.monotonic() < deadline:
                    if start_page(f)[0]:
                        break
                    action = preparation_action(f)
                    if action is not None and time.monotonic()-last_tap > 2:
                        name, point = action
                        if name not in meta['navigation']:
                            meta['navigation'].append(name)
                            cam.tap(*point)
                            last_tap = time.monotonic()
                    f, _, _ = cam.read()
            cv2.imwrite(str(out / 'start.png'), f)
            ok, why = start_page(f)
            meta['start_check'] = why
            print('start page:', ok, why, flush=True)
            if not ok or a.dry:
                meta['stop_reason'] = 'dry' if a.dry else 'start_page_not_verified'
                return out
            cam.tap(*START_XY)
            t_tap = time.monotonic()
            t_start = start_pts = None
            last_dec, still_since = -1., None
            cmd, prev_th = False, None
            with (out / 'decisions.jsonl').open('w') as journal:
                while time.monotonic() - t_tap < a.max_time:
                    f, recv, pts = cam.read()
                    th = thumb(f)
                    cut = prev_th is not None and float(np.abs(th-prev_th).mean()) > CUT
                    prev_th = th
                    if t_start is None:
                        score = go_score(f)
                        if score > 1500:
                            t_start, start_pts = recv, pts
                            meta.update(start_pts=pts, start_received=recv, tap_to_go=recv-t_tap, go_score=score)
                            cv2.imwrite(str(out / 'go.png'), f)
                            est.reset()
                            print(f'GO detected {recv-t_tap:.2f}s after tap', flush=True)
                        else:
                            if recv-t_tap > 12:
                                meta['stop_reason'] = 'go_not_detected'
                                break
                            continue
                    if cut:
                        meta['stop_reason'] = 'scene_cut'
                        break
                    # Timer and capture intervals live on the device clock, independent of delivery jitter.
                    est.update(f, pts, cmd)
                    dd = est.last_d
                    frames_log.append([round(recv-t_start, 4), pts] +
                                      ([round(float(x), 3) for x in dd[:3]] if dd is not None else [None]*3) + [int(cmd)])
                    if est.speed() < 300 and pts-start_pts > 2:
                        still_since = pts if still_since is None else still_since
                        if pts-still_since > 1:
                            meta['stop_reason'] = 'car_stopped'
                            break
                    else:
                        still_since = None
                    # Collect a short motion history before the first decision; GO occurs at game t~0.
                    if pts-start_pts < 0.12 or pts-last_dec < 1/30-0.004:
                        continue
                    last_dec = pts
                    t0 = time.monotonic()
                    grid, state = est.observe()
                    with torch.no_grad():
                        logits, _ = net(grid, state)
                    p = float(torch.softmax(logits, 1)[0, 1])
                    if action_script is not None and pts-start_pts >= script_after:
                        elapsed = pts-start_pts
                        if reference_clock is not None:
                            match = reference_clock.match(f)
                            if match['ready']:
                                script_shift = elapsed-match['phase']
                                meta['reference_alignment'] = {**match, 'elapsed': elapsed,
                                                               'shift_s': float(script_shift)}
                                reference_clock = None
                            elif elapsed > 2.:
                                meta['stop_reason'] = 'reference_start_not_matched'
                                break
                        index = np.searchsorted(action_script[:, 0], elapsed-script_shift+1e-4, side='right')-1
                        p = float(action_script[max(index, 0), 1]) if reference_clock is None else 0.
                    cmd = p > 0.5
                    changed = cam.set_pressed(cmd)
                    sent = time.monotonic()
                    grids.append((grid[0, 0].numpy()*2).astype(np.uint8))
                    row = {"recv": round(recv-t_start, 4), "pts": pts, "sent": round(sent-t_start, 4),
                           "elapsed": round(pts-start_pts, 4), "decide_ms": round(1000*(sent-t0), 1),
                           "age_ms": round(1000*(sent-recv), 1), "p": round(p, 4), "cmd": int(cmd),
                           "changed": changed, "heading": round(np.degrees(est.heading()), 1),
                           "speed": round(est.speed()), "pos": [round(float(v), 1) for v in est.pos],
                           "airborne": bool(est.airborne), "boost": int(state[0, 5] > 0),
                           "motion_valid": bool(dd is not None and dd[3] >= 20 and dd[4] < 2),
                           "flow_inliers": int(dd[3]) if dd is not None else 0,
                           "vertical_shift": round(float(dd[2]), 2) if dd is not None else None}
                    rows.append(row)
                    journal.write(json.dumps(row)+'\n')
                    journal.flush()
                    if changed:
                        print(f'{row["elapsed"]:.2f}s {"B press" if cmd else "A release"} p={p:.2f} heading={row["heading"]} speed={row["speed"]}', flush=True)
                else:
                    meta['stop_reason'] = 'max_time'
                cam.set_pressed(False)
                # Keep the final animation/result as evidence, without clicking its buttons.
                deadline = time.monotonic()+2.0
                while time.monotonic() < deadline:
                    f, _, _ = cam.read()
    except BaseException as exc:
        meta['stop_reason'] = 'error'
        meta['error'] = repr(exc)
        raise
    finally:
        if f is not None:
            cv2.imwrite(str(out/'end.png'), f)
            meta['outcome'] = outcome(f)
        meta['race_time'] = rows[-1]['elapsed'] if rows else 0
        (out/'log.json').write_text(json.dumps({'meta': meta, 'rows': rows, 'frames': frames_log}, indent=1)+'\n')
        if grids:
            np.savez_compressed(out/'grids.npz', grids=np.stack(grids))
        print(f'{len(rows)} decisions, {meta["stop_reason"]} -> {out}', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--action-script', help='timestamped open-loop calibration inputs; never a promoted policy')
    ap.add_argument('run')
    ap.add_argument('--checkpoint', default='best.pt')
    ap.add_argument('--serial', required=True, help='ADB serial from adb devices')
    ap.add_argument('--loop-delay', type=float, default=0.11)
    ap.add_argument('--max-time', type=float, default=75.)
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--out')
    ap.add_argument('--prepare', action='store_true', help='return from a verified result/main page before starting')
    drive(ap.parse_args())


if __name__ == '__main__':
    main()
