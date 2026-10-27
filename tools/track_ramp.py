"""Track the flat landing surface to recover camera motion through a ramp event.

Uses a metric pose fitted from visible corners. It does not assume all ramp
pixels are on the ground plane, and does not use the car trajectory as geometry.
"""
import argparse
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kart.projection3d import project, unproject_plane
from kart.scenes import sha256


def top_texture_mask(frame, region):
    """Reject darker sidewalls before selecting features on the flat landing.

    Median colour keeps the road grid texture eligible while excluding wall
    stripes whose individual bright pixels overlap the road-top palette.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = cv2.medianBlur(hsv[..., 1], 7)
    value = cv2.medianBlur(hsv[..., 2], 7)
    samples = (region > 0) & (saturation > 180) & (value >= 110) & (value < 220)
    if samples.sum() < 100:
        return np.zeros_like(region)
    hist = np.convolve(np.bincount(value[samples], minlength=256), np.ones(7), mode='same')
    peak = int(np.argmax(hist[110:220])+110)
    return ((region > 0) & (saturation > 180) & (value >= .95*peak)).astype(np.uint8)*255


def flat_approach_mask(shape, camera, end):
    """Conservative reviewed flat support, well behind the uncertain slope foot."""
    y, x = np.mgrid[0:shape[0]:4, 0:shape[1]:4]
    world = unproject_plane(np.c_[x.ravel(), y.ravel()], camera)
    valid = (np.isfinite(world).all(1) & (world[:, 1] < end) & (world[:, 1] > -2200)
             & (np.abs(world[:, 0]) < 2200)).reshape(x.shape)
    return cv2.resize(valid.astype(np.uint8)*255, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def frames(run, start, end):
    run = Path(run)
    log = json.loads((run/'log.json').read_text())
    times = [json.loads(s)['pts']-log['meta']['start_pts'] for s in (run/'frames.jsonl').read_text().splitlines()]
    with av.open(str(run/'capture.h264')) as stream:
        for index, f in enumerate(stream.decode(video=0)):
            if times[index] < start-.01:
                continue
            if times[index] > end:
                break
            yield times[index], f.to_ndarray(format='bgr24')


def main(a):
    cv2.setNumThreads(2)
    cfg = json.loads(Path(a.geometry).read_text())
    if a.flat_approach_end is not None and (cfg.get('origin') != 'lip' or a.flat_approach_end >= 0):
        raise ValueError('Flat approach support requires a lip-origin scene and a negative bound')
    cam = np.array(cfg['train_cameras'][0] if a.source == 'crossing' else cfg['validation_camera'])
    annotation = None
    if a.annotation:
        from tools.fit_ramp_geometry import landmarks
        if cfg.get('landmark_model') == 'lip_and_landing':
            from tools.fit_ramp_landmarks import landmarks
        annotation = json.loads(Path(a.annotation).read_text())
        points = (landmarks(cfg['shape']) if cfg.get('landmark_model') == 'lip_and_landing' else
                  landmarks([cfg['shape'][k] for k in ['width', 'length', 'height', 'gap']]))
        names = [k for k in annotation['points'] if k in points]
        if a.initial_height is not None:
            result = least_squares(lambda xy: (project([points['land_right']],
                [*xy, a.initial_height])-annotation['points']['land_right']).ravel(), cam[:2])
            candidate = np.r_[result.x, a.initial_height]
        else:
            result = least_squares(lambda c: (project([points[k] for k in names], c)-
                                   np.array([annotation['points'][k] for k in names])).ravel(), cam)
            candidate = result.x
        error = project([points[k] for k in names], candidate)-[annotation['points'][k] for k in names]
        if not result.success or np.max(np.linalg.norm(error, axis=1)) > 10:
            raise ValueError('Initial landmark pose does not fit the fixed geometry')
        cam = candidate
    lip_y = (0 if cfg.get('origin') == 'lip' else cfg['shape']['length'])+cfg['shape']['gap']
    first, pts, world = None, None, None
    rows = []
    stopped = None
    for t, frame in frames(a.run, a.start, a.end):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if first is None:
            mask = np.zeros_like(gray)
            # The landing's visible flat top is bounded by its front edge.
            polygon = np.array([[0, 256], [215, 333], [0, 459]]) if a.source == 'crossing' else np.array([[0, 236], [312, 346], [90, 488], [0, 470]])
            if annotation:
                if abs(t-annotation['time']) > .02:
                    raise ValueError('Annotation and first decoded frame are not aligned')
                polygon = np.array(annotation['landing_polygon'], np.int32)
            cv2.fillConvexPoly(mask, polygon, 255)
            if a.flat_approach_end is not None:
                mask |= flat_approach_mask(gray.shape, cam, a.flat_approach_end)
            mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
            mask = top_texture_mask(frame, mask)
            mask[:330] = 0
            mask[540:920, 270:590] = 0
            mask[1150:, 380:] = 0
            features = cv2.goodFeaturesToTrack(gray, 250, .001, 5, mask=mask)
            if features is None or len(features) < 20:
                stopped = {'t': t, 'reason': 'insufficient_initial_features'}
                break
            pts = features[:, 0]
            world = np.c_[unproject_plane(pts, cam), np.zeros(len(pts))]
            first = t
            prev = gray
            rows.append(dict(t=t, camera=cam.tolist(), count=len(pts), rms=0.))
            continue
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts[:, None], None, winSize=(25, 25), maxLevel=4)
        back, sb, _ = cv2.calcOpticalFlowPyrLK(gray, prev, nxt, None, winSize=(25, 25), maxLevel=4)
        nxt, back = nxt[:, 0], back[:, 0]
        good = (st[:, 0] > 0) & (sb[:, 0] > 0) & (np.linalg.norm(back-pts, axis=1) < .8)
        good &= (nxt[:, 0] > 10) & (nxt[:, 0] < 710) & (nxt[:, 1] > 300) & (nxt[:, 1] < 1550)
        good &= ~((nxt[:, 0] > 270) & (nxt[:, 0] < 590) & (nxt[:, 1] > 540) & (nxt[:, 1] < 920))
        pts, world = nxt[good], world[good]
        if len(pts) < 20:
            stopped = {'t': t, 'reason': 'insufficient_tracked_features', 'count': len(pts)}
            break
        result = least_squares(lambda c: (project(world, c)-pts).ravel(), cam,
                               loss='soft_l1', f_scale=1.5, max_nfev=80)
        candidate = result.x
        err = np.linalg.norm(project(world, candidate)-pts, axis=1)
        good = err < 4
        dt = t-rows[-1]['t']
        # A low residual on a few collapsed points cannot identify camera pose.
        # These are rejection bounds, not forces that pull the camera onto a path.
        if (not result.success or good.sum() < 20 or
                np.linalg.norm(candidate-cam) > 3500*dt+10):
            stopped = {'t': t, 'reason': 'unidentifiable_pose', 'inliers': int(good.sum())}
            break
        cam = candidate
        rows.append(dict(t=t, camera=cam.tolist(), count=int(good.sum()),
                         rms=float(np.sqrt(np.mean(err[good]**2)))))
        pts, world = pts[good], world[good]
        if len(pts) < 120:
            # Reseed only the reconstructed flat landing, never the rising ramp.
            w = cfg['shape']['width']
            polygon = project([[0, lip_y+20, 0], [w, lip_y+20, 0],
                               [w, lip_y+1800, 0], [0, lip_y+1800, 0]], cam).astype(np.int32)
            mask = np.zeros_like(gray)
            cv2.fillConvexPoly(mask, polygon, 255)
            if a.flat_approach_end is not None:
                mask |= flat_approach_mask(gray.shape, cam, a.flat_approach_end)
            mask = cv2.erode(mask, np.ones((15, 15), np.uint8))
            mask = top_texture_mask(frame, mask)
            mask[:330] = 0
            mask[540:920, 270:590] = 0
            mask[1150:, 380:] = 0
            for xy in pts:
                cv2.circle(mask, tuple(xy.astype(int)), 10, 0, -1)
            new = cv2.goodFeaturesToTrack(gray, 150-len(pts), .003, 8, mask=mask)
            if new is not None:
                new = new[:, 0]
                pts = np.concatenate([pts, new])
                world = np.concatenate([world, np.c_[unproject_plane(new, cam), np.zeros(len(new))]])
        prev = gray
    out = Path(a.out)
    inputs = [Path(a.run)/p for p in ['capture.h264', 'frames.jsonl', 'log.json']]
    inputs += [Path(a.geometry), Path(__file__), Path('kart/projection3d.py')]
    if a.annotation:
        inputs.append(Path(a.annotation))
    out.write_text(json.dumps({'run': a.run, 'rows': rows, 'stopped': stopped,
                              'initial_height': a.initial_height,
                              'flat_approach_end': a.flat_approach_end,
                              'feature_surface': 'median-colour top palette; sidewalls rejected',
                              'source_sha256': {str(p): sha256(p) for p in inputs},
                              'limitation': 'camera pose is not car height; later samples are unknown'}, indent=2)+'\n')
    print('samples', len(rows), 'last', rows[-1] if rows else None, 'stopped', stopped)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    p.add_argument('--geometry', default='artifacts/diagnostics/sim-gap-closure/ramp-fit.json')
    p.add_argument('--source', choices=['crossing', 'failure'], required=True)
    p.add_argument('--annotation', help='Independent first-frame landmarks and flat landing polygon')
    p.add_argument('--initial-height', type=float,
                   help='Known initial camera height; pose uses landing corner only, lips are held out')
    p.add_argument('--flat-approach-end', type=float,
                   help='Reviewed conservative flat region v <= bound, well behind the uncertain slope foot')
    p.add_argument('--start', type=float, required=True)
    p.add_argument('--end', type=float, required=True)
    p.add_argument('--out', required=True)
    main(p.parse_args())
