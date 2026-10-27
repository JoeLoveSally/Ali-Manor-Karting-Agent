"""Known game UI evidence, calibrated on captured 720x1600 frames.

Only the ordinary claim and drift-mode entry are used to prepare another race.
No advertisement, purchase, revival, or unrecognised page is clicked.
"""
from pathlib import Path

import cv2
_ROOT = Path(__file__).resolve().parent.parent/'artifacts'/'ui'
_REGIONS = {'success': (240, 755, 480, 845),
            'failure': (240, 755, 480, 845), 'revive': (210, 720, 500, 855),
            'ordinary_claim': (250, 1290, 470, 1370),
            'drift_mode': (365, 775, 705, 915)}
_TEMPLATES = {k: cv2.imread(str(_ROOT/f'{k}.png'), cv2.IMREAD_GRAYSCALE) for k in _REGIONS}


def score(frame, name):
    if frame.shape[:2] != (1600, 720):
        return 0.
    x0, y0, x1, y1 = _REGIONS[name]
    patch = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    template = _TEMPLATES[name]
    if template is None:
        raise FileNotFoundError(_ROOT/f'{name}.png')
    return float(cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED).max())


def preparation_action(frame):
    if outcome(frame) in ('success', 'failure') and score(frame, 'ordinary_claim') > .90:
        return 'ordinary_claim', (.5, .83)
    if score(frame, 'drift_mode') > .90:
        return 'drift_mode', (.74, .53)
    return None


def outcome(frame):
    success = score(frame, 'success') > .90
    failure = max(score(frame, 'failure'), score(frame, 'revive')) > .90
    if success and not failure:
        return 'success'
    if failure and not success:
        return 'failure'
    return 'unknown'
