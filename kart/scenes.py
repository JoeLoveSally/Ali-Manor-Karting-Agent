"""Versioned reconstruction artifacts. Camera occupancy is a draft, not ground truth."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .sim import Track


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load_scene(directory, training=False):
    directory = Path(directory)
    meta = json.loads((directory/'scene.json').read_text())
    if meta.get('schema') != 1:
        raise ValueError('Unsupported scene schema')
    if sha256(directory/'scene.npz') != meta['data_sha256']:
        raise ValueError('Scene data hash mismatch')
    if training and not meta['training_ready']:
        raise ValueError('Scene is a reconstruction draft: ' + '; '.join(meta['blockers']))
    with np.load(directory/'scene.npz') as src:
        data = {k: src[k] for k in src.files}
    track = Track(data['road'], data['origin'], float(data['res']),
                  known=data['known'], jumps=data['jumps'], ramps=data.get('ramps', ()),
                  reliefs=data.get('reliefs', ()))
    return track, data, meta
