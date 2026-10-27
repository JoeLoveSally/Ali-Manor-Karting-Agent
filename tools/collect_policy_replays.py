"""Capture ten labelled best-policy examples through the original evaluator.

Selection is stratified by type/outcome (7 pass, 3 fail), not a success-rate
sample. All 512 episodes are replayed per suite to preserve batch/noise parity.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kart.ppo import ActorCritic, load_params, limit_cuda_memory
from kart.scenes import sha256
from kart.vecenv import VecKart
from tools.eval_survival import SUITES, rollout


FIELDS = ['t', 'x', 'y', 'theta', 'height', 'speed', 'command', 'effective', 'air',
          'boost', 'progress', 'observation', 'perfect', 'launch', 'land', 'fell',
          'finished', 'unsafe_landing']
LABELS = {'standard16': '普通路线', 'long30': '长路线', 'bends16': '密集连续弯', 'ramp': '坡道',
          'relief': '整幅起伏路面', 'middle16': '中等长度连续弯', 'level645': '645关失败弯段'}
QUOTAS = [('standard16', 3, 0), ('long30', 2, 1), ('bends16', 1, 1), ('ramp', 1, 1)]


class RecordedEnv:
    """Read-only observation hook; the unmodified rollout still drives VecKart."""
    def __init__(self, env, indices):
        self.env, self.indices = env, indices
        self.ids = torch.tensor(indices, device=env.dev)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def load_state_dict(self, state):
        self.env.load_state_dict(state)
        self.frames = [[] for _ in self.indices]
        self.grids = [[] for _ in self.indices]
        self.states = [[] for _ in self.indices]
        self.active = np.ones(len(self.indices), bool)
        self.capture()

    def capture(self, info=None):
        e = self.env
        base = [e.t, e.x, e.y, e.th, e.z, e.speed_now, e.pressed, e.eff,
                e.air.float(), e.boost, e.prog]
        values = torch.stack(base, 1)[self.ids].cpu().numpy()
        flags = ['perfect', 'launch', 'land', 'fell', 'finished', 'unsafe_landing']
        events = (torch.stack([info[k].float() for k in flags], 1)[self.ids].cpu().numpy()
                  if info else np.zeros((len(self.indices), len(flags))))
        for j in range(len(self.indices)):
            if self.active[j]:
                self.frames[j].append(values[j].tolist()+[max(0, len(self.grids[j])-1)]+events[j].tolist())

    def observe(self):
        grid, state = self.env.observe()
        images = (grid[self.ids, 0]*255).round().to(torch.uint8).cpu().numpy()
        states = state[self.ids].cpu().numpy()
        for j in range(len(self.indices)):
            if self.active[j]:
                self.grids[j].append(images[j])
                self.states[j].append(states[j].tolist())
                self.frames[j][-1][11] = len(self.grids[j])-1
        return grid, state

    def step(self, action):
        result = self.env.step(action)
        self.capture(result[2])
        self.active &= ~result[1][self.ids].cpu().numpy()
        return result


def collect(args):
    limit_cuda_memory(args.device,getattr(args,'gpu_memory_gb',0.))
    root = Path(args.evaluation)
    report = json.loads((root/'report.json').read_text())
    model_name = getattr(args, 'model_name', 'best')
    balanced = getattr(args, 'balanced', False)
    if sha256(args.model) != report['model_sha256'][model_name]:
        raise ValueError('Model is not the evaluated checkpoint')
    if sha256(args.params) != report['params_sha256']:
        raise ValueError('Evaluation parameters changed')
    from kart import vecenv, sim, ppo, projection3d, geometry, tracks, cases
    for module in [vecenv, sim, ppo, projection3d, geometry, tracks, cases]:
        if module is cases and module.__name__ not in report['source_sha256']:
            continue  # Compatibility with immutable pre-v3 evaluations.
        if sha256(module.__file__) != report['source_sha256'][module.__name__]:
            raise ValueError('Kernel changed: '+module.__name__)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    net = ActorCritic().to(args.device)
    net.load_state_dict(torch.load(args.model, map_location=args.device, weights_only=True))
    net.eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    selection = np.random.default_rng(args.seed)
    cases, images = [], {}
    selected = getattr(args, 'suites', None) or ['standard16', 'long30', 'bends16', 'ramp', 'relief']
    quotas = [(suite, 1, 1) for suite in selected] if balanced else QUOTAS
    for suite, successes, failures in quotas:
        expected = json.loads((root/f'{suite}-{model_name}.json').read_text())
        good = np.flatnonzero(np.asarray(expected['finished']) > 0)
        bad = np.flatnonzero(np.asarray(expected['fell']) > 0)
        if suite == 'ramp' and not balanced:
            bad = bad[np.asarray(expected['unsafe_landings'])[bad] > 0]
        if balanced:
            indices = ([int(selection.choice(good))] if len(good) else [])
            indices += ([int(selection.choice(bad))] if len(bad) else [])
            remaining = np.setdiff1d(np.arange(len(expected['finished'])), indices)
            indices += selection.choice(remaining, 2-len(indices), replace=False).tolist()
        else:
            indices = (selection.choice(good, successes, replace=False).tolist()
                       + selection.choice(bad, failures, replace=False).tolist())
        bank = root/f'{suite}-episodes.pt'
        if sha256(bank) != report['suites'][suite]['bank_sha256']:
            raise ValueError('Initial state bank changed')
        initial = torch.load(bank, map_location=args.device, weights_only=True)
        count = initial['tensors']['x'].numel()
        env = VecKart(count, device=args.device, params=load_params(args.params),
                      n_segments=SUITES[suite], max_time=report['config']['max_time'], loop_delay=(0, 0))
        if suite in ('ramp', 'relief'):
            env.startup_elapsed = None
        recorded = RecordedEnv(env, indices)
        result = rollout(recorded, net, initial, report['suites'][suite]['seed']+101,
                         report['config']['frame_skip'])
        # Check every original episode, not just the ten displayed examples.
        for key, value in expected.items():
            if key != 'summary' and not np.allclose(result[key], value, rtol=0, atol=1e-5):
                raise AssertionError('Replay differs from prior evaluation: '+suite+' '+key)
        for j, episode in enumerate(indices):
            ident = f'{len(cases)+1:02d}-{suite}-{episode}'
            tensors = initial['tensors']
            def array(name):
                return tensors[name][episode].cpu().numpy()
            holes, jumps, ramps, distract = array('neg_r'), array('jumps'), array('ramps'), array('distract')
            reliefs = array('reliefs') if 'reliefs' in tensors else np.empty((0, 6))
            case = dict(id=ident, number=len(cases)+1, suite=suite, label=LABELS[suite], episode_index=episode,
                        finished=bool(expected['finished'][episode]), seconds=expected['seconds'][episode],
                        timeout=bool(expected['timeout'][episode]), unknown=bool(expected['unknown'][episode]),
                        progress_fraction=expected['progress_fraction'][episode],
                        launches=int(expected['launches'][episode]), safe_landings=int(expected['safe_landings'][episode]),
                        unsafe_landings=int(expected['unsafe_landings'][episode]),
                        air_control_mode=('responsive' if bool(tensors.get('air_responds',
                            torch.zeros(count, dtype=torch.bool, device=args.device))[episode]) else 'latched')
                            if report['params'].get('air_turn_continues') else 'frozen_heading',
                        rects=array('pos_r').tolist(), holes=holes[holes[:, 0] < 1e8].tolist(),
                        jumps=jumps[jumps[:, 0] >= 0].tolist(), ramps=ramps[ramps[:, 0] >= 0].tolist(),
                        reliefs=reliefs[reliefs[:, 0] >= 0].tolist(),
                        distract=distract[distract[:, 0] < 1e8].tolist(), corners=array('corners').tolist(),
                        params={k: v[episode].item() for k, v in initial['P'].items()},
                        frames=recorded.frames[j], observed_states=recorded.states[j],
                        observation_count=len(recorded.grids[j]), evaluation_parity=True)
            cases.append(case)
            images[ident] = np.stack(recorded.grids[j])
        print(f'{suite}: {indices}; all {count} original episode records match', flush=True)
    selection_note = (f'{len(quotas)} suites, two examples each; one pass and one failure where available, '
                      'otherwise sampled without replacement. Not a success-rate sample.' if balanced else
                      'Stratified by suite/outcome: 7 passes, 3 failures; ramp failure sampled among '
                      'conservative landing rejections. Not a success-rate sample.')
    data = dict(model_sha256=sha256(args.model), params_sha256=sha256(args.params), fields=FIELDS,
                model_name=model_name, evaluation_envs=report['config']['envs'],
                selection_seed=args.seed, selection=selection_note,
                cases=cases, source_sha256={'collector': sha256(__file__),
                                           'evaluation_report': sha256(root/'report.json')})
    (out/'traces.json').write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':'))+'\n')
    np.savez_compressed(out/'observations.npz', **images)
    print(f'Saved {len(cases)} complete replays and actual model-input grids', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--suites', nargs='+', choices=LABELS)
    parser.add_argument('--model', required=True)
    parser.add_argument('--params', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--gpu-memory-gb',type=float,default=0.)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--model-name', default='best')
    parser.add_argument('--balanced', action='store_true', help='Two examples from each of five suites')
    collect(parser.parse_args())
