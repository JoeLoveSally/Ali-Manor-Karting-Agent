"""Paired deterministic policy evaluation on frozen, unseen simulator episodes.

Uses the deployed argmax action rule and the shared noisy/lagged observation
and physics kernel. No training, device access, or deployment occurs here.
"""
import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kart.ppo import ActorCritic, load_params, limit_cuda_memory
from kart.scenes import sha256
from kart.vecenv import VecKart, jump_distance_bounds


SUITES = {'standard16': 16, 'long30': 30, 'bends16': 16, 'ramp': 1, 'relief': 1,
          'middle16': 16, 'level645': 4}


def prepare(env, suite, seed, robust=False):
    """Keep production random tracks, or build explicitly labelled stress cases."""
    if suite not in ('bends16', 'middle16', 'ramp', 'relief'):
        return
    if robust and suite in ('ramp', 'relief'):
        from kart.curriculum import local_straight
        local_straight(env, torch.arange(env.n, device=env.dev), suite,
                       rng=np.random.default_rng(seed), stratified=True)
        return
    rng = np.random.default_rng(seed)
    env.neg_r.fill_(1e9)
    env.jumps.fill_(-1)
    env.ramps.fill_(-1)
    env.reliefs.fill_(-1)
    env.distract.fill_(1e9)
    for i in range(env.n):
        half = rng.uniform(200, 250)
        if suite in ('bends16', 'middle16'):
            lengths = np.r_[rng.uniform(800, 1400), rng.uniform(
                180 if suite == 'bends16' else 620, 620 if suite == 'bends16' else 1200, env.n_segments-1)]
        elif suite == 'relief':
            start = rng.uniform(450, 750)
            length = env.base.relief_length*rng.uniform(.7, 1.3)
            height = env.base.relief_height*rng.uniform(.5, 1.5)*rng.choice([-1, 1])
            lengths = [start+length+600]
            env.reliefs[i, 0] = torch.tensor([1, start, start+length, -half, half, height], device=env.dev)
            env.x[i] = rng.uniform(-.6, .6)*half
            env.th[i] = rng.uniform(0, .15)
        else:
            p = env.base
            speed, gain = env.P['speed'][i].item(), env.P['boost_gain'][i].item()
            from dataclasses import replace
            bounds = jump_distance_bounds(replace(p, speed=speed, boost_gain=gain))
            lip = rng.uniform(850, 1050)
            lengths = [lip+bounds[1]+500]
            gap = rng.uniform(200, min(350, bounds[0]-p.landing_inset-16))
            env.neg_r[i, 0] = torch.tensor([-half-8, lip, half+8, lip+gap], device=env.dev)
            env.jumps[i, 0] = torch.tensor([1, lip, lip+gap, -half, half, 0], device=env.dev)
            env.ramps[i, 0] = torch.tensor([1, lip-p.ramp_length, lip, -half, half, p.ramp_height], device=env.dev)
            # Recoverable left-edge approaches, with a partially completed B->A
            # turn in some episodes. This is a synthetic local stress case.
            # Ground clearance must not make the stress suite dead on reset.
            inset = env.base.ground_inset
            env.x[i] = -half+rng.uniform(max(8, inset+8), max(80, inset+80))
            env.th[i] = rng.uniform(0, np.pi/2)
        points, rects = [np.zeros(2)], [[-half, -400, half, 0]]
        for k, length in enumerate(lengths):
            a = points[-1]
            b = a+([0., length] if k % 2 == 0 else [length, 0.])
            rects.append([min(a[0], b[0])-half, min(a[1], b[1])-half,
                          max(a[0], b[0])+half, max(a[1], b[1])+half])
            points.append(b)
        env.pos_r[i] = torch.as_tensor(np.asarray(rects), dtype=torch.float32, device=env.dev)
        env.corners[i] = torch.as_tensor(np.asarray(points), dtype=torch.float32, device=env.dev)
    if suite in ('ramp', 'relief'):
        env.startup_elapsed = None
        env.startup_age.fill_(env.base.startup_duration)
        env.speed_now.copy_(env.P['speed'])
        env.velocity_history[:, :, 0] = (env.speed_now*torch.sin(env.th))[:, None]
        env.velocity_history[:, :, 1] = (env.speed_now*torch.cos(env.th))[:, None]
        env.map_last_xy[:, 0] = env.x
    env.prog.zero_()
    env.seg.zero_()


@torch.inference_mode()
def rollout(env, net, initial, seed, frame_skip, action_rule='argmax'):
    env.load_state_dict(initial)
    torch.manual_seed(seed)  # identical observation-noise stream for each model
    if action_rule not in ('argmax', 'sampled'):
        raise ValueError('Unknown action rule')
    action_rng = torch.Generator(device=env.dev).manual_seed(seed+1777)
    n, dev = env.n, env.dev
    active = torch.ones(n, dtype=torch.bool, device=dev)
    record = {k: torch.zeros(n, device=dev) for k in
              ['finished', 'fell', 'timeout', 'unknown', 'progress', 'seconds',
               'launches', 'safe_landings', 'unsafe_landings', 'switches', 'perfect']}
    xy = torch.zeros(n, 2, device=dev)
    for _ in range(math.ceil(env.max_time/(env.dt*frame_skip))+2):
        logits, _ = net(*env.observe())
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Non-finite policy logits')
        action = (logits.argmax(1) if action_rule == 'argmax' else
                  torch.multinomial(logits.softmax(1), 1, generator=action_rng).squeeze(1))
        record['switches'] += (active & (action != env.pressed)).float()
        for _ in range(frame_skip):
            if not active.any():
                break
            _, done, info = env.step(torch.where(active, action, env.pressed.long()))
            unknown = info.get('unknown', torch.zeros_like(active))
            record['launches'] += (active & info['launch']).float()
            record['safe_landings'] += (active & info['land'] & ~info['fell'] & ~unknown).float()
            record['unsafe_landings'] += (active & info['unsafe_landing'] & ~unknown).float()
            record['perfect'] += (active & info['perfect']).float()
            terminal = active & done
            for key in ['finished', 'fell', 'timeout']:
                record[key][terminal] = info[key][terminal].float()
            record['unknown'][terminal] = unknown[terminal].float()
            record['progress'][terminal] = info['progress'][terminal]
            record['seconds'][terminal] = env.t[terminal]
            xy[terminal] = torch.stack([env.x, env.y], 1)[terminal]
            active &= ~done
        if not active.any():
            break
    if active.any():
        raise RuntimeError('Evaluation ended with nonterminal episodes')
    length = (env.corners[:, 1:]-env.corners[:, :-1]).abs().sum((1, 2))
    record['progress_fraction'] = (record['progress']/length).clamp(0, 1)
    result = {k: v.cpu().tolist() for k, v in record.items()}
    result['terminal_xy'] = xy.cpu().tolist()
    attempted = int(record['launches'].sum().item())
    safe = int(record['safe_landings'].sum().item())
    result['summary'] = dict(episodes=n, successes=int(record['finished'].sum().item()),
                             success_rate=record['finished'].mean().item(),
                             fell=int(record['fell'].sum().item()),
                             timeouts=int(record['timeout'].sum().item()),
                             unknown=int(record['unknown'].sum().item()),
                             mean_progress_fraction=record['progress_fraction'].mean().item(),
                             mean_perfect=record['perfect'].mean().item(),
                             successful_perfect_fraction=(record['finished'] *
                                 (record['perfect']/max(1,env.n_segments-1)).clamp(0,1)).mean().item(),
                             launches=attempted, safe_landings=safe,
                             safe_landing_rate=safe/attempted if attempted else None,
                             unsafe_landings=int(record['unsafe_landings'].sum().item()))
    if (initial['tensors'].get('episode_kind', torch.zeros(n, device=dev)) == 2).all():
        strata = {}
        for axis in (0, 1):
            for response in (False, True):
                mask = ((initial['tensors']['jumps'][:, 0, 0] == axis) &
                        (initial['tensors']['air_responds'] == response))
                count = int(mask.sum())
                if count:
                    strata[f'{"B" if axis == 0 else "A"}_{"responsive" if response else "latched"}'] = dict(
                        episodes=count, successes=int(record['finished'][mask].sum()),
                        success_rate=float(record['finished'][mask].mean()))
        result['summary']['strata'] = strata
    return result


def evaluate(args):
    limit_cuda_memory(args.device,getattr(args,'gpu_memory_gb',0.))
    if not Path(args.params).is_file():
        raise FileNotFoundError(args.params)
    if args.envs <= 0 or args.frame_skip <= 0 or args.max_time <= 0:
        raise ValueError('Environment count, frame skip and time limit must be positive')
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    params = load_params(args.params)
    models = {}
    for spec in args.models:
        name, path = spec.split('=', 1)
        if name in models or not name:
            raise ValueError('Unique nonempty model names required')
        net = ActorCritic().to(args.device)
        net.load_state_dict(torch.load(path, map_location=args.device, weights_only=True))
        net.eval()
        models[name] = (path, net)
    if 'baseline' not in models:
        raise ValueError('A baseline model is required for paired comparisons')
    report = dict(config=vars(args), params=asdict(params), params_sha256=sha256(args.params),
                  model_sha256={name: sha256(path) for name, (path, _) in models.items()},
                  source_sha256={'evaluator': sha256(__file__)}, suites={},
                  action_rule='argmax, matching real-phone policy; no sampling or training',
                  pairing='Frozen initial environment state and identical observation-noise RNG per suite; '
                          'no episode resets, so model-dependent terminations cannot change future maps.',
                  scope='Synthetic simulator evaluation only. standard16 is held-out training distribution; '
                        'long30 tests longer routes; bends16 and ramp are constructed stress cases. '
                        'level645, when included, tests only the reconstructed local planar section with '
                        'randomized dynamics/observations, not the full level. No real-device result.')
    from kart import vecenv, sim, ppo, projection3d, geometry, tracks, cases
    report['source_sha256'].update({module.__name__: sha256(module.__file__) for module in
                                   [vecenv, sim, ppo, projection3d, geometry, tracks, cases]})
    if getattr(args, 'case', None):
        report['case_sha256'] = sha256(args.case)
    begin = time.monotonic()
    for index, suite in enumerate(args.suites):
        seed = args.seed+index*1009
        case = None
        if suite == 'level645':
            from kart.cases import load_case
            if not getattr(args, 'case', None):
                raise ValueError('level645 evaluation requires --case')
            case = load_case(args.case)
        env = VecKart(args.envs, device=args.device, seed=seed, params=params, randomize=1.,
                      n_segments=SUITES[suite], max_time=args.max_time, loop_delay=(0., 0.),
                      case=case, case_probability=float(case is not None))
        prepare(env, suite, seed+37, robust=getattr(args, 'robust_scenes', False))
        initial = env.state_dict()
        bank = out/f'{suite}-episodes.pt'
        torch.save(initial, bank)
        entry = dict(seed=seed, bank_sha256=sha256(bank), models={}, paired={})
        for name, (path, net) in models.items():
            result = rollout(env, net, initial, seed+101, args.frame_skip)
            result_file = out/f'{suite}-{name}.json'
            result_file.write_text(json.dumps(result)+'\n')
            entry['models'][name] = result['summary']
            if getattr(args, 'sampled_diagnostics', False) and suite == 'ramp':
                sampled = rollout(env, net, initial, seed+101, args.frame_skip, action_rule='sampled')
                (out/f'{suite}-{name}-sampled.json').write_text(json.dumps(sampled)+'\n')
                entry.setdefault('sampled_models', {})[name] = sampled['summary']
            if name == 'baseline':
                baseline = result
            print(json.dumps(dict(suite=suite, model=name, **result['summary'])), flush=True)
        baseline = json.loads((out/f'{suite}-baseline.json').read_text())
        base_success = np.asarray(baseline['finished'], bool)
        for name in models:
            if name == 'baseline':
                continue
            candidate = json.loads((out/f'{suite}-{name}.json').read_text())
            success = np.asarray(candidate['finished'], bool)
            entry['paired'][name] = dict(wins=int((success & ~base_success).sum()),
                                        losses=int((~success & base_success).sum()),
                                        both_pass=int((success & base_success).sum()),
                                        both_fail=int((~success & ~base_success).sum()),
                                        success_rate_delta=float(success.mean()-base_success.mean()))
        report['suites'][suite] = entry
        report['seconds'] = round(time.monotonic()-begin, 2)
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params', required=True)
    parser.add_argument('--models', nargs='+', required=True, help='NAME=WEIGHTS, including baseline')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--gpu-memory-gb',type=float,default=0.)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--envs', type=int, default=512)
    parser.add_argument('--seed', type=int, default=27182818)
    parser.add_argument('--max-time', type=float, default=60.)
    parser.add_argument('--frame-skip', type=int, default=2)
    parser.add_argument('--case', help='approved local reconstruction case')
    parser.add_argument('--robust-scenes', action='store_true', help='stratified A/B ramps and relief with explicit air-control uncertainty')
    parser.add_argument('--sampled-diagnostics', action='store_true', help='compare sampled and deployed argmax actions on the same ramp bank')
    parser.add_argument('--suites', nargs='+', choices=SUITES,
                        default=['standard16', 'long30', 'bends16', 'ramp', 'relief'])
    parser.add_argument('--out', required=True)
    evaluate(parser.parse_args())
