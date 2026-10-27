"""PPO on VecKart. Policy: small CNN on the ego occupancy grid + MLP on the state vector.

Usage (Spark):  python -m kart.ppo --device cuda --envs 1024 --iters 2000 --out runs/ppo-001
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .sim import Params
from .vecenv import G, STATE_DIM, VecKart


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ReLU(),   # 24
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 12
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),  # 6
            nn.Flatten(), nn.Linear(32 * 6 * 6, 128), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(128 + STATE_DIM, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.pi = nn.Linear(128, 2)
        self.v = nn.Linear(128, 1)

    def forward(self, grid, state):
        h = self.mlp(torch.cat([self.cnn(grid), state], 1))
        return self.pi(h), self.v(h).squeeze(1)


def load_params(path="artifacts/sim_params.json"):
    p = Path(path)
    if not p.exists():
        return Params()
    cfg = json.loads(p.read_text())
    cfg.pop("release_shift", None)
    return Params(**{k: v for k, v in cfg.items() if k in Params.__dataclass_fields__})


def limit_cuda_memory(device, gib):
    """Bound only this process's allocator; never alter other GPU processes."""
    if not np.isfinite(gib) or gib < 0:
        raise ValueError('GPU memory limit must be finite and nonnegative')
    dev = torch.device(device)
    if dev.type == 'cuda' and gib:
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        total = torch.cuda.get_device_properties(index).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., gib*1024**3/total), index)


def save_best(out, net, iteration, score, guard_result, validation=None):
    from .scenes import sha256
    torch.save(net.state_dict(), out/'best.pt')
    record = dict(iter=iteration, score=score, model_sha256=sha256(out/'best.pt'),
                  recording_guard_enabled=guard_result is not None, **(guard_result or {}))
    if validation is not None:
        record['fixed_validation'] = validation
    name = 'best-guard.json' if guard_result is not None else 'best-selection.json'
    (out/name).write_text(json.dumps(record, indent=2)+'\n')


def train(a):
    # Backwards-compatible callers omit the optional recording guard.
    for key, value in dict(replay_data=None, replay_coef=1., replay_batch=256, guard_every=25,
                          case=None, case_probability=0., validation_envs=0, eval_every=25,
                          early_stop_patience=0, early_stop_min_iters=500, early_stop_delta=.002,
                          target_kl=0., frame_skip_max=0, timing=False, deterministic=False,
                          pure_ppo=False, gpu_memory_gb=0., balanced_scenes=False,
                          ent_final=None, ent_anneal_iters=0).items():
        if not hasattr(a, key):
            setattr(a, key, value)
    if a.pure_ppo and (a.init or a.replay_data):
        raise ValueError('Pure PPO requires random initialization and no human action data')
    if a.balanced_scenes and (not a.pure_ppo or a.scene):
        raise ValueError('Balanced scenes require pure PPO with generated tracks')
    if a.ent_anneal_iters < 0 or (a.ent_final is not None and (a.ent_final < 0 or not np.isfinite(a.ent_final))):
        raise ValueError('Invalid entropy schedule')
    if a.replay_coef < 0 or a.replay_batch < 1 or a.guard_every < 1:
        raise ValueError('Invalid replay guard configuration')
    if (a.validation_envs < 0 or a.eval_every < 1 or a.early_stop_patience < 0
            or a.early_stop_min_iters < 0 or a.early_stop_delta < 0 or a.target_kl < 0
            or (a.frame_skip_max and a.frame_skip_max < a.frame_skip)
            or (a.early_stop_patience and not a.validation_envs)):
        raise ValueError('Invalid fixed validation or early stopping configuration')
    dev = torch.device(a.device)
    limit_cuda_memory(dev,a.gpu_memory_gb)
    if a.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    env_args = dict(device=dev, seed=a.seed, params=load_params(a.params), randomize=a.randomize,
                    max_time=a.max_time, loop_delay=tuple(a.loop_delay))
    from .scenes import sha256
    # Stable names permit moving a checkpoint between WSL and the training host.
    provenance = {'params': sha256(a.params)}
    provenance.update({name: sha256(Path(__file__).with_name(name)) for name in
                       ['ppo.py', 'vecenv.py', 'rasterenv.py', 'sim.py', 'scenes.py', 'tracks.py', 'geometry.py',
                        'projection3d.py', 'replay_guard.py', 'cases.py', 'selection.py', 'curriculum.py']})
    case = None
    if a.validation_envs:
        provenance['validation_evaluator'] = sha256(Path(__file__).parent.parent/'tools/eval_survival.py')
    if a.case:
        from .cases import load_case
        case = load_case(a.case)
        provenance['case'] = sha256(a.case)
    guard = None
    if a.replay_data:
        from .replay_guard import ReplayGuard
        guard = ReplayGuard(a.replay_data, dev)
        provenance['recordings'] = sha256(a.replay_data)
    if a.scene:
        from .scenes import load_scene
        from .rasterenv import RasterKart
        track, data, scene = load_scene(a.scene, training=True)
        if not scene.get('finish') or 'start' not in scene:
            raise ValueError('A training scene needs explicit start and finish geometry')
        provenance['scene_data'] = scene['data_sha256']
        provenance['scene_meta'] = sha256(Path(a.scene)/'scene.json')
        env = RasterKart(track, a.envs, start=scene['start'], theta=scene.get('theta', 0.),
                         reference=data['pos'], finish=scene['finish'],
                         startup_elapsed=scene.get('startup_elapsed'), **env_args)
    else:
        env = VecKart(a.envs, n_segments=a.segments, case=case, case_probability=a.case_probability,
                      balanced_scenes=a.balanced_scenes, **env_args)
    net = ActorCritic().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    checkpoint = torch.load(a.resume, map_location=dev, weights_only=True) if a.resume else None
    out = Path(a.resume).parent if checkpoint else Path(a.out)
    config = {**vars(a), 'provenance': provenance}
    if checkpoint:
        for key, value in config.items():
            if key not in ['iters', 'out', 'resume', 'device', 'threads'] and checkpoint['config'][key] != value:
                raise ValueError(f'Resume configuration changed: {key}')
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    log = open(out / "log.jsonl", "a")
    T, N = a.horizon, a.envs
    # decisions every `frame_skip` sim steps (the real loop runs at ~30 Hz, sim at 60 Hz)
    buf = {k: torch.zeros((T, N) + s, device=dev) for k, s in
           [("grid", (1, G, G)), ("state", (STATE_DIM,)), ("act", ()), ("logp", ()), ("rew", ()),
            ("done", ()), ("val", ())]}
    ep_ret = torch.zeros(N, device=dev)
    stats = {"ret": [], "finished": [], "fell": [], "perfect": [], "progress": [], "unknown": []}
    ep_perf = torch.zeros(N, device=dev)
    best = -1e9
    start_iter = 0
    if checkpoint:
        net.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['optimizer'])
        env.load_state_dict(checkpoint['environment'])
        ep_ret, ep_perf = checkpoint['ep_ret'], checkpoint['ep_perf']
        stats, best, start_iter = checkpoint['stats'], checkpoint['best'], checkpoint['iter']+1
        torch.set_rng_state(checkpoint['rng'].cpu())
        if dev.type == 'cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng']])
    elif a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev, weights_only=True))
    initial_guard = None
    if guard and not checkpoint:
        initial_guard = guard.evaluate(net)
        (out/'initial-guard.json').write_text(json.dumps(initial_guard, indent=2)+'\n')
        if not initial_guard['passed']:
            raise ValueError('Initial model fails real recording regression gate')
    if a.horizon*a.envs < 2*a.minibatches:
        raise ValueError('Need at least two samples per minibatch')
    if a.iters <= start_iter or a.save_every < 1:
        raise ValueError('Need remaining iterations and a positive checkpoint interval')
    from .selection import FixedValidation, Patience, SuiteArchive
    selector = Patience(**checkpoint.get('selection', {})) if checkpoint else Patience()
    validator = (FixedValidation(load_params(a.params), dev, a.seed+1000003, a.validation_envs, case, a.balanced_scenes)
                 if a.validation_envs else None)
    archive = SuiteArchive(out) if a.balanced_scenes else None
    validation_result = None
    if validator and not checkpoint:
        validation_result = validator.evaluate(net)
        qualified = (initial_guard is None or initial_guard['passed']) and validation_result.get('selection_qualified', True)
        selector.update(validation_result['score'], -1, qualified,
                        a.early_stop_min_iters, a.early_stop_patience, a.early_stop_delta)
        best = (validation_result['score'] if a.balanced_scenes else selector.best)
        save_best(out, net, -1, best, initial_guard, validation_result)
        (out/'validation.jsonl').write_text(json.dumps(dict(iter=-1, qualified=qualified, **validation_result))+'\n')
        if archive:
            archive.update(net, validation_result, -1)
    t0 = time.time()
    elapsed_before = float(checkpoint.get('elapsed_seconds', 0.)) if checkpoint else 0.
    stopped_early = False
    def stamp():
        if a.timing and dev.type == 'cuda':
            torch.cuda.synchronize(dev)
        return time.perf_counter()
    for it in range(start_iter, a.iters):
        iteration_start = stamp()
        scene_decisions = torch.zeros(5, device=dev, dtype=torch.long)
        scene_terminals = torch.zeros(5, device=dev, dtype=torch.long)
        scene_successes = torch.zeros(5, device=dev, dtype=torch.long)
        for t in range(T):
            grid, state = env.observe()
            if a.balanced_scenes:
                scene_decisions += torch.bincount(env.episode_kind, minlength=5)
            with torch.no_grad():
                logits, val = net(grid, state)
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
            r_tot = torch.zeros(N, device=dev)
            d_tot = torch.zeros(N, dtype=torch.bool, device=dev)
            skip = (int(torch.randint(a.frame_skip, a.frame_skip_max+1, (1,)).item())
                    if a.frame_skip_max else a.frame_skip)
            for _ in range(skip):
                r, d, info = env.step(torch.where(d_tot, env.pressed.long(), act))
                r_tot += r * (~d_tot)
                ep_perf += (info["perfect"] & ~d_tot).float()
                newly = d & ~d_tot
                if a.balanced_scenes:
                    scene_terminals += torch.bincount(env.episode_kind[newly], minlength=5)
                    scene_successes += torch.bincount(env.episode_kind[newly & info['finished']], minlength=5)
                if newly.any():
                    for k, src in [("finished", info["finished"]), ("fell", info["fell"])]:
                        stats[k] += src[newly].float().tolist()
                    stats['unknown'] += info.get('unknown', torch.zeros_like(d))[newly].float().tolist()
                    stats["progress"] += info["progress"][newly].tolist()
                    stats["perfect"] += ep_perf[newly].tolist()
                    stats["ret"] += (ep_ret + r_tot)[newly].tolist()
                d_tot |= d
            for k, v in [("grid", grid), ("state", state), ("act", act.float()),
                         ("logp", dist.log_prob(act)), ("rew", r_tot), ("done", d_tot.float()), ("val", val)]:
                buf[k][t] = v
            ep_ret = (ep_ret + r_tot) * (~d_tot)
            ep_perf = ep_perf * (~d_tot)
            if d_tot.any():
                env.reset_envs(torch.nonzero(d_tot).squeeze(1))
        with torch.no_grad():
            _, last_v = net(*env.observe())
        sampling_end = stamp()
        adv = torch.zeros(T, N, device=dev)
        gae = torch.zeros(N, device=dev)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else buf["val"][t + 1]
            nonterm = 1 - buf["done"][t]
            delta = buf["rew"][t] + a.gamma * nv * nonterm - buf["val"][t]
            gae = delta + a.gamma * a.lam * nonterm * gae
            adv[t] = gae
        ret = adv + buf["val"]
        flat = {k: v.reshape((T * N,) + v.shape[2:]) for k, v in buf.items()}
        fadv, fret = adv.reshape(-1), ret.reshape(-1)
        idx = torch.randperm(T * N, device=dev)
        mb = T * N // a.minibatches
        grad_norm = torch.tensor(0.)
        update_stopped, approximate_kl = False, 0.
        for _ in range(a.epochs):
            idx = torch.randperm(T * N, device=dev)
            for s in range(0, T * N, mb):
                j = idx[s:s + mb]
                logits, v = net(flat["grid"][j], flat["state"][j])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(flat["act"][j].long())
                ratio = (lp - flat["logp"][j]).exp()
                approximate_kl = float((ratio-1-(lp-flat['logp'][j])).mean().detach())
                ad = fadv[j]
                ad = (ad - ad.mean()) / (ad.std(unbiased=False) + 1e-8)
                pl = -torch.min(ratio * ad, ratio.clamp(1 - a.clip, 1 + a.clip) * ad).mean()
                vl = 0.5 * (fret[j] - v).pow(2).mean()
                ent = dist.entropy().mean()
                entropy_weight = a.ent
                if a.ent_final is not None:
                    fraction = min(1., (it+1)/max(1, a.ent_anneal_iters))
                    entropy_weight += fraction*(a.ent_final-a.ent)
                loss = pl + a.vf * vl - entropy_weight * ent
                if guard:
                    rehearsal = guard.loss(net, a.replay_batch)
                    loss = loss+a.replay_coef*rehearsal
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite PPO loss at iteration {it}')
                if a.target_kl and approximate_kl > 1.5*a.target_kl:
                    update_stopped = True
                    break
                opt.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(net.parameters(), 0.5, error_if_nonfinite=True)
                opt.step()
            if update_stopped:
                break
        update_end = stamp()
        guard_result = None
        validate_now = validator and ((it+1) % a.eval_every == 0 or it == a.iters-1)
        if guard and ((it+1) % a.guard_every == 0 or it == a.iters-1 or validate_now):
            guard_result = guard.evaluate(net)
            with (out/'guard.jsonl').open('a') as guard_log:
                guard_log.write(json.dumps(dict(iter=it, **guard_result))+'\n')
        if validate_now:
            validation_result = validator.evaluate(net)
            qualified = (guard is None or guard_result['passed']) and validation_result.get('selection_qualified', True)
            if archive:
                validation_result['archive_improved'] = archive.update(net, validation_result, it)
            improved, stopped_early = selector.update(validation_result['score'], it, qualified,
                a.early_stop_min_iters, a.early_stop_patience, a.early_stop_delta)
            with (out/'validation.jsonl').open('a') as validation_log:
                validation_log.write(json.dumps(dict(iter=it, qualified=qualified,
                    no_improvement=selector.no_improvement, **validation_result))+'\n')
            if improved:
                best = selector.best
                save_best(out, net, it, best, guard_result, validation_result)
            elif a.balanced_scenes and selector.best <= -1e9 and validation_result['score'] > best+a.early_stop_delta:
                # Keep useful progress before the first qualified checkpoint,
                # while explicitly marking it ineligible and not early-stopping.
                best = validation_result['score']
                save_best(out, net, it, best, guard_result, validation_result)
        validation_end = stamp()
        timings = dict(sampling_seconds=sampling_end-iteration_start,
                       update_seconds=update_end-sampling_end,
                       validation_seconds=validation_end-update_end)
        if stats["ret"]:
            row = {"iter": it, "time": round(elapsed_before+time.time()-t0, 1), "episodes": len(stats["ret"])}
            row.update({k: float(np.mean(v)) for k, v in stats.items()})
            row["entropy"] = float(ent.detach())
            if a.balanced_scenes:
                from .curriculum import KINDS
                counts = torch.stack([scene_decisions, scene_terminals, scene_successes]).cpu().tolist()
                row['scenes'] = {name: dict(decisions=counts[0][i], episodes=counts[1][i], successes=counts[2][i])
                                 for i, name in enumerate(KINDS)}
                row['entropy_weight'] = entropy_weight
            if a.timing:
                row['timings'] = timings
            log.write(json.dumps(row) + "\n")
            log.flush()
            if it % 10 == 0:
                print(json.dumps(row), flush=True)
            # Profiles with zero perfect_bonus retain physical boosts without
            # rewarding them. Pure PPO rewards perfect drifts explicitly.
            score = (row["finished"] + 0.04 * env.base.perfect_bonus * row["perfect"]
                     + 1e-9 * row["progress"])
            allowed = guard is None or (guard_result is not None and guard_result['passed'])
            if validator is None and allowed and score > best and len(stats["ret"]) >= 50:
                best = score
                save_best(out, net, it, score, guard_result)
            stats = {k: [] for k in stats}
        # Log progress even when long episodes have not terminated yet.
        status = {'iter': it, 'total_iters': a.iters, 'seconds_this_session': round(time.time()-t0, 1),
                  'elapsed_seconds': round(elapsed_before+time.time()-t0, 1),
                  'device': str(dev), 'scene': a.scene, 'status': 'running',
                  'loss': float(loss.detach()), 'gradient_norm': float(grad_norm),
                  'approximate_kl': approximate_kl, 'update_stopped_by_kl': update_stopped,
                  'selection': selector.state_dict() if validator else None}
        if a.timing:
            status['timings'] = timings
        if dev.type == 'cuda':
            status['cuda_memory'] = dict(allocated_mib=torch.cuda.memory_allocated(dev)/1024**2,
                reserved_mib=torch.cuda.memory_reserved(dev)/1024**2,
                peak_allocated_mib=torch.cuda.max_memory_allocated(dev)/1024**2,
                allocator_limit_gib=a.gpu_memory_gb)
        if validate_now:
            status['fixed_validation'] = validation_result
        if guard_result is not None:
            status['recording_guard'] = guard_result
        (out/'status.tmp.json').write_text(json.dumps(status)+'\n')
        (out/'status.tmp.json').replace(out/'status.json')
        if (it+1) % a.save_every == 0 or it == a.iters-1 or stopped_early:
            torch.save(net.state_dict(), out / "last.pt")
            saved = dict(model=net.state_dict(), optimizer=opt.state_dict(), environment=env.state_dict(),
                         ep_ret=ep_ret, ep_perf=ep_perf, stats=stats, best=best, iter=it,
                         config=config, selection=selector.state_dict(), rng=torch.get_rng_state(),
                         elapsed_seconds=elapsed_before+time.time()-t0,
                         cuda_rng=torch.cuda.get_rng_state_all() if dev.type == 'cuda' else [])
            torch.save(saved, out/'checkpoint.tmp.pt')
            (out/'checkpoint.tmp.pt').replace(out/'checkpoint.pt')
        if stopped_early:
            break
    torch.save(net.state_dict(), out / "last.pt")
    (out/'status.json').write_text(json.dumps({'iter': it, 'total_iters': a.iters,
        'status': 'completed', 'stop_reason': 'no_qualified_validation_improvement' if stopped_early else 'max_iters',
        'elapsed_seconds': elapsed_before+time.time()-t0,
        'cuda_memory': status.get('cuda_memory'),
        'selection': selector.state_dict() if validator else None})+'\n')
    log.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--scene', help='reviewed scene directory; reconstruction drafts are rejected')
    ap.add_argument('--init', help='initial policy weights')
    ap.add_argument('--resume', help='resume a full checkpoint, including environment and optimizer')
    ap.add_argument('--save-every', type=int, default=25)
    ap.add_argument('--replay-data', help='immutable human rehearsal and separate recording regression bank')
    ap.add_argument('--pure-ppo', action='store_true', help='random initialization, no human action loss or checks; same-run recovery allowed')
    ap.add_argument('--balanced-scenes', action='store_true', help='mix independent ramp, bends and relief starting states; robust deployed-action validation')
    ap.add_argument('--ent-final', type=float, help='optional final exploration coefficient; does not penalize switches')
    ap.add_argument('--ent-anneal-iters', type=int, default=0)
    ap.add_argument('--gpu-memory-gb', type=float, default=0., help='cap this process PyTorch CUDA allocator in GiB; zero disables the cap')
    ap.add_argument('--replay-coef', type=float, default=1.)
    ap.add_argument('--replay-batch', type=int, default=256)
    ap.add_argument('--guard-every', type=int, default=25)
    ap.add_argument('--case', help='approved reconstructed local case, mixed with random tracks')
    ap.add_argument('--case-probability', type=float, default=0.)
    ap.add_argument('--validation-envs', type=int, default=0)
    ap.add_argument('--eval-every', type=int, default=25)
    ap.add_argument('--early-stop-patience', type=int, default=0, help='consecutive checks without qualified improvement')
    ap.add_argument('--early-stop-min-iters', type=int, default=500)
    ap.add_argument('--early-stop-delta', type=float, default=.002)
    ap.add_argument('--target-kl', type=float, default=0., help='stop a PPO update, not the whole run, if policy drift is excessive')
    ap.add_argument('--timing', action='store_true', help='synchronized sampling/update/validation wall times')
    ap.add_argument('--deterministic', action='store_true', help='reproducible CUDA algorithms and matmul workspace')
    ap.add_argument('--frame-skip-max', type=int, default=0, help='optional variable decision interval up to this many physics steps')
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--segments", type=int, default=30)
    ap.add_argument("--max-time", type=float, default=60.0)
    ap.add_argument("--randomize", type=float, default=1.0)
    ap.add_argument("--loop-delay", type=float, nargs=2, default=[0.05, 0.18],
                    help="range of the live loop delay added to the game latencies (s)")
    ap.add_argument("--params", default="artifacts/sim_params.json")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/ppo")
    train(ap.parse_args())
