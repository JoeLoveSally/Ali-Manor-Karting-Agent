"""Fixed validation selection and resumable patience, independent of PPO loss."""
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass
class Patience:
    best: float = -1e9
    best_iter: int = -1
    no_improvement: int = 0
    checks: int = 0

    def update(self, score, iteration, qualified, minimum, patience, delta):
        self.checks += 1
        improved = qualified and np.isfinite(score) and score > self.best+delta
        if improved:
            self.best, self.best_iter, self.no_improvement = score, iteration, 0
        else:
            self.no_improvement += 1
        stop = (patience > 0 and self.best > -1e9 and iteration+1 >= minimum
                and self.no_improvement >= patience)
        return bool(improved), bool(stop)

    def state_dict(self):
        return asdict(self)


def balanced_score(suites, perfect_credit=.005):
    """Selection on deployed argmax outcomes; no reference-action constraints."""
    success = float(np.mean([x['success_rate'] for x in suites.values()]))
    progress = float(np.mean([x['mean_progress_fraction'] for x in suites.values()]))
    perfect = float(np.mean([x.get('successful_perfect_fraction', 0.) for x in suites.values()]))
    ramp = suites['ramp']
    strata = ramp.get('strata', {})
    ramp_min = min((x['success_rate'] for x in strata.values()), default=ramp['success_rate'])
    worst = min(x['success_rate'] for x in suites.values())
    qualified = ramp['success_rate'] >= .98 and ramp_min >= .95 and suites['relief']['success_rate'] >= .98
    return dict(success_rate=success, successful_perfect_fraction=perfect,
                score=.55*success+.25*worst+.20*ramp_min+.01*progress+perfect_credit*perfect,
                selection_qualified=qualified, worst_suite_success_rate=worst,
                worst_ramp_stratum_success_rate=ramp_min,
                selection_scope='Argmax only; 55% mean success + 25% weakest suite + 20% weakest ramp stratum; '
                                'ramp/relief >=98%, each ramp stratum >=95%; progress/perfect tie breakers')


class SuiteArchive:
    """Retain best argmax checkpoint per suite independently of the aggregate winner."""
    def __init__(self, root):
        import json
        self.root = root/'candidates'
        self.root.mkdir(exist_ok=True)
        self.path = self.root/'by-suite.json'
        self.records = json.loads(self.path.read_text()) if self.path.exists() else {}

    def update(self, net, validation, iteration):
        import json
        from .scenes import sha256
        improved = []
        for name, summary in validation['suites'].items():
            key = (summary['success_rate'], validation['score'])
            old = self.records.get(name, {})
            previous = (old.get('success_rate', -1), old.get('aggregate_score', -1e9))
            if key > previous:
                path = self.root/f'{name}.pt'
                temporary = path.with_suffix('.tmp.pt')
                torch.save(net.state_dict(), temporary);temporary.replace(path)
                self.records[name] = dict(iter=iteration, success_rate=key[0], aggregate_score=key[1],
                                          model=str(path.name), model_sha256=sha256(path))
                improved.append(name)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.records, indent=2)+'\n');temporary.replace(self.path)
        return improved


class FixedValidation:
    def __init__(self, params, device, seed, count, case=None, robust=False):
        from .vecenv import VecKart
        from tools.eval_survival import prepare
        self.perfect_credit = .005 if params.perfect_bonus > 0 else 0.
        self.robust = robust
        self.banks = []
        suites = ['standard16', 'long30', 'bends16', 'middle16', 'ramp', 'relief']
        if case is not None:
            suites.append('level645')
        # Evaluation must not advance the training policy's noise/sample stream.
        self.devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
        with torch.random.fork_rng(devices=self.devices):
            for k, name in enumerate(suites):
                nseg = 4 if name == 'level645' else 30 if name == 'long30' else 1 if name in ('ramp', 'relief') else 16
                env = VecKart(count, device=device, seed=seed+1009*k, params=params,
                              n_segments=nseg, max_time=60., loop_delay=(0., 0.),
                              case=case if name == 'level645' else None,
                              case_probability=float(name == 'level645'))
                if name != 'level645':
                    prepare(env, name, seed+1009*k+37, robust=robust)
                self.banks.append((name, env, env.state_dict(), seed+1009*k+101))

    @torch.inference_mode()
    def evaluate(self, net):
        from tools.eval_survival import rollout
        suites = {}
        with torch.random.fork_rng(devices=self.devices):
            for name, env, initial, seed in self.banks:
                result = rollout(env, net, initial, seed, frame_skip=2)
                suites[name] = result['summary']
                if self.robust and name == 'ramp':
                    sampled = rollout(env, net, initial, seed, frame_skip=2, action_rule='sampled')['summary']
        if self.robust:
            return dict(suites=suites, **balanced_score(suites, self.perfect_credit),
                        sampled_suites={'ramp': sampled},
                        ramp_sampling_gap=sampled['success_rate']-suites['ramp']['success_rate'])
        success = float(np.mean([x['success_rate'] for x in suites.values()]))
        progress = float(np.mean([x['mean_progress_fraction'] for x in suites.values()]))
        perfect = float(np.mean([x.get('successful_perfect_fraction',0.) for x in suites.values()]))
        return {'suites': suites, 'success_rate': success,
                'score': success+.01*progress+self.perfect_credit*perfect,
                'successful_perfect_fraction': perfect,
                'selection_scope': 'Fixed simulator validation: success + 0.01 progress; '
                    'when perfect reward is enabled, add up to 0.005 for drifts in successful episodes'}
