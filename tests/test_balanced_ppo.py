import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kart.cases import load_case
from kart.curriculum import KINDS, local_straight
from kart.ppo import load_params
from kart.selection import Patience, SuiteArchive, balanced_score
from kart.vecenv import VecKart
from tools.eval_survival import prepare, rollout


def parameters():
    return replace(load_params('configs/train-params.json'), air_input_response_probability=.5)


@pytest.mark.parametrize('seed', [17, 521, 20261031])
def test_independent_ramp_starts_have_a_safe_action_in_all_four_strata(seed):
    env = VecKart(128, seed=seed, params=parameters(), n_segments=1, loop_delay=(0, 0))
    prepare(env, 'ramp', seed+1, robust=True)
    assert env.episode_kind.eq(KINDS.index('ramp')).all()
    assert env.air_responds.sum() == 64
    active = torch.ones(env.n, dtype=torch.bool)
    success = torch.zeros_like(active)
    launches = torch.zeros_like(active)
    for _ in range(350):
        _, done, info = env.step((env.case_finish[:, 0] == 0).long())
        assert not (active & info['fell']).any()
        launches |= active & info['launch']
        success |= active & info['finished']
        active &= ~done
        if not active.any():
            break
    assert launches.all() and success.all()


def test_training_mix_covers_specialists_and_restores_rng_and_scene_state():
    case = load_case('configs/level645-case.json')
    args = dict(n=256, seed=37, params=parameters(), n_segments=6, loop_delay=(0, 0),
                balanced_scenes=True, case=case, case_probability=.15)
    env = VecKart(**args)
    counts = torch.bincount(env.episode_kind, minlength=5)
    for _ in range(3):
        env.reset_envs(torch.arange(env.n))
        counts += torch.bincount(env.episode_kind, minlength=5)
    proportions = counts.numpy()/counts.sum().item()
    np.testing.assert_allclose(proportions, [.25, .15, .30, .20, .10], atol=.055)
    saved = env.state_dict()
    other = VecKart(**args);other.load_state_dict(saved)
    for source in (env, other):
        source.reset_envs(torch.arange(source.n))
    for name, value in env.state_dict()['tensors'].items():
        torch.testing.assert_close(value, other.state_dict()['tensors'][name])


def test_air_response_modes_are_explicit_and_checkpointed():
    env = VecKart(2, params=parameters(), loop_delay=(0, 0), n_segments=4)
    env.air.fill_(True);env.z.fill_(1000);env.vz.zero_()
    env.th.fill_(np.pi/4);env.air_target.fill_(np.pi/2)
    env.air_responds[:] = torch.tensor([False, True])
    env.step(torch.zeros(2, dtype=torch.long))
    assert env.th[0] > np.pi/4 and env.th[1] < np.pi/4
    assert torch.equal(env.state_dict()['tensors']['air_responds'], torch.tensor([False, True]))


def test_local_straight_progress_does_not_reward_lateral_motion():
    env = VecKart(4, params=parameters(), n_segments=1, loop_delay=(0, 0))
    prepare(env, 'ramp', 521, robust=True)
    before = env._progress()
    axis_b = env.case_finish[:, 0] == 0
    env.x += (~axis_b).float()*10
    env.y += axis_b.float()*10
    torch.testing.assert_close(before, env._progress())
    env.x += axis_b.float()*10
    env.y += (~axis_b).float()*10
    torch.testing.assert_close(env._progress(), before+10)


def summaries(ramp=1., bends=.8):
    return {name: dict(success_rate=value, mean_progress_fraction=1., successful_perfect_fraction=0.)
            for name, value in [('ramp', ramp), ('relief', 1.), ('bends16', bends)]}


def test_average_gain_cannot_qualify_a_regressed_ramp_and_patience_waits_for_qualification():
    assert balanced_score(summaries())['selection_qualified']
    bad = summaries(ramp=.94, bends=1.)
    assert not balanced_score(bad)['selection_qualified']
    bad = summaries();bad['ramp']['strata'] = {'A': {'success_rate': .90}, 'B': {'success_rate': 1.}}
    assert not balanced_score(bad)['selection_qualified']
    patience = Patience()
    for it in range(20):
        assert patience.update(.8, it, False, 5, 2, .001) == (False, False)


def test_archive_preserves_suite_winner_when_aggregate_changes(tmp_path):
    archive = SuiteArchive(tmp_path)
    net = torch.nn.Linear(1, 1)
    archive.update(net, dict(suites=summaries(), score=.8), 10)
    archive.update(net, dict(suites=summaries(ramp=.94, bends=1.), score=.9), 20)
    saved = json.loads((tmp_path/'candidates/by-suite.json').read_text())
    assert saved['ramp']['iter'] == 10 and saved['bends16']['iter'] == 20
    assert SuiteArchive(tmp_path).records == saved


def test_sampled_actions_do_not_consume_observation_random_stream():
    class Net(torch.nn.Module):
        def forward(self, grid, state):
            return torch.zeros(len(state), 2), torch.zeros(len(state))
    env = VecKart(8, params=replace(parameters(), speed=0), n_segments=4, max_time=.15, loop_delay=(0, 0))
    initial = env.state_dict()
    rollout(env, Net(), initial, 719, 2, 'argmax')
    expected = torch.get_rng_state()
    rollout(env, Net(), initial, 719, 2, 'sampled')
    assert torch.equal(expected, torch.get_rng_state())


def test_robust_ramp_report_covers_both_axes_and_both_uncertain_modes():
    class HoldTarget(torch.nn.Module):
        def forward(self, grid, state):
            return torch.stack([1-state[:, 2], state[:, 2]], 1)*20, torch.zeros(len(state))
    env = VecKart(8, params=parameters(), n_segments=1, loop_delay=(0, 0))
    prepare(env, 'ramp', 521, robust=True)
    result = rollout(env, HoldTarget(), env.state_dict(), 517, 2)
    assert result['summary']['successes'] == 8
    assert len(result['summary']['strata']) == 4
    assert all(x['episodes'] == 2 and x['success_rate'] == 1. for x in result['summary']['strata'].values())


def test_unqualified_training_keeps_best_progress_without_early_stop(tmp_path, monkeypatch):
    from kart.ppo import train
    import kart.selection
    class Fixed:
        checks = 0
        def __init__(self, *args): pass
        def evaluate(self, net):
            self.checks += 1
            return dict(suites=summaries(ramp=.5), score=self.checks/10,
                        success_rate=.5, selection_qualified=False)
    monkeypatch.setattr(kart.selection, 'FixedValidation', Fixed)
    args = SimpleNamespace(device='cpu', threads=1, seed=7, envs=4, randomize=1.,
        params='configs/train-params.json', max_time=.2, loop_delay=[0, 0],
        scene=None, segments=6, lr=0., horizon=4, frame_skip=2, gamma=.995, lam=.95,
        minibatches=2, epochs=1, clip=.2, vf=.5, ent=.01, init=None, save_every=1,
        validation_envs=1, eval_every=1, early_stop_patience=1, early_stop_min_iters=1,
        early_stop_delta=.001, out=str(tmp_path/'run'), resume=None, iters=2,
        pure_ppo=True, balanced_scenes=True)
    train(args)
    status = json.loads((tmp_path/'run/status.json').read_text())
    best = json.loads((tmp_path/'run/best-selection.json').read_text())
    assert status['iter'] == 1 and status['stop_reason'] == 'max_iters'
    assert best['iter'] == 1 and not best['fixed_validation']['selection_qualified']
