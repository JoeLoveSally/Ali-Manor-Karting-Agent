import numpy as np
import torch

from kart.sim import Params
from kart.vecenv import VecKart
from tools.eval_survival import rollout


class ReleasePolicy(torch.nn.Module):
    def forward(self, grid, state):
        logits = torch.tensor([1., 0.], device=grid.device).expand(len(grid), -1)
        return logits, torch.zeros(len(grid), device=grid.device)


def test_evaluation_keeps_first_terminal_event_and_pairs_noise():
    env = VecKart(2, params=Params(speed=100, dt=.1, perfect_dist=-1),
                  n_segments=1, randomize=0, max_time=2, loop_delay=(0, 0))
    env.pos_r[:] = torch.tensor([-50., -100., 50., 100.])
    env.neg_r.fill_(1e9)
    env.corners[:] = torch.tensor([[0., 0.], [0., 20.]])
    env.x[1] = 100  # One episode falls on step 1; the other finishes on step 2.
    env.O['noise'].fill_(.03)
    initial = env.state_dict()
    a = rollout(env, ReleasePolicy(), initial, seed=333, frame_skip=2)
    b = rollout(env, ReleasePolicy(), initial, seed=333, frame_skip=2)
    assert a == b
    assert a['finished'] == [1., 0.] and a['fell'] == [0., 1.]
    np.testing.assert_allclose(a['seconds'], [.2, .1])
    np.testing.assert_allclose(a['terminal_xy'], [[0, 20], [100, 10]])
    assert a['summary']['successes'] == 1 and a['summary']['fell'] == 1


def test_replay_capture_preserves_rollout_and_stops_at_each_terminal():
    from tools.collect_policy_replays import RecordedEnv, FIELDS
    env = VecKart(2, params=Params(speed=100, dt=.1, perfect_dist=-1),
                  n_segments=1, randomize=0, max_time=2, loop_delay=(0, 0))
    env.pos_r[:] = torch.tensor([-50., -100., 50., 100.])
    env.neg_r.fill_(1e9)
    env.corners[:] = torch.tensor([[0., 0.], [0., 20.]])
    env.x[1] = 100
    initial = env.state_dict()
    original = rollout(env, ReleasePolicy(), initial, seed=333, frame_skip=2)
    capture = RecordedEnv(env, [0, 1])
    recorded = rollout(capture, ReleasePolicy(), initial, seed=333, frame_skip=2)
    assert recorded == original
    assert [len(v) for v in capture.frames] == [3, 2]
    for i in range(2):
        assert len(capture.frames[i][-1]) == len(FIELDS)
        np.testing.assert_allclose(capture.frames[i][-1][1:3], original['terminal_xy'][i])
        assert capture.grids[i][0].shape == (48, 48)
