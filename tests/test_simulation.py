import json
from types import SimpleNamespace

import numpy as np
import torch
import pytest

from kart.rasterenv import RasterKart
from kart.sim import Params, Sim, Track
from kart.vecenv import VecKart
from tools.validate_sim import load
from kart.scenes import load_scene, sha256


def test_recorded_trajectory_does_not_fill_holes(tmp_path):
    road = np.ones((20, 20))*10
    road[8:12, 8:12] = 0
    seen = np.ones((20, 20))*10
    seen[0] = 0
    file = tmp_path/'hole.npz'
    np.savez(file, road=road, seen=seen, origin=[0, 0], res=8,
             pos=[[80, 80]], t=[0], touch=[False])
    track, *_ = load(file)
    assert track.is_known([80, 80]) and not track.on_road([80, 80])
    assert not track.is_known([1, 1])


def test_unknown_contact_is_neither_fall_nor_success():
    tr = Track(np.zeros((10, 10)), [0, 0], known=np.zeros((10, 10)))
    env = RasterKart(tr, params=Params(speed=0), start=(10, 10))
    reward, done, info = env.step(torch.zeros(1))
    assert done.item() and info['unknown'].item()
    assert not info['fell'].item() and not info['finished'].item()
    assert reward.item() == 0 and torch.all(env.observe()[0] == .5)
    sim = Sim(tr, Params(speed=0), start=(10, 10))
    assert not sim.step(False) and sim.s.unknown and sim.s.alive


def test_partial_scene_start_does_not_reward_untraveled_reference_distance():
    tr = Track(np.ones((100, 100)), [-400, -400])
    env = RasterKart(tr, params=Params(speed=0), start=(0., 100.),
                     reference=np.array([[0., 0.], [0., 100.], [0., 200.]]))
    for _ in range(2):
        reward, _, _ = env.step(torch.zeros(1))
        assert reward.item() == 0.
        env.reset_envs(torch.tensor([0]))


def test_scalar_replay_and_batch_share_delay_drift_boost_and_jump():
    road = np.ones((200, 200), bool)
    road[70:90] = False
    tr = Track(road, [-1000, -100], res=10,
               jumps=[[1, 600, 800, -1000, 1000, 20]])
    p = Params(speed=500, turn_rate=4, latency_press=.035, latency_release=.02,
               perfect_dist=10000, margin=0, loop_delay=.04)
    sim = Sim(tr, p)
    env = RasterKart(tr, n=3, params=p, loop_delay=(.04, .04))
    launched, boosted = False, False
    for i in range(160):
        act = 95 <= i < 125
        sim.step(act)
        _, _, info = env.step(torch.full((3,), int(act)))
        np.testing.assert_allclose(env.x.numpy(), sim.s.pos[0], atol=1e-5)
        np.testing.assert_allclose(env.y.numpy(), sim.s.pos[1], atol=1e-5)
        assert torch.all(env.th == sim.s.theta)
        assert torch.all(env.air == sim.s.airborne)
        assert torch.all(env.eff == sim.s.effective)
        launched |= bool(info['launch'].any())
        boosted |= bool(info['perfect'].any())
    assert launched and boosted


def test_gap_requires_explicit_jump_and_landing_requires_road():
    p = Params(speed=100, dt=.1, margin=0)
    road = np.ones((40, 10), bool)
    road[10:20] = False
    plain = Sim(Track(road, [-50, 0], res=10), p)
    jump = Sim(Track(road, [-50, 0], res=10, jumps=[[1, 100, 200, -50, 50, 0]]), p)
    no_landing = road.copy()
    no_landing[20:] = False
    missing = Sim(Track(no_landing, [-50, 0], res=10, jumps=[[1, 100, 200, -50, 50, 0]]), p)
    for _ in range(22):
        plain.step(False)
        jump.step(False)
        missing.step(False)
    assert not plain.s.alive and jump.s.alive and not jump.s.airborne
    assert not missing.s.alive


def test_conservative_landing_rejects_edge_but_preserves_safe_flight():
    from dataclasses import replace
    road = np.ones((100, 40), bool)
    road[20:40] = False
    track = Track(road, [0, 0], res=10,
                  jumps=[[1, 200, 400, 0, 400, 100]])
    p = Params(speed=200, dt=.05, perfect_dist=-1, landing_inset=32, perfect_bonus=0)
    edge = RasterKart(track, params=p, start=(10, 190))
    center = RasterKart(track, params=p, start=(200, 190))
    legacy = RasterKart(track, params=replace(p, landing_inset=0), start=(10, 190))
    landed = False
    for _ in range(40):
        _, _, info = edge.step(torch.zeros(1))
        _, _, safe = center.step(torch.zeros(1))
        _, _, old = legacy.step(torch.zeros(1))
        assert torch.equal(edge.y, legacy.y) and torch.equal(edge.z, legacy.z)
        assert not safe['fell'].item() and not old['fell'].item()
        if info['land'].item():
            assert info['fell'].item() and info['unsafe_landing'].item()
            assert not info['unknown'].item()
            landed = True
            break
    assert landed


def test_conservative_landing_unknown_footprint_is_unscorable():
    road = np.ones((50, 50), bool)
    known = np.ones_like(road)
    known[:, :5] = False
    env = RasterKart(Track(road, [0, 0], res=10, known=known),
                     params=Params(speed=0, landing_inset=32), start=(70, 200))
    env.air.fill_(True)
    env.z.fill_(.01)
    env.vz.fill_(-1)
    reward, done, info = env.step(torch.zeros(1))
    assert done.item() and info['land'].item() and info['unknown'].item()
    assert not info['fell'].item() and not info['finished'].item()
    assert reward.item() == 0


def test_zero_perfect_bonus_preserves_physical_boost():
    from dataclasses import replace
    track = Track(np.ones((200, 200)), [-800, -800])
    p = Params(speed=100, perfect_dist=10000, perfect_bonus=0, turn_rate=1000,
               latency_press=0, latency_release=0)
    survival = RasterKart(track, params=p)
    rewarded = RasterKart(track, params=replace(p, perfect_bonus=.5))
    r0, _, i0 = survival.step(torch.ones(1))
    r1, _, i1 = rewarded.step(torch.ones(1))
    assert i0['perfect'].item() and i1['perfect'].item()
    assert r1.item()-r0.item() == pytest.approx(.5)
    for _ in range(5):
        survival.step(torch.ones(1))
        rewarded.step(torch.ones(1))
        assert torch.equal(survival.speed_now, rewarded.speed_now)
    assert survival.speed_now.item() > p.speed


def test_conservative_generated_jumps_have_slow_and_fast_landing_room():
    from kart.vecenv import random_rects, jump_distance_bounds
    rng = np.random.default_rng(20261023)
    p = Params(speed=1280, boost_gain=.4, landing_inset=32, perfect_bonus=0)
    count = 0
    for _ in range(64):
        sample = p.randomized(rng)
        bounds = jump_distance_bounds(sample)
        tr = random_rects(rng, 30, jump_bounds=bounds, landing_inset=p.landing_inset)
        assert tr['pos'].shape == (31, 4) and tr['corners'].shape == (31, 2)
        for axis, lip, _, l0, l1, _ in tr['jumps']:
            count += 1
            # For an aligned flight, every point between these limits is on
            # the same straight landing road, with room before the next bend.
            for distance in np.linspace(*bounds, 5):
                along, lateral = lip+distance, (l0+l1)/2
                x, y = (along, lateral) if axis == 0 else (lateral, along)
                for dx in [-32, 0, 32]:
                    for dy in [-32, 0, 32]:
                        def inside(rects):
                            return ((x+dx >= rects[:, 0]) & (x+dx < rects[:, 2]) &
                                    (y+dy >= rects[:, 1]) & (y+dy < rects[:, 3])).any()
                        assert inside(tr['pos']) and not inside(tr['neg'])
    assert count > 100


def test_input_never_acts_before_its_delay():
    tr = Track(np.ones((100, 100)), [-400, -400])
    sim = Sim(tr, Params(speed=0, dt=.01, latency_press=.025))
    for _ in range(3):
        sim.step(True)
        assert sim.s.theta == 0
    sim.step(True)
    assert sim.s.theta > 0


def test_last_segment_is_not_finish():
    env = VecKart(1, n_segments=2, randomize=0)
    env.seg.fill_(1)
    env.prog.fill_(1.)
    assert not env._finished().item()
    length = (env.corners[:, 1:]-env.corners[:, :-1]).abs().sum()
    env.prog.fill_(length)
    assert env._finished().item()


def test_observation_averages_velocity_and_resets_without_episode_leak():
    p = Params(speed=100, turn_rate=np.pi/2/.1, latency_press=0,
               dt=.05, observation_window=.2, observation_window_jitter=.04)
    tr = Track(np.ones((200, 200)), [-800, -800])
    env = RasterKart(tr, params=p)
    env.step(torch.ones(1))
    _, sensed = env.observe()
    _, true = env.observe(clean=True)
    assert 0 < sensed[0, 0] < true[0, 0]
    for _ in range(10):
        env.step(torch.ones(1))
    assert env.observe()[1][0, 0] > .99
    env.reset_envs(torch.tensor([0]))
    assert env.observe()[1][0, 0] == 0
    assert env.obs_boost.item() == 0 and torch.isinf(env.obs_banner_at).all()


def test_physical_ramp_flight_is_ballistic_and_bad_landing_falls():
    p = Params(speed=1000, gravity=5000, dt=.01, margin=0, perfect_dist=0)
    road = np.ones((250, 100), bool)
    road[:, :30] = False
    road[:, 70:] = False
    road[76:101] = False
    tr = Track(road, [-500, -500], res=10,
               jumps=[[1, 260, 510, -200, 200, 0]],
               ramps=[[1, 0, 260, -200, 200, 122]])
    env = RasterKart(tr, params=p, start=(0, 0))
    launched = None
    for _ in range(120):
        _, _, info = env.step(torch.zeros(1))
        if info['launch'].item():
            launched = (env.t.item(), env.z.item(), env.vz.item())
        elif launched and env.air.item():
            elapsed = env.t.item()-launched[0]
            expected = launched[1]+launched[2]*elapsed-.5*p.gravity*elapsed**2
            assert abs(env.z.item()-expected) < .01
        if info['land'].item():
            assert launched and not info['fell'].item()
            break
    else:
        pytest.fail('No landing')
    # The same lip cannot save a car whose lateral velocity carries it off the landing.
    bad = RasterKart(tr, params=p, start=(190, 250), theta=.6)
    bad.air.fill_(True)
    bad.z.fill_(122)
    bad.vz.fill_(940)
    for _ in range(100):
        _, _, info = bad.step(torch.zeros(1))
        if info['land'].item():
            assert info['fell'].item()
            break
    else:
        pytest.fail('No bad landing')


def test_completed_drift_can_earn_one_later_reward_and_survive_checkpoint():
    road = np.ones((200, 200), bool)
    road[:20] = False  # inner boundary v=0 on the approach
    road[:116, 40:] = False  # inner boundary rises to v=96 at u=20
    track = Track(road, [-20, -20], res=1)
    p = Params(speed=100, dt=.05, turn_rate=np.pi/.1,
               latency_press=0, drift_slow=1, perfect_dist=12, boost_gain=0)
    env = RasterKart(track, params=p, start=(0, 100))
    for _ in range(2):
        assert not env.step(torch.ones(1))[2]['perfect'].item()
    assert env.drifting.item() == 0 and env.drift_pending.item()
    restored = RasterKart(track, params=p, start=(0, 100))
    restored.load_state_dict(env.state_dict())
    hits = []
    for _ in range(8):
        info = env.step(torch.ones(1))[2]
        other = restored.step(torch.ones(1))[2]
        assert info['perfect'].item() == other['perfect'].item()
        hits.append(info['perfect'].item())
    assert sum(hits) == 1 and not hits[0]
    assert not env.drift_pending.item()
    env.reset_envs(torch.tensor([0]))
    assert not env.drift_pending.item()


def test_initial_straight_near_edge_has_no_drift_to_credit():
    p = Params(speed=0, perfect_min_angle=0, perfect_dist=10000)
    env = RasterKart(Track(np.ones((100, 100)), [-400, -400]), params=p,
                     theta=np.pi/2)
    for _ in range(3):
        assert not env.step(torch.ones(1))[2]['perfect'].item()


def test_launch_uses_crossing_position_and_remaining_step_time():
    p = Params(speed=100, dt=.1, gravity=100, drift_slow=1,
               turn_rate=0, perfect_dist=0, margin=0)
    road = np.ones((100, 100), bool)
    tr = Track(road, [-100, -100], res=4,
               jumps=[[1, 5, 25, -1, 1, 0]])
    env = RasterKart(tr, params=p, start=(0, 0))
    _, _, info = env.step(torch.zeros(1))
    assert info['launch'].item()
    # Legacy gap: 0.2 s total flight, take-off halfway through this step.
    assert env.z.item() == pytest.approx(.375, abs=1e-5)
    assert env.vz.item() == pytest.approx(5., abs=1e-5)
    # The end-of-step lateral coordinate lies outside, but the actual crossing
    # is inside the launch span. Conversely, entering after the lip is too late.
    theta = np.pi/4
    inside = RasterKart(tr, params=p, start=(-5., 0), theta=theta)
    assert inside.step(torch.ones(1))[2]['launch'].item()
    outside = RasterKart(tr, params=p, start=(-7., 0), theta=theta)
    assert not outside.step(torch.ones(1))[2]['launch'].item()


def test_airborne_target_is_latched_while_existing_turn_can_finish():
    p = Params(speed=0, dt=.1, gravity=0, turn_rate=1, latency_press=0,
               air_turn_continues=True)
    env = RasterKart(Track(np.ones((100, 100)), [-400, -400]), params=p, theta=.5)
    env.air.fill_(True)
    env.z.fill_(100)
    env.air_target.fill_(0)
    for expected in [.4, .3, .2, .1, 0.]:
        _, _, info = env.step(torch.ones(1))
        assert env.th.item() == pytest.approx(expected, abs=1e-6)
        assert not info['perfect'].item()
    assert env.drifting.item() == 1  # credit/speed state settles only on the road
    env.z.zero_()
    assert env.step(torch.ones(1))[2]['land'].item()
    env.step(torch.ones(1))
    assert env.th.item() > 0  # the still-held command applies after landing
    env.reset_envs(torch.tensor([0]))
    assert env.air_target.item() == 0


def test_startup_acceleration_preserves_distance_and_section_initialization():
    for dt in [.1, .05]:
        p = Params(speed=100, startup_duration=.4, dt=dt, perfect_dist=0)
        env = VecKart(1, params=p, randomize=0, loop_delay=(0, 0))
        assert env.speed_now.item() == 0
        for _ in range(round(.4/dt)):
            env.step(torch.zeros(1))
        assert env.y.item() == pytest.approx(20., abs=1e-4)
        for _ in range(round(.4/dt)):
            env.step(torch.zeros(1))
        assert env.y.item() == pytest.approx(60., abs=1e-4)
        env.reset_envs(torch.tensor([0]))
        assert env.startup_age.item() == 0 and env.speed_now.item() == 0
    tr = Track(np.ones((100, 100)), [-400, -400])
    section = RasterKart(tr, params=p)
    race = RasterKart(tr, params=p, startup_elapsed=0.)
    section.step(torch.zeros(1))
    race.step(torch.zeros(1))
    assert section.y.item() == pytest.approx(5.)
    assert race.y.item() == pytest.approx(.3125)


def test_startup_speed_profile_changes_sensor_only_and_expires():
    p = Params(speed=100, dt=.05, startup_duration=.4, startup_phase=.4,
               startup_speed_profiles=((.5, .5, 1.),), startup_profile_dt=.2)
    env = VecKart(1, params=p, randomize=0, loop_delay=(0, 0))
    env.step(torch.zeros(1))
    assert env.y.item() == pytest.approx(5.)
    assert env.observe(clean=True)[1][0, 4].item() == pytest.approx(.1)
    assert env.observe()[1][0, 4].item() == pytest.approx(.05)
    section = RasterKart(Track(np.ones((100, 100)), [-400, -400]), params=p)
    section.step(torch.zeros(1))
    assert section.observe()[1][0, 4].item() == pytest.approx(.1)
    for _ in range(9):
        env.step(torch.zeros(1))
    assert env.y.item() == pytest.approx(50., abs=1e-4)
    assert env.observe()[1][0, 4].item() == pytest.approx(.1)


def test_outer_edge_does_not_award_inner_edge_drift():
    road = np.zeros((100, 100), bool)
    road[20:80, 20:80] = True
    env = RasterKart(Track(road, [-400, -400]), params=Params(speed=0), start=(0, 225))
    env.eff.fill_(1)
    # Outer +v edge is 15 units away, but the B-turn's inner -v edge is far away.
    assert env.edge_distance().item() > 400


def test_observation_projection_matches_camera_and_unknown_memory_stays_unknown():
    from kart.geometry import M
    from kart.projection3d import project
    tr = Track(np.ones((200, 200)), [-800, -800])
    env = RasterKart(tr, params=Params(speed=0, observation_height=True, observation_memory=True))
    env.z.fill_(120)
    px, py = env.observation_points()
    screen = project(np.c_[px[0].numpy(), py[0].numpy(), np.zeros(px.shape[1])], camera=(0, 0, 120))
    flat = np.c_[env.grid_off.numpy(), np.ones(px.shape[1])] @ M.T
    np.testing.assert_allclose(screen, flat[:, :2]/flat[:, 2:], atol=.002)
    unknown = RasterKart(Track(np.ones((200, 200)), [-800, -800], known=np.zeros((200, 200))),
                         params=Params(speed=0, observation_memory=True))
    for _ in range(3):
        unknown.step(torch.zeros(1))
        assert torch.all(unknown.observe()[0] == .5)
        assert torch.all(unknown.map_memory == .5)


def test_fixed_camera_does_not_move_flat_landing_when_car_jumps():
    from kart.geometry import M
    from kart.projection3d import project
    road = Track(np.ones((200, 200)), [-800, -800])
    env = RasterKart(road, params=Params(speed=0, observation_height=True, camera_height_gain=0.))
    before = env.observation_points()
    env.z.fill_(120)
    after = env.observation_points()
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    # The independent camera projection remains at baseline height despite car lift.
    screen = project(np.c_[after[0][0].numpy(), after[1][0].numpy(), np.zeros(after[0].shape[1])])
    flat = np.c_[env.grid_off.numpy(), np.ones(after[0].shape[1])] @ M.T
    np.testing.assert_allclose(screen, flat[:, :2]/flat[:, 2:], atol=.002)


def test_draft_scene_cannot_silently_train(tmp_path):
    np.savez(tmp_path/'scene.npz', road=np.ones((2, 2)), known=np.zeros((2, 2)),
             origin=[0, 0], res=8, jumps=np.empty((0, 6)))
    meta = {'schema': 1, 'training_ready': False, 'blockers': ['ramp height unresolved'],
            'data_sha256': sha256(tmp_path/'scene.npz')}
    (tmp_path/'scene.json').write_text(json.dumps(meta))
    load_scene(tmp_path)
    with pytest.raises(ValueError, match='ramp height unresolved'):
        load_scene(tmp_path, training=True)
    (tmp_path/'scene.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_scene(tmp_path)


def test_ppo_resume_preserves_optimizer_environment_and_rng(tmp_path):
    from kart.ppo import train
    params = json.loads(open('artifacts/sim_params.json').read())
    params.update(observation_window=.21, observation_window_jitter=.03, banner_delay=.25,
                  observation_memory=True, observation_height=True, camera_height_gain=0., air_turn_continues=True,
                  startup_duration=.4, startup_phase=.05, startup_phase_jitter=.01,
                  startup_speed_profiles=[[.5, .6, 1.], [1., .8, 1.]], startup_profile_dt=.2,
                  startup_profile_shift=.05, startup_profile_noise=.1)
    param_path = tmp_path/'params.json'
    param_path.write_text(json.dumps(params))
    options = dict(device='cpu', threads=1, seed=7, envs=4, randomize=1.,
                   params=str(param_path), max_time=.3, loop_delay=[.05, .18],
                   scene=None, segments=3, lr=3e-4, horizon=8, frame_skip=2, gamma=.995,
                   lam=.95, minibatches=2, epochs=1, clip=.2, vf=.5, ent=.01,
                   init=None, save_every=1)
    split, full = tmp_path/'split', tmp_path/'full'
    train(SimpleNamespace(**options, out=str(split), iters=1, resume=None))
    train(SimpleNamespace(**options, out=str(split), iters=2, resume=str(split/'checkpoint.pt')))
    train(SimpleNamespace(**options, out=str(full), iters=2, resume=None))
    a = torch.load(split/'checkpoint.pt', weights_only=True)
    b = torch.load(full/'checkpoint.pt', weights_only=True)
    for k in a['model']:
        assert torch.equal(a['model'][k], b['model'][k]), k
    for k in a['environment']['tensors']:
        assert torch.equal(a['environment']['tensors'][k], b['environment']['tensors'][k]), k
    assert torch.equal(a['rng'], b['rng'])
    assert a['environment']['rng'] == b['environment']['rng']
    assert a['iter'] == b['iter'] == 1
