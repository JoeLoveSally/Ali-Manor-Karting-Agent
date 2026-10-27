import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kart.ppo import load_params, train
from kart.rasterenv import RasterKart
from kart.sim import Params, Track
from kart.vecenv import VecKart


@pytest.mark.parametrize('device,index',[('cuda',0),('cuda:3',3)])
def test_cuda_memory_limit_uses_explicit_visible_device_index(monkeypatch,device,index):
    from kart.ppo import limit_cuda_memory
    calls=[]
    monkeypatch.setattr(torch.cuda,'current_device',lambda:0)
    def properties(i):
        assert i==index
        return SimpleNamespace(total_memory=128*1024**3)
    monkeypatch.setattr(torch.cuda,'get_device_properties',properties)
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda f,i:calls.append((f,i)))
    limit_cuda_memory(device,2)
    assert calls==[(2/128,index)]


def test_zero_clearance_keeps_center_on_road_but_rejects_outside_on_ground_and_landing():
    p=load_params('configs/train-params.json')
    assert p.ground_inset == p.landing_inset == p.margin == p.margin_jitter == 0
    for seed in range(20):
        assert p.randomized(np.random.default_rng(seed)).margin == 0
    track=Track(np.ones((100,40),bool),[0,0],10)
    for airborne in [False,True]:
        for x,expected in [(0.1,False),(-0.1,True)]:
            env=RasterKart(track,params=replace(p,speed=0),start=(x,200))
            if airborne:
                env.air.fill_(True);env.z.fill_(.01);env.vz.fill_(-1)
            _,_,info=env.step(torch.zeros(1))
            assert info['fell'].item() == expected


def test_switching_has_no_reward_cost_and_perfect_drift_is_rewarded_once():
    track=Track(np.ones((200,200),bool),[-800,-800])
    p=Params(speed=0,perfect_dist=-1,switch_penalty=0,latency_press=0,latency_release=0)
    env=RasterKart(track,params=p)
    for action in [0,1,0,1,0]:
        assert env.step(torch.tensor([action]))[0].item() == 0
    p=replace(p,speed=100,perfect_dist=10000,perfect_bonus=.5,turn_rate=1000)
    rewarded=RasterKart(track,params=p)
    plain=RasterKart(track,params=replace(p,perfect_bonus=0))
    bonuses=[]
    for _ in range(6):
        r,_,info=rewarded.step(torch.ones(1))
        base,_,_=plain.step(torch.ones(1))
        bonuses.append((r-base).item())
    assert bonuses == pytest.approx([.5,0,0,0,0,0])


def test_zero_inset_does_not_disable_feasible_jump_generation():
    from kart.vecenv import jump_distance_bounds
    p=load_params('configs/train-params.json')
    assert p.constrain_jump_geometry
    env=VecKart(32,params=p,n_segments=16,seed=3,loop_delay=(0,0))
    count=0
    for i in range(env.n):
        sample=replace(p,**{k:env.P[k][i].item() for k in ['speed','boost_gain']})
        for axis,lip,end,l0,l1,_ in env.jumps[i].tolist():
            if axis < 0: continue
            count+=1
            # Straight flights, from slow to boosted, must find landing road.
            for d in jump_distance_bounds(sample):
                x,y=(lip+d,(l0+l1)/2) if axis==0 else ((l0+l1)/2,lip+d)
                assert env.on_road(torch.tensor([[x]]),torch.tensor([[y]]),env=[i]).item()
    assert count > 5


@pytest.mark.parametrize('forbidden', ['init','replay_data'])
def test_pure_ppo_rejects_prior_weights_or_human_labels(forbidden):
    a=SimpleNamespace(pure_ppo=True,init=None,replay_data=None)
    setattr(a,forbidden,'forbidden-input')
    with pytest.raises(ValueError,match='random initialization'):
        train(a)


@pytest.mark.parametrize('profile', ['v5', 'v6'])
def test_pure_job_runs_without_recordings_or_prior_initialization(tmp_path,monkeypatch,profile):
    from tools import train_survival_job
    from kart.scenes import sha256
    monkeypatch.chdir(tmp_path)
    commands=[]
    def fake_run(args):
        commands.append(args)
        folder=tmp_path/args[args.index('--out')+1];folder.mkdir(exist_ok=True)
        if args[0]=='tools/eval_survival.py':
            assert 'previous=previous.pt' in args
            suites={s:{'models':{m:{'success_rate':.95 if m=='best' else .8}
                        for m in ['baseline','previous','best']}}
                    for s in ['standard16','long30','bends16','middle16','level645','ramp','relief']}
            (folder/'report.json').write_text(json.dumps({'suites':suites}));return
        assert '--pure-ppo' in args and '--init' not in args and '--replay-data' not in args
        assert args[args.index('--lr')+1]=='0.0003'
        if profile == 'v6':
            assert '--balanced-scenes' in args and '--ent-final' in args
            if folder.name == 'run':
                assert args[args.index('--validation-envs')+1] == '128'
                assert args[args.index('--early-stop-min-iters')+1] == '1000'
        it=int(args[args.index('--iters')+1])-1
        torch.save(dict(iter=it,config=dict(pure_ppo=True,init=None,replay_data=None),
                        model={'w':torch.zeros(1)}),folder/'checkpoint.pt')
        (folder/'status.json').write_text(json.dumps({'status':'completed','iter':it}))
        (folder/'best.pt').write_bytes(b'best');(folder/'last.pt').write_bytes(b'last')
        (folder/'best-selection.json').write_text(json.dumps({'model_sha256':sha256(folder/'best.pt'),
                                                           'recording_guard_enabled':False,
                                                           'fixed_validation':{'selection_qualified':True}}))
    monkeypatch.setattr(train_survival_job,'run',fake_run)
    train_survival_job.main(SimpleNamespace(device='cpu',envs=4,iters=2,profile=profile))
    report=json.loads((tmp_path/'candidate.json').read_text())
    assert report['eligible_for_phone'] and 'recording_guard' not in report
    assert report['strict_evaluation'] is None
    assert not list(tmp_path.rglob('*guard*'))
    assert any('--resume' in c for c in commands if 'gpu-smoke' in c)
    assert all('--resume' not in c for c in commands if 'run' in c)
