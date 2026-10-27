"""Detached training job: smoke/resume, PPO, paired simulation evaluation and report.

v5 is pure PPO from scratch. Earlier profiles remain available for reproduction.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kart.scenes import sha256


def run(args):
    subprocess.run([sys.executable, *args], check=True)


def completed(path):
    return path.exists() and json.loads(path.read_text()).get('status') == 'completed'


def restore_initial_run(root):
    """Initialize mutable output from immutable, audited resume inputs once."""
    root=Path(root)
    if (root/'run').exists() or not (root/'resume').exists():
        return
    migration=json.loads((root/'checkpoint-migration.json').read_text())
    if sha256(root/'resume/checkpoint.pt')!=migration['migrated_checkpoint_sha256']:
        raise ValueError('Resume checkpoint differs from the audited migration')
    staging=root/f'run-initializing-{os.getpid()}'
    shutil.copytree(root/'resume',staging)
    staging.rename(root/'run')


def main(a):
    root=Path.cwd();status=root/'job-status.json'
    profile = getattr(a, 'profile', 'v2')
    pure = profile in ('v5', 'v6')
    balanced = profile == 'v6'
    v3 = profile in ('v3','v4','v5','v6')
    def phase(name, **extra):
        row=dict(status=name,updated_at=time.time(),**extra)
        temporary=status.with_suffix('.tmp');temporary.write_text(json.dumps(row,indent=2)+'\n');temporary.replace(status)
        print(json.dumps(row),flush=True)
    common=['-m','kart.ppo','--device',a.device,'--threads','4','--params','params.json',
            '--loop-delay','0','0','--save-every','10']
    if pure:
        common += ['--pure-ppo','--lr','0.0003','--seed','20261031' if balanced else '20261029']
        if balanced:
            common += ['--balanced-scenes','--ent-final','.001','--ent-anneal-iters','1000']
    else:
        common += ['--init','initial.pt' if v3 else 'baseline.pt',
                   '--lr','0.00001' if profile=='v4' else '0.00005' if v3 else '0.0001',
                   '--seed','20261027' if profile=='v4' else '20261026' if v3 else '20261024',
                   '--replay-data','recordings.npz','--replay-coef','2' if v3 else '1']
    if getattr(a,'gpu_memory_gb',0):
        common += ['--gpu-memory-gb',str(a.gpu_memory_gb)]
    if v3:
        common += ['--case','level645-case.json','--case-probability','.15' if balanced else '.25',
                   '--target-kl','.015','--frame-skip-max','3','--timing','--deterministic']
    try:
        if a.device == 'cuda':
            import torch
            if not torch.cuda.is_available():raise RuntimeError('Spark CUDA is unavailable')
        phase('smoke')
        smoke=common+['--envs','8','--horizon','16','--segments','6','--max-time','5',
                      '--epochs','2','--minibatches','2','--out','gpu-smoke']
        if not pure: smoke += ['--replay-batch','128','--guard-every','1']
        if v3:
            smoke += ['--validation-envs','4' if balanced else '2','--eval-every','1']
        if not (root/'gpu-smoke-resumed.json').exists():
            cp=root/'gpu-smoke/checkpoint.pt'
            if not cp.exists():run(smoke+['--iters','3'])
            if json.loads((root/'gpu-smoke/status.json').read_text()).get('iter',-1)<3:
                run(smoke+['--iters','4','--resume',str(cp)])
            if pure:
                import torch
                checkpoint=torch.load(cp,map_location='cpu',weights_only=True)
                if (checkpoint['iter'] != 3 or not checkpoint['config']['pure_ppo']
                        or checkpoint['config']['init'] or checkpoint['config']['replay_data']
                        or not all(torch.isfinite(t).all() for t in checkpoint['model'].values())):
                    raise RuntimeError('Smoke update/resume verification failed')
                smoke_result=dict(passed=True,iter=3,finite_weights=True,resume_verified=True,
                                  recording_guard_enabled=False)
            else:
                smoke_result=json.loads((root/'gpu-smoke/guard.jsonl').read_text().splitlines()[-1])
                if not smoke_result['passed']:raise RuntimeError('Smoke update failed recording guard')
            (root/'gpu-smoke-resumed.json').write_text(json.dumps(smoke_result,indent=2)+'\n')
        phase('training')
        if pure:
            if (root/'resume').exists():
                raise ValueError('Pure PPO cannot import a previous experiment checkpoint')
        else:
            restore_initial_run(root)
        train=common+['--envs',str(a.envs),'--horizon','64','--segments','16','--max-time','60',
                      '--iters',str(a.iters),'--out','run']
        if not pure: train += ['--replay-batch','256','--guard-every','25']
        if v3:
            train += ['--validation-envs','128' if balanced else '32',
                      '--eval-every','50' if balanced else '25',
                      '--early-stop-patience','16' if balanced else '12',
                      '--early-stop-min-iters','1000' if balanced else '500',
                      '--early-stop-delta','.0001' if pure else '.002']
        if not completed(root/'run/status.json'):
            if (root/'run/checkpoint.pt').exists():train+=['--resume','run/checkpoint.pt']
            run(train)
        phase('evaluation')
        models=['baseline=baseline.pt','last=run/last.pt']
        if v3: models.insert(1,'previous=previous.pt' if pure else 'previous=initial.pt')
        if (root/'run/best.pt').exists():models.insert(1,'best=run/best.pt')
        if balanced and (root/'prior.pt').exists():models.append('prior=prior.pt')
        evaluations=[('params.json','evaluation')]
        if profile=='v4':evaluations.append(('params-strict48.json','evaluation-strict48'))
        for params_name,output_name in evaluations:
            evaluation=root/output_name
            if evaluation.exists() and not (evaluation/'report.json').exists():
                evaluation.rename(root/f'{output_name}-incomplete-{time.time_ns()}')
            if (evaluation/'report.json').exists():
                existing=json.loads((evaluation/'report.json').read_text())
                required={'standard16','long30','bends16','ramp','relief','middle16','level645'}
                if not balanced or required <= set(existing.get('suites', {})):
                    continue
                evaluation.rename(root/f'{output_name}-incomplete-{time.time_ns()}')
            evaluation_args=['tools/eval_survival.py','--device',a.device,'--threads','4','--envs','128',
                 '--params',params_name,'--models',*models,'--suites',
                 'standard16','long30','bends16','ramp','relief']
            if v3: evaluation_args += ['middle16','level645','--case','level645-case.json']
            if balanced: evaluation_args += ['--robust-scenes','--sampled-diagnostics']
            if getattr(a,'gpu_memory_gb',0):
                evaluation_args += ['--gpu-memory-gb',str(a.gpu_memory_gb)]
            run(evaluation_args+['--seed','20261101' if balanced else '20261030' if pure else '20261028' if profile=='v4' else '20261027' if v3 else '20261025','--out',output_name])
        report=json.loads((root/'evaluation/report.json').read_text())
        gate=json.loads((root/'run/best-guard.json').read_text()) if (root/'run/best-guard.json').exists() else None
        selection=json.loads((root/'run/best-selection.json').read_text()) if pure else gate
        eligible=bool(selection and (pure or selection['passed'])
                      and selection['model_sha256']==sha256(root/'run/best.pt'))
        if balanced:
            eligible &= bool(selection and selection.get('fixed_validation', {}).get('selection_qualified'))
        if eligible:
            for suite in ['standard16','long30','relief']:
                scores=report['suites'][suite]['models']
                eligible &= scores['best']['success_rate'] >= scores['baseline']['success_rate']
            if v3:
                for suite in ['standard16','long30','bends16','middle16','level645','ramp','relief']:
                    scores=report['suites'][suite]['models']
                    eligible &= scores['best']['success_rate'] >= scores['previous']['success_rate']
                    if balanced and 'prior' in scores:
                        eligible &= scores['best']['success_rate'] >= scores['prior']['success_rate']
                eligible &= report['suites']['level645']['models']['best']['success_rate'] >= .9
        candidate=dict(stage='await_phone_comparison' if eligible else 'rejected_offline',
            deployed=False,phone_test_performed=False,eligible_for_phone=eligible,
            model='run/best.pt' if selection else None,selection=selection,
            evaluation='evaluation/report.json',
            profile=profile, training_status=json.loads((root/'run/status.json').read_text()),
            strict_evaluation='evaluation-strict48/report.json' if profile=='v4' else None,
            limitation='All rollout results are synthetic; phone use requires separate comparison.')
        if pure:
            candidate['training_mode']='pure_ppo_from_random_initialization'
        else:
            candidate['recording_guard']=gate
        (root/'candidate.json').write_text(json.dumps(candidate,indent=2)+'\n')
        phase('completed',eligible_for_phone=eligible,candidate='candidate.json')
    except BaseException as error:
        phase('failed',error=str(error))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda');p.add_argument('--envs',type=int,default=256)
    p.add_argument('--iters',type=int,default=2000)
    p.add_argument('--gpu-memory-gb',type=float,default=0.)
    p.add_argument('--profile',choices=['v2','v3','v4','v5','v6'],default='v6')
    config=Path('job-config.json')
    if config.exists():
        defaults=json.loads(config.read_text())
        if set(defaults)-{'device','envs','iters','profile','gpu_memory_gb'}: raise ValueError('Unknown job configuration key')
        p.set_defaults(**defaults)
    main(p.parse_args())
