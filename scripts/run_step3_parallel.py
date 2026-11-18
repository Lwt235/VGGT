#!/usr/bin/env python3
"""Live-resource checked torchrun resume and independent stream evaluation.

No process is signalled here. Each evaluation worker owns exactly one plan
directory; only this coordinator merges manifests and summaries after exit.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_step3_pipeline import resources, available


def select_gpus(first, second, count, fraction, *, minimum_free_mib=20480):
    before = {g.uuid:g for g in first.gpus}
    return [g for g in available(first,second,fraction,allow_shared=True)
        if min(g.free_mib,before[g.uuid].free_mib) >= minimum_free_mib][:count]


def merge_evaluations(output, plans, checkpoint):
    """Merge in immutable protocol order; workers never share root metadata."""
    summaries = []
    for plan in plans:
        path = output/plan['name']/'comparison.json'
        value = json.loads(path.read_text())
        if value['plan'] != plan:
            raise ValueError('Parallel evaluation plan mismatch: '+plan['name'])
        summaries.append(value)
    from src.vis.step2 import dump_json
    dump_json(output/'stream_plans.json',plans)
    dump_json(output/'summary.json',dict(checkpoint=checkpoint,
        world_gauge='seed I, metric m; no posterior alignment',
        seed_inputs='RGB/K + seed-only RGB-D; later RGB/K only',streams=summaries))
    dump_json(output/'image_manifest.json',dict(images=sorted(str(p.relative_to(output)) for p in output.rglob('*.png')),
        opened_by_agent=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/step3_v2.yaml')
    parser.add_argument('--run',default='step3_v2_seed17')
    parser.add_argument('--protocol',default='step3_v2_protocol')
    parser.add_argument('--gpus',type=int,default=3,choices=(2,3,4))
    parser.add_argument('--stage',choices=('train','evaluate','all','profile'),default='all')
    parser.add_argument('--skip-tests-before',action='store_true')
    parser.add_argument('--fresh',action='store_true',help='Start a new run from the Full18k base')
    parser.add_argument('--profile-first',action='store_true',help='Require a real DDP profile before the training run')
    parser.add_argument('--minimum-training-gpus',type=int,choices=(2,3,4),default=2)
    args = parser.parse_args()
    results = (ROOT/'results').resolve()
    run_root = (results/args.run).resolve()
    folder = (results/args.protocol).resolve()
    if any(p.parent != results for p in (run_root, folder)):
        parser.error('Use separate run and protocol names directly under results')
    if run_root == folder:
        parser.error('The run and protocol directories must be separate')
    if args.minimum_training_gpus > args.gpus:
        parser.error('Minimum training GPUs exceeds requested GPUs')
    if args.fresh and run_root.exists():
        parser.error('A fresh training run requires a new results directory')
    import yaml
    config = yaml.safe_load((ROOT/args.config).read_text())
    if config.get('streaming_revision') == 'v3' and (
            not args.run.startswith('step3_v3_') or not args.protocol.startswith('step3_v3_')):
        parser.error('v3 requires its own run and protocol directories')
    folder.mkdir(exist_ok=True)
    lock = (folder/'parallel_pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    log = (folder/'parallel_pipeline.log').open('a',buffering=1)
    def report(message):
        line = datetime.now(timezone.utc).isoformat(timespec='seconds')+' '+message
        print(line,flush=True); log.write(line+'\n')

    def checked(count, fraction, minimum=20480):
        if shutil.disk_usage(ROOT).free < 512*2**20:
            raise RuntimeError('Disk guard: less than 512 MiB free')
        first = resources(); time.sleep(3); second = resources()
        return select_gpus(first,second,count,fraction,minimum_free_mib=minimum)

    def single_test(stage):
        while True:
            cards = checked(1,.16,12288)
            if cards:break
            report('Waiting for strict-test GPU memory');time.sleep(30)
        env = os.environ.copy();env['REQUIRE_GSPLAT']='1';env['MKL_THREADING_LAYER']='GNU'
        command = [sys.executable,'-B','scripts/run_on_available_gpu.py','--gpu',str(cards[0].index),
            '--memory-fraction','.16','--','/home/liwt/miniColmap/.venv/bin/python','-B','-m','pytest','-q']
        with (folder/(stage+'.log')).open('a') as out:
            report('Launch '+stage+f' GPU {cards[0].index}')
            subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
        report('Completed '+stage)

    def train(profile):
        while True:
            cards = checked(args.gpus,.28)
            if len(cards) >= args.minimum_training_gpus:break
            report(f'Waiting for {args.minimum_training_gpus} GPUs with >=20 GiB free')
            time.sleep(30)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=','.join(g.uuid for g in cards),
            VGGT_GPU_MEMORY_FRACTION='.28',MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='2',
            NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1')
        run = args.run+'_ddp_profile' if profile else args.run
        if profile and (ROOT/'results'/run).exists():
            run += '_'+datetime.now(timezone.utc).strftime('%m%d%H%M%S')
        command = [sys.executable,'-B','-m','torch.distributed.run','--standalone',
            '--nproc_per_node='+str(len(cards)),'--','scripts/train_step3.py','--config',args.config,'--run',run]
        if profile:
            command += ['--profile','3']
        elif not args.fresh:
            command += ['--resume']
        report('Launch trajectory DDP '+str([g.index for g in cards])+' '+str(command))
        with (folder/('ddp_profile.log' if profile else 'ddp_training.log')).open('a') as out:
            subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
        report('DDP naturally exited')
        if profile:
            rows = [json.loads(line) for line in (results/run/'metrics.jsonl').read_text().splitlines()]
            if len(rows) != 3 or max(row['peak_allocated_gib'] for row in rows) > 15.:
                raise RuntimeError('DDP profile failed the three-update, 15 GiB requirement')
            for module in rows[0]['module_grad_norms']:
                if not any(row['module_grad_norms'][module] > 0 for row in rows):
                    raise RuntimeError('DDP profile did not exercise '+module)
            report(f'DDP profile passed: {len(cards)} ranks, {max(row["peak_allocated_gib"] for row in rows):.2f} GiB peak')

    if args.stage in ('train','all','profile'):
        if not args.skip_tests_before:
            single_test('strict_ddp_before')
        if args.stage == 'profile' or args.profile_first:
            train(profile=True)
        if args.stage != 'profile':
            train(profile=False)
        if args.stage in ('train','profile'):return

    checkpoint = f'results/{args.run}/training_state.pt'
    summary = json.loads((run_root/'summary.json').read_text())
    if summary['completed_steps'] != config['steps'] or summary.get('profile'):
        raise ValueError('Full training must finish before evaluation')
    main_plans = json.loads((ROOT/'results/step3_streaming_seed17/evaluation_final/stream_plans.json').read_text())
    loiter_plans = json.loads((ROOT/'results/step3_streaming_seed17/loiter_diagnostic/plans.json').read_text())
    jobs = deque([(p,'evaluation_final',True) for p in main_plans]+
        [(p,'loiter_diagnostic',False) for p in loiter_plans])
    running = {}
    while jobs or running:
        for gpu,(process,handle,plan) in list(running.items()):
            code = process.poll()
            if code is None:continue
            handle.close();del running[gpu]
            if code:
                raise RuntimeError(f'Evaluation {plan} failed {code}; other workers remain independent, no signals sent')
            report('Completed evaluation '+plan+f' on GPU {gpu}')
        if jobs and len(running) < args.gpus:
            cards = checked(args.gpus,.16,12288)
            for card in cards:
                if card.index in running:continue
                if not jobs or len(running)>=args.gpus:break
                plan,subfolder,visualize = jobs.popleft()
                output = f'results/{args.run}/{subfolder}'
                command = [sys.executable,'-B','scripts/run_on_available_gpu.py','--gpu',str(card.index),
                    '--memory-fraction','.16','--',sys.executable,'-B','scripts/evaluate_step3.py',
                    '--config',args.config,'--checkpoint',checkpoint,'--output',output,
                    '--plans',('results/step3_streaming_seed17/evaluation_final/stream_plans.json' if visualize
                               else 'results/step3_streaming_seed17/loiter_diagnostic/plans.json'),
                    '--streams',plan['name'],'--worker']
                if not visualize:command.append('--no-visualize')
                handle = (folder/('eval_'+plan['name']+'.log')).open('a')
                env = os.environ.copy();env['MKL_THREADING_LAYER']='GNU'
                report('Launch independent evaluation '+plan['name']+f' GPU {card.index}')
                process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                running[card.index] = (process,handle,plan['name'])
        if jobs or running:time.sleep(10)
    for subfolder,plans in (('evaluation_final',main_plans),('loiter_diagnostic',loiter_plans)):
        output = run_root/subfolder
        from src.vis.streaming import training_curves
        rows = [json.loads(x) for x in (run_root/'metrics.jsonl').read_text().splitlines() if x.strip()]
        training_curves(output/'vis',rows)
        merge_evaluations(output,plans,checkpoint)
    single_test('strict_tests_after_parallel')
    report('All GPU workers naturally exited. Image review and final report remain.')


if __name__=='__main__':main()
