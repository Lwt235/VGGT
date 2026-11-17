#!/usr/bin/env python3
"""Wait for idle resources; strict tests -> profile -> train -> evaluate.

This coordinator never signals any process. Every stage obtains two fresh GPU
snapshots and a disk check. Only this project's new Step3 directories are used.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_step2_ablations import read_resources, Snapshot


def resources():
    snapshot = read_resources()
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    # A brief utilization dip in another user's distributed training does not
    # make that GPU idle. Require absence of existing compute contexts too.
    occupied = frozenset(line.strip() for line in active.splitlines() if line.strip())
    return Snapshot(snapshot.gpus, occupied)


def available(first, second, fraction):
    before = {g.uuid: g for g in first.gpus}
    def idle(g):
        cap = g.total_mib*fraction
        return g.utilization <= 15 and cap <= 20*1024 and g.free_mib >= max(12288, cap+2048)
    occupied = first.project_variant_uuids | second.project_variant_uuids
    return sorted([g for g in second.gpus if g.uuid in before and g.uuid not in occupied and idle(g) and idle(before[g.uuid])],
                  key=lambda g: (-g.free_mib, g.index))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--run', default='step3_streaming_seed17')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--skip-profile', action='store_true')
    parser.add_argument('--tests-only', action='store_true')
    args = parser.parse_args()
    folder = ROOT/'results/step3_protocol'
    folder.mkdir(exist_ok=True)
    lock = (folder/'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    log = (folder/'pipeline.log').open('a', buffering=1)
    def report(message):
        line = datetime.now(timezone.utc).isoformat(timespec='seconds')+' '+message
        print(line, flush=True); log.write(line+'\n')

    def run(stage, arguments, *, fraction=.38, require_gsplat=False):
        while True:
            free = shutil.disk_usage(ROOT).free
            if free < 300*2**20:
                raise RuntimeError('Step3 disk guard: less than 300 MiB free; no stage launched')
            first = resources()
            time.sleep(3)
            second = resources()
            candidates = available(first, second, fraction)
            if candidates:
                gpu = candidates[0]
                # run_on_available_gpu performs its own live memory/process check.
                report(f'Launch {stage}: GPU {gpu.index}, free {gpu.free_mib} MiB, cap {fraction}, disk {free//2**20} MiB')
                command = [sys.executable, '-B', 'scripts/run_on_available_gpu.py', '--gpu', str(gpu.index),
                    '--min-free-mib', '12288', '--memory-fraction', str(fraction), '--', sys.executable, '-B', *arguments]
                import os
                environment = os.environ.copy()
                if require_gsplat:
                    environment['REQUIRE_GSPLAT'] = '1'
                with (folder/(stage+'.log')).open('a', buffering=1) as output:
                    code = subprocess.call(command, cwd=ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT)
                report(f'Exit {stage}: {code}')
                if code:
                    raise RuntimeError(f'{stage} failed ({code}); inspect {folder/(stage+".log")}')
                return
            report('Waiting: no repeatedly idle GPU; utilizations '+','.join(f'{g.index}:{g.utilization:g}%' for g in second.gpus))
            time.sleep(max(1, min(args.poll_seconds, 60)))

    run('strict_tests_before', ['-m', 'pytest', '-q'], fraction=.16, require_gsplat=True)
    if args.tests_only:
        return
    if not args.skip_profile:
        profile_run = args.run.replace('seed17', 'profile_seed17')
        if (ROOT/'results'/profile_run).exists():
            profile_run += '_'+datetime.now(timezone.utc).strftime('%m%d%H%M%S')
        run('unroll_profile', ['scripts/train_step3.py', '--run', profile_run, '--profile', '3'])
        profile = json.loads((ROOT/'results'/profile_run/'summary.json').read_text())
        if not profile.get('profile') or profile['completed_steps'] != 3:
            raise RuntimeError('Streaming profile did not complete all three real unroll updates')
    training = ['scripts/train_step3.py', '--run', args.run]
    if args.resume:
        training.append('--resume')
    run('stream_training', training)
    summary = json.loads((ROOT/'results'/args.run/'summary.json').read_text())
    if summary['completed_steps'] != 3000 or summary.get('profile'):
        raise RuntimeError('Step3 training is incomplete; evaluation not launched')
    run('stream_evaluation', ['scripts/evaluate_step3.py', '--checkpoint', f'results/{args.run}/training_state.pt',
        '--output', f'results/{args.run}/evaluation_final'])
    run('strict_tests_after', ['-m', 'pytest', '-q'], fraction=.16, require_gsplat=True)
    report('GPU pipeline finished. Actual image inspection, scientific report, and commit remain agent tasks.')


if __name__ == '__main__':
    main()
