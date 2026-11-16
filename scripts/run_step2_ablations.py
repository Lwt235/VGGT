#!/usr/bin/env python3
"""Run three fresh/resumed Step2b variants only on repeatedly idle GPUs.

Source the project's environment before invoking this script. This coordinator
never terminates a process and never downloads or loads model weights.
"""
import argparse
import fcntl
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('render_only', 'pe_only', 'isotropic_point')
MEMORY_FRACTION = .16
MIN_FREE_MIB = 12288
RESERVE_MIB = 2048
MAX_UTILIZATION = 20


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    utilization: float
    free_mib: int
    used_mib: int
    total_mib: int


@dataclass(frozen=True)
class Snapshot:
    gpus: tuple
    project_variant_uuids: frozenset = frozenset()


@dataclass(frozen=True)
class Job:
    variant: str
    resume: bool = False

    @property
    def run(self):
        return 'step2_ablation_'+self.variant+'_seed17'

    @property
    def folder(self):
        return PROJECT_ROOT/'results'/self.run


def parse_gpus(text):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(',')]
        if len(fields) != 6:
            raise ValueError('Malformed nvidia-smi GPU row')
        index, uuid, utilization, free, used, total = fields
        gpu = GPU(int(index), uuid, float(utilization), int(free), int(used), int(total))
        if (not 0 <= gpu.utilization <= 100 or min(gpu.free_mib, gpu.used_mib) < 0
                or gpu.total_mib <= 0 or gpu.free_mib > gpu.total_mib):
            raise ValueError('Invalid nvidia-smi GPU values')
        rows.append(gpu)
    if len({row.index for row in rows}) != len(rows) or len({row.uuid for row in rows}) != len(rows):
        raise ValueError('Duplicate nvidia-smi GPU identity')
    return tuple(rows)


def is_project_variant(pid):
    """Read ownership only; never signal a PID or inspect unrelated arguments."""
    try:
        process = Path('/proc')/str(int(pid))
        command = (process/'cmdline').read_bytes().split(b'\0')
        has_trainer = any(Path(arg.decode(errors='replace')).name == 'train_step2_ablation.py' for arg in command)
        return has_trainer and (process/'cwd').resolve() == PROJECT_ROOT
    except (OSError, ValueError):
        return False


def read_resources():
    gpu_text = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,utilization.gpu,memory.free,memory.used,memory.total',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    process_text = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'],
        text=True, timeout=15)
    occupied = set()
    for line in process_text.splitlines():
        fields = [item.strip() for item in line.split(',')]
        if len(fields) >= 2 and is_project_variant(fields[1]):
            occupied.add(fields[0])
    return Snapshot(parse_gpus(gpu_text), frozenset(occupied))


def idle(gpu):
    # The fixed .16 cap is below 20 GiB on A6000. Reject larger cards if that
    # fraction would violate the project's absolute per-run 20 GiB ceiling.
    cap = gpu.total_mib*MEMORY_FRACTION
    return (gpu.utilization <= MAX_UTILIZATION and cap <= 20*1024
            and gpu.free_mib >= max(MIN_FREE_MIB, cap+RESERVE_MIB))


def eligible_gpus(first, second, occupied=()):
    before = {gpu.uuid:gpu for gpu in first.gpus}
    blocked = set(occupied) | set(first.project_variant_uuids) | set(second.project_variant_uuids)
    return sorted([gpu for gpu in second.gpus if gpu.uuid in before and gpu.uuid not in blocked
        and gpu.index == before[gpu.uuid].index and idle(before[gpu.uuid]) and idle(gpu)],
        key=lambda gpu:(-gpu.free_mib, gpu.index))


def command(job, gpu_index):
    result = [sys.executable, '-B', 'scripts/run_on_available_gpu.py', '--gpu', str(gpu_index),
        '--min-free-mib', str(MIN_FREE_MIB), '--memory-fraction', str(MEMORY_FRACTION), '--',
        sys.executable, '-B', 'scripts/train_step2_ablation.py', '--config',
        'configs/step2_ablation_'+job.variant+'.yaml', '--run', job.run, '--until', '18000']
    if job.resume:
        result.append('--resume')
    return result


def completed(job):
    try:
        summary = json.loads((job.folder/'summary.json').read_text())
        evaluation = json.loads((job.folder/'evaluation_18000_full.json').read_text())
        groups = {row['group']:(row['queries'], row['blind_queries']) for row in evaluation['aggregates']}
        return (summary['completed_steps'] == 18000 and summary.get('profile') is False
            and summary['ablation'] == job.variant and evaluation['step'] == 18000
            and evaluation['ablation'] == job.variant and evaluation['query_inputs'] == ['rgb', 'K']
            and groups == {'cross_scene_val':(45, 0), 'same_scene_heldout':(79, 2)}
            and len(evaluation['episodes']) == 126)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def validate_run(job):
    if not job.folder.resolve().is_relative_to(PROJECT_ROOT/'results'):
        raise ValueError('Run directory must stay inside this project results')
    if completed(job):
        return 'complete'
    if job.folder.exists():
        if not job.resume:
            raise ValueError(job.run+' exists and is incomplete; explicitly name its variant with --resume')
        if not (job.folder/'training_state.pt').is_file():
            raise ValueError(job.run+' has no own training_state.pt to resume')
        return 'resume'
    if job.resume:
        raise ValueError(job.run+' does not exist; --resume requires its own checkpoint')
    return 'fresh'


def acquire_coordinator_lock():
    path = PROJECT_ROOT/'results/step2b_protocol/scheduler.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        handle.close()
        raise
    return handle


class Reporter:
    def __init__(self, *, dry_run=False):
        self.handle = None
        if not dry_run:
            path = PROJECT_ROOT/'results/step2b_protocol/ablation_scheduler.log'
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = path.open('a', buffering=1)

    def __call__(self, message):
        line = datetime.now(timezone.utc).isoformat(timespec='seconds')+' '+message
        print(line, flush=True)
        if self.handle is not None:
            self.handle.write(line+'\n')

    def close(self):
        if self.handle is not None:
            self.handle.close()


def describe(snapshot):
    return '; '.join(f'gpu={gpu.index} uuid={gpu.uuid} util={gpu.utilization:g}% free={gpu.free_mib}MiB '
        f'used={gpu.used_mib}MiB total={gpu.total_mib}MiB own_variant={gpu.uuid in snapshot.project_variant_uuids}'
        for gpu in snapshot.gpus) or 'no GPUs reported'


def collect_finished(running, report):
    failed = False
    for item in list(running):
        code = item['process'].poll()
        if code is None:
            continue
        item['console'].close(); running.remove(item)
        if code != 0 or not completed(item['job']):
            failed = True
            report(f"FAILURE {item['job'].run} pid={item['process'].pid} exit={code}; further launches stopped; existing children continue")
        else:
            report(f"COMPLETED {item['job'].run} gpu={item['gpu'].index} pid={item['process'].pid}")
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', nargs='+', choices=VARIANTS, default=[], metavar='VARIANT',
                        help='Explicitly resume only the named variants from their own checkpoint')
    parser.add_argument('--max-parallel', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--probe-interval', type=float, default=2., help='Seconds between two idle observations (2–30)')
    parser.add_argument('--wait-seconds', type=float, default=15., help='Idle/resource/completion polling interval (1–30)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 2 <= args.probe_interval <= 30 or not 1 <= args.wait_seconds <= 30:
        parser.error('Probe interval must be 2–30 seconds; wait interval must be 1–30 seconds')
    jobs = [Job(variant, variant in args.resume) for variant in VARIANTS]
    try:
        states = {job.variant:validate_run(job) for job in jobs}
    except ValueError as error:
        parser.error(str(error))
    coordinator_lock = None
    if not args.dry_run:
        try:
            coordinator_lock = acquire_coordinator_lock()
        except BlockingIOError:
            parser.error('Another Step2b coordinator holds results/step2b_protocol/scheduler.lock; no work launched')
    report = Reporter(dry_run=args.dry_run)
    pending = [job for job in jobs if states[job.variant] != 'complete']
    for job in jobs:
        report(('SKIPPED verified 18k complete ' if states[job.variant]=='complete' else 'PLANNED '+states[job.variant]+' ')+job.run)
    running = []
    failed = False
    try:
        if args.dry_run:
            first = read_resources(); report('RESOURCE first '+describe(first))
            time.sleep(args.probe_interval)
            second = read_resources(); report('RESOURCE second '+describe(second))
            available = eligible_gpus(first, second)
            report('IDLE_CANDIDATES '+','.join(str(gpu.index) for gpu in available))
            for index, job in enumerate(pending):
                gpu = available[index].index if index < min(len(available), args.max_parallel) else '<next-idle-gpu>'
                report('DRY_RUN '+shlex.join(command(job, gpu)))
            return 0
        while pending or running:
            failed = collect_finished(running, report) or failed
            if failed:
                if not running:
                    report('FAILED pending_unlaunched='+','.join(job.run for job in pending))
                    return 1
                time.sleep(args.wait_seconds)
                continue
            if not pending:
                if running:
                    time.sleep(args.wait_seconds)
                continue
            if len(running) >= args.max_parallel:
                time.sleep(args.wait_seconds)
                continue
            try:
                first = read_resources()
                time.sleep(args.probe_interval)
                second = read_resources()
                occupied = {item['gpu'].uuid for item in running}
                candidates = eligible_gpus(first, second, occupied)
                launched = False
                for gpu in candidates:
                    if not pending or len(running) >= args.max_parallel:
                        break
                    # A third observation immediately before every launch avoids
                    # using stale idle/free-memory data when several cards qualify.
                    live = read_resources()
                    failed = collect_finished(running, report) or failed
                    if failed:
                        break
                    occupied = {item['gpu'].uuid for item in running}
                    if gpu.uuid not in {row.uuid for row in eligible_gpus(second, live, occupied)}:
                        continue
                    job = pending[0]
                    state = validate_run(job)
                    if state == 'complete':
                        pending.pop(0); report('SKIPPED verified 18k complete '+job.run)
                        continue
                    console_path = PROJECT_ROOT/'results'/(job.run+'_console.log')
                    console_path.parent.mkdir(parents=True, exist_ok=True)
                    console = console_path.open('a', buffering=1)
                    try:
                        process = subprocess.Popen(command(job, gpu.index), cwd=PROJECT_ROOT,
                            stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
                    except BaseException:
                        console.close()
                        raise
                    pending.pop(0)
                    running.append(dict(job=job, gpu=gpu, process=process, console=console))
                    launched = True
                    report(f'STARTED {job.run} gpu={gpu.index} uuid={gpu.uuid} pid={process.pid} console={console_path}')
                if not launched and not failed:
                    report('WAIT no repeatedly idle unoccupied GPU; '+describe(second))
                    time.sleep(args.wait_seconds)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                report('RESOURCE_OR_LAUNCH_FAILURE '+str(error)+'; further launches stopped; no process terminated')
                failed = True
        report('ALL_ABLATIONS_COMPLETE')
        return 0
    except KeyboardInterrupt:
        report('INTERRUPTED coordinator only; existing children remain running: '+','.join(str(item['process'].pid) for item in running))
        return 130
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report('FAILURE '+str(error)+'; no process terminated')
        return 1
    finally:
        for item in running:
            item['console'].close()
        report.close()
        if coordinator_lock is not None:
            fcntl.flock(coordinator_lock, fcntl.LOCK_UN)
            coordinator_lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
