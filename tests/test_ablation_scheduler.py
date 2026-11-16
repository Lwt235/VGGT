"""Resource coordination tests with synthetic GPUs and child processes only."""
import json
from pathlib import Path
import sys
import pytest
import scripts.run_step2_ablations as scheduler


def gpu(index=0, *, util=0, free=40000, total=49140):
    return scheduler.GPU(index, 'GPU-'+str(index), util, free, total-free, total)


def snapshot(*gpus, busy=()):
    return scheduler.Snapshot(tuple(gpus), frozenset(busy))


def write_completion(job):
    job.folder.mkdir(parents=True, exist_ok=True)
    (job.folder/'summary.json').write_text(json.dumps(dict(completed_steps=18000, profile=False, ablation=job.variant)))
    (job.folder/'evaluation_18000_full.json').write_text(json.dumps(dict(step=18000, ablation=job.variant,
        query_inputs=['rgb', 'K'], aggregates=[dict(group='cross_scene_val', queries=45, blind_queries=0),
        dict(group='same_scene_heldout', queries=79, blind_queries=2)], episodes=[{}]*126)))


def test_parse_resources_and_require_two_idle_observations():
    rows = scheduler.parse_gpus('0, GPU-0, 0, 40000, 9140, 49140\n1, GPU-1, 100, 20000, 29140, 49140\n')
    assert rows[0].uuid == 'GPU-0' and rows[1].utilization == 100
    assert scheduler.eligible_gpus(snapshot(gpu(util=100)), snapshot(gpu(util=0))) == []
    assert scheduler.eligible_gpus(snapshot(gpu(util=0)), snapshot(gpu(util=100))) == []
    assert scheduler.eligible_gpus(snapshot(gpu(util=20)), snapshot(gpu(util=20))) == [gpu(util=20)]
    assert scheduler.eligible_gpus(snapshot(gpu()), snapshot(gpu()), occupied=['GPU-0']) == []
    assert scheduler.eligible_gpus(snapshot(gpu()), snapshot(gpu(), busy=['GPU-0'])) == []


def test_memory_threshold_reserve_and_absolute_cap_are_all_enforced():
    assert not scheduler.idle(gpu(free=12287))
    assert scheduler.idle(gpu(free=12288))
    # At 120 GiB total the .16 cap plus reserve exceeds the fixed 12 GiB floor.
    assert not scheduler.idle(gpu(free=21000, total=122880))
    assert scheduler.idle(gpu(free=22000, total=122880))
    assert not scheduler.idle(gpu(free=140000, total=150000))
    with pytest.raises(ValueError):
        scheduler.parse_gpus('0, GPU-0, N/A, 40000, 9000, 49000')


def test_commands_are_fixed_and_resume_is_per_variant():
    fresh = scheduler.command(scheduler.Job('render_only'), 2)
    resumed = scheduler.command(scheduler.Job('pe_only', True), 3)
    assert fresh[fresh.index('--gpu')+1] == '2'
    assert fresh[fresh.index('--memory-fraction')+1] == '0.16'
    assert fresh[fresh.index('--min-free-mib')+1] == '12288'
    assert fresh[fresh.index('--run')+1] == 'step2_ablation_render_only_seed17'
    assert fresh[fresh.index('--until')+1] == '18000' and '--resume' not in fresh
    assert resumed[-1] == '--resume'
    assert resumed[resumed.index('--config')+1] == 'configs/step2_ablation_pe_only.yaml'


def test_only_one_coordinator_can_hold_nonblocking_lifetime_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, 'PROJECT_ROOT', tmp_path)
    first = scheduler.acquire_coordinator_lock()
    try:
        with pytest.raises(BlockingIOError):
            scheduler.acquire_coordinator_lock()
    finally:
        first.close()
    second = scheduler.acquire_coordinator_lock()
    second.close()


def test_existing_runs_require_own_explicit_resume_and_complete_runs_skip(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, 'PROJECT_ROOT', tmp_path)
    fresh = scheduler.Job('render_only')
    assert scheduler.validate_run(fresh) == 'fresh'
    fresh.folder.mkdir(parents=True)
    with pytest.raises(ValueError, match='explicitly'):
        scheduler.validate_run(fresh)
    with pytest.raises(ValueError, match='own training_state'):
        scheduler.validate_run(scheduler.Job('render_only', True))
    (fresh.folder/'training_state.pt').write_bytes(b'own state validated by trainer')
    assert scheduler.validate_run(scheduler.Job('render_only', True)) == 'resume'
    write_completion(fresh)
    assert scheduler.validate_run(fresh) == 'complete'
    data = json.loads((fresh.folder/'evaluation_18000_full.json').read_text())
    data['aggregates'][0]['queries'] = 44
    (fresh.folder/'evaluation_18000_full.json').write_text(json.dumps(data))
    assert not scheduler.completed(fresh)


@pytest.mark.parametrize('gpu_count, expected_parallel', [(1, 1), (2, 2), (3, 3)])
def test_scheduler_serializes_one_card_and_spreads_parallel_jobs(tmp_path, monkeypatch, gpu_count, expected_parallel):
    monkeypatch.setattr(scheduler, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['run_step2_ablations.py'])
    monkeypatch.setattr(scheduler.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(scheduler, 'read_resources', lambda: snapshot(*(gpu(i) for i in range(gpu_count))))
    active, launches, peaks = {}, [], []
    class Process:
        def __init__(self, args, **kwargs):
            self.pid = 100+len(launches)
            self.run = args[args.index('--run')+1]
            self.gpu = int(args[args.index('--gpu')+1])
            assert self.gpu not in active
            assert kwargs['start_new_session'] is True
            active[self.gpu] = self
            self.polls = 0
            launches.append(self)
            peaks.append(len(active))
        def poll(self):
            self.polls += 1
            if self.polls <= 3:
                return None
            active.pop(self.gpu)
            variant = next(name for name in scheduler.VARIANTS if self.run == scheduler.Job(name).run)
            write_completion(scheduler.Job(variant))
            return 0
    monkeypatch.setattr(scheduler.subprocess, 'Popen', Process)
    assert scheduler.main() == 0
    assert len(launches) == 3 and max(peaks) == expected_parallel
    text = (tmp_path/'results/step2b_protocol/ablation_scheduler.log').read_text()
    assert 'ALL_ABLATIONS_COMPLETE' in text


def test_child_failure_stops_new_launches_without_signaling_other_children(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['run_step2_ablations.py', '--max-parallel', '2'])
    monkeypatch.setattr(scheduler.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(scheduler, 'read_resources', lambda: snapshot(gpu(0), gpu(1)))
    launches = []
    class Process:
        def __init__(self, args, **kwargs):
            self.pid = 100+len(launches)
            self.variant = scheduler.VARIANTS[len(launches)]
            self.calls = 0
            launches.append(self)
        def poll(self):
            self.calls += 1
            if self.variant == 'render_only':
                return None if self.calls == 1 else 7
            if self.calls == 1:
                return None
            write_completion(scheduler.Job(self.variant))
            return 0
        def terminate(self):
            raise AssertionError('No child may be terminated by the coordinator')
        kill = terminate
    monkeypatch.setattr(scheduler.subprocess, 'Popen', Process)
    assert scheduler.main() == 1
    assert len(launches) == 2 and launches[1].calls == 2
    text = (tmp_path/'results/step2b_protocol/ablation_scheduler.log').read_text()
    assert 'FAILURE' in text and 'pending_unlaunched=step2_ablation_isotropic_point_seed17' in text
    assert 'ALL_ABLATIONS_COMPLETE' not in text


def test_dry_run_shows_commands_without_starting_child_or_writing_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['run_step2_ablations.py', '--dry-run'])
    monkeypatch.setattr(scheduler.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(scheduler, 'read_resources', lambda: snapshot(gpu()))
    def no_launch(*args, **kwargs):
        raise AssertionError('dry run must not launch a process')
    monkeypatch.setattr(scheduler.subprocess, 'Popen', no_launch)
    assert scheduler.main() == 0
    assert not (tmp_path/'results').exists()
