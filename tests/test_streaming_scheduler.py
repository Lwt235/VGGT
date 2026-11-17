"""A utilization dip in somebody else's training is not an idle card."""
from scripts.run_step2_ablations import GPU, Snapshot
from scripts.run_step3_pipeline import available


def snapshot(*, busy=False, utilization=0, free=48000, total=49140):
    return Snapshot((GPU(0, 'card0', utilization, free, total-free, total),),
                    frozenset({'card0'}) if busy else frozenset())


def test_scheduler_requires_empty_compute_contexts_and_two_idle_snapshots():
    empty = snapshot()
    assert len(available(empty, empty, .38)) == 1
    assert available(snapshot(busy=True), empty, .38) == []
    assert available(empty, snapshot(busy=True), .38) == []
    assert available(snapshot(utilization=100), empty, .38) == []


def test_scheduler_preserves_memory_reserve_and_absolute_twenty_gib_cap():
    empty = snapshot()
    assert available(empty, snapshot(free=19000), .38) == []
    huge = snapshot(total=96*1024, free=90*1024)
    assert available(huge, huge, .38) == []


def test_explicit_shared_authorization_allows_serial_busy_gpu_with_memory_only():
    busy = snapshot(busy=True, utilization=100, free=40000)
    assert available(busy, busy, .38) == []
    assert len(available(busy, busy, .38, allow_shared=True)) == 1
    insufficient = snapshot(busy=True, utilization=100, free=19000)
    assert available(insufficient, insufficient, .38, allow_shared=True) == []
