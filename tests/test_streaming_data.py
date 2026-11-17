"""Train/val isolation, seed gauge and small exact spatial checkpoint tests."""
from pathlib import Path
import random

import pytest
import torch
import yaml

from src.data import QueryInput, Supervision
from src.geometry import unproject
from src.streaming_data import StreamingDataset


def split_fixture(tmp_path):
    frames = list(range(10, 20))
    train = [10, 11, 12, 14, 15, 16, 18, 19]
    item = dict(id='train/room', path='/readonly/train', partition='train',
        frame_ids=frames, train_frame_ids=train, heldout_frame_ids=[13, 17],
        surface_overlap=[[0. if i == j else .8 for j in frames] for i in frames])
    val = dict(item, id='val/unseen', path='/readonly/val', partition='val')
    path = tmp_path/'splits.yaml'
    path.write_text(yaml.safe_dump(dict(sequences=[item, val])))
    return path


def test_stream_sampling_uses_train_frames_for_seed_and_queries(tmp_path):
    dataset = StreamingDataset(split_fixture(tmp_path), (28, 42))
    rng = random.Random(17)
    modes = set()
    for _ in range(100):
        stream = dataset.sample(rng)
        assert stream['scene'] == 'train/room'
        assert 5 <= len(stream['frame_ids']) <= 8
        assert set(stream['frame_ids']).issubset({10, 11, 12, 14, 15, 16, 18, 19})
        modes.add(stream['mode'])
    assert modes == {'explore', 'loiter', 'revisit'}
    # Metadata sampling decodes no image and retains no RGB cache.
    assert dataset.scenes == {}
    assert not hasattr(dataset, 'cache')


def test_stream_training_rejects_heldout_and_val_before_loading(tmp_path):
    dataset = StreamingDataset(split_fixture(tmp_path), (28, 42))
    with pytest.raises(ValueError, match='val/heldout'):
        dataset.load_stream('train/room', [10, 13], training=True)
    with pytest.raises(ValueError, match='val/heldout'):
        dataset.load_stream('val/unseen', [10, 11], training=True)
    assert dataset.scenes == {}


def test_stream_load_is_lazy_and_first_camera_defines_gauge(tmp_path, monkeypatch):
    loaded = []

    class FakeScene:
        frame_ids = list(range(10, 20))

        def __init__(self, path):
            assert path == '/readonly/train'

        def load(self, index, resize_hw):
            loaded.append(index)
            pose = torch.eye(4)
            pose[0, 3] = 10.+index*.2
            K = torch.tensor([[20., 0., 1.], [0., 20., .5], [0., 0., 1.]])
            depth = torch.ones(2, 3)*2.
            return QueryInput(torch.zeros(3, 2, 3), K), Supervision(
                depth, pose, unproject(depth, K, pose), torch.ones(2, 3, dtype=torch.bool))

    monkeypatch.setattr('src.streaming_data.Sun3DScene', FakeScene)
    dataset = StreamingDataset(split_fixture(tmp_path), (2, 3))
    stream = dataset.load_stream('train/room', [14, 15, 16], training=True)
    assert not loaded
    observations = iter(stream['observations'])
    query, seed = next(observations)
    torch.testing.assert_close(seed.c2w, torch.eye(4))
    assert loaded == [4]
    _, later = next(observations)
    torch.testing.assert_close(later.c2w[:3, 3], torch.tensor([.2, 0., 0.]), atol=1e-6, rtol=0.)
    assert set(vars(query)) == {'rgb', 'K'}
    assert loaded == [4, 5]


def test_formal_step2_split_streaming_windows_do_not_include_val_or_heldout():
    root = Path(__file__).resolve().parents[1]
    dataset = StreamingDataset(root/'configs/step2_splits.yaml', (168, 224))
    assert {'harvard_c5/hv_c5_1', 'hotel_umd/maryland_hotel3',
            'mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika'} <= set(dataset.training_windows)
    assert all(dataset.sequences[sid]['partition'] == 'train' for sid in dataset.training_windows)
    for _ in range(50):
        stream = dataset.sample(random.Random(_))
        item = dataset.sequences[stream['scene']]
        assert not set(stream['frame_ids']) & set(item['heldout_frame_ids'])


def test_spatial_delta_restore_is_exact_and_rejects_names():
    from scripts.train_step3 import load_spatial_delta
    engine = torch.nn.Linear(3, 2)
    delta = {name:parameter.detach().clone()+.3 for name,parameter in engine.named_parameters()}
    load_spatial_delta(engine, delta)
    for name, parameter in engine.named_parameters():
        torch.testing.assert_close(parameter, delta[name], rtol=0., atol=0.)
    with pytest.raises(ValueError, match='names'):
        load_spatial_delta(engine, {})


def test_checkpoint_retains_only_latest_and_final_hardlink(tmp_path, monkeypatch):
    from scripts.train_step3 import save_checkpoint
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    engine = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(engine.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    config = dict(steps=3, base_checkpoint='results/immutable_full/training_state_18000.pt')
    save_checkpoint(engine, optimizer, scheduler, config, tmp_path, 1)
    save_checkpoint(engine, optimizer, scheduler, config, tmp_path, 2)
    assert sorted(path.name for path in tmp_path.iterdir()) == ['training_state.pt']
    save_checkpoint(engine, optimizer, scheduler, config, tmp_path, 3)
    final, latest = tmp_path/'training_state_00003.pt', tmp_path/'training_state.pt'
    assert final.stat().st_ino == latest.stat().st_ino
    checkpoint = torch.load(latest, weights_only=True)
    assert checkpoint['step'] == 3
    assert checkpoint['base_checkpoint'] == config['base_checkpoint']
    assert checkpoint['trainable'].keys() == dict(engine.named_parameters()).keys()


def test_resume_replays_interrupted_metric_tail_without_duplicate_updates(tmp_path):
    import json
    from scripts.train_step3 import reconcile_metrics_for_resume
    path=tmp_path/'metrics.jsonl'
    path.write_text(''.join(json.dumps(dict(step=i,loss=1/i))+'\n' for i in range(1,6)))
    with path.open('a') as handle:handle.write('{"step":6,"loss":')
    reconcile_metrics_for_resume(tmp_path,3)
    assert [json.loads(line)['step'] for line in path.read_text().splitlines()]==[1,2,3]
    interrupted=(tmp_path/'metrics_interrupted.jsonl').read_text()
    assert '"step": 4' in interrupted and '"step": 5' in interrupted and '"step":6' in interrupted
    with path.open('a') as handle:
        for i in (4,5):handle.write(json.dumps(dict(step=i,loss=1/i))+'\n')
    reconcile_metrics_for_resume(tmp_path,5)
    assert [json.loads(line)['step'] for line in path.read_text().splitlines()]==[1,2,3,4,5]
    assert not (tmp_path/'metrics.jsonl.tmp').exists()


def test_resume_rejects_missing_checkpoint_metric_prefix(tmp_path):
    from scripts.train_step3 import reconcile_metrics_for_resume
    with pytest.raises(ValueError,match='no corresponding metric prefix'):
        reconcile_metrics_for_resume(tmp_path,3)
    (tmp_path/'metrics.jsonl').write_text('{"step":1}\n{"step":3}\n')
    with pytest.raises(ValueError,match='do not agree'):
        reconcile_metrics_for_resume(tmp_path,3)
