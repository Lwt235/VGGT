"""Step2b paired-training protocol checks independent of GPU/assets."""
from pathlib import Path
import copy
import json
import numpy as np
import pytest
import torch
import yaml
from src.data import QueryInput, Supervision
from src.models.ellipsoid_pose import local_se3_exp
from src.geometry import transform_points
from src.vis.ablations import sphere_shell
from scripts.train_step2_ablation import (validate_protocol, validate_replay, validate_episode,
    replay_world, replay_read_jitter, reader_statistics, module_gradient_norms, check_connected, active_count, resume_logs)
import scripts.train_step2_ablation as trainer

ROOT = Path(__file__).resolve().parents[1]


def configuration(name):
    return yaml.safe_load((ROOT/'configs'/name).read_text())


@pytest.mark.parametrize('variant', ('render_only', 'pe_only', 'isotropic_point'))
def test_ablation_configs_share_full_budget_losses_and_optimizer(variant):
    config = configuration('step2_ablation_'+variant+'.yaml')
    validate_protocol(config, configuration('step2_v5.yaml'))
    assert config['seed'] == 17 and config['steps'] == 18000 and config['image_hw'] == [168, 224]
    assert config['full_eval_steps'] == [0, 1500, 3000, 6000, 12000, 18000]
    assert config['checkpoint_milestones'] == [18000]
    assert config['systemic_epe_limit_m'] is None and config['stop_on_epe'] is False
    assert config['mapping_self_observation'] == (variant != 'pe_only')
    broken = copy.deepcopy(config); broken['optimizer']['native_lr'] *= 2
    with pytest.raises(ValueError, match='optimizer'):
        validate_protocol(broken, configuration('step2_v5.yaml'))
    broken = copy.deepcopy(config); broken['systemic_epe_limit_m'] = .5
    with pytest.raises(ValueError, match='must not stop'):
        validate_protocol(broken, configuration('step2_v5.yaml'))


def test_replay_world_changes_only_mapping_and_supervision_leaves_query_rgb_k():
    query = QueryInput(torch.rand(3, 2, 3), torch.eye(3))
    target = Supervision(torch.ones(2, 3), torch.eye(4), torch.randn(2, 3, 3), torch.ones(2, 3, dtype=torch.bool))
    G = torch.eye(4)
    G[:3, :3], G[:3, 3] = local_se3_exp(torch.tensor([.3, -.1, .2]), torch.tensor([1., -2., .5]))
    mapping, changed = replay_world([(query, target)], target, G)
    assert mapping[0][0] is query
    assert list(vars(query)) == ['rgb', 'K']
    torch.testing.assert_close(changed.world_points, transform_points(target.world_points, G))
    torch.testing.assert_close(changed.c2w, G@target.c2w)
    torch.testing.assert_close(mapping[0][1].world_points, changed.world_points)
    assert changed.depth is target.depth and changed.valid is target.valid


def test_replay_uses_recorded_application_decision_and_rejects_invalid_se3():
    replay = dict(world_gauges=torch.eye(4).repeat(2, 1, 1), read_jitters=torch.eye(4).repeat(2, 1, 1),
                  read_jitter_applied=torch.tensor([False, True]), paired_radius_m=torch.tensor([.6, 0.]),
                  paired_requested=torch.tensor([True, False]))
    validate_replay(replay, 2)
    target = Supervision(None, torch.eye(4), None, None)
    assert replay_read_jitter(replay, 0, target) is None
    torch.testing.assert_close(replay_read_jitter(replay, 1, target), torch.eye(4))
    replay['world_gauges'][1, 0, 0] = -1
    with pytest.raises(ValueError, match=r'SE\(3\)'):
        validate_replay(replay, 2)


def test_paired_replay_zero_sentinel_never_requests_extra_erasure():
    replay = dict(world_gauges=torch.eye(4).repeat(2, 1, 1), read_jitters=torch.eye(4).repeat(2, 1, 1),
                  read_jitter_applied=torch.tensor([False, False]), paired_radius_m=torch.tensor([0., .75]),
                  paired_requested=torch.tensor([False, True]))
    validate_replay(replay, 2)
    replay['paired_requested'][0] = True
    with pytest.raises(ValueError, match='radius'):
        validate_replay(replay, 2)


def test_episode_schedule_cannot_leak_heldout_frames_or_query_into_map():
    scenes = {'train': (None, dict(partition='train', train_frame_ids=[1, 2, 3]))}
    validate_episode(dict(scene='train', query_frame_id=1, mapping_frame_ids=[2, 3]), scenes)
    for maps, query in (([1, 2], 1), ([2, 4], 1), ([2, 3], 4), ([], 1)):
        with pytest.raises(ValueError, match='schedule'):
            validate_episode(dict(scene='train', query_frame_id=query, mapping_frame_ids=maps), scenes)


def test_disabled_reader_statistics_and_gradient_checks():
    active = torch.nn.Linear(2, 2)
    active(torch.ones(1, 2)).sum().backward()
    modules = dict(active=active, o5=None)
    norms = module_gradient_norms(modules)
    assert norms['active'] > 0 and norms['o5'] == 'disabled'
    assert reader_statistics(None) == 'disabled'
    check_connected([dict(module_grad_norms=norms)], modules)
    with pytest.raises(RuntimeError, match='active module active'):
        check_connected([dict(module_grad_norms=dict(active=0., o5='disabled'))], modules)
    assert active_count({}) == 0 and active_count(dict(attention=dict(indices=torch.arange(3)))) == 3


def test_sphere_visualization_uses_scalar_support_without_eigenvectors(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('A sphere does not have a unique eigenbasis')
    monkeypatch.setattr(np.linalg, 'eigh', forbidden)
    center = np.array([1., 2., 3.])
    shell = sphere_shell(center, .04, detail=8)
    np.testing.assert_allclose(np.linalg.norm(shell-center, axis=-1), .3, atol=1e-10)


def test_evaluator_handles_no_render_reader_and_preserves_episode_schema(tmp_path, monkeypatch):
    class Dataset:
        scenes = {'a': (None, {}), 'b': (None, {})}
        def evaluation_episodes(self, **kwargs):
            return [dict(scene=scene, split=split, query_frame_id=3, mapping_frame_ids=[1, 2])
                    for scene, split in (('a', 'cross_scene_val'), ('b', 'same_scene_heldout'))]
        def episode(self, scene, mapping_frame_ids, query_frame_id, **kwargs):
            return dict(name=scene, mapping=[], query=(None, None))
    class Model:
        render_reader = None
        pe_reader = None
        def eval(self): return self
        def train(self): return self
        def build_memory(self, mapping): return []
        def __call__(self, query, memory):
            return dict(local_geometry_confidence=torch.ones(1), attention=None)
    monkeypatch.setattr(trainer, 'benchmark_observable', lambda *args: True)
    monkeypatch.setattr(trainer, 'criterion', lambda *args: (None, {key:1. for key in trainer.SUMMARY_KEYS}))
    aggregates = trainer.evaluate(Model(), Dataset(), dict(ablation='pe_only'), tmp_path, 500, quick=True, device='cpu')
    assert [row['queries'] for row in aggregates] == [1, 1]
    output = json.loads((tmp_path/'evaluation_00500_quick.json').read_text())
    assert output['query_inputs'] == ['rgb', 'K']
    assert all(row['active_gaussians'] == 0 for row in output['episodes'])
    assert all('empty_same_weights' not in row for row in output['episodes'])


def test_resume_retains_interrupted_logs_and_replays_from_own_state(tmp_path):
    text = ''.join(json.dumps(dict(step=step))+'\n' for step in (1, 2, 3))
    validation = ''.join(json.dumps(dict(step=step, group='cross_scene_val'))+'\n' for step in (0, 2, 3))
    (tmp_path/'metrics.jsonl').write_text(text)
    (tmp_path/'validation.jsonl').write_text(validation)
    existing = tmp_path/'metrics_interrupted_00003.jsonl'
    existing.write_text('previous interruption\n')
    metrics, rows = resume_logs(tmp_path, 2)
    assert [row['step'] for row in metrics] == [1, 2]
    assert [row['step'] for row in rows] == [0, 2]
    assert existing.read_text() == 'previous interruption\n'
    assert (tmp_path/'metrics_interrupted_00003_1.jsonl').read_text() == text
    assert (tmp_path/'validation_interrupted_00003.jsonl').read_text() == validation
    assert [json.loads(line)['step'] for line in (tmp_path/'metrics.jsonl').read_text().splitlines()] == [1, 2]
    with pytest.raises(ValueError, match='precede own checkpoint'):
        resume_logs(tmp_path, 3)
