"""CPU tests of fixed six-way protocol, pixel bins and uncropped montage inputs."""
import json
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from scripts.compare_step2_ablations import (TAGS, DEFAULT_RUNS, METRICS, STEPS, SCENES,
    validate_episodes, metric_value, summarize_run, make_montage, image_source, collect_curves)
from scripts.diagnose_step2_ablations import accumulate_bins, summarize_bins, DEPTH_BINS


def protocol():
    sequences = {}
    records = []
    for scene, split, observed, blind in [(SCENES[2], 'cross_scene_val', 45, 0),
            (SCENES[0], 'same_scene_heldout', 68, 2),
            (SCENES[3], 'same_scene_heldout', 11, 0)]:
        n = observed+blind
        frame_ids = list(range(n+1))
        overlap = [[.5]*(n+1) for _ in frame_ids]
        for q in range(observed+1, n+1):
            overlap[q][0] = 0.
        sequences[scene] = dict(frame_ids=frame_ids, surface_overlap=overlap)
        for q in range(1, n+1):
            records.append(dict(scene=scene, split=split, query_frame_id=q, mapping_frame_ids=[0],
                                **{key:float(q)/100 for key in METRICS}))
    return sequences, dict(step=18000, episodes=records)


def test_six_method_defaults_and_fixed_physical_45_79_2():
    assert len(TAGS) == 6
    assert DEFAULT_RUNS['RenderOnly'] == 'step2_ablation_render_only_seed17'
    assert DEFAULT_RUNS['PEOnly'] == 'step2_ablation_pe_only_seed17'
    assert DEFAULT_RUNS['IsotropicPoint'] == 'step2_ablation_isotropic_point_seed17'
    sequences, data = protocol()
    rows = validate_episodes(data, sequences)
    report = summarize_run(rows, sequences, 'Full')
    assert report['observable_all']['point_epe_m']['count'] == 124
    assert report['cross_scene_val']['point_epe_m']['count'] == 45
    assert report['same_scene_heldout']['point_epe_m']['count'] == 79
    assert len(report['same_scene_heldout']['tail_queries']) == 10
    assert report['robotics']['point_epe_m']['count'] == 11
    # Learned support is irrelevant to benchmark membership.
    rows[0]['surface_support_rate'] = 0.
    assert len(validate_episodes(data, sequences)) == 126
    with pytest.raises(ValueError, match='order'):
        validate_episodes(dict(step=18000, episodes=rows[::-1]), sequences, reference=rows)
    rows[0]['benchmark_observable'] = False
    with pytest.raises(ValueError, match='observability'):
        validate_episodes(data, sequences)


def test_no_memory_final_camera_identity_is_explicit_and_other_missing_values_fail():
    row = dict(camera_epe_m=.25)
    assert metric_value(row, 'anchored_camera_epe_m', 'No-Memory') == .25
    with pytest.raises(KeyError):
        metric_value(row, 'anchored_camera_epe_m', 'Full')
    with pytest.raises(ValueError, match='Nonfinite'):
        metric_value(dict(camera_epe_m=float('nan')), 'camera_epe_m', 'Full')


def test_curves_use_exact_126_evaluations_and_single_map_has_no_trajectory(tmp_path):
    sequences, data = protocol()
    roots = {tag:tmp_path/tag for tag in TAGS}
    for tag, root in roots.items():
        if tag == 'Single-map':
            continue
        root.mkdir()
        for step in STEPS:
            (root/f'evaluation_{step:05d}_full.json').write_text(json.dumps(dict(data, step=step)))
    curves = collect_curves(roots, sequences, data['episodes'])
    assert curves['Single-map']['status'] == 'N/A'
    assert 'stages' not in curves['Single-map']
    assert [r['step'] for r in curves['PEOnly']['stages']] == list(STEPS)
    (roots['PEOnly']/'evaluation_03000_full.json').unlink()
    with pytest.raises(FileNotFoundError):
        collect_curves(roots, sequences, data['episodes'])


def test_pixel_depth_bins_half_open_masks_world_l1_and_signed_depth_bias():
    depth = torch.tensor([[.09, .1, 2.9, 3., 4.9, 5., 9.9, 10.]])
    truth = torch.zeros(1, 8, 3)
    delta = torch.tensor([1., 2., 2.]).expand_as(truth)
    raw = truth.clone();raw[..., 2] = depth-.5
    final = truth.clone();final[..., 2] = depth+.25
    target = SimpleNamespace(world_points=truth, depth=depth, valid=torch.ones_like(depth, dtype=torch.bool))
    output = dict(world_points=truth+delta, raw_camera_points=raw, camera_points=final)
    bins = {label:[] for label in DEPTH_BINS}
    counts = accumulate_bins(output, target, bins)
    assert list(counts.values()) == [2, 2, 2]
    summary = summarize_bins(bins)
    for row in summary.values():
        assert row['valid_pixels'] == 2
        assert row['world_epe_m'] == pytest.approx(3.)
        assert row['world_l1_m'] == pytest.approx(5/3)
        assert row['raw_depth_bias_m'] == pytest.approx(-.5)
        assert row['final_depth_bias_m'] == pytest.approx(.25)
        assert row['raw_depth_l1_p90_m'] == pytest.approx(.5)
        assert row['final_depth_l1_p90_m'] == pytest.approx(.25)


def test_six_way_montage_keeps_source_pngs_and_marks_no_memory_v2_na(tmp_path):
    roots = {tag:tmp_path/tag for tag in TAGS}
    assert image_source('No-Memory', 'V2', SCENES[0], roots, tmp_path) is None
    assert 'single_map_frame_vis' in str(image_source('Single-map', 'V2', SCENES[0], roots, tmp_path))
    sources, original = [], {}
    for i, tag in enumerate(TAGS):
        if tag == 'No-Memory':
            sources.append((tag, None));continue
        path = tmp_path/(tag+'.png')
        image = Image.new('RGB', (180, 100), (i*30, 100, 150))
        image.putpixel((0, 0), (255, 0, 0))
        image.save(path)
        original[path] = path.read_bytes()
        sources.append((tag, path))
    destination = tmp_path/'montage.png'
    manifest = make_montage(sources, destination, 'CPU structure test')
    assert destination.is_file() and len(manifest) == 6
    assert manifest[1]['source'] is None and manifest[1]['status'].startswith('N/A')
    assert all(path.read_bytes() == content for path, content in original.items())
    assert len([r for r in manifest if r['status'] == 'original_png']) == 5


def test_diagnostic_loader_uses_official_and_exact_ablation_trainable_cpu_mock(tmp_path, monkeypatch):
    import scripts.diagnose_step2_ablations as diagnostic
    captured = {}
    config = dict(architecture_revision='step2_v5', receive_layer=4, ablation='pe_only')
    state = dict(step=18000, config=config, trainable={'weight':torch.tensor([.7])})

    class MockMemory(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def cuda(self):
            # This is a CPU structural mock; no actual CUDA/model initialization.
            return self

    monkeypatch.setattr(diagnostic, 'VGGTSpatialMemory', MockMemory)
    monkeypatch.setattr(diagnostic, 'load_checkpoint_readonly', lambda _:state)
    assets = dict(official_vggt_source={'path':'official_source'}, vggt_weights={'path':'official_weights'})
    model = diagnostic.load_memory_model(tmp_path, config, assets)
    assert captured == dict(source='official_source', weights='official_weights', revision='step2_v5',
                            receive_layer=4, pose_jitter_probability=0., ablation='pe_only')
    torch.testing.assert_close(model.weight, torch.tensor([.7]))
    state['trainable'] = {}
    with pytest.raises(ValueError, match='parameter names'):
        diagnostic.load_memory_model(tmp_path, config, assets)
