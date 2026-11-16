"""CPU tests of fixed six-way protocol, pixel bins and uncropped montage inputs."""
import json
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from scripts.compare_step2_ablations import (TAGS, DEFAULT_RUNS, METRICS, STEPS, SCENES,
    validate_episodes, metric_value, summarize_run, make_montage, image_source, collect_curves,
    BIN_REQUIRED_FIELDS, depth_bin_comparison, blind_query_record, BLIND_DIAGNOSTICS)
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


def test_blind_summary_preserves_own_rejection_fields_and_marks_unavailable_null():
    _, data = protocol()
    base = data['episodes'][-1]
    full = dict(base, dustbin_mean=.82, dustbin_rejection_rate=.91,
                refined_dustbin_mean=.87, surface_support_rate=.012)
    single = dict(base, dustbin_mean=.73, dustbin_rejection_rate=.78)
    no_memory = dict(base, surface_support_rate=.012)
    result = {tag:blind_query_record(row, tag) for tag, row in
              (('Full', full), ('Single-map', single), ('No-Memory', no_memory))}
    for key in BLIND_DIAGNOSTICS:
        assert result['Full'][key] == full[key]
    assert result['Single-map']['dustbin_mean'] == .73
    assert result['Single-map']['dustbin_rejection_rate'] == .78
    assert result['Single-map']['refined_dustbin_mean'] is None
    assert result['Single-map']['surface_support_rate'] is None
    assert all(result['No-Memory'][key] is None for key in BLIND_DIAGNOSTICS if key != 'surface_support_rate')
    assert result['No-Memory']['surface_support_rate'] == .012
    assert json.loads(json.dumps(result))['No-Memory']['dustbin_mean'] is None
    assert result['Full']['query_frame_id'] == base['query_frame_id']


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


@pytest.mark.parametrize('no_memory', [False, True])
def test_diagnostic_loader_uses_official_and_exact_trainable_cpu_mock(tmp_path, monkeypatch, no_memory):
    import scripts.diagnose_step2_ablations as diagnostic
    captured = {}
    config = dict(architecture_revision='single_frame_nomemory' if no_memory else 'step2_v5', receive_layer=4)
    if not no_memory:
        config['ablation'] = 'pe_only'
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
    monkeypatch.setattr(diagnostic, 'VGGTNoMemory', MockMemory)
    monkeypatch.setattr(diagnostic, 'load_checkpoint_readonly', lambda _:state)
    assets = dict(official_vggt_source={'path':'official_source'}, vggt_weights={'path':'official_weights'})
    model = diagnostic.load_memory_model(tmp_path, config, assets, no_memory=no_memory)
    expected = dict(source='official_source', weights='official_weights', receive_layer=4)
    if not no_memory:
        expected.update(revision='step2_v5', pose_jitter_probability=0., ablation='pe_only')
    assert captured == expected
    torch.testing.assert_close(model.weight, torch.tensor([.7]))
    state['trainable'] = {}
    with pytest.raises(ValueError, match='parameter names'):
        diagnostic.load_memory_model(tmp_path, config, assets, no_memory=no_memory)


@pytest.mark.parametrize('tag', TAGS)
def test_diagnostic_prediction_selects_only_permitted_memory_branch(tag, monkeypatch):
    import scripts.diagnose_step2_ablations as diagnostic
    query, target, memory, result = object(), object(), object(), object()
    mapping = [object(), object()]
    calls = []

    class Model:
        def build_memory(self, observations):
            calls.append(('fusion', observations))
            return memory

        def __call__(self, *inputs):
            calls.append(('forward', inputs))
            return result

    class Episode:
        def __getitem__(self, key):
            if key == 'query':
                return query, target
            assert tag != 'No-Memory', 'No-Memory must never access mapping observations'
            assert key == 'mapping'
            return mapping

    def one_frame(model, observation):
        calls.append(('unfused_first', observation))
        return memory

    monkeypatch.setattr(diagnostic, 'single_frame_unfused', one_frame)
    assert diagnostic.predict_episode(Model(), Episode(), tag) is result
    if tag == 'No-Memory':
        assert calls == [('forward', (query,))]
    elif tag == 'Single-map':
        assert calls == [('unfused_first', mapping[0]), ('forward', (query, memory))]
    else:
        assert calls == [('fusion', mapping), ('forward', (query, memory))]


def test_depth_statistics_measure_nonconstant_errors_and_signed_bias():
    # Two pixels give a nontrivial P90 distinct from mean/median/max.
    values = torch.tensor([[1., .2, -1., .5], [4., .8, 1., -.5]]).numpy()
    result = summarize_bins({label:[values] for label in DEPTH_BINS})
    for row in result.values():
        assert set(BIN_REQUIRED_FIELDS) <= set(row)
        assert row['world_epe_m'] == pytest.approx(2.5)
        assert row['world_epe_median_m'] == pytest.approx(2.5)
        assert row['world_epe_p90_m'] == pytest.approx(3.7)
        assert row['world_epe_max_m'] == pytest.approx(4.)
        assert row['world_l1_p90_m'] == pytest.approx(.74)
        assert row['raw_depth_l1_m'] == pytest.approx(1.)
        assert row['raw_depth_bias_m'] == pytest.approx(0.)
        assert row['final_depth_l1_m'] == pytest.approx(.5)
        assert row['final_depth_bias_m'] == pytest.approx(0.)


def complete_bins(count=11):
    return {label:dict(valid_pixels=count, **{key:.1 for key in BIN_REQUIRED_FIELDS}) for label in DEPTH_BINS}


def test_final_bins_require_measured_historical_fields_partial_preview_may_show_na(tmp_path):
    bins = {tag:complete_bins() for tag in TAGS}
    merged = tmp_path/'robotics_depth_bins.json'
    del bins['Full']['near_0.1_3m']['world_l1_m']
    merged.write_text(json.dumps(dict(status='complete', robotics_depth_bins=bins)))
    with pytest.raises(ValueError, match='world_l1_m'):
        depth_bin_comparison(tmp_path, plot=False)
    partial = depth_bin_comparison(tmp_path, allow_incomplete=True, plot=False)
    assert partial['status'] == 'incomplete'
    assert 'N/A' in (tmp_path/'robotics_depth_bins_comparison.csv').read_text()
    bins['Full']['near_0.1_3m']['world_l1_m'] = .1
    merged.write_text(json.dumps(dict(status='complete', robotics_depth_bins=bins)))
    assert depth_bin_comparison(tmp_path, plot=False)['status'] == 'complete'
    csv_text = (tmp_path/'robotics_depth_bins_comparison.csv').read_text()
    assert 'N/A' not in csv_text
    assert all(key in csv_text.splitlines()[0] for key in BIN_REQUIRED_FIELDS)


def test_matching_cache_skips_forward_and_binds_method_weights_config_split_specs(tmp_path, monkeypatch):
    import scripts.diagnose_step2_ablations as diagnostic
    import yaml
    sequences, data = protocol()
    specs = [{key:row[key] for key in ('scene', 'mapping_frame_ids', 'query_frame_id', 'split')}
             for row in data['episodes']]
    root = tmp_path/'full_run';root.mkdir()
    split_path = tmp_path/'split.yaml';split_path.write_text('fixed split')
    config = dict(architecture_revision='step2_v5', receive_layer=4,
                  split=str(split_path), image_hw=[168, 224])
    config_path = root/'config.yaml';config_path.write_text(yaml.safe_dump(config))
    checkpoint = root/'training_state_18000.pt';checkpoint.write_bytes(b'fake readonly checkpoint')
    monkeypatch.setattr(diagnostic, 'OUTPUT', tmp_path/'matrix')
    metadata = diagnostic.checkpoint_metadata(root, config, specs, tag='Full')
    robotics = [r for r in specs if r['scene'].startswith('harvard_robotics_lab/')]
    cache = dict(metadata=metadata, episodes=[dict(**r, valid_pixels_by_bin={label:1 for label in DEPTH_BINS})
        for r in robotics], robotics_depth_bins=complete_bins())
    destination = diagnostic.cache_path('Full')
    diagnostic.write_json(destination, cache)

    def must_not_load(*args, **kwargs):
        raise AssertionError('Matching cache must skip all model/GPU loading')

    monkeypatch.setattr(diagnostic, 'load_memory_model', must_not_load)
    assert diagnostic.evaluate_variant('Full', root, object(), specs, {}) == cache
    for tag in ('No-Memory', 'Single-map'):
        changed = diagnostic.checkpoint_metadata(root, config, specs, tag=tag)
        assert diagnostic.read_matching_cache(destination, changed, tag, specs) is None
    changed_specs = [dict(r) for r in specs]
    changed_specs[0]['mapping_frame_ids'] = [1, 0]
    changed = diagnostic.checkpoint_metadata(root, config, changed_specs, tag='Full')
    assert diagnostic.read_matching_cache(destination, changed, 'Full', changed_specs) is None
    checkpoint.write_bytes(b'changed checkpoint')
    changed = diagnostic.checkpoint_metadata(root, config, specs, tag='Full')
    assert diagnostic.read_matching_cache(destination, changed, 'Full', specs) is None
    config_path.write_text(config_path.read_text()+'\n# changed config\n')
    assert diagnostic.checkpoint_metadata(root, config, specs, tag='Full')['config_sha256'] != metadata['config_sha256']
    split_path.write_text('changed split')
    assert diagnostic.checkpoint_metadata(root, config, specs, tag='Full')['split_sha256'] != metadata['split_sha256']


def test_final_merge_uses_all_six_new_bound_caches_without_historical_fallback(tmp_path, monkeypatch):
    import scripts.diagnose_step2_ablations as diagnostic
    import yaml
    _, data = protocol()
    specs = [{key:row[key] for key in ('scene', 'mapping_frame_ids', 'query_frame_id', 'split')}
             for row in data['episodes']]
    robotics = [r for r in specs if r['scene'].startswith('harvard_robotics_lab/')]
    monkeypatch.setattr(diagnostic, 'OUTPUT', tmp_path/'matrix')
    split = tmp_path/'split.yaml';split.write_text('fixed split')
    roots = {tag:tmp_path/tag for tag in TAGS}
    roots['Single-map'] = roots['Full']
    for root in set(roots.values()):
        root.mkdir()
        (root/'training_state_18000.pt').write_bytes(b'fixed checkpoint')
    for tag, root in roots.items():
        config = dict(architecture_revision='single_frame_nomemory' if tag == 'No-Memory' else 'step2_v5',
                      split=str(split), image_hw=[168, 224], receive_layer=4)
        if tag in diagnostic.VARIANTS:
            config['ablation'] = diagnostic.VARIANTS[tag]
        (root/'config.yaml').write_text(yaml.safe_dump(config))
        metadata = diagnostic.checkpoint_metadata(root, config, specs, tag=tag)
        cache = dict(metadata=metadata, episodes=[dict(**r, valid_pixels_by_bin={label:1 for label in DEPTH_BINS})
            for r in robotics], robotics_depth_bins=complete_bins())
        diagnostic.write_json(diagnostic.cache_path(tag), cache)
    historical = roots['Full']/'final_comparison/robotics_depth_bins.json'
    historical.parent.mkdir()
    historical.write_text('must never be read or overwritten')
    result, sources = diagnostic.collect_cached_diagnostics(roots, specs)
    assert set(result) == set(TAGS)
    assert all(source['method'] == tag and '/diagnostics/' in source['cache'] for tag, source in sources.items())
    assert sources['Full']['method'] != sources['Single-map']['method']
    assert historical.read_text() == 'must never be read or overwritten'
    cache_path = diagnostic.cache_path('Full')
    incomplete = diagnostic.read_json(cache_path)
    del incomplete['robotics_depth_bins']['near_0.1_3m']['world_l1_m']
    diagnostic.write_json(cache_path, incomplete)
    result, _ = diagnostic.collect_cached_diagnostics(roots, specs)
    assert 'Full' not in result and len(result) == 5
