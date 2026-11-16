#!/usr/bin/env python3
"""CPU-only six-way Step2b comparison; existing runs and PNGs stay read-only.

Run after diagnose_step2_ablations.py and all three 18k ablation evaluations.
--allow-incomplete produces an explicitly partial preview while training runs.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import PROJECT_ROOT, configure_local_storage
configure_local_storage()
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from src.evaluation import benchmark_observable

OUTPUT = PROJECT_ROOT/'results/step2b_matrix_seed17'
STEPS = (0, 1500, 3000, 6000, 12000, 18000)
TAGS = ('Full', 'No-Memory', 'Single-map', 'RenderOnly', 'PEOnly', 'IsotropicPoint')
VARIANTS = {'RenderOnly': 'render_only', 'PEOnly': 'pe_only', 'IsotropicPoint': 'isotropic_point'}
DEFAULT_RUNS = dict(zip(TAGS, ('step2_v5_full_seed17', 'step2_nomemory_seed17',
    'step2_v5_full_seed17', 'step2_ablation_render_only_seed17',
    'step2_ablation_pe_only_seed17', 'step2_ablation_isotropic_point_seed17')))
SCENES = ('harvard_c5/hv_c5_1', 'harvard_c6/hv_c6_1',
          'harvard_c11/hv_c11_2', 'harvard_robotics_lab/hv_s1_1')
METRICS = ('point_epe_m', 'point_l1_m', 'coarse_point_epe_m',
           'rotation0_deg', 'rotation1_deg', 'translation0_m', 'translation1_m',
           'camera_epe_m', 'anchored_camera_epe_m')
BLIND_DIAGNOSTICS = ('dustbin_mean', 'dustbin_rejection_rate', 'refined_dustbin_mean',
                     'surface_support_rate')
CURVE_METRICS = ('point_epe_m', 'point_l1_m', 'rotation0_deg', 'rotation1_deg',
                 'translation0_m', 'translation1_m', 'camera_epe_m', 'anchored_camera_epe_m')
LABELS = ('World EPE [m]', 'World coordinate L1 [m]', 'T0 rotation [deg]', 'T1 rotation [deg]',
          'T0 translation [m]', 'T1 translation [m]', 'Native camera EPE [m]', 'Final camera EPE [m]')
GROUPS = ('observable_all', 'cross_scene_val', 'same_scene_heldout', 'C11', 'robotics', 'C5', 'C6')
DEPTH_LABELS = ('near_0.1_3m', 'mid_3_5m', 'far_5_10m')
BIN_STATISTICS = ('_m', '_median_m', '_p90_m', '_max_m')
BIN_METRICS = ('world_epe', 'world_l1', 'raw_depth_l1', 'final_depth_l1')
BIN_REQUIRED_FIELDS = tuple(name+suffix for name in BIN_METRICS for suffix in BIN_STATISTICS)+(
    'raw_depth_bias_m', 'final_depth_bias_m')


def validate_depth_bins(bins, *, required_tags=TAGS):
    """Complete final tables require measured finite statistics for every bin."""
    for tag in required_tags:
        if tag not in bins:
            raise ValueError('Missing depth-bin method: '+tag)
        for label in DEPTH_LABELS:
            row = bins[tag].get(label, {})
            if not isinstance(row.get('valid_pixels'), int) or row['valid_pixels'] <= 0:
                raise ValueError('Missing valid pixel count: '+tag+'/'+label)
            for key in BIN_REQUIRED_FIELDS:
                if key not in row or row[key] is None or not np.isfinite(float(row[key])):
                    raise ValueError('Missing/nonfinite depth-bin field: '+tag+'/'+label+'/'+key)
    for label in DEPTH_LABELS:
        counts = {bins[tag][label]['valid_pixels'] for tag in required_tags}
        if len(counts) != 1:
            raise ValueError('Depth-bin valid pixel masks differ across methods: '+label)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_root(name):
    root = (PROJECT_ROOT/'results'/name).resolve()
    if not root.is_relative_to((PROJECT_ROOT/'results').resolve()):
        raise ValueError('Run must be inside project results: '+str(name))
    return root


def identities(rows):
    return [(r['scene'], tuple(r['mapping_frame_ids']), r['query_frame_id'], r['split']) for r in rows]


def validate_episodes(data, sequences, *, reference=None, step=18000):
    rows = data['episodes']
    ids = identities(rows)
    if data['step'] != step or len(rows) != 126 or len(set(ids)) != 126:
        raise ValueError('Expected exactly 126 unique fixed queries at step '+str(step))
    if reference is not None and ids != identities(reference):
        raise ValueError('Episode identities, order, or mapping-frame order changed')
    counts = {group: [0, 0] for group in ('cross_scene_val', 'same_scene_heldout')}
    for row in rows:
        observable = benchmark_observable(row, sequences)
        if 'benchmark_observable' in row and bool(row['benchmark_observable']) != observable:
            raise ValueError('Stored benchmark observability differs from fixed physical overlap')
        counts[row['split']][0 if observable else 1] += 1
    if counts != {'cross_scene_val': [45, 0], 'same_scene_heldout': [79, 2]}:
        raise ValueError('Fixed 45+79+2 protocol changed: '+str(counts))
    return rows


def metric_value(row, key, tag):
    # No-Memory defines raw_camera_points == camera_points; old JSON stores
    # that one error only. This is an explicit identity, not an estimate.
    if key == 'anchored_camera_epe_m' and tag == 'No-Memory' and key not in row:
        key = 'camera_epe_m'
    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError('Nonfinite metric '+key+' in '+tag)
    return value


def blind_query_record(row, tag):
    """Preserve each method's measured rejection; absent diagnostics are null."""
    return dict(scene=row['scene'], query_frame_id=row['query_frame_id'],
        mapping_frame_ids=row['mapping_frame_ids'], split=row['split'], benchmark_observable=False,
        **{key:metric_value(row, key, tag) for key in METRICS},
        **{key:row.get(key) for key in BLIND_DIAGNOSTICS})


def stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(count=0, mean=None, median=None, p90=None, maximum=None)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite summary values')
    return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p90=float(np.quantile(values, .9)), maximum=float(values.max()))


def group_rows(rows, sequences, group):
    rows = [r for r in rows if benchmark_observable(r, sequences)]
    if group == 'observable_all':
        return rows
    if group in ('cross_scene_val', 'same_scene_heldout'):
        return [r for r in rows if r['split'] == group]
    prefix = {'C11': 'harvard_c11/', 'robotics': 'harvard_robotics_lab/',
              'C5': 'harvard_c5/', 'C6': 'harvard_c6/'}[group]
    return [r for r in rows if r['scene'].startswith(prefix)]


def summarize_run(rows, sequences, tag):
    result = {}
    for group in GROUPS:
        selected = group_rows(rows, sequences, group)
        result[group] = {key: stats([metric_value(r, key, tag) for r in selected]) for key in METRICS}
        result[group]['tail_queries'] = [dict(scene=r['scene'], query_frame_id=r['query_frame_id'],
            mapping_frame_ids=r['mapping_frame_ids'], **{k:metric_value(r, k, tag) for k in METRICS})
            for r in sorted(selected, key=lambda r:r['point_epe_m'], reverse=True)[:10]]
    return result


def collect_curves(roots, sequences, reference, *, allow_incomplete=False):
    curves = {'Single-map': dict(status='N/A', reason='Same Full 18k weights, one unfused map view; no independent training trajectory')}
    for tag, root in roots.items():
        if tag == 'Single-map':
            continue
        stages = []
        for step in STEPS:
            path = root/f'evaluation_{step:05d}_full.json'
            if not path.exists() and allow_incomplete:
                stages.append(dict(step=step, status='missing', source=str(path)))
                continue
            rows = validate_episodes(read_json(path), sequences, reference=reference, step=step)
            stages.append(dict(step=step, status='complete', source=str(path), sha256=digest(path),
                               groups=summarize_run(rows, sequences, tag)))
        curves[tag] = dict(stages=stages)
    return curves


def plot_curves(curves, output):
    for group in ('observable_all', 'cross_scene_val', 'same_scene_heldout'):
        fig, axes = plt.subplots(2, 4, figsize=(19, 8), layout='constrained')
        for tag in TAGS:
            if tag == 'Single-map':
                continue
            stages = [r for r in curves[tag]['stages'] if r['status'] == 'complete']
            for ax, key, label in zip(axes.flat, CURVE_METRICS, LABELS):
                ax.plot([r['step'] for r in stages], [r['groups'][group][key]['mean'] for r in stages], 'o-', label=tag, ms=4)
                ax.set(xlabel='Updates', ylabel=label)
                ax.set_xticks(STEPS, ['0', '1.5k', '3k', '6k', '12k', '18k'], rotation=30)
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(group+' / fixed physical overlap; query means; no alignment / Single-map has no separate training curve')
        fig.savefig(output/f'curves_{group}.png', dpi=150)
        plt.close(fig)


def image_source(tag, kind, scene, roots, output):
    key = scene.replace('/', '__')
    filename = 'V1_world_overlap.png' if kind == 'V1' else 'V2_ellipsoid_shells.png'
    if tag == 'No-Memory' and kind == 'V2':
        return None
    if tag == 'Single-map':
        base = roots['Full']/'final_comparison' if kind == 'V1' else output
        return base/'single_map_frame_vis'/key/filename
    return roots[tag]/'vis/step_18000'/key/filename


def make_montage(sources, destination, title, *, allow_incomplete=False):
    """Keep every original panel and its coordinate labels; no crops/axis edits."""
    images, manifest = [], []
    for tag, path in sources:
        if path is not None and path.is_file():
            with Image.open(path) as source:
                images.append(source.convert('RGB'))
            manifest.append(dict(method=tag, source=str(path), sha256=digest(path), status='original_png'))
        elif path is None:
            images.append(None)
            manifest.append(dict(method=tag, source=None, status='N/A: no public memory; no shells exist'))
        elif allow_incomplete:
            images.append(None)
            manifest.append(dict(method=tag, source=str(path), status='missing: incomplete preview'))
        else:
            raise FileNotFoundError(path)
    width = max((im.width for im in images if im is not None), default=1650)
    sizes = [(width, round(im.height*width/im.width)) if im is not None else None for im in images]
    panel_height = max((size[1] for size in sizes if size is not None), default=750)
    bar, gap = 60, 18
    canvas = Image.new('RGB', (2*width+3*gap, 3*(panel_height+bar+gap)+bar), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(font_manager.findfont('DejaVu Sans'), 23)
    draw.text((gap, 15), title, fill='black', font=font)
    for index, (im, tag, record, size) in enumerate(zip(images, TAGS, manifest, sizes)):
        x = gap+(index % 2)*(width+gap)
        y = bar+(index//2)*(panel_height+bar+gap)
        draw.text((x, y+12), tag, fill='black', font=font)
        if im is None:
            draw.rectangle((x, y+bar, x+width, y+bar+panel_height), fill='#f0f0f0')
            draw.text((x+35, y+bar+panel_height//2), record['status'], fill='black', font=font)
        else:
            resized = im.resize(size, Image.Resampling.LANCZOS) if im.size != size else im
            canvas.paste(resized, (x, y+bar+(panel_height-size[1])//2))
    canvas.save(destination)
    return manifest


def write_tables(summary, output):
    with (output/'six_way_metrics.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['group', 'method', 'metric', 'queries', 'mean', 'median', 'p90', 'maximum'])
        for tag in TAGS:
            if tag not in summary:
                continue
            for group in GROUPS:
                for key in METRICS:
                    s = summary[tag][group][key]
                    writer.writerow([group, tag, key, s['count'], s['mean'], s['median'], s['p90'], s['maximum']])
    lines = ['# Step2b six-way final comparison', '',
        'Fixed physical overlap: 45 cross-scene + 79 heldout observable queries; 2 blind queries reported separately. Query metrics are unaligned, in meters/degrees. EPE uses Euclidean norm; L1 averages the three absolute world-coordinate errors.', '',
        'No-Memory final camera EPE equals its native camera EPE by model construction. Single-map uses Full 18k weights and the first mapping frame without fusion; it has no separate training curve.', '']
    for group in ('observable_all', 'cross_scene_val', 'same_scene_heldout', 'robotics', 'C11'):
        lines += ['## '+group, '', '| Method | EPE mean / median / P90 / max [m] | L1 [m] | R0 / R1 [deg] | t0 / t1 [m] | Native / final camera EPE [m] |',
                  '|---|---:|---:|---:|---:|---:|']
        for tag in TAGS:
            if tag not in summary:
                lines.append('| '+tag+' | N/A | N/A | N/A | N/A | N/A |')
                continue
            g = summary[tag][group]
            f = lambda key: f"{g[key]['mean']:.4f}"
            epe = ' / '.join(f"{g['point_epe_m'][s]:.4f}" for s in ('mean', 'median', 'p90', 'maximum'))
            lines.append(f"| {tag} | {epe} | {f('point_l1_m')} | {f('rotation0_deg')} / {f('rotation1_deg')} | {f('translation0_m')} / {f('translation1_m')} | {f('camera_epe_m')} / {f('anchored_camera_epe_m')} |")
        lines.append('')
    lines += ['Per-query tails and both blind queries are preserved in JSON. Final depth bins use fresh read-only forwards of each method\'s 18k weights, cached under diagnostics/. All six methods require complete measured bin statistics; partial previews may show N/A.', '']
    (output/'six_way_summary.md').write_text('\n'.join(lines))


def depth_bin_comparison(output, *, allow_incomplete=False, plot=True):
    path = output/'robotics_depth_bins.json'
    if not path.exists():
        if allow_incomplete:
            return dict(status='missing', source=str(path))
        raise FileNotFoundError('Run diagnose_step2_ablations.py first: '+str(path))
    data = read_json(path)
    bins = data['robotics_depth_bins']
    if not allow_incomplete and (data['status'] != 'complete' or set(bins) != set(TAGS)):
        raise ValueError('Final depth-bin diagnostic must include all six methods')
    try:
        validate_depth_bins(bins)
        complete = data['status'] == 'complete'
    except (ValueError, TypeError):
        if not allow_incomplete:
            raise
        complete = False
    keys = BIN_REQUIRED_FIELDS
    labels = DEPTH_LABELS
    with (output/'robotics_depth_bins_comparison.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['method', 'bin', 'valid_pixels', *keys])
        for tag in TAGS:
            for label in labels:
                row = bins.get(tag, {}).get(label, {})
                writer.writerow([tag, label, row.get('valid_pixels', 'N/A'), *[row.get(k, 'N/A') for k in keys]])
    if plot:
        fig, axes = plt.subplots(2, 3, figsize=(17, 8), layout='constrained')
        selected = ('world_epe_m', 'world_l1_m', 'raw_depth_l1_m',
                    'final_depth_l1_m', 'raw_depth_bias_m', 'final_depth_bias_m')
        for ax, key in zip(axes.flat, selected):
            for index, tag in enumerate(TAGS):
                values = [bins.get(tag, {}).get(label, {}).get(key, np.nan) for label in labels]
                ax.bar(np.arange(3)+(index-2.5)*.13, values, width=.13, label=tag)
            ax.set_xticks(np.arange(3), ['Near [.1,3)', 'Mid [3,5)', 'Far [5,10)'])
            ax.set(ylabel=key+' [m]')
            ax.grid(axis='y', alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle('11 fixed robotics queries / pooled valid pixels / no alignment; incomplete-preview fields are N/A, not zero')
        fig.savefig(output/'robotics_depth_bins_six_way.png', dpi=150)
        plt.close(fig)
    return dict(status='complete' if complete else 'incomplete', source=str(path), sha256=digest(path))


def make_parser(*, comparison=True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', default=DEFAULT_RUNS['Full'])
    parser.add_argument('--baseline', default=DEFAULT_RUNS['No-Memory'])
    parser.add_argument('--render-only', default=DEFAULT_RUNS['RenderOnly'])
    parser.add_argument('--pe-only', default=DEFAULT_RUNS['PEOnly'])
    parser.add_argument('--isotropic-point', default=DEFAULT_RUNS['IsotropicPoint'])
    if comparison:
        parser.add_argument('--allow-incomplete', action='store_true')
        parser.add_argument('--no-plots', action='store_true', help='JSON/CSV/Markdown only')
    return parser


def roots_from_args(args):
    return {tag:run_root(name) for tag, name in zip(TAGS, (args.full, args.baseline, args.full,
                args.render_only, args.pe_only, args.isotropic_point))}


def main():
    args = make_parser().parse_args()
    roots = roots_from_args(args)
    full_config = yaml.safe_load((roots['Full']/'config.yaml').read_text())
    sequences = {r['id']:r for r in yaml.safe_load((PROJECT_ROOT/full_config['split']).read_text())['sequences']}
    reference = validate_episodes(read_json(roots['Full']/'evaluation_18000_full.json'), sequences)
    data, sources, missing = {}, {}, []
    for tag in TAGS:
        path = (roots['Full']/'final_comparison/single_map_frame_same_weights.json' if tag == 'Single-map'
                else roots[tag]/'evaluation_18000_full.json')
        if not path.exists() and args.allow_incomplete:
            missing.append(str(path))
            continue
        rows = validate_episodes(read_json(path), sequences, reference=reference)
        config = yaml.safe_load((roots[tag]/'config.yaml').read_text())
        if config['image_hw'] != full_config['image_hw'] or digest(PROJECT_ROOT/config['split']) != digest(PROJECT_ROOT/full_config['split']):
            raise ValueError('Run split or resolution changed: '+tag)
        if tag in VARIANTS and config['ablation'] != VARIANTS[tag]:
            raise ValueError('Wrong ablation configuration: '+tag)
        data[tag] = rows
        sources[tag] = dict(path=str(path), sha256=digest(path))
    summary = {tag:summarize_run(rows, sequences, tag) for tag, rows in data.items()}
    blinds = {tag:[blind_query_record(r, tag)
                    for r in rows if not benchmark_observable(r, sequences)] for tag, rows in data.items()}
    curves = collect_curves(roots, sequences, reference, allow_incomplete=args.allow_incomplete)
    for tag, curve in curves.items():
        missing.extend(stage['source'] for stage in curve.get('stages', []) if stage['status'] == 'missing')
    if not args.no_plots:
        for scene in SCENES:
            for kind in ('V1', 'V2'):
                for tag in TAGS:
                    path = image_source(tag, kind, scene, roots, OUTPUT)
                    if path is not None and not path.is_file():
                        if not args.allow_incomplete:
                            raise FileNotFoundError(path)
                        missing.append(str(path))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    depth_status = depth_bin_comparison(OUTPUT, allow_incomplete=args.allow_incomplete, plot=not args.no_plots)
    if depth_status['status'] != 'complete':
        missing.append(depth_status['source'])
    write_json(OUTPUT/'six_way_summary.json', dict(step=18000, protocol=dict(observable_cross_scene=45,
        observable_heldout=79, blind=2, image_hw=full_config['image_hw'], split=full_config['split'],
        split_sha256=digest(PROJECT_ROOT/full_config['split']), alignment='none', aggregation='equal query weight'),
        status='incomplete' if missing else 'complete', missing=missing, sources=sources, groups_by_method=summary))
    write_json(OUTPUT/'blind_queries.json', blinds)
    write_json(OUTPUT/'validation_curves.json', curves)
    write_tables(summary, OUTPUT)
    write_json(OUTPUT/'diagnostic_source.json', depth_status)
    if not args.no_plots:
        plot_curves(curves, OUTPUT)
        manifest = {}
        for scene in SCENES:
            manifest[scene] = {}
            for kind in ('V1', 'V2'):
                panels = [(tag, image_source(tag, kind, scene, roots, OUTPUT)) for tag in TAGS]
                manifest[scene][kind] = make_montage(panels,
                    OUTPUT/(scene.replace('/', '__')+f'_{kind}_six_way.png'),
                    f'{scene} / {kind} / fixed query / step 18000 / original axes retained',
                    allow_incomplete=args.allow_incomplete)
        write_json(OUTPUT/'figure_sources.json', manifest)
    print(json.dumps(dict(output=str(OUTPUT), methods=list(data), missing=missing), indent=2))


if __name__ == '__main__':
    main()
