"""Actual unseen-query figures and explicit training/validation curves."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.vis.smoke import v1, v2, v3, v4, v5, save, cloud, COLD, ORANGE


def unseen_visualizations(model, episode, memory, full, empty, erased, box, mask, folder):
    from src.geometry import intrinsics_cv_to_edge
    query, target = episode['query']
    scene = dict(name=episode['name'], mapping=episode['mapping'])
    folder.mkdir(parents=True, exist_ok=True)
    h, w = query.rgb.shape[-2:]
    render0 = model.render_reader.render(memory, full['T0'], intrinsics_cv_to_edge(query.K), h, w)
    render1 = model.render_reader.render(memory, full['T1'], intrinsics_cv_to_edge(query.K), h, w)
    try:
        wall = v1(folder, scene, query, target, full)
    except RuntimeError as error:
        # Some SUN3D scenes have no sufficiently sampled vertical wall ROI.
        # Still show the real unaligned clouds and clearly report no section.
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
        for _, truth in scene['mapping']:
            p = cloud(truth.world_points, truth.valid)
            axes[0].scatter(p[:, 0], p[:, 2], s=.5, c=COLD, alpha=.3)
        p = cloud(full['world_points'], target.valid)
        axes[0].scatter(p[:, 0], p[:, 2], s=.5, c=ORANGE, alpha=.5)
        axes[0].set(xlabel='world x [m]', ylabel='world z [m]', title='Direct world coordinates; NO alignment')
        axes[1].text(.05, .5, str(error), wrap=True); axes[1].axis('off')
        fig.suptitle(episode['name']+'; wall section unavailable')
        save(fig, folder/'V1_world_overlap.png')
        wall = dict(unavailable_reason=str(error))
    v2(folder, scene, memory)
    v3(folder, scene, query, target, full, empty, render0, render1)
    v4(folder, scene, query, memory, full, model.receive_layer)
    erasure = v5(folder, scene, query, target, memory, full, erased, box, mask)
    return wall, erasure


def curves(folder, train, validation):
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout='constrained')
    for ax, key, title in zip(axes[0], ('loss', 'point_epe_m', 'camera_epe_m'),
                               ('Training loss', 'Training world EPE [m]', 'Raw camera geometry EPE [m]')):
        data = [r for r in train if r.get('world_supervision_enabled', 1)]
        y = np.array([r[key] for r in data]); x = np.array([r['step'] for r in data])
        ax.plot(x, y, alpha=.15)
        if len(y) >= 40:
            ax.plot(x[39:], np.convolve(y, np.ones(40)/40, mode='valid'))
        ax.set(title=title, xlabel='Updates')
    for ax, key, title in zip(axes[1], ('point_epe_m', 'rotation1_deg', 'translation1_m'),
                               ('Unaligned world EPE [m]', 'Rotation error [deg]', 'Translation error [m]')):
        for group in ('cross_scene_val', 'same_scene_heldout'):
            full = [r for r in validation if r['scope']=='full' and r['group']==group]
            ax.plot([r['step'] for r in full], [r[key] for r in full], 'o-', label=group)
            quick = [r for r in validation if r['scope']=='quick' and r['group']==group]
            if quick:
                ax.scatter([r['step'] for r in quick], [r[key] for r in quick], marker='x', alpha=.5, label=group+' quick subset')
            if key in ('rotation1_deg', 'translation1_m'):
                ax.plot([r['step'] for r in full], [r[key.replace('1_', '0_')] for r in full], '--', alpha=.6)
        ax.set(title=title+'; dashed=T0, solid=T1', xlabel='Updates'); ax.legend(fontsize=6)
    fig.suptitle('Fresh official VGGT / seed17; fixed disjoint validation; evaluation G=I, no read noise')
    save(fig, folder/'training_validation_curves.png')


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
