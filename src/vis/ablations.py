"""Readout-independent V1/V2 for Step2b; never invokes an O5 reader."""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.vis.smoke import v1, v2, array, cloud, axes3d, save, COLD, ORANGE


def sphere_shell(mean, variance, *, detail=10):
    """CPU scalar-radius shell: repeated spherical eigenvalues need no eigh."""
    theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, detail), np.linspace(0, np.pi, detail))
    unit = np.stack((np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)), -1)
    return np.asarray(mean) + unit * (1.5*np.sqrt(max(float(variance), 1e-10)))


def spherical_v2(folder, scene, memory):
    means, covars, alpha = array(memory.means), array(memory.covars), array(memory.alpha)
    variances = np.trace(covars, axis1=-2, axis2=-1)/3
    nearest = np.argsort(np.linalg.norm(means, axis=-1))[:12]
    fig = plt.figure(figsize=(11, 5), layout='constrained')
    for panel, indices in enumerate((range(len(memory)), nearest)):
        ax = fig.add_subplot(1, 2, panel+1, projection='3d')
        for index in indices:
            shell = sphere_shell(means[index], variances[index], detail=9 if panel == 0 else 14)
            ax.plot_surface(shell[..., 0], shell[..., 2], -shell[..., 1], color='#0072b2',
                            alpha=float(np.clip(alpha[index], .05, .95)), linewidth=.15, shade=False)
        axes3d(ax)
        ax.set_title('All token spheres (1.5 sigma)' if panel == 0 else 'Inset: 12 nearest spheres')
    fig.suptitle(f"V2 · {scene['name']} · isotropic scalar support; transparency=learned alpha")
    save(fig, folder/'V2_ellipsoid_shells.png')


@torch.no_grad()
def ablation_visualizations(episode, memory, prediction, folder, *, isotropic=False):
    query, target = episode['query']
    scene = dict(name=episode['name'], mapping=episode['mapping'])
    folder.mkdir(parents=True, exist_ok=True)
    try:
        wall = v1(folder, scene, query, target, prediction)
    except RuntimeError as error:
        plt.close('all')  # v1 may have constructed a figure before rejecting its wall ROI.
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
        for _, truth in scene['mapping']:
            points = cloud(truth.world_points, truth.valid)
            axes[0].scatter(points[:, 0], points[:, 2], s=.5, c=COLD, alpha=.3)
        points = cloud(prediction['world_points'], target.valid)
        axes[0].scatter(points[:, 0], points[:, 2], s=.5, c=ORANGE, alpha=.5)
        axes[0].set(xlabel='world x [m]', ylabel='world z [m]', title='Direct world coordinates; NO alignment')
        axes[1].text(.05, .5, str(error), wrap=True); axes[1].axis('off')
        fig.suptitle(episode['name']+'; wall section unavailable')
        save(fig, folder/'V1_world_overlap.png')
        wall = dict(unavailable_reason=str(error))
    if isotropic:
        spherical_v2(folder, scene, memory)
    else:
        v2(folder, scene, memory)
    return wall
