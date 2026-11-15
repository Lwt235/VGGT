"""V1–V5 from actual model outputs and fixed-weight interventions."""

import itertools
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

from src.geometry import patch_intrinsics, world_rays, ray_ellipsoid_distance
from src.losses.world import world_loss, pose_errors

COLD = "#8c969f"
ORANGE = "#ff851b"


def array(value):
    return value.detach().float().cpu().numpy()


def cloud(points, mask=None, max_points=6000):
    points = array(points)
    if mask is not None:
        points = points[array(mask).astype(bool)]
    points = points.reshape(-1, 3)
    points = points[np.isfinite(points).all(-1)]
    return points[::max(1, len(points)//max_points)]


def ellipsoid(ax, mean, covariance, color, alpha, detail=10):
    eigenvalues, rotation = np.linalg.eigh(covariance)
    theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, detail), np.linspace(0, np.pi, detail))
    sphere = np.stack((np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)), -1)
    shell = mean + (sphere * (1.5*np.sqrt(np.maximum(eigenvalues, 1e-10)))) @ rotation.T
    ax.plot_surface(shell[..., 0], shell[..., 2], -shell[..., 1], color=color,
                    alpha=float(np.clip(alpha, .05, .95)), linewidth=.15, edgecolor=color,
                    antialiased=True, shade=False)


def axes3d(ax):
    ax.set_xlabel("world x [m]", fontsize=8, labelpad=2)
    ax.set_ylabel("world z [m]", fontsize=8, labelpad=2)
    ax.set_zlabel("")
    # Keep the vertical coordinate label inside the axes: Matplotlib's 3D
    # tight bounding box otherwise clips it on narrow physical-aspect panels.
    ax.text2D(.03, .97, "vertical = -world y [m]", transform=ax.transAxes, va="top", fontsize=7)
    ax.view_init(elev=25, azim=-55)
    spans = [np.ptp(ax.get_xlim3d()), np.ptp(ax.get_ylim3d()), np.ptp(ax.get_zlim3d())]
    ax.set_box_aspect(np.maximum(spans, .1))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(3))
    ax.tick_params(labelsize=7)


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def v1(folder, scene, query, target, prediction):
    historical = np.concatenate([cloud(gt.world_points, gt.valid, 2500) for _, gt in scene["mapping"]])
    predicted = cloud(prediction["world_points"], target.valid)
    fig = plt.figure(figsize=(14, 4.5), layout="constrained")
    ax = fig.add_subplot(131, projection="3d")
    ax.scatter(historical[:, 0], historical[:, 2], -historical[:, 1], s=.5, color=COLD, alpha=.35)
    ax.scatter(predicted[:, 0], predicted[:, 2], -predicted[:, 1], s=.5, color=ORANGE, alpha=.5)
    axes3d(ax)
    ax.set_title("History RGB-D / model query; NO alignment")
    ax = fig.add_subplot(132)
    ax.scatter(historical[:, 0], historical[:, 2], s=.5, color=COLD, alpha=.4)
    ax.scatter(predicted[:, 0], predicted[:, 2], s=.5, color=ORANGE, alpha=.5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set(xlabel="world x [m]", ylabel="world z [m]", title="Bird's-eye in fixed mapping gauge")
    # Reference plane comes only from a fixed mapping-image wall ROI.
    _, reference = scene["mapping"][0]
    h, w = reference.depth.shape
    gt_world = array(target.world_points)
    # Select a sufficiently sampled vertical planar patch from mapping GT only.
    # It is a fixed diagnostic plane, NEVER a transform fitted to predictions.
    candidates = []
    ref_points = array(reference.world_points)
    ref_valid = array(reference.valid).astype(bool)
    for row in range(3):
        for col in range(4):
            y0, y1, x0, x1 = row*h//4, (row+1)*h//4, col*w//4, (col+1)*w//4
            patch = ref_points[y0:y1, x0:x1][ref_valid[y0:y1, x0:x1]]
            if len(patch) < 40:
                continue
            origin = np.median(patch, axis=0)
            _, _, basis = np.linalg.svd(patch-origin, full_matrices=False)
            n = basis[-1]
            if abs(n[1]) > .65:
                continue  # exclude floor/table surfaces in the mapping gauge
            overlap = (np.abs((gt_world-origin)@n) < .15) & array(target.valid).astype(bool)
            if overlap.sum() < 100:
                continue
            thickness = np.ptp(np.quantile((patch-origin)@n, [.1, .9]))
            candidates.append((thickness, patch, origin, n, overlap, [x0, y0, x1, y1]))
    if not candidates:
        raise RuntimeError("No adequately sampled vertical reference surface for wall cross-section")
    _, ref_wall, center, normal, selected, rectangle = min(candidates, key=lambda item: item[0])
    ref_residual = (ref_wall-center)@normal
    pred_residual = (array(prediction["world_points"])[selected]-center)@normal
    ax = fig.add_subplot(133)
    ax.hist(ref_residual, bins=40, density=True, color=COLD, alpha=.6, label="Mapping reference ROI")
    ax.hist(pred_residual, bins=40, density=True, color=ORANGE, alpha=.6, label="Predicted overlap ROI")
    ax.set(xlabel="Signed distance to fixed reference plane [m]", ylabel="Density", title="Wall ROI cross-section (no output fitting)")
    ax.legend(fontsize=7, loc="upper left")
    inset = ax.inset_axes([.6, .65, .38, .3])
    inset.imshow(np.clip(array(scene["mapping"][0][0].rgb).transpose(1, 2, 0), 0, 1))
    from matplotlib.patches import Rectangle
    x0, y0, x1, y1 = rectangle
    inset.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="cyan", linewidth=1))
    inset.axis("off")
    inset.set_title("Fixed reference ROI", fontsize=6)
    fig.suptitle(f"V1 · {scene['name']} · direct metric world coordinates")
    save(fig, folder/"V1_world_overlap.png")
    return {"mapping_wall_p90_p10_m": float(np.ptp(np.quantile(ref_residual, [.1, .9]))),
            "predicted_wall_p90_p10_m": float(np.ptp(np.quantile(pred_residual, [.1, .9]))),
            "wall_evaluation_pixels": int(selected.sum()), "mapping_roi_xyxy": rectangle,
            "reference_plane_center": center.tolist(), "reference_plane_normal": normal.tolist()}


def v2(folder, scene, memory):
    means, covars, alpha = array(memory.means), array(memory.covars), array(memory.alpha)
    _, vectors = np.linalg.eigh(covars)
    normal_rgb = np.abs(vectors[..., 0])
    fig = plt.figure(figsize=(11, 5), layout="constrained")
    for panel, indices in enumerate((range(len(memory)), np.argsort(np.linalg.norm(means, axis=-1))[:12])):
        ax = fig.add_subplot(1, 2, panel+1, projection="3d")
        for idx in indices:
            ellipsoid(ax, means[idx], covars[idx], normal_rgb[idx], alpha[idx], detail=9 if panel == 0 else 14)
        if panel == 0:
            close = np.argsort(np.linalg.norm(means, axis=-1))[:12]
            lower, upper = means[close].min(0)-.1, means[close].max(0)+.1
            corners = np.array(list(itertools.product(*zip(lower, upper))))
            for a, b in itertools.combinations(corners, 2):
                if np.sum(a != b) == 1:
                    ax.plot([a[0], b[0]], [a[2], b[2]], [-a[1], -b[1]], color="#e9b818", lw=1)
        axes3d(ax)
        ax.set_title("All token Gaussians (1.5 sigma)" if panel == 0 else "Inset: 12 nearest token shells")
    fig.suptitle(f"V2 · {scene['name']} · RGB=|thinnest axis|; transparency=learned alpha")
    save(fig, folder/"V2_ellipsoid_shells.png")


def edges(image, valid=None):
    dy, dx = np.gradient(image.astype(float))
    gradient = np.hypot(dx, dy)
    if valid is None:
        valid = np.ones_like(gradient, dtype=bool)
    if not valid.any():
        return np.zeros_like(gradient, dtype=bool)
    return (gradient > max(np.quantile(gradient[valid], .8), 1e-4)) & valid


def calibration_panel(ax, rgb, rendered, title):
    depth = array(rendered.depth)[..., 0]
    opacity = array(rendered.opacity)[..., 0]
    h, w = depth.shape
    resized = torch.nn.functional.interpolate(torch.from_numpy(rgb).permute(2, 0, 1)[None],
                                               size=(h, w), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy()
    gray = resized.mean(-1)
    actual_edge = edges(gray)
    expected_edge = edges(depth, opacity > .05)
    overlay = np.repeat((gray*.25)[..., None], 3, -1)
    overlay[..., 0] += expected_edge*.85
    overlay[..., 1] += actual_edge*.85
    ax.imshow(np.clip(overlay, 0, 1))
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    inset = ax.inset_axes([.02, .02, .4, .32])
    inset.imshow(depth, cmap="viridis", vmin=0, vmax=6)
    inset.set_title("Memory z [m]", fontsize=6, color="white", backgroundcolor="black")
    inset.axis("off")
    inset = ax.inset_axes([.56, .02, .4, .32])
    inset.imshow(1-opacity, cmap="gray", vmin=0, vmax=1)
    inset.set_title("Unseen 1-O", fontsize=6, color="white", backgroundcolor="black")
    inset.axis("off")


def v3(folder, scene, query, target, full, empty, render0, render1):
    rgb = np.clip(array(query.rgb).transpose(1, 2, 0), 0, 1)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), layout="constrained")
    axes[0].imshow(rgb)
    axes[0].set_title("a · Actual query RGB")
    axes[0].axis("off")
    calibration_panel(axes[1], rgb, render0, "b · T0: red=geometry edge; green=RGB edge")
    calibration_panel(axes[2], rgb, render1, "c · T1: same edge definitions")
    for output, color, label in ((empty, "#527bbe", "Empty memory / same weights"), (full, ORANGE, "Full memory")):
        p = cloud(output["world_points"], target.valid, 2500)
        axes[3].scatter(p[:, 0], p[:, 2], s=1, color=color, alpha=.4, label=label)
    axes[3].set(xlabel="world x [m]", ylabel="world z [m]", title="d · World output comparison")
    axes[3].legend(fontsize=6)
    axes[3].set_aspect("equal", adjustable="datalim")
    fig.suptitle(f"V3 · {scene['name']} · geometric display render 112x140; trained read 8x10; no RGB memory")
    r0, t0 = pose_errors(full['T0'], target.c2w)
    r1, t1 = pose_errors(full['T1'], target.c2w)
    fig.supxlabel(f"Predicted pose only · R0→R1={float(r0):.2f}→{float(r1):.2f} deg; t0→t1={float(t0):.3f}→{float(t1):.3f} m", fontsize=10)
    save(fig, folder/"V3_bootstrap_calibration.png")


def v4(folder, scene, query, memory, prediction, layer):
    rgb = np.clip(array(query.rgb).transpose(1, 2, 0), 0, 1)
    hp, wp = rgb.shape[0]//14, rgb.shape[1]//14
    selection = [(hp//2, wp//4), (hp//2, wp//2), (hp//2, wp-2)]
    info = prediction["attention"]
    weights = array(info["attention"]).mean(0) if info["attention"] is not None else None
    indices = array(info["indices"]).astype(int)
    means, covars = array(memory.means), array(memory.covars)
    fig = plt.figure(figsize=(12, 5), layout="constrained")
    ax_image = fig.add_subplot(121)
    ax_image.imshow(rgb)
    ax_image.set_title("Selected patches: interior / structure / margin")
    ax_image.axis("off")
    ax = fig.add_subplot(122, projection="3d")
    ax.scatter(means[:, 0], means[:, 2], -means[:, 1], s=2, color=COLD, alpha=.2)
    origins, directions, _ = world_rays(patch_intrinsics(query.K), prediction["T0"], hp, wp)
    origins, directions = array(origins), array(directions)
    for number, (y, x) in enumerate(selection):
        ax_image.plot((x+.5)*14-.5, (y+.5)*14-.5, "o", mfc="none", mec="yellow", markersize=12)
        ax_image.text((x+.5)*14, (y+.5)*14, str(number+1), color="yellow", fontsize=10)
        qi = y*wp+x
        end = origins[qi]+directions[qi]*4
        ax.plot([origins[qi, 0], end[0]], [origins[qi, 2], end[2]], [-origins[qi, 1], -end[1]], lw=1.5)
        if weights is not None and len(indices):
            top = np.argsort(weights[qi])[-min(30, len(indices)):]
            for local_idx in top:
                actual_idx = indices[local_idx]
                heat = weights[qi, local_idx]/max(weights[qi].max(), 1e-8)
                ellipsoid(ax, means[actual_idx], covars[actual_idx], plt.cm.inferno(heat), .3+.4*heat, detail=8)
    axes3d(ax)
    ax.set_title(f"Actual attention Top-30 per ray; layer {layer}; mean of 8 heads")
    fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap="inferno"),
                 ax=ax, shrink=.5, pad=.02, label="Attention / per-ray maximum")
    fig.suptitle(f"V4 · {scene['name']} · active set {len(indices)} / public {len(memory)}; heat normalized per ray")
    save(fig, folder/"V4_attention_rays.png")


def v5(folder, scene, query, target, memory, full, erased, box, erased_mask):
    lower, upper = (array(x) for x in box)
    before, after = array(full["world_points"]), array(erased["world_points"])
    valid = array(target.valid).astype(bool)
    delta = np.linalg.norm(after-before, axis=-1)
    target_points = array(target.world_points)
    error_before = np.linalg.norm(before-target_points, axis=-1)
    error_after = np.linalg.norm(after-target_points, axis=-1)
    error_delta = error_after-error_before
    local = ((target_points >= lower) & (target_points <= upper)).all(-1) & valid
    outside = valid & ~local
    fig = plt.figure(figsize=(15, 4), layout="constrained")
    ax = fig.add_subplot(141, projection="3d")
    means = array(memory.means)
    mask = array(erased_mask).astype(bool)
    ax.scatter(means[~mask, 0], means[~mask, 2], -means[~mask, 1], s=4, color=COLD)
    ax.scatter(means[mask, 0], means[mask, 2], -means[mask, 1], s=7, color="red")
    corners = np.array(list(itertools.product(*zip(lower, upper))))
    for a, b in itertools.combinations(corners, 2):
        if np.sum(a != b) == 1:
            ax.plot([a[0], b[0]], [a[2], b[2]], [-a[1], -b[1]], color="red", lw=1)
    axes3d(ax)
    ax.set_title(f"Erase {mask.sum()} / {len(memory)} Gaussians")
    for panel, points, name in ((2, before, "Full memory"), (3, after, "Same weights, bbox erased")):
        ax = fig.add_subplot(1, 4, panel)
        p = points[valid][::3]
        ax.scatter(p[:, 0], p[:, 2], s=1, color=ORANGE, alpha=.4)
        ax.set(xlabel="world x [m]", ylabel="world z [m]", title=name)
        ax.set_aspect("equal", adjustable="datalim")
        # Shared explicit limits reveal global movement; no view recentering.
        both = np.concatenate((before[valid], after[valid]))
        limits = np.quantile(both[:, [0, 2]], [.01, .99], axis=0)
        ax.set_xlim(limits[0, 0]-.1, limits[1, 0]+.1)
        ax.set_ylim(limits[0, 1]-.1, limits[1, 1]+.1)
    ax = fig.add_subplot(144)
    bound = max(.001, float(np.quantile(np.abs(error_delta[valid]), .98)))
    heat = ax.imshow(np.where(valid, error_delta, np.nan), cmap="coolwarm", vmin=-bound, vmax=bound)
    if local.any():
        ax.contour(local.astype(float), levels=[.5], colors=["cyan"], linewidths=.7)
    ax.set_title("EPE change [m]; red=worse; cyan=bbox")
    ax.axis("off")
    fig.colorbar(heat, ax=ax, fraction=.05)
    fig.suptitle(f"V5 · {scene['name']} · 3D bbox intervention, zero retraining, fixed model")
    fig.supxlabel(f"EPE increase inside/outside = {error_delta[local].mean():+.4f}/{error_delta[outside].mean():+.4f} m; full camera prediction is allowed to change", fontsize=10)
    save(fig, folder/"V5_local_erasure.png")
    rotation_change, translation_change = pose_errors(erased["T1"], full["T1"])
    return {"erased_count": int(mask.sum()), "local_pixels": int(local.sum()),
            "mean_change_inside_m": float(delta[local].mean()) if local.any() else None,
            "mean_change_outside_m": float(delta[outside].mean()) if outside.any() else None,
            "full_epe_inside_m": float(error_before[local].mean()) if local.any() else None,
            "erased_epe_inside_m": float(error_after[local].mean()) if local.any() else None,
            "epe_increase_inside_m": float(error_delta[local].mean()) if local.any() else None,
            "full_epe_outside_m": float(error_before[outside].mean()) if outside.any() else None,
            "erased_epe_outside_m": float(error_after[outside].mean()) if outside.any() else None,
            "epe_increase_outside_m": float(error_delta[outside].mean()) if outside.any() else None,
            "mean_abs_epe_change_inside_m": float(np.abs(error_delta[local]).mean()) if local.any() else None,
            "mean_abs_epe_change_outside_m": float(np.abs(error_delta[outside]).mean()) if outside.any() else None,
            "pose_rotation_change_deg": float(rotation_change), "pose_translation_change_m": float(translation_change),
            "box_lower": lower.tolist(), "box_upper": upper.tolist()}


@torch.no_grad()
def deliver_visualizations(model, samples, output_dir, records, *, camera_weight=.5, refinement_safety_weight=0.):
    folder = output_dir/"vis"
    folder.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    def criterion(prediction, target):
        return world_loss(prediction, target, camera_weight=camera_weight, refinement_safety_weight=refinement_safety_weight)
    for scene in samples:
        scene_folder = folder/scene["name"]
        scene_folder.mkdir(exist_ok=True)
        query, target = scene["queries"][1]
        memory = model.build_memory(scene["mapping"])
        full = model(query, memory, return_attention=True)
        empty = model(query, memory.subset(slice(0, 0)))
        from src.geometry import intrinsics_cv_to_edge
        h, w = query.rgb.shape[-2:]
        render0 = model.render_reader.render(memory, full["T0"], intrinsics_cv_to_edge(query.K), h, w)
        render1 = model.render_reader.render(memory, full["T1"], intrinsics_cv_to_edge(query.K), h, w)
        # Select a bbox near the central query surface only in the evaluator;
        # neither query GT nor this evaluator is part of model.forward().
        valid_pixels = target.valid.nonzero()
        desired = valid_pixels.new_tensor([h//2, 3*w//4])
        chosen_pixel = valid_pixels[(valid_pixels-desired).float().square().sum(-1).argmin()]
        center_point = target.world_points[chosen_pixel[0], chosen_pixel[1]]
        center_index = (memory.means-center_point).norm(dim=-1).argmin()
        nearest = memory.means[center_index].detach()
        center = .5*(nearest+center_point)
        radius = torch.maximum(center.new_tensor([.4, .5, .4]), (nearest-center_point).abs()*.5+.1)
        box = (center-radius, center+radius)
        erased_memory, mask = memory.erase_box(*box)
        erased = model(query, erased_memory)
        _, full_metrics = criterion(full, target)
        _, empty_metrics = criterion(empty, target)
        _, erased_metrics = criterion(erased, target)
        metrics = {"full": full_metrics, "empty_same_weights": empty_metrics, "erased_same_weights": erased_metrics}
        attention = full["attention"]["attention"]
        entropy = None
        if attention is not None and attention.shape[-1] > 1:
            distribution = attention.float().mean(0).clamp_min(1e-10)
            entropy = float((-(distribution*distribution.log()).sum(-1)/np.log(distribution.shape[-1])).mean())
        scales = torch.linalg.eigvalsh(memory.covars).clamp_min(0).sqrt()
        metrics["memory"] = {"public_gaussians": len(memory), "active_gaussians": len(full["attention"]["indices"]),
                             "alpha_range": [float(memory.alpha.min()), float(memory.alpha.max())],
                             "axis_scale_range_m": [float(scales.min()), float(scales.max())],
                             "attention_normalized_entropy_mean": entropy,
                             "attention_entropy_per_head": (-(attention.float().clamp_min(1e-10)*attention.float().clamp_min(1e-10).log()).sum(-1)/np.log(attention.shape[-1])).mean(-1).cpu().tolist() if attention is not None and attention.shape[-1]>1 else None,
                             "o5_gate": gate_statistics(model.render_reader.gate), "o6_gate": gate_statistics(model.pe_reader.gate)}
        metrics['memory']['local_geometry_confidence_mean'] = float(full['local_geometry_confidence'].mean())
        if attention is not None:
            origin, ray, dz = world_rays(patch_intrinsics(query.K), full['T0'], h//14, w//14)
            selected = memory.subset(full['attention']['indices'])
            rho2, distance = ray_ellipsoid_distance(origin, ray, selected.means, selected.covars)
            relative_z = distance*dz[:, None]-full['rendered'].depth.reshape(-1, 1)
            weights = attention.float().mean(0)
            neighbor = rho2 <= 9.
            metrics['memory'].update(attention_mass_in_3sigma=float((weights*neighbor).sum(-1).mean()),
                uniform_mass_in_3sigma=float(neighbor.float().mean()),
                attention_log1p_rho2=float((weights*torch.log1p(rho2)).sum(-1).mean()),
                uniform_log1p_rho2=float(torch.log1p(rho2).mean()),
                attention_mass_behind_50cm=float((weights*(relative_z>.5)).sum(-1).mean()),
                gamma_positive=torch.nn.functional.softplus(model.pe_reader.gamma).cpu().tolist(),
                beta_positive=torch.nn.functional.softplus(model.pe_reader.beta).cpu().tolist())
        metrics["all_training_queries"] = []
        for query_index, (training_query, training_target) in enumerate(scene["queries"]):
            training_prediction = full if query_index == 1 else model(training_query, memory)
            _, query_metrics = criterion(training_prediction, training_target)
            query_metrics['rotation_gain_deg'] = query_metrics['rotation0_deg']-query_metrics['rotation1_deg']
            query_metrics['translation_gain_m'] = query_metrics['translation0_m']-query_metrics['translation1_m']
            empty_prediction = empty if query_index == 1 else model(training_query, memory.subset(slice(0, 0)))
            _, empty_query_metrics = criterion(empty_prediction, training_target)
            query_metrics['empty_epe_m'] = empty_query_metrics['point_epe_m']
            metrics["all_training_queries"].append(dict(query_index=query_index, **query_metrics))
        metrics["wall"] = v1(scene_folder, scene, query, target, full)
        v2(scene_folder, scene, memory)
        v3(scene_folder, scene, query, target, full, empty, render0, render1)
        v4(scene_folder, scene, query, memory, full, model.receive_layer)
        metrics["erasure"] = v5(scene_folder, scene, query, target, memory, full, erased, box, mask)
        if not metrics["erasure"]["local_pixels"]:
            raise RuntimeError("Erasure bbox has no visible evaluation surface")
        # Same-scene held-out query is diagnostic, not disjoint-scene validation.
        heldout_query, heldout_target = scene["heldout"][0]
        _, metrics["heldout_query_diagnostic"] = criterion(model(heldout_query, memory), heldout_target)
        _, metrics["heldout_empty_diagnostic"] = criterion(model(heldout_query, memory.subset(slice(0, 0))), heldout_target)
        # Evaluator-only coverage of this final learned memory, separate from
        # the initial native-resolution v1 read-only coverage diagnostic.
        points = heldout_target.world_points[heldout_target.valid]
        precision = torch.linalg.inv(memory.covars)
        covered = 0
        for start in range(0, len(points), 2048):
            difference = points[start:start+2048, None]-memory.means[None]
            rho2 = torch.einsum('pmi,mij,pmj->pm', difference, precision, difference)
            covered += int((rho2 <= 9).any(-1).sum())
        metrics['heldout_query_diagnostic']['final_memory_3sigma_coverage_smoke_resolution'] = covered/len(points)
        all_metrics[scene["name"]] = metrics
        print("Visualizations generated:", scene_folder, flush=True)
    plot_training_curves(output_dir, records)
    (output_dir/"evaluation.json").write_text(json.dumps(all_metrics, indent=2)+"\n")
    print("V1–V5 images exist; open and inspect them before reporting completion.", flush=True)


def gate_statistics(gate):
    value = gate.detach().tanh()
    return dict(mean=float(value.mean()), mean_abs=float(value.abs().mean()), max_abs=float(value.abs().max()), channels=value.numel())


def plot_training_curves(output_dir, records):
    fig, grid = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    fig.suptitle('Training curves follow the run augmentation config; fixed evaluation has no pose jitter or key dropout', fontsize=10)
    axes = grid.flatten()
    palette = ["#0072b2", "#d55e00", "#009e73"]
    for index, name in enumerate(dict.fromkeys(record["scene"] for record in records)):
        data = [r for r in records if r["scene"] == name]
        for ax, key, title in zip(axes, ["loss", "point_epe_m", "rotation1_deg", "translation1_m"], ["World + pose loss (random training gauge)", "World EPE [m]", "Rotation: dashed T0, solid T1 [deg]", "Translation: dashed T0, solid T1 [m]"]):
            values = np.array([r[key] for r in data])
            steps = np.array([r["step"] for r in data])
            ax.plot(steps, values, alpha=.18, color=palette[index])
            if len(values) >= 10:
                ax.plot(steps[9:], np.convolve(values, np.ones(10)/10, mode="valid"), label=name, color=palette[index])
            ax.set(xlabel="Total updates", title=title)
            if key in ('rotation1_deg', 'translation1_m'):
                coarse_key = key.replace('1_', '0_')
                coarse = np.array([r[coarse_key] for r in data])
                if len(coarse) >= 10:
                    ax.plot(steps[9:], np.convolve(coarse, np.ones(10)/10, mode='valid'), ls='--', color=palette[index], alpha=.7)
    axes[0].legend(fontsize=7)
    for key, color in [('o5', '#0072b2'), ('o6', '#d55e00')]:
        axes[4].plot([r['step'] for r in records], [r[key+'_gate'] for r in records], color=color, label=key+' mean |tanh gate|')
        maxkey = key+'_gate_max_abs'
        if maxkey in records[0]:
            axes[4].plot([r['step'] for r in records], [r[maxkey] for r in records], ls='--', color=color, label=key+' max |tanh gate|')
    axes[4].set(title='Per-channel gate opening', xlabel='Total updates')
    axes[4].legend(fontsize=7)
    axes[5].plot([r['step'] for r in records], [r['min_covariance_eigenvalue'] for r in records], lw=.6)
    axes[5].set(title='Minimum covariance eigenvalue [m²]', xlabel='Total updates', yscale='log')
    alpha_axes = axes[5].twinx()
    alpha_axes.plot([r['step'] for r in records], [r['alpha_min'] for r in records], color='#d55e00', ls='--', lw=.6)
    alpha_axes.plot([r['step'] for r in records], [r['alpha_max'] for r in records], color='#d55e00', lw=.6)
    alpha_axes.set(ylabel='alpha min/max (orange)', ylim=(0, 1))
    save(fig, output_dir/"vis/training_curves.png")
