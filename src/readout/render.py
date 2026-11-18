"""O5: gsplat attribute moments with alpha-normalized expectations."""

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from src.geometry import invert_se3, transform_points
from src.memory.gaussians import GaussianMemory


@dataclass
class RenderedMemory:
    world: Tensor
    features: Tensor
    depth: Tensor
    depth_variance: Tensor
    opacity: Tensor
    normal: Tensor | None = None


def normalize_rendered_moments(accumulated: Tensor, opacity: Tensor,
                               latent_dim: int, eps=1e-6, *, with_normals=False) -> RenderedMemory:
    valid = opacity > eps
    expectation = torch.where(valid, accumulated / opacity.clamp_min(eps), 0.)
    depth = expectation[..., 3+latent_dim:4+latent_dim]
    second = expectation[..., 4+latent_dim:5+latent_dim]
    normal = F.normalize(expectation[..., 5+latent_dim:8+latent_dim], dim=-1, eps=eps) if with_normals else None
    return RenderedMemory(expectation[..., :3], expectation[..., 3:3+latent_dim], depth,
                          (second - depth.square()).clamp_min(0), opacity, normal)


def camera_facing_normals(covars, means, center):
    # Detached eigenvectors select the unoriented thin axis only. Two inverse
    # iterations keep a finite differentiable covariance path without the
    # undefined eigenvector backward of equal tangent eigenvalues.
    with torch.no_grad():
        seed = torch.linalg.eigh(covars.detach()).eigenvectors[..., 0]
    eye = torch.eye(3, device=covars.device, dtype=covars.dtype)
    normal = seed
    for _ in range(2):
        normal = F.normalize(torch.linalg.solve(covars+1e-8*eye, normal[..., None])[..., 0], dim=-1)
    facing = ((center-means)*normal).sum(-1, keepdim=True)
    return normal*torch.where(facing >= 0, 1., -1.)


class RenderReader(nn.Module):
    def __init__(self, latent_dim=256, token_dim=1024, *, metric_depth=True, surface_normals=False, camera_frame=False, local_multiplicative=False, physical_local=False, isotropic=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.surface_normals = surface_normals
        self.camera_frame = camera_frame
        self.local_multiplicative = local_multiplicative
        self.physical_local = physical_local
        self.isotropic = isotropic
        self.project = nn.Sequential(nn.Linear(latent_dim+(8 if camera_frame else 5), token_dim), nn.GELU(), nn.Linear(token_dim, token_dim))
        self.unseen = nn.Parameter(torch.zeros(token_dim))
        self.norm = nn.LayerNorm(token_dim)
        self.gate = nn.Parameter(torch.zeros(token_dim))
        self.depth_confidence = None
        if metric_depth:
            self.depth_confidence = nn.Sequential(nn.Linear(latent_dim+4, 64), nn.GELU(), nn.Linear(64, 1))
            nn.init.zeros_(self.depth_confidence[-1].weight)
            nn.init.constant_(self.depth_confidence[-1].bias, 1. if local_multiplicative or physical_local else 0.)

    def calibrate_camera_scale(self, camera_points, rendered):
        """Map-only metric scale inside forward, with smooth unseen fallback.

        This is not output/GT alignment: only O5 predictions and native camera
        points enter. Empty memory gives exactly gamma=1 with finite gradients.
        """
        hp, wp = rendered.depth.shape[:2]
        raw_z = F.adaptive_avg_pool2d(camera_points[..., 2][None, None], (hp, wp))[0, 0, ..., None]
        rel_var = rendered.depth_variance/rendered.depth.square().clamp_min(.04)
        valid = (rendered.depth > .05) & (raw_z > .05)
        weight = rendered.opacity/(1+rel_var)*valid
        # A small identity pseudo-observation avoids a discontinuous ratio at
        # zero support and interpolates over total visible weight, not GT.
        prior = .05
        gamma = ((weight*rendered.depth).sum()+prior)/((weight*raw_z).sum()+prior)
        gamma = gamma.clamp(.25, 4.)
        return gamma*camera_points, gamma

    def anchor_camera_geometry(self, camera_points, rendered, K_cv, *, reliability=None):
        """Local O5 metric depth correction along calibrated camera rays.

        A learned reliability head uses O5 latent/depth moments and current
        geometry. Unseen locations contribute zero. No GT or image is stored.
        This keeps a spatial geometry path after DPT instead of forcing all
        memory effects through one global camera transform.
        """
        hp, wp = rendered.depth.shape[:2]
        h, w = camera_points.shape[:2]
        native_depth = F.adaptive_avg_pool2d(camera_points[..., 2][None, None], (hp, wp))[0, 0, ..., None]
        if self.camera_frame:
            variance = rendered.depth_variance/rendered.depth.square().clamp_min(.04)
            attrs = torch.cat((rendered.features, rendered.opacity, torch.log1p(variance),
                               torch.log1p(rendered.depth),
                               (rendered.depth-native_depth)/rendered.depth.clamp_min(.2)), -1)
        else:
            variance = rendered.depth_variance
            attrs = torch.cat((rendered.features, rendered.opacity, torch.log1p(variance), rendered.depth, native_depth), -1)
        confidence = self.depth_confidence(attrs).sigmoid()
        if self.local_multiplicative:
            # Each patch owns its ratio: no reduction over space. Upsample each
            # 1x1 patch independently with bilinear interpolation, so erasure
            # cannot change an adjacent patch or leak into an opacity=0 hole.
            ratio = (rendered.depth/native_depth.clamp_min(.05)).clamp(.25, 4.)
            reliable = confidence*rendered.opacity/(1+variance)*(native_depth > .05)
            factor = reliable*(ratio-1)
            if h % hp or w % wp:
                raise ValueError('Local anchoring requires an integer patch grid')
            tiles = F.interpolate(factor.reshape(hp*wp, 1, 1, 1), size=(h//hp, w//wp), mode='bilinear', align_corners=False)
            scale = tiles.reshape(hp, wp, h//hp, w//wp).permute(0, 2, 1, 3).reshape(h, w)
            return camera_points*(1+scale[..., None]), confidence
        modulation = .65+.35*confidence if self.physical_local else confidence
        correction = modulation*rendered.opacity/(1+variance)*(rendered.depth-native_depth)
        if reliability is not None:
            correction = correction*reliability
        correction = F.interpolate(correction.permute(2, 0, 1)[None], size=(h, w), mode='bilinear', align_corners=False)[0, 0]
        from src.geometry import pixel_grid
        grid = pixel_grid(h, w, device=K_cv.device, dtype=K_cv.dtype, center=0.)
        rays = grid @ torch.linalg.inv(K_cv).T
        return camera_points+rays*correction[..., None], confidence

    def render(self, memory: GaussianMemory, query_c2w: Tensor, K_patch: Tensor,
               height: int, width: int) -> RenderedMemory:
        # query_c2w must be a MODEL prediction. No GT camera parameter exists here.
        if len(memory) == 0:
            accumulated = query_c2w.new_zeros(height, width, self.latent_dim+5+3*self.surface_normals)
            opacity = query_c2w.new_zeros(height, width, 1)
        else:
            if not memory.means.is_cuda:
                raise RuntimeError("O5 requires real CUDA gsplat; no silent CPU substitute")
            from gsplat import rasterization
            w2c = invert_se3(query_c2w)
            depth = transform_points(memory.means, w2c)[:, 2:3]
            attributes = torch.cat((memory.means, memory.features, depth, depth.square()), -1)
            covars = memory.covars
            if self.isotropic:
                variance = covars.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-10)
                covars = variance[:, None, None]*torch.eye(3, device=covars.device, dtype=covars.dtype)
            if self.surface_normals:
                normals = (torch.zeros_like(memory.means) if self.isotropic else
                           camera_facing_normals(covars, memory.means, query_c2w[:3, 3]))
                attributes = torch.cat((attributes, normals), -1)
            accumulated, opacity, _ = rasterization(
                means=memory.means, quats=None, scales=None, covars=covars,
                opacities=memory.alpha, colors=attributes, viewmats=w2c[None],
                Ks=K_patch[None], width=width, height=height, render_mode="RGB", packed=False)
            accumulated, opacity = accumulated[0], opacity[0]
        return normalize_rendered_moments(accumulated, opacity, self.latent_dim, with_normals=self.surface_normals)

    def forward(self, tokens: Tensor, memory: GaussianMemory, query_c2w: Tensor,
                K_patch: Tensor, height: int, width: int):
        rendered = self.render(memory, query_c2w, K_patch, height, width)
        attributes = torch.cat((rendered.world, rendered.features,
                                rendered.depth, rendered.depth_variance), -1)
        if self.camera_frame:
            w2c = invert_se3(query_c2w)
            camera = transform_points(rendered.world, w2c)
            camera = torch.where(rendered.opacity > 1e-6, camera, 0.)
            normal = (rendered.normal @ w2c[:3, :3].T if rendered.normal is not None
                      else torch.zeros_like(camera))
            attributes = torch.cat((camera, normal, rendered.features, torch.log1p(rendered.depth),
                rendered.depth_variance/(rendered.depth.square()+.05)), -1)
        visible = self.project(attributes.reshape(-1, self.project[0].in_features))
        opacity = rendered.opacity.reshape(-1, 1)
        received = opacity*visible + (1-opacity)*self.unseen
        return tokens + torch.tanh(self.gate)*self.norm(received), rendered
