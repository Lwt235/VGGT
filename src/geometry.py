"""OpenCV camera-z geometry; explicit integer/half-integer pixel conventions.

Points use (..., 3), homogeneous transforms use column-vector semantics.
No fitting, scale recovery or post-prediction alignment is performed here.
"""

import torch
from torch import Tensor
from torch.nn import functional as F


def invert_se3(transform: Tensor) -> Tensor:
    if transform.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError("Expected c2w/w2c with shape (...,3/4,4)")
    rotation = transform[..., :3, :3].transpose(-1, -2)
    translation = -(rotation @ transform[..., :3, 3:4])
    bottom = transform.new_zeros((*transform.shape[:-2], 1, 4))
    bottom[..., 0, 3] = 1
    return torch.cat((torch.cat((rotation, translation), -1), bottom), -2)


def transform_points(points: Tensor, transform: Tensor) -> Tensor:
    batch_ndim = transform.ndim - 2
    flat = points.reshape(*points.shape[:batch_ndim], -1, 3)
    out = flat @ transform[..., :3, :3].transpose(-1, -2)
    out = out + transform[..., None, :3, 3]
    return out.reshape(points.shape)


def pixel_grid(height: int, width: int, *, device=None, dtype=torch.float32,
               center: float = 0.) -> Tensor:
    y, x = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                          torch.arange(width, device=device, dtype=dtype), indexing="ij")
    return torch.stack((x + center, y + center, torch.ones_like(x)), -1)


def unproject(depth: Tensor, K: Tensor, c2w: Tensor, *, center=0.) -> Tensor:
    """Depth is camera z, NOT Euclidean distance; returns (...,H,W,3)."""
    grid = pixel_grid(*depth.shape[-2:], device=depth.device, dtype=depth.dtype, center=center)
    camera = torch.einsum("...ij,hwj->...hwi", torch.linalg.inv(K), grid)
    return transform_points(camera * depth[..., None], c2w)


def project(world: Tensor, K: Tensor, w2c: Tensor) -> tuple[Tensor, Tensor]:
    camera = transform_points(world, w2c)
    batch_ndim = K.ndim - 2
    flat = camera.reshape(*camera.shape[:batch_ndim], -1, 3)
    image = (flat @ K.transpose(-1, -2)).reshape(camera.shape)
    return image[..., :2] / image[..., 2:3], camera[..., 2]


def intrinsics_cv_to_edge(K: Tensor) -> Tensor:
    """OpenCV integer centers -> gsplat index+0.5 raster sampling."""
    shift = torch.eye(3, device=K.device, dtype=K.dtype)
    shift[0, 2] = shift[1, 2] = .5
    return shift @ K


def patch_intrinsics(K_cv: Tensor, patch_size=14, *, output_center=.5) -> Tensor:
    """Patch rays address original cv coords (index+0.5)*patch_size-0.5.

    output_center=.5: gsplat patch raster K; output_center=0: cv patch K.
    Image resizing/cropping must be reflected in the input K_cv beforehand.
    """
    if patch_size <= 0 or output_center not in (0., .5):
        raise ValueError("Positive patch size and center in {0,0.5} required")
    scale = K_cv.new_tensor([1 / patch_size, 1 / patch_size, 1]).diag()
    K_edge_patch = scale @ intrinsics_cv_to_edge(K_cv)
    shift = torch.eye(3, device=K_cv.device, dtype=K_cv.dtype)
    shift[0, 2] = shift[1, 2] = output_center - .5
    return shift @ K_edge_patch


def quat_wxyz_to_matrix(quaternion: Tensor) -> Tensor:
    identity = torch.zeros_like(quaternion)
    identity[..., 0] = 1
    norm = quaternion.norm(dim=-1, keepdim=True)
    q = torch.where(norm > 1e-8, quaternion / norm.clamp_min(1e-8), identity)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def covariance_from_quat_scale(q: Tensor, scale: Tensor) -> Tensor:
    R = quat_wxyz_to_matrix(q)
    return (R * scale.square()[..., None, :]) @ R.transpose(-1, -2)


def world_rays(K: Tensor, c2w: Tensor, height: int, width: int, *, center=.5):
    if K.ndim != 2 or c2w.ndim != 2:
        raise ValueError("Use one camera per spatial-memory read")
    grid = pixel_grid(height, width, device=K.device, dtype=K.dtype, center=center).reshape(-1, 3)
    camera_d = F.normalize(grid @ torch.linalg.inv(K).T, dim=-1)
    d = camera_d @ c2w[:3, :3].T
    return c2w[:3, 3].expand_as(d), d, camera_d[:, 2]


def ray_ellipsoid_distance(origins: Tensor, directions: Tensor, means: Tensor, covars: Tensor):
    """Closed-form closest point on a half-ray, for Q rays and M ellipsoids.

    Returns squared Mahalanobis distance and t (ray length for unit direction).
    The caller converts t to camera-z before comparing with rasterized depth.
    """
    precision = torch.linalg.inv(covars)
    delta = means[None] - origins[:, None]
    Pd = torch.einsum("mij,qj->qmi", precision, directions)
    denom = torch.einsum("qi,qmi->qm", directions, Pd).clamp_min(1e-10)
    t = (torch.einsum("qmi,qmi->qm", delta, Pd) / denom).clamp_min(0)
    residual = t[..., None] * directions[:, None] - delta
    distance = torch.einsum("qmi,mij,qmj->qm", residual, precision, residual)
    return distance.clamp_min(0), t
