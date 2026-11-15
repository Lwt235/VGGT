import numpy as np
import torch

from src.data import decode_sun3d_depth, resize_intrinsics, sun3d_K_to_cv


def test_sun3d_uint16_rotation_preserves_metric_depth_and_invalid_zero():
    mm = np.array([0, 1000, 2345, 10000], dtype=np.uint16)
    encoded = (mm << 3) | (mm >> 13)
    np.testing.assert_allclose(decode_sun3d_depth(encoded), mm.astype(np.float32)/1000)


def test_sun3d_matlab_to_cv_then_half_pixel_resize():
    raw = torch.tensor([[570., 0., 320.], [0., 570., 240.], [0., 0., 1.]])
    cv = sun3d_K_to_cv(raw)
    torch.testing.assert_close(cv[:2, 2], torch.tensor([319., 239.]))
    scaled = resize_intrinsics(cv, (480, 640), (240, 320))
    torch.testing.assert_close(scaled[:2, 2], torch.tensor([159.25, 119.25]))
