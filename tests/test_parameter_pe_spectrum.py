"""CPU regression for O6 thin-axis backward at repeated tangent eigenvalues."""
import pytest
import torch

from src.memory.gaussians import GaussianMemory
from src.models.ellipsoid_pose import local_se3_exp
from src.readout.parameter_pe import ParameterPEReader, covariance_spectrum_and_thin_axis


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_repeated_tangent_roots_keep_exact_forward_and_finite_covariance_gradient(dtype):
    covariance = torch.diag(torch.tensor([.04, .04, .0025], dtype=dtype)).requires_grad_()
    with torch.no_grad():
        expected_values, expected_vectors = torch.linalg.eigh(covariance)
    values, normal = covariance_spectrum_and_thin_axis(covariance)
    torch.testing.assert_close(values, expected_values, atol=0, rtol=0)
    torch.testing.assert_close(normal, expected_vectors[:, 0], atol=0, rtol=0)
    torch.testing.assert_close(values.sqrt().log(), expected_values.sqrt().log(), atol=0, rtol=0)
    loss = (normal*normal.new_tensor([.2, .3, .7])).sum()+values.log().sum()
    loss.backward()
    assert torch.isfinite(covariance.grad).all() and covariance.grad.abs().sum() > 0
    # A tilted normal must still differentiate through covariance off-diagonals.
    assert covariance.grad[:2, 2].abs().sum() > 0


def test_simple_spectrum_first_gradient_matches_torch_eigh_for_symmetric_parameters():
    raw = torch.tensor([[.07, .006, -.002], [.006, .04, .004], [-.002, .004, .009]],
                       dtype=torch.float64, requires_grad=True)
    covariance = (raw+raw.T)*.5
    values, normal = covariance_spectrum_and_thin_axis(covariance)
    reference_values, reference_vectors = torch.linalg.eigh(covariance)
    normal_weight = raw.new_tensor([.2, -.3, .7])
    value_weight = raw.new_tensor([.7, -.4, .3])
    loss = (normal*normal_weight).sum()+(values.log()*value_weight).sum()
    reference_loss = (reference_vectors[:, 0]*normal_weight).sum()+(reference_values.log()*value_weight).sum()
    actual, = torch.autograd.grad(loss, raw, retain_graph=True)
    expected, = torch.autograd.grad(reference_loss, raw)
    # Only the relative 1e-6 implicit damping changes the simple-root derivative.
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-5)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_fully_repeated_spectrum_retains_finite_geometry_gradient(dtype):
    covariance = (.04*torch.eye(3, dtype=dtype)).requires_grad_()
    values, normal = covariance_spectrum_and_thin_axis(covariance)
    (values.log().sum()+(normal*normal.new_tensor([.2, .3, .7])).sum()).backward()
    assert torch.isfinite(covariance.grad).all()
    assert covariance.grad.abs().sum() > 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_pe_camera_frame_repeated_tangents_backward_reaches_covariance_and_query_pose(dtype):
    torch.manual_seed(17)
    K = torch.tensor([[3., 0., 1.], [0., 3., 1.], [0., 0., 1.]], dtype=dtype)
    means = torch.tensor([[-.3, -.25, 2.], [.4, -.2, 2.4], [-.2, .4, 2.8], [.5, .4, 3.1]], dtype=dtype)
    covariance = torch.diag(torch.tensor([.04, .04, .0025], dtype=dtype)).repeat(4, 1, 1).requires_grad_()
    memory = GaussianMemory(means, covariance, torch.full((4,), .7, dtype=dtype), torch.randn(4, 4, dtype=dtype))
    rotation = torch.tensor([.12, -.08, .04], dtype=dtype, requires_grad=True)
    translation = torch.tensor([.02, -.03, .01], dtype=dtype, requires_grad=True)
    R, t = local_se3_exp(rotation, translation)
    pose = torch.cat((torch.cat((R, t[:, None]), -1), R.new_tensor([[0., 0., 0., 1.]])), 0)
    pose.retain_grad()
    reader = ParameterPEReader(16, 4, heads=2, bands=3, camera_frame=True).to(dtype=dtype)
    with torch.no_grad():
        reader.gate.fill_(.3)
    tokens = torch.randn(4, 16, dtype=dtype)
    # Exactly the PE-only path: no rendered O5 depth, opacity or position PE.
    output, _ = reader(tokens, memory, pose, K, 2, 2, rendered=None)
    (output*torch.randn_like(output)).sum().backward()
    for value in (covariance, pose, rotation, translation):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0
