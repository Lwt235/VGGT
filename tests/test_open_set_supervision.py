import torch
from src.data import Supervision
from src.losses.world import world_loss, open_set_matching_loss
from src.losses.pose_consistency import pose_consistency_loss
from src.readout.render import RenderedMemory


def example(blind=False):
    points = torch.tensor([[[0., 0., 2.], [.3, 0., 2.], [.6, 0., 2.]]])
    truth = Supervision(torch.full((1, 3), 2.), torch.eye(4), points, torch.ones(1, 3, dtype=torch.bool))
    t = torch.tensor(.2, requires_grad=True)
    T = torch.eye(4); T[0, 3] = t
    camera = (points+.1).requires_grad_()
    means = points.reshape(-1, 3).clone()+(20 if blind else 0)
    logits = torch.zeros(3, 4, requires_grad=True)
    W = logits.softmax(-1)
    z = torch.full((3,), 2.2, requires_grad=True)
    output = dict(T0=T, T1=T, world_points=camera+T[:3, 3], coarse_world_points=camera+T[:3, 3],
        camera_points=camera, anchors=W[:, :-1]@means, anchor_support_means=means,
        anchor_support_covars=torch.eye(3).repeat(3, 1, 1)*.01, bootstrap_attention=W[:, :-1]/.75,
        bootstrap_support=torch.ones(3, dtype=torch.bool), matching_attention=W, z_cam=z,
        c_depth=torch.ones(3)*.8, dustbin_probability=W[:, -1], match_mass=(1-W[:, -1]).sum(),
        rendered=RenderedMemory(points, torch.zeros(1, 3, 8), torch.ones(1, 3, 1)*2., torch.zeros(1, 3, 1), torch.ones(1, 3, 1)))
    return truth, output, t, logits, z


def test_zero_overlap_has_no_random_world_pose_gradient_but_trains_depth_and_rejection():
    truth, output, t, logits, z = example(True)
    loss, metrics = world_loss(output, truth)
    loss.backward()
    assert t.grad is None or t.grad.abs() == 0
    assert output['camera_points'].grad.abs().sum() > 0 and z.grad.abs().sum() > 0
    assert (logits.grad[:, -1] < 0).all()
    assert metrics['world_supervision_enabled'] == 0 and metrics['surface_support_rate'] == 0


def test_retained_support_is_positive_and_only_true_support_loss_is_negative():
    truth, output, _, logits, _ = example()
    match, _, coverage, visible = open_set_matching_loss(output, truth)
    gradient = torch.autograd.grad(match, logits, retain_graph=True)[0]
    assert visible and coverage == 1 and (gradient[:, -1] > 0).all()
    original = (output['anchor_support_means'], output['anchor_support_covars'], torch.tensor([False, True, False]))
    match, _, _, _ = open_set_matching_loss(output, truth, original)
    gradient = torch.autograd.grad(match, logits)[0]
    assert (gradient[:, -1] > 0).all()  # Remaining support forbids a conflicting negative.
    means, covariance, erased = original
    covariance = covariance*.1
    output['anchor_support_means'] = means[~erased]
    output['anchor_support_covars'] = covariance[~erased]
    logits = torch.zeros(3, 3, requires_grad=True)
    output['matching_attention'] = logits.softmax(-1)
    match, _, _, _ = open_set_matching_loss(output, truth, (means, covariance, erased))
    gradient = torch.autograd.grad(match, logits)[0]
    assert gradient[1, -1] < 0 and gradient[0, -1] > 0


def test_open_set_loss_balances_positive_and_negative_classes():
    truth, output, _, _, _ = example()
    means = output['anchor_support_means']
    covariance = output['anchor_support_covars']*.1
    erased = torch.tensor([False, True, False])
    output['anchor_support_means'] = means[~erased]
    output['anchor_support_covars'] = covariance[~erased]
    output['matching_attention'] = torch.tensor([[.4, .4, .2]]).expand(3, -1)
    match, _, _, _ = open_set_matching_loss(output, truth, (means, covariance, erased))
    expected = -(torch.tensor(.4).log()+torch.tensor(.2).log())
    torch.testing.assert_close(match, expected)


def test_raw_native_camera_points_receive_blind_local_geometry_supervision():
    truth, output, t, _, _ = example(True)
    raw = (truth.world_points+.2).requires_grad_()
    output['raw_camera_points'] = raw
    loss, metrics = world_loss(output, truth)
    loss.backward()
    assert t.grad is None or t.grad.abs() == 0
    assert raw.grad.abs().sum() > 0
    assert output['camera_points'].grad is None
    assert metrics['camera_epe_m'] > metrics['anchored_camera_epe_m']


def test_partial_zero_overlap_cannot_restore_world_pose_supervision():
    truth, output, t, logits, _ = example(True)
    teacher = dict(T0=torch.eye(4), T1=torch.eye(4))
    loss, _ = pose_consistency_loss(output, teacher, truth)
    loss.backward()
    assert t.grad is None or t.grad.abs() == 0
    assert logits.grad[:, -1].sum() < 0
