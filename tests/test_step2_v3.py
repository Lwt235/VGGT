"""Planar support, map-only scale and pose/geometry gradient separation."""
import torch
import pytest
from src.augmentation import sample_world_gauge
from src.geometry import transform_points, world_rays
from src.losses.world import pose_errors
from src.models.ellipsoid_pose import planar_safe_initial, anisotropic_ray_pose, local_se3_exp
from src.readout.render import RenderReader, RenderedMemory


@pytest.mark.parametrize('normal_axis', [0, 1, 2, 'oblique'])
def test_planar_initialization_retains_all_axes_and_world_equivariance(normal_axis):
    torch.manual_seed(17)
    points = torch.randn(100, 3, dtype=torch.float64)
    axis = 0 if normal_axis == 'oblique' else normal_axis
    points[:, axis] = 2+1e-5*points[:, axis]
    if normal_axis == 'oblique':
        camera_rotation,_ = local_se3_exp(torch.tensor([.8,.3,.4],dtype=points.dtype), points.new_zeros(3))
        points = points @ camera_rotation.T
    points[:, 2] += 7
    G = torch.eye(4, dtype=torch.float64)
    G[:3, :3], _ = local_se3_exp(torch.tensor([.6, -.8, 1.1], dtype=torch.float64), torch.zeros(3, dtype=torch.float64))
    G[:3, 3] = torch.tensor([1., -2., .3])
    anchors = transform_points(points, G)
    # A few cross-room outliers must not dominate robust scatter scale.
    anchors[-3:] += torch.tensor([3., -2., 2.])
    weights = torch.ones(100, dtype=torch.float64); weights[-3:] = .1
    R,t,scale,_ = planar_safe_initial(points, anchors, weights)
    T = torch.eye(4, dtype=torch.float64); T[:3,:3],T[:3,3] = R,t
    angle,distance = pose_errors(T,G)
    assert angle < .5 and distance < .03
    torch.testing.assert_close(R.T@R,torch.eye(3,dtype=R.dtype),atol=1e-6,rtol=1e-6)
    assert torch.linalg.det(R) > .99999 and abs(float(scale)-1) < .01
    H = G.clone(); H[:3,:3] = local_se3_exp(torch.tensor([-.4,.3,-.2],dtype=R.dtype),t*0)[0]
    moved = transform_points(anchors,H)
    RG,tG,_,_ = planar_safe_initial(points,moved,weights)
    torch.testing.assert_close(RG,H[:3,:3]@R,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(tG,H[:3,:3]@t+H[:3,3],atol=1e-6,rtol=1e-6)


def test_side_wall_lm_initialization_and_gradients():
    torch.manual_seed(17)
    P = torch.randn(80,3,dtype=torch.float64)
    P[:,0] = 1.2+P[:,0]*1e-5;P[:,2] += 7
    rays = torch.nn.functional.normalize(P,dim=-1)
    G=sample_world_gauge('cpu').double()
    G[:3,:3]=local_se3_exp(torch.tensor([.1,.7,-.2],dtype=P.dtype),P.new_zeros(3))[0]
    X=transform_points(P,G).requires_grad_()
    C=torch.diag(P.new_tensor([.03**2,.2**2,.2**2])).repeat(80,1,1)
    C=G[:3,:3]@C@G[:3,:3].T
    T,_,d=anisotropic_ray_pose(P/2.5,rays,X,C,P.new_ones(80),metric_scale=True,planar_safe=True,return_diagnostic=True)
    angle,distance=pose_errors(T,G)
    assert angle < .1 and distance < .01
    assert (d['energy_history'][1:]<=d['energy_history'][:-1]+1e-8).all()
    T[:3,3].square().sum().backward()
    assert torch.isfinite(X.grad).all() and X.grad.abs().sum()>0


@pytest.mark.parametrize('factor',[2.,3.])
def test_visible_scale_recovers_far_depth_and_empty_is_exact_identity(factor):
    h,w=4,5
    reader=RenderReader(4,16,camera_frame=True,surface_normals=True)
    points=torch.randn(h,w,3);points[...,2]=6+torch.rand(h,w)*2
    raw=(points/factor).requires_grad_()
    render=RenderedMemory(points,torch.zeros(h,w,4),points[...,2:3],torch.ones(h,w,1)*.03,torch.ones(h,w,1))
    scaled,gamma=reader.calibrate_camera_scale(raw,render)
    assert abs(float(gamma)-factor)<.01
    torch.testing.assert_close(scaled,points,atol=.02,rtol=.003)
    scaled.sum().backward();assert torch.isfinite(raw.grad).all()
    render.opacity.zero_()
    empty,gamma=reader.calibrate_camera_scale(raw,render)
    assert gamma.item()==1.
    torch.testing.assert_close(empty,raw,atol=0,rtol=0)


def test_relative_refinement_keeps_rotation_and_metric_translation_across_scale():
    from src.geometry import unproject
    from src.models.spatial_pose import SpatialRefinement
    torch.manual_seed(17)
    h,w=4,5
    K=torch.tensor([[6.,0.,2.5],[0.,6.,2.],[0.,0.,1.]])
    points=unproject(torch.full((h,w),3.),K,torch.eye(4),center=.5)
    normal=torch.zeros_like(points);normal[...,2]=-1
    reader=SpatialRefinement(16,residual_only=True,point_to_plane=True,relative_geometry=True)
    torch.nn.init.normal_(reader.residual_pose.weight,std=.05)
    tokens=torch.randn(h*w,16)
    results=[]
    for scale in [1.,3.]:
        rendered=RenderedMemory(scale*(points+torch.tensor([0.,0.,.05])),torch.zeros(h,w,4),torch.ones(h,w,1)*3*scale,
            torch.ones(h,w,1)*.04*scale**2,torch.ones(h,w,1),normal)
        T,_,_=reader(tokens,scale*points,torch.eye(4),rendered,K,h,w)
        results.append(T)
    torch.testing.assert_close(results[0][:3,:3],results[1][:3,:3],atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(results[0][:3,3]*3,results[1][:3,3],atol=2e-6,rtol=2e-6)
    assert results[0][:3,3].norm()>1e-5


def test_final_camera_residual_opens_depth_confidence():
    reader=RenderReader(4,16,surface_normals=True,camera_frame=True)
    K=torch.eye(3)
    points=torch.zeros(1,2,3);points[...,2]=torch.tensor([[1.8,3.6]])
    depth=torch.tensor([[[2.],[6.]]])
    render=RenderedMemory(torch.zeros(1,2,3),torch.zeros(1,2,4),depth,torch.zeros(1,2,1),torch.ones(1,2,1))
    scaled,_=reader.calibrate_camera_scale(points,render)
    anchored,confidence=reader.anchor_camera_geometry(scaled,render,K)
    assert (confidence==.5).all()
    (anchored[...,2:3]-depth).abs().mean().backward()
    assert reader.depth_confidence[-1].bias.grad < 0
    assert torch.isfinite(reader.depth_confidence[-1].weight.grad).all()
