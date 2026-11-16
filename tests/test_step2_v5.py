"""Deep matching LM and smooth physically weighted calibrated-ray anchoring."""
import inspect
import torch
from src.models.deep_refinement import DeepEllipsoidRefinement
from src.models.ellipsoid_pose import anisotropic_ray_pose, local_se3_exp
from src.memory.gaussians import GaussianMemory
from src.readout.render import RenderReader, RenderedMemory
from src.geometry import transform_points


def fixture():
    torch.manual_seed(17)
    xy=torch.cartesian_prod(torch.linspace(-.5,.5,4),torch.linspace(-.6,.6,6))
    p=torch.cat((xy,torch.ones(24,1)),1)*(3+torch.rand(24,1)*2)
    rays=torch.nn.functional.normalize(p,dim=-1)
    features=torch.eye(24)
    memory=GaussianMemory(p.clone(),torch.eye(3)[None].repeat(24,1,1)*.02,torch.ones(24),features)
    bootstrap=dict(matching_query=features, matching_keys=features, matching_temperature=torch.tensor(60.),
        matching_dustbin_logit=torch.full((24,1),-10.),camera_rays=rays,c_depth=torch.ones(24))
    return p,rays,memory,bootstrap


def test_deep_lm_clean_pose_fixed_point_monotone_and_no_dpt_input():
    p,rays,memory,bootstrap=fixture()
    model=DeepEllipsoidRefinement(32,24)
    final=torch.randn(24,32)
    T,_,_,diag=model(final,memory,bootstrap,torch.eye(4))
    torch.testing.assert_close(T,torch.eye(4),atol=2e-6,rtol=0)
    assert (torch.diff(diag['refinement_lm_energy_history'])<=1e-9).all()
    # No camera-point parameter: changing a DPT prediction cannot reach T1.
    assert list(inspect.signature(model.forward).parameters)==['final','memory','bootstrap','init_pose']
    biased=p/3
    a,_,da=anisotropic_ray_pose(p,rays,memory.means,memory.covars,torch.ones(24),init_pose=torch.eye(4),iterations=2,return_diagnostic=True)
    b,_,db=anisotropic_ray_pose(biased,rays,memory.means,memory.covars,torch.ones(24),init_pose=torch.eye(4),iterations=2,return_diagnostic=True)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    for d in (da,db):assert (torch.diff(d['energy_history'])<=1e-9).all()
    init=torch.eye(4);init[:3,3]=torch.tensor([.06,-.04,.05])
    changed,_,_,d=model(final,memory,bootstrap,init)
    assert changed[:3,3].norm()<init[:3,3].norm()
    assert (torch.diff(d['refinement_lm_energy_history'])<=1e-9).all()
    (changed[:3,3].square().sum()+d['refined_matching_attention'][:,-1].mean()).backward()
    assert model.query_delta[-1].weight.grad is not None
    assert torch.isfinite(model.query_delta[-1].weight.grad).all()


def test_deep_lm_world_gauge_equivariance_and_empty_identity():
    p,rays,memory,bootstrap=fixture()
    model=DeepEllipsoidRefinement(32,24);final=torch.randn(24,32)
    init=torch.eye(4);init[:3,3]=torch.tensor([.05,.02,-.03])
    T,_,_,_=model(final,memory,bootstrap,init)
    G=torch.eye(4);G[:3,:3],_=local_se3_exp(torch.tensor([.7,-.4,.2]),torch.zeros(3));G[:3,3]=torch.tensor([1.,-2.,.6])
    rotated=GaussianMemory(transform_points(memory.means,G),G[:3,:3][None]@memory.covars@G[:3,:3].T[None],memory.alpha,memory.features)
    Tg,_,_,_=model(final,rotated,bootstrap,G@init)
    torch.testing.assert_close(Tg,G@T,atol=3e-6,rtol=2e-6)
    empty,_,_,_=model(final,memory.subset(slice(0,0)),{},init)
    torch.testing.assert_close(empty,init,atol=0,rtol=0)


def test_smooth_local_ray_anchor_no_tiles_no_leak_beyond_half_patch():
    torch.manual_seed(17)
    reader=RenderReader(8,16,camera_frame=True,physical_local=True)
    h,w=56,112;raw=torch.randn(h,w,3);raw[...,2]=2.
    K=torch.tensor([[90.,0.,56.],[0.,90.,28.],[0.,0.,1.]])
    depth=torch.linspace(3.,7.,8)[None,:,None].expand(4,8,1).clone().requires_grad_()
    rendered=RenderedMemory(torch.zeros(4,8,3),torch.randn(4,8,8),depth,torch.zeros(4,8,1),torch.ones(4,8,1))
    full,c=reader.anchor_camera_geometry(raw,rendered,K)
    opacity=rendered.opacity.clone();opacity[:,4:]=0
    partial=RenderedMemory(rendered.world,rendered.features,depth,rendered.depth_variance,opacity)
    after,_=reader.anchor_camera_geometry(raw,partial,K)
    torch.testing.assert_close(full[:,:49],after[:,:49],atol=0,rtol=0)
    torch.testing.assert_close(after[:,63:],raw[:,63:],atol=0,rtol=0)
    # All interior x coordinates interpolate; no 14-pixel constant steps.
    difference=torch.diff(full[28,8:103,2]);assert (difference>0).all()
    assert full[...,2].mean()>4.5 and c.mean()>.7
    from src.geometry import pixel_grid
    rays=pixel_grid(h,w,device=K.device,dtype=K.dtype,center=0.)@torch.linalg.inv(K).T
    correction=full-raw
    torch.testing.assert_close(correction,rays*correction[...,2:3],atol=3e-7,rtol=1e-6)
    full.sum().backward();assert depth.grad.abs().sum()>0 and torch.isfinite(depth.grad).all()
    partial.opacity.zero_();empty,_=reader.anchor_camera_geometry(raw,partial,K)
    torch.testing.assert_close(empty,raw,atol=0,rtol=0)
