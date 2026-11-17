"""Conservative camera-local refinement for self-predicted streaming maps."""
import torch
from torch import nn
from src.geometry import invert_se3
from src.models.spatial_pose import rotation_6d


class StreamPoseGate(nn.Module):
    """Only predicted correspondence/LM consistency controls the correction.

    The gate acts on a LOCAL pose increment, hence commutes with any world
    gauge. It is evaluated before final O5 rendering, never as post-alignment.
    """
    def __init__(self):
        super().__init__()
        self.confidence = nn.Linear(4, 1)
        nn.init.zeros_(self.confidence.weight)
        nn.init.constant_(self.confidence.bias, -2.)

    def forward(self, base, proposed, diagnostic):
        reference = base.detach()
        delta = invert_se3(reference) @ proposed
        p = 1-diagnostic['dustbin_probability']
        quality = p.mean()
        W = diagnostic.get('refined_matching_attention', diagnostic['matching_attention'])
        agreement = (1-W[:, -1]).mean()
        history = diagnostic.get('refinement_lm_energy_history')
        improvement = quality*0 if history is None else (
            (history[0]-history[-1])/history[0].abs().clamp_min(1e-6)).clamp(0, 1)
        # Small LM systems deliberately run in CPU float64; diagnostics must
        # join the GPU feature tensor explicitly (no change to pose solving).
        improvement = improvement.to(quality)
        distance = delta[:3, 3].norm()
        trace = delta[:3, :3].diagonal().sum()
        # Rotation trace is used only for magnitude limiting, not pose loss.
        angle = torch.acos(((trace-1)/2).detach().clamp(-1+1e-7, 1-1e-7))
        attrs = torch.stack((quality, agreement, improvement, distance.detach()))
        learned = self.confidence(attrs).sigmoid().squeeze()
        reliability = (quality/.25).clamp(0, 1)*(agreement/.25).clamp(0, 1)
        limit = torch.minimum((.06/distance.clamp_min(1e-6)).clamp(max=1),
                              (torch.deg2rad(angle.new_tensor(3.))/angle).clamp(max=1))
        gate = learned*reliability*limit
        eye = torch.eye(3, dtype=base.dtype, device=base.device)
        mixed = eye+gate*(delta[:3, :3]-eye)
        rotation = rotation_6d(torch.cat((mixed[:, 0], mixed[:, 1])))
        top = torch.cat((rotation, (gate*delta[:3, 3])[:, None]), -1)
        local = torch.cat((top, base.new_tensor([[0., 0., 0., 1.]])), 0)
        return reference@local, gate
