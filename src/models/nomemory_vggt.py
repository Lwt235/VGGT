"""Single-frame official VGGT baseline; forward accepts RGB/K only.

The first mapping camera defines supervision, never an additional model input.
No Gaussian state, mapping images, memory readers, frame KV or anchoring exists.
"""
from pathlib import Path
import torch
from torch import nn
from src.data import QueryInput
from src.geometry import transform_points
from src.models.pose import PoseHead
from src.models.vggt_memory import official_vggt_class,frozen_patch_tokens,aggregate_single
from src.runtime import load_checkpoint_readonly


class VGGTNoMemory(nn.Module):
    def __init__(self,source,weights,*,receive_layer=4):
        super().__init__()
        if not Path(weights).is_file():raise FileNotFoundError(weights)
        self.native=official_vggt_class(source)(enable_camera=False,enable_depth=False,enable_track=False)
        checkpoint=load_checkpoint_readonly(weights)
        if 'model' in checkpoint:checkpoint=checkpoint['model']
        expected=self.native.state_dict()
        selected={k:v for k,v in checkpoint.items() if k in expected}
        if not any(k.startswith('aggregator.patch_embed.') for k in selected):raise ValueError('Official DINOv2 weights required')
        self.native.load_state_dict(selected,strict=True)
        del checkpoint,selected,expected
        self.native.requires_grad_(False)
        self.native.point_head.requires_grad_(True)
        self.native.aggregator.global_blocks[receive_layer].requires_grad_(True)
        self.native.aggregator.global_blocks[11].requires_grad_(True)
        self.pose=PoseHead(4100)  # native camera token + query patch summary + K

    def train(self,mode=True):
        super().train(mode);self.native.aggregator.patch_embed.eval();return self

    def forward(self,query:QueryInput):
        if type(query) is not QueryInput:raise TypeError('Single-frame baseline accepts QueryInput(rgb,K) only')
        rgb,K=query.rgb,query.K
        shallow=frozen_patch_tokens(self.native.aggregator,rgb)
        with torch.autocast(device_type=rgb.device.type,dtype=torch.bfloat16,enabled=rgb.is_cuda):
            outputs,start=aggregate_single(self.native.aggregator,rgb,shallow)
        with torch.autocast(device_type=rgb.device.type,enabled=False):
            outputs=[None if x is None else x.float() for x in outputs]
            points,_=self.native.point_head(outputs,images=rgb[None,None],patch_start_idx=start)
            camera=points[0,0]
            final=outputs[-1][0,0]
            h,w=rgb.shape[-2:]
            intrinsic=torch.stack((K[0,0]/w,K[1,1]/h,K[0,2]/w,K[1,2]/h))
            pose=self.pose(torch.cat((final[0],final[start:].mean(0),intrinsic)))
            # Match Full's isolation of pose supervision from depth bias.
            world=transform_points(camera,pose.detach())
        return dict(world_points=world,camera_points=camera,raw_camera_points=camera,coarse_world_points=world,T0=pose,T1=pose)
