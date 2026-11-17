"""Read-only streaming clips from the unchanged Step 2 scene/frame split.

RGB-D and poses are dataset/criterion objects, never streaming engine inputs.
Only the seed frame supplies depth to the engine. Loading is lazy and has no
RGB cache, so evaluation need not retain historical images outside the engine.
"""
from pathlib import Path
import random

import torch
import yaml

from src.data import QueryInput, Sun3DScene, Supervision
from src.geometry import invert_se3, transform_points


class StreamingDataset:
    def __init__(self, split_path, image_hw, minimum_frames=5, maximum_frames=8):
        self.split_path = Path(split_path)
        with self.split_path.open() as handle:
            self.spec = yaml.safe_load(handle)
        self.image_hw = tuple(image_hw)
        self.minimum_frames, self.maximum_frames = minimum_frames, maximum_frames
        if not 5 <= minimum_frames <= maximum_frames <= 8:
            raise ValueError('Training unroll must contain 5–8 frames')
        self.sequences = {item['id']: item for item in self.spec['sequences']
                          if item['partition'] != 'excluded'}
        self.scenes = {}
        self.training_windows = {}
        for sid, item in self.sequences.items():
            if item['partition'] != 'train':
                continue
            allowed = sorted(item['train_frame_ids'])
            if set(allowed) & set(item['heldout_frame_ids']):
                raise ValueError('Train and heldout frame IDs overlap')
            # Consecutive available training observations; missing heldouts are
            # never decoded or used as unsupervised bridge frames.
            indices = {frame: i for i, frame in enumerate(item['frame_ids'])}
            windows = []
            for start in range(len(allowed)-minimum_frames+1):
                frames = allowed[start:start+minimum_frames]
                adjacent = [max(item['surface_overlap'][indices[a]][indices[b]],
                                item['surface_overlap'][indices[b]][indices[a]])
                            for a, b in zip(frames, frames[1:])]
                if min(adjacent) >= .10:
                    windows.append(start)
            if windows:
                self.training_windows[sid] = windows
        if not self.training_windows:
            raise ValueError('No train-only continuous five-frame streaming clips')
        self.training_examples = [(sid, start) for sid, starts in self.training_windows.items() for start in starts]

    def _scene(self, sid):
        if sid not in self.scenes:
            item = self.sequences[sid]
            scene = Sun3DScene(item['path'])
            if scene.frame_ids != item['frame_ids']:
                raise ValueError('Shared data frame IDs changed after fixed split')
            self.scenes[sid] = scene
        return self.scenes[sid]

    def validate_frames(self, sid, frame_ids, *, training=False):
        item = self.sequences[sid]
        if not frame_ids:
            raise ValueError('Empty stream')
        if not set(frame_ids).issubset(item['frame_ids']):
            raise ValueError('Unknown stream frame')
        if training and (item['partition'] != 'train' or
                         not set(frame_ids).issubset(item.get('train_frame_ids', []))):
            raise ValueError('Streaming training cannot read val/heldout frames')

    def sample(self, rng=random, *, device='cpu'):
        # Uniform physical clip starts, rather than giving a five-frame smoke
        # room the same sampling mass as a 100-frame real exploration sequence.
        sid, start = rng.choice(self.training_examples)
        item = self.sequences[sid]
        allowed = sorted(item['train_frame_ids'])
        length = rng.randint(self.minimum_frames, self.maximum_frames)
        mode = rng.choice(('explore', 'loiter', 'revisit'))
        base = allowed[start:start+self.minimum_frames]
        if mode == 'explore':
            # Extend only across the same precomputed physical continuity test.
            frames = list(base)
            ids = {f: i for i, f in enumerate(item['frame_ids'])}
            for frame in allowed[start+len(base):start+length]:
                a, b = ids[frames[-1]], ids[frame]
                if max(item['surface_overlap'][a][b], item['surface_overlap'][b][a]) < .10:
                    break
                frames.append(frame)
        elif mode == 'loiter':
            path = base[:3]+base[1:2]
            frames = [path[i % len(path)] for i in range(length)]
        else:
            outward = base[:max(3, (length+1)//2)]
            path = outward+outward[-2::-1]
            frames = [path[i % len(path)] for i in range(length)]
        result = self.load_stream(sid, frames, device=device, training=True)
        result['mode'] = mode
        return result

    def load_stream(self, sid, frame_ids, *, device='cpu', training=False):
        """Return metadata and a lazy one-pass observations iterator.

        All target poses/points are expressed in the first seed camera gauge.
        The consumer must pass ONLY RGB/K to engine.forward after the seed.
        """
        frame_ids = list(frame_ids)
        self.validate_frames(sid, frame_ids, training=training)

        def observations():
            scene = self._scene(sid)
            gauge = None
            for frame_id in frame_ids:
                q, truth = scene.load(scene.frame_ids.index(frame_id), resize_hw=self.image_hw)
                if gauge is None:
                    gauge = invert_se3(truth.c2w)
                pose = gauge @ truth.c2w
                target = Supervision(truth.depth.to(device), pose.to(device),
                    transform_points(truth.world_points, gauge).to(device), truth.valid.to(device))
                yield QueryInput(q.rgb.to(device), q.K.to(device)), target

        return dict(scene=sid, frame_ids=frame_ids, training=training,
                    world_gauge='seed camera, c2w_0=I, metric metres', observations=observations())
