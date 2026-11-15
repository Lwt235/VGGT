"""Fixed disjoint scene/frame splits and read-only, covisible RGB-D episodes.

GT projection is a dataset sampling/benchmark construction operation. Only
mapping RGB-D/c2w enter the writer; the query object still contains RGB/K only.
"""
from collections import OrderedDict
from pathlib import Path
import random
import torch
import yaml
from src.data import Sun3DScene, QueryInput, Supervision
from src.geometry import invert_se3, transform_points, project


def frame_partition(frame_ids):
    if len(frame_ids) < 4:
        return [], [], 'fewer than four paired frames: no disjoint map/query/heldout episode'
    heldout = frame_ids[4::5] if len(frame_ids) >= 8 else frame_ids[-1:]
    return [f for f in frame_ids if f not in heldout], heldout, None


def reframe(observation, world_from_old):
    query, truth = observation
    return query, Supervision(truth.depth, world_from_old@truth.c2w,
                              transform_points(truth.world_points, world_from_old), truth.valid)


@torch.no_grad()
def surface_overlap(frames):
    """Row=query, col=mapping, visible valid surface agreement, not just FOV."""
    n = len(frames)
    result = torch.zeros(n, n)
    for i, (_, target) in enumerate(frames):
        points = target.world_points[::3, ::3].reshape(-1, 3)
        mask = target.valid[::3, ::3].reshape(-1)
        points = points[mask]
        for j, (query, mapping) in enumerate(frames):
            if i == j or not len(points):
                continue
            uv, z = project(points, query.K, invert_se3(mapping.c2w))
            h, w = mapping.depth.shape
            x, y = uv[:, 0].round().long(), uv[:, 1].round().long()
            visible = (z > .1) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
            x, y = x.clamp(0, w-1), y.clamp(0, h-1)
            agree = mapping.valid[y, x] & ((z-mapping.depth[y, x]).abs() < .10+.02*z)
            result[i, j] = (visible & agree).float().mean()
    return result


class EpisodeDataset:
    def __init__(self, split_path, image_hw, *, cache_frames=96):
        with Path(split_path).open() as f:
            self.spec = yaml.safe_load(f)
        self.image_hw = tuple(image_hw)
        self.cache_frames = cache_frames
        self.cache = OrderedDict()
        self.scenes = {}
        for item in self.spec['sequences']:
            if item['partition'] == 'excluded':
                continue
            scene = Sun3DScene(item['path'])
            if scene.frame_ids != item['frame_ids']:
                raise ValueError('Shared data frame IDs changed after the fixed split')
            self.scenes[item['id']] = (scene, item)
        self.training_queries = []
        for sid, (scene, item) in self.scenes.items():
            if item['partition'] != 'train':
                continue
            ids = item['train_frame_ids']
            for frame in ids:
                if len(self.candidates(sid, frame, ids)) >= 2:
                    self.training_queries.append((sid, frame))
        if not self.training_queries:
            raise ValueError('No covisible disjoint training episodes')

    def candidates(self, sid, query_frame, allowed):
        _, item = self.scenes[sid]
        row = item['frame_ids'].index(query_frame)
        overlap = item['surface_overlap'][row]
        return [(f, overlap[item['frame_ids'].index(f)]) for f in allowed
                if f != query_frame and overlap[item['frame_ids'].index(f)] >= .15]

    def frame(self, sid, frame):
        key = (sid, frame)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        scene, item = self.scenes[sid]
        value = scene.load(item['frame_ids'].index(frame), resize_hw=self.image_hw)
        self.cache[key] = value
        if len(self.cache) > self.cache_frames:
            self.cache.popitem(last=False)
        return value

    def episode(self, sid, map_ids, query_id, *, device='cpu'):
        if query_id in map_ids:
            raise ValueError('Query must never occur in mapping observations')
        mapping = [self.frame(sid, f) for f in map_ids]
        query = self.frame(sid, query_id)
        gauge = invert_se3(mapping[0][1].c2w)
        def move(item):
            q, t = reframe(item, gauge)
            return QueryInput(q.rgb.to(device), q.K.to(device)), Supervision(
                t.depth.to(device), t.c2w.to(device), t.world_points.to(device), t.valid.to(device))
        return dict(name=sid, mapping=[move(x) for x in mapping], query=move(query),
                    mapping_frame_ids=map_ids, query_frame_id=query_id)

    def sample(self, rng=random, *, device='cpu'):
        # Uniform frame sampling uses all eligible frames instead of repeatedly
        # oversampling the tiny smoke sequences. Maps vary on every update.
        sid, query = rng.choice(self.training_queries)
        allowed = self.scenes[sid][1]['train_frame_ids']
        possible = self.candidates(sid, query, allowed)
        count = min(len(possible), rng.randint(2, 4))
        selected = rng.sample(possible, count)
        # At least one strong surface overlap; no query truth enters forward.
        strongest = max(possible, key=lambda x: x[1])
        if max(x[1] for x in selected) < .3 and strongest[1] >= .3:
            selected[0] = strongest
        return self.episode(sid, [x[0] for x in selected], query, device=device)

    def evaluation_episodes(self, *, max_per_scene=None):
        episodes = []
        for sid, (_, item) in self.scenes.items():
            if item['partition'] == 'val':
                mapping_pool, query_ids = item['map_frame_ids'], item['query_frame_ids']
                label = 'cross_scene_val'
            else:
                mapping_pool, query_ids = item['train_frame_ids'], item['heldout_frame_ids']
                label = 'same_scene_heldout'
            if max_per_scene:
                query_ids = query_ids[::max(1, len(query_ids)//max_per_scene)][:max_per_scene]
            for query in query_ids:
                candidates = self.candidates(sid, query, mapping_pool)
                # Preserve blind/low-overlap heldouts and label their group in
                # the criterion; never silently discard an evaluation query.
                if not candidates:
                    candidates = [(f, 0.) for f in mapping_pool if f != query]
                map_ids = [f for f, _ in sorted(candidates, key=lambda x: (-x[1], x[0]))[:3]]
                if map_ids:
                    episodes.append(dict(scene=sid, mapping_frame_ids=map_ids,
                                         query_frame_id=query, split=label))
        return episodes
