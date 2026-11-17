"""Train-only long strided physical walks; GT is confined to data/criterion."""
import random
from src.streaming_data import StreamingDataset


class StridedStreamingDataset(StreamingDataset):
    def __init__(self, split_path, image_hw, minimum_frames=12, maximum_frames=16):
        if not 12 <= minimum_frames <= maximum_frames <= 16:
            raise ValueError('Long streaming clips require 12–16 observations')
        super().__init__(split_path, image_hw)
        self.minimum_frames, self.maximum_frames = minimum_frames, maximum_frames
        self.graphs, self.longest = {}, {}
        self.training_examples = []
        for sid, item in self.sequences.items():
            if item['partition'] != 'train':
                continue
            allowed = set(item['train_frame_ids'])
            # Stride is measured in original sequence indices, not in a list
            # with heldouts removed. Skipped/heldout images are never decoded.
            graph = {i: [] for i, f in enumerate(item['frame_ids']) if f in allowed}
            for i in graph:
                for j in range(i+1, min(i+5, len(item['frame_ids']))):
                    if j in graph and max(item['surface_overlap'][i][j], item['surface_overlap'][j][i]) >= .10:
                        graph[i].append(j)
            longest = {}
            for i in sorted(graph, reverse=True):
                longest[i] = 1+max((longest[j] for j in graph[i]), default=0)
            self.graphs[sid], self.longest[sid] = graph, longest
            self.training_examples.extend((sid, i) for i in graph if longest[i] >= 8)
        if not self.training_examples:
            raise ValueError('No train-only long co-visible strided walks')

    def sample(self, rng=random, *, device='cpu'):
        sid, start = rng.choice(self.training_examples)
        item, graph, longest = self.sequences[sid], self.graphs[sid], self.longest[sid]
        length = rng.randint(self.minimum_frames, self.maximum_frames)
        mode = rng.choices(('strided_explore', 'wide_revisit', 'loiter'), weights=(.45, .40, .15))[0]
        outward_length = length if mode == 'strided_explore' else max(8, (length+2)//2)
        outward = [start]
        while len(outward) < outward_length and graph[outward[-1]]:
            choices = graph[outward[-1]]
            needed = min(outward_length-len(outward), longest[outward[-1]]-1)
            viable = [j for j in choices if longest[j] >= needed]
            outward.append(rng.choice(viable or choices))
        if mode == 'loiter':
            local = [outward[-2], outward[-3], outward[-2], outward[-1]]
            indices = (outward+[local[i % len(local)] for i in range(length)])[:length]
        else:
            path = outward+outward[-2:0:-1]
            indices = [path[i % len(path)] for i in range(length)]
        frames = [item['frame_ids'][i] for i in indices]
        result = self.load_stream(sid, frames, device=device, training=True)
        result.update(mode=mode, source_indices=indices,
            strides=[abs(b-a) for a,b in zip(indices, indices[1:])],
            seed_surface_overlap=[1. if i == start else item['surface_overlap'][start][i] for i in indices])
        return result
