#!/usr/bin/env python3
"""Read only registered SUN3D assets; write independent Step2 inventory/splits."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import PROJECT_ROOT, configure_local_storage, load_config
configure_local_storage()
import hashlib
import yaml
import torch
from src.data import Sun3DScene
from src.step2_data import frame_partition, surface_overlap


def main():
    torch.set_num_threads(2)
    output = PROJECT_ROOT/'configs/step2_splits.yaml'
    if output.exists():
        raise SystemExit('Existing fixed Step2 split is preserved')
    seen_paths, content_scenes, records = set(), {}, []
    for root in load_config()['datasets']['sun3d']['roots']:
        for intrinsics in sorted(Path(root).rglob('intrinsics.txt')):
            path = intrinsics.parent.resolve(strict=True)
            if path in seen_paths:
                continue
            seen_paths.add(path)
            scene = Sun3DScene(path)
            sid = path.parent.name+'/'+path.name
            train_ids, heldout, reason = frame_partition(scene.frame_ids)
            partition = 'val' if path.parent.name in ('harvard_c11', 'harvard_robotics_lab') else 'train'
            if reason:
                partition = 'excluded'
            record = dict(id=sid, path=str(path), partition=partition, frame_ids=scene.frame_ids,
                          paired_frames=len(scene), pose_file=str(scene.pose_file))
            # Detect duplicate RGB content across scene groups before splitting.
            for frame in scene.frame_ids:
                digest = hashlib.sha256(scene.images[frame].read_bytes()).hexdigest()
                previous = content_scenes.get(digest)
                if previous and previous[0] != partition and 'excluded' not in (previous[0], partition):
                    raise RuntimeError(f'Cross-partition duplicate RGB: {previous[1]} and {sid}')
                content_scenes[digest] = (partition, sid)
            if reason:
                record['exclusion_reason'] = reason
            else:
                frames = [scene.load(i, resize_hw=(84, 112)) for i in range(len(scene))]
                record['valid_depth_fractions'] = [round(float(t.valid.float().mean()), 4) for _, t in frames]
                record['surface_overlap'] = [[round(v, 4) for v in row] for row in surface_overlap(frames).tolist()]
                if partition == 'val':
                    # All validation observations belong exclusively to val.
                    record['map_frame_ids'] = scene.frame_ids[::3]
                    record['query_frame_ids'] = [f for f in scene.frame_ids if f not in record['map_frame_ids']]
                else:
                    record['train_frame_ids'], record['heldout_frame_ids'] = train_ids, heldout
                del frames
            records.append(record)
            print(sid, partition, len(scene), flush=True)
    spec = dict(seed=17, policy='scene-disjoint val; heldout frames excluded from BOTH training maps and queries',
                units='m', world_gauge='first mapping camera per episode; no post-alignment',
                overlap_definition='84x112 decoded depth, sampled valid query surfaces; mapping z agreement < 0.10+0.02*z m',
                sampling='uniform eligible training query frames, random 2–4 covisible maps; overlap >=0.15',
                sequences=records)
    output.write_text(yaml.safe_dump(spec, sort_keys=False, default_flow_style=None, width=120))
    print('Fixed split:', output, flush=True)


if __name__ == '__main__':
    main()
