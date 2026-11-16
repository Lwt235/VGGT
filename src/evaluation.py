"""Fixed evaluator-only surface-overlap groups from the original data split.

Never supplied to a model. Learned Gaussian size must not redefine which
benchmark queries count as observable or silently remove a tail failure.
"""
def benchmark_observable(spec, sequences):
    scene=sequences[spec['scene']]
    row=scene['frame_ids'].index(spec['query_frame_id'])
    support=max(scene['surface_overlap'][row][scene['frame_ids'].index(f)] for f in spec['mapping_frame_ids'])
    return support>=.05
