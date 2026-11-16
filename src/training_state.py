"""Exact training continuation with one explicitly mutable safety setting."""
import random
import torch


def resume_config_compatible(saved, requested):
    return {k:v for k,v in saved.items() if k!='systemic_epe_limit_m'} == {
        k:v for k,v in requested.items() if k!='systemic_epe_limit_m'}


def restore_training_state(model, optimizer, scheduler, checkpoint, config):
    if not resume_config_compatible(checkpoint['config'], config) or checkpoint.get('initialization')!='fresh_official_vggt':
        raise ValueError('Resume permits only systemic_epe_limit_m metadata changes')
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    if parameters.keys()!=checkpoint['trainable'].keys():
        raise ValueError('Trainable parameter names changed at resume')
    if any(parameters[n].shape!=v.shape for n,v in checkpoint['trainable'].items()):
        raise ValueError('Trainable parameter shape changed at resume')
    with torch.no_grad():
        for name,value in checkpoint['trainable'].items():parameters[name].copy_(value)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    random.setstate(checkpoint['python_rng'])
    torch.set_rng_state(checkpoint['torch_rng'])
    torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    return checkpoint['step']
