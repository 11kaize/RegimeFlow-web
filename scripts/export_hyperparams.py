"""Export checkpoint parameters recursively, plus the shared Web inference config.

The Web config is generated from this checkpoint and explicit demo/data rules;
Python and JavaScript both read web/models/inference-config.json.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def export(checkpoint, output_root=ROOT):
    checkpoint = Path(checkpoint)
    hp = plain(torch.load(checkpoint, map_location='cpu', weights_only=False)['hyper_parameters'])
    prior = hp['prior_params']
    if isinstance(prior['saturation_scales'], str):
        prior['saturation_scales'] = json.loads(prior['saturation_scales'])
    config = {key: hp[key] for key in ('context_length', 'prediction_length', 'cond_dim', 'use_condition')}
    config.update({key: prior[key] for key in ('alpha', 'beta', 'noise_scale', 'min_variance', 'num_harmonics', 'saturation_scales')})
    config.update(poly_degree=prior.get('poly_degree', 2), flow_sigma=hp['sigm'],
                  min_scale=1.0, denoise_steps=4, checkpoint_num_steps=hp['num_steps'],
                  transform='minmax', dataset_clip=[0, 1e9],
                  min_relative_range=1e-3, min_unique_ratio=0.01,
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  schema_version=1, model_version='inference-parity-2')
    for relative, data in [('ckpt/RegimeFlow/hyperparams.json', hp), ('web/models/inference-config.json', config)]:
        dest = Path(output_root) / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'ckpt/RegimeFlow/seed53_best.ckpt')
    parser.add_argument('--output-root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoint, args.output_root), indent=2))
