"""
Extract hyperparameters from checkpoint and save as JSON.
This enables torch-free inference on Render.
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import omegaconf

# Monkey-patch mamba_ssm
class _FakeMamba(torch.nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

class _FakeMambaSSM:
    Mamba = _FakeMamba
    __version__ = "2.2.6-py"
sys.modules['mamba_ssm'] = _FakeMambaSSM()

import pytorch_lightning as pl
from models.FlowMatching.RegimeFlow.RegimeFlow import RegimeFlowCond

ckpt_path = str(PROJECT_ROOT / "ckpt" / "RegimeFlow" / "seed53_best.ckpt")
model = RegimeFlowCond.load_from_checkpoint(ckpt_path, map_location='cpu', strict=False, weights_only=False)
hp = model.hparams

# Convert to plain dict
hp_dict = {}
for k, v in hp.items():
    if isinstance(v, (dict, omegaconf.DictConfig)):
        hp_dict[k] = dict(v)
    elif isinstance(v, (list, omegaconf.ListConfig)):
        hp_dict[k] = list(v)
    else:
        hp_dict[k] = v

output_path = str(PROJECT_ROOT / "ckpt" / "RegimeFlow" / "hyperparams.json")
with open(output_path, 'w') as f:
    json.dump(hp_dict, f, indent=2, default=str)

print(f"Saved hyperparameters to {output_path}")
print(f"Keys: {list(hp_dict.keys())}")
print(f"context_length={hp_dict.get('context_length')}, prediction_length={hp_dict.get('prediction_length')}")
