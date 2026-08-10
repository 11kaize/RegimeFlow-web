"""
Export ConditionEncoder to ONNX for torch-free inference.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

# Monkey-patch mamba_ssm (needed for model imports)
class _FakeMamba(torch.nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
        super().__init__()
        self.d_model = d_model
    def forward(self, x):
        return x

class _FakeMambaSSM:
    Mamba = _FakeMamba
    __version__ = "2.2.6-py"
sys.modules['mamba_ssm'] = _FakeMambaSSM()

import pytorch_lightning as pl
from models.FlowMatching.RegimeFlow.RegimeFlow import RegimeFlowCond

# Load model
ckpt_path = str(PROJECT_ROOT / "ckpt" / "RegimeFlow" / "seed53_best.ckpt")
model = RegimeFlowCond.load_from_checkpoint(ckpt_path, map_location='cpu', strict=False, weights_only=False)
model.eval()

ce = model.condition_encoder
if ce is None:
    print("No condition encoder in this model")
    sys.exit(1)

print(f"ConditionEncoder: d_model={ce.d_model}")

# Create export wrapper
class CondEncoderForONNX(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, traj_pattern, period):
        return self.encoder(batch_size=1, device='cpu',
                           traj_pattern=traj_pattern, period=period)

wrapper = CondEncoderForONNX(ce.cpu())
wrapper.eval()

# Export to ONNX
dummy_pattern = torch.tensor([0], dtype=torch.long)
dummy_period = torch.tensor([0.0], dtype=torch.float32)

onnx_path = str(PROJECT_ROOT / "ckpt" / "RegimeFlow" / "cond_encoder.onnx")

torch.onnx.export(
    wrapper,
    (dummy_pattern, dummy_period),
    onnx_path,
    input_names=['traj_pattern', 'period'],
    output_names=['cond_emb'],
    dynamic_axes={
        'traj_pattern': {0: 'batch'},
        'period': {0: 'batch'},
        'cond_emb': {0: 'batch'},
    },
    opset_version=17,
    do_constant_folding=True,
)

print(f"Exported to {onnx_path}")
print(f"File size: {Path(onnx_path).stat().st_size / 1024:.1f} KB")

# Verify
import onnx
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("ONNX check: ✅ valid")

# Quick inference test
import onnxruntime as ort
session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
out = session.run(None, {
    'traj_pattern': np.array([3], dtype=np.int64),
    'period': np.array([12.5], dtype=np.float32),
})
print(f"Test output shape: {out[0].shape}")
print("✅ Condition encoder export complete")
