"""Strict EMA-aware CPU export; shares the corrected Mamba implementation.

Use --output-root /tmp/regimeflow-export to validate without replacing assets.
The legacy entrypoint scripts/export_onnx.py delegates here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from web.backend.engine import RegimeFlowEngine
from scripts.export_hyperparams import export as export_config


class ConditionExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.enc = model

    def forward(self, traj_pattern, period):
        return self.enc(batch_size=traj_pattern.shape[0], device='cpu',
                        traj_pattern=traj_pattern, period=period)


class BackboneExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.bb = model

    def forward(self, t, x_in, cond_emb):
        return self.bb(t, x_in, cond_emb)


def export(checkpoint, output_root, component='all'):
    torch.set_num_threads(1)
    torch.manual_seed(20261011)
    engine = RegimeFlowEngine(str(checkpoint))  # strict keys; EMA when present
    output_root = Path(output_root)
    model_dir = output_root/'ckpt/RegimeFlow'
    browser_dir = output_root/'web/models'
    model_dir.mkdir(parents=True, exist_ok=True)
    browser_dir.mkdir(parents=True, exist_ok=True)
    total = engine.context_length + engine.prediction_length
    cases = []
    if component in ('all', 'condition'):
        cases.append(('cond_encoder', ConditionExport(engine.condition_encoder),
                      (torch.tensor([3]), torch.tensor([12.5])),
                      ['traj_pattern', 'period'], ['cond_emb']))
    if component in ('all', 'backbone'):
        cases.append(('backbone', BackboneExport(engine.backbone),
                      (torch.tensor([.25]), torch.randn(1, total, 1), torch.randn(1, engine.cond_dim)),
                      ['t', 'x_in', 'cond_emb'], ['velocity']))
    report = {}
    for name, wrapper, args, inputs, outputs in cases:
        wrapper.eval()
        filename = model_dir/(name+'.onnx')
        with torch.no_grad():
            torch.onnx.export(wrapper, args, str(filename), input_names=inputs,
                              output_names=outputs, opset_version=17,
                              do_constant_folding=True, dynamo=False)
        model = onnx.load(filename)
        if name == 'backbone':
            marker = model.metadata_props.add()
            marker.key, marker.value = 'regimeflow.mamba_projection_layout', 'x,z;dt,B,C'
        onnx.save(model, browser_dir/(name+'.onnx'))
        # Use the existing two-file backend asset convention.
        external = model_dir/(name+'.onnx.data')
        if external.exists():
            external.unlink()
        onnx.save_model(model, filename, save_as_external_data=True,
                        all_tensors_to_one_file=True, location=external.name, size_threshold=1024)
        onnx.checker.check_model(str(filename))
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(str(filename), options, providers=['CPUExecutionProvider'])
        with torch.no_grad():
            expected = wrapper(*args).numpy()
        actual = session.run(None, {key: value.numpy() for key, value in zip(inputs, args)})[0]
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
        report[name] = {'max_abs_error': float(np.max(np.abs(actual-expected))), 'strict_weight_load': True}
    export_config(checkpoint, output_root)
    (output_root/'export_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'ckpt/RegimeFlow/seed53_best.ckpt')
    parser.add_argument('--output-root', type=Path, default=ROOT)
    parser.add_argument('--component', choices=['all', 'backbone', 'condition'], default='all')
    args = parser.parse_args()
    export(args.checkpoint, args.output_root, args.component)
