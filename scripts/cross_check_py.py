"""Deterministic reference output from the Python engine, for cross-checking
the JS port (engine-web.js). Zeros out numpy RNG so the pipeline is
deterministic; dumps intermediates (loc/scale/mu/x0) + final prediction to JSON.
"""
import json
import sys
from pathlib import Path

import numpy as np

# Project root on sys.path so `web.backend.engine_onnx` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Deterministic: zero out all Gaussian draws ──
def _zeros_randn(*shape):
    return np.zeros(shape, dtype=np.float64)

np.random.randn = _zeros_randn

from web.backend.engine_onnx import init_engine

CKPT = "ckpt/RegimeFlow/seed53_best.ckpt"
eng = init_engine(CKPT)

# Synthetic context: trend + oscillation, 120 points (exercises the truncate-to-96 path)
t = np.arange(120, dtype=np.float64)
context = (0.5 * t + 2.0 * np.sin(2 * np.pi * t / 40.0)).tolist()

traj_pattern = 3   # oscillation
period = 12.5

pred = eng.predict(context, traj_pattern=traj_pattern, period=period)

# ── Recompute intermediates the same way for comparison ──
# 引擎恒等归一化：loc=ctx[0]、scale=1.0（见 engine_onnx.py StdScaler 的 axis 细节）
ctx = np.asarray(context, dtype=np.float32).flatten()
ctx = ctx[-96:]  # already >= 96
loc = float(ctx[0])
scale = 1.0
scaled = (ctx - loc) / scale
mu = float((10.0 / (1.0 + 10.0 * 96)) * scaled.sum())

out = {
    "context_head": context[:3],
    "traj_pattern": traj_pattern,
    "period": period,
    "loc": loc,
    "scale": scale,
    "mu": mu,
    # x0 construction: first 3 context (ctx/scale), boundary, first 3 future (mu)
    "x0_ctx_head": [float(ctx[i] / scale) for i in range(3)],
    "x0_ctx_tail": [float(ctx[i] / scale) for i in range(93, 96)],
    "x0_fut_head": [mu, mu, mu],
    "pred_len": int(len(pred)),
    "pred_head": [float(v) for v in pred[:6]],
    "pred_tail": [float(v) for v in pred[-6:]],
}

print(json.dumps(out, indent=2))
