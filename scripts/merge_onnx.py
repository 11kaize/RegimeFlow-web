"""Merge ONNX external-data models into single-file .onnx for browser use.

onnxruntime-web (WASM) cannot load models that use external data
(backbone.onnx + backbone.onnx.data), so we inline the weights into a single
file. Output goes to web/models/ (43MB + 400KB) which is then hosted somewhere
CORS-fetchable (HuggingFace / GitHub raw) for the client-side inference engine.
"""
import onnx
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "ckpt" / "RegimeFlow"
OUT = Path(__file__).resolve().parent.parent / "web" / "models"

OUT.mkdir(parents=True, exist_ok=True)

for name in ["backbone", "cond_encoder"]:
    src = SRC / f"{name}.onnx"
    dst = OUT / f"{name}.onnx"
    print(f"Merging {name}: {src} -> {dst}")
    model = onnx.load(str(src), load_external_data=True)
    onnx.save(model, str(dst), save_as_external_data=False)
    print(f"  done: {dst.stat().st_size / 1e6:.2f} MB")

print("All merged.")
