"""Repair the committed legacy ONNX slice layout without changing learned weights.

Requires numpy and onnx. Run from any directory with:
    python scripts/repair_onnx_mamba_layout.py

This migration accepts only the known legacy graph. Future exports must use
the reference Mamba projection layout. The backend's external weight file is kept
unchanged; the browser's merged model is rebuilt from that backend model.
"""
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ckpt/RegimeFlow/backbone.onnx"
MERGED = ROOT / "web/models/backbone.onnx"
LEGACY_SHA256 = "ad56a8a2b0c4e5e02dc3f3d20ca175f5c148e3590b24527aae9a8fd2b47927e7"
MARKER = "regimeflow.mamba_projection_layout"


def repair():
    # Do not load/rewrite external tensors: only Slice index constants change.
    model = onnx.load(SOURCE, load_external_data=False)
    if any(p.key == MARKER and p.value == "x,z;dt,B,C" for p in model.metadata_props):
        print("Backend graph already repaired; rebuilding browser model.")
    else:
        if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != LEGACY_SHA256:
            raise ValueError("Unknown ONNX graph; this migration only accepts the released legacy export.")
        arrays = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer if not t.external_data}
        tensors = {t.name: t for t in model.graph.initializer}
        producers = {out: n for n in model.graph.node for out in n.output}

        def replace_bounds(node, start, end):
            # Shared old constants are used elsewhere, so attach new constants
            # to this Slice only. Its output shape and consumers stay unchanged.
            assert arrays[node.input[3]].tolist() == [2]
            assert arrays[node.input[4]].tolist() == [1]
            for index, value, suffix in ((1, start, "start"), (2, end, "end")):
                name = node.name + "_mamba_correct_" + suffix
                model.graph.initializer.append(numpy_helper.from_array(np.array([value], dtype=np.int64), name))
                node.input[index] = name

        count = 0
        for block in range(4):
            prefix = f"bb.residual_blocks.{block}.mamba_layer.mamba."
            conv = next(n for n in model.graph.node if prefix + "conv1d.weight" in n.input)
            xp_slice = producers[producers[conv.input[0]].input[0]]
            in_slices = [n for n in model.graph.node if n.op_type == "Slice" and n.input[0] == xp_slice.input[0]]
            assert len(in_slices) == 2
            for node in in_slices:
                start = int(arrays[node.input[1]][0])
                assert start in (0, 192)
                # Legacy first half feeds gate z, second half feeds conv x.
                replace_bounds(node, 192 if start == 0 else 0, 384 if start == 0 else 192)
                count += 1

            namespace = f"bb.residual_blocks.{block}:"
            x_proj = next(n for n in model.graph.node if n.op_type == "MatMul"
                          and n.input[1] in tensors and list(tensors[n.input[1]].dims) == [192, 38]
                          and any(p.key == "namespace" and namespace in p.value for p in n.metadata_props))
            proj_slices = [n for n in model.graph.node if n.op_type == "Slice" and n.input[0] == x_proj.output[0]]
            assert len(proj_slices) == 3
            for node in proj_slices:
                start = int(arrays[node.input[1]][0])
                # Existing B, C, dt consumers must read [6:22], [22:38], [0:6].
                bounds = {0: (6, 22), 16: (22, 38), 32: (0, 6)}[start]
                replace_bounds(node, *bounds)
                count += 1
        assert count == 20
        marker = model.metadata_props.add()
        marker.key, marker.value = MARKER, "x,z;dt,B,C"
        onnx.save(model, SOURCE)
        print(f"Repaired {count} Slice nodes; learned weights unchanged.")

    onnx.checker.check_model(str(SOURCE))
    merged = onnx.load(SOURCE, load_external_data=True)
    onnx.save(merged, MERGED, save_as_external_data=False)
    print(f"Rebuilt {MERGED.relative_to(ROOT)}")


if __name__ == "__main__":
    repair()
