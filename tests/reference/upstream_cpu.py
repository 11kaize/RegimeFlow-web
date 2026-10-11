"""Run upstream Mamba forward + reference scan without CUDA/Triton imports."""
import ast
import math
from pathlib import Path
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange, repeat


def upstream_mamba():
    directory = Path(__file__).resolve().parent
    namespace = dict(torch=torch, nn=nn, Tensor=Tensor, F=F, math=math,
                     Optional=Optional, rearrange=rearrange, repeat=repeat,
                     causal_conv1d_fn=None, causal_conv1d_update=None,
                     selective_state_update=None, RMSNorm=None,
                     layer_norm_fn=None, rms_norm_fn=None)
    for file, symbol in [('selective_scan_v2_2_6.py', 'selective_scan_ref'),
                         ('mamba_simple_v2_2_6.py', 'Mamba')]:
        tree = ast.parse((directory/file).read_text())
        selected = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == symbol)
        exec(compile(ast.Module(body=[selected], type_ignores=[]), file, 'exec'), namespace)
        if symbol == 'selective_scan_ref':
            namespace['selective_scan_fn'] = namespace[symbol]
    return namespace['Mamba']
