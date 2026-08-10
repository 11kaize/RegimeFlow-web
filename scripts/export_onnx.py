"""
RegimeFlow ONNX 导出 + CPU 推理脚本
====================================
将 RegimeFlowCond 导出为 ONNX 格式，支持 CPU 推理。

策略：
1. 用纯 PyTorch 实现替换 mamba_ssm 的 CUDA Mamba（CPU 不兼容）
2. 将 backbone 导出为 ONNX（BLR prior 和 ODE solver 太复杂，保留在 PyTorch 中）
3. 用 ONNX Runtime 运行推理

Usage:
    # 导出
    python scripts/export_onnx.py --checkpoint ckpt/RegimeFlow/seed53_best.ckpt --export

    # 导出 + 测试推理
    python scripts/export_onnx.py --checkpoint ckpt/RegimeFlow/seed53_best.ckpt --export --test
"""

import os
import sys
import time
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Pure PyTorch Mamba Implementation (CPU-compatible)
# ===========================================================================

class MambaPy(nn.Module):
    """
    Pure PyTorch implementation of Mamba SSM v2 (compatible with mamba_ssm==2.2.6).

    Architecture (v2, differs from v1):
        x → in_proj → (z, x')        # 2*d_inner output
        x' → conv1d → SiLU → x_proj → (B, C, dt)    # dt_rank + 2*d_state
        dt → dt_proj → softplus
        x', dt, A, B, C, D → selective_scan → y
        z → SiLU → gate
        y × gate → out_proj → output

    Weight shapes (verified against seed53_best.ckpt):
        in_proj.weight:  (2*d_inner, d_model)
        conv1d.weight:   (d_inner, 1, d_conv)  — depthwise
        x_proj.weight:   (dt_rank + 2*d_state, d_inner)
        dt_proj.weight:  (d_inner, dt_rank)
        dt_proj.bias:    (d_inner,)
        out_proj.weight: (d_model, d_inner)
        A_log:           (d_inner, d_state)
        D:               (d_inner,)

    The selective scan is implemented with a simple for-loop (OK for CPU).
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(d_model * expand)
        self.dt_rank = max(1, d_model // 16)  # "auto" in mamba_ssm v2

        # ── v2 architecture ──

        # Step 1: project input → (z, x')
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Step 2: depthwise causal conv on x'
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # Activation
        self.act_fn = nn.SiLU()

        # Step 3: project x' → (B, C, dt)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)

        # Step 4: project dt → d_inner (with bias = dt_bias from mamba_ssm)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # SSM parameters
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Step 6: output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _selective_scan(self, u, delta, A, B, C, D):
        """
        Pure PyTorch selective scan (SSM core).

        Args:
            u:  (B, L, D) — input
            delta: (B, L, D) — discretization step size
            A:    (D, N) — diagonal state matrix (stored as real log-values)
            B:    (B, L, N) — input-dependent B
            C:    (B, L, N) — input-dependent C
            D:    (D,) — skip connection

        Returns:
            y: (B, L, D)
        """
        B_dim, L, D = u.shape
        N = A.shape[1]
        device = u.device
        dtype = u.dtype

        # A is stored as log of negative values, so -exp(A_log) is the real A
        A_real = -torch.exp(A.float())

        # Discretize
        delta_A = delta.unsqueeze(-1) * A_real.unsqueeze(0).unsqueeze(1)  # (B, L, D, N)
        A_bar = torch.exp(delta_A)
        B_bar = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, D, N)

        # Sequential scan (CPU-friendly for-loop)
        h = torch.zeros(B_dim, D, N, device=device, dtype=dtype)
        ys = []

        for t in range(L):
            At = A_bar[:, t]        # (B, D, N)
            Bt = B_bar[:, t]        # (B, D, N)
            ut = u[:, t].unsqueeze(-1)  # (B, D, 1)
            Ct = C[:, t].unsqueeze(1)   # (B, 1, N)

            h = At * h + Bt * ut    # (B, D, N)
            yt = (Ct * h).sum(-1)   # (B, D)
            ys.append(yt)

        y = torch.stack(ys, dim=1)  # (B, L, D)

        if D is not None:
            y = y + u * D.unsqueeze(0).unsqueeze(1)

        return y

    def forward(self, x):
        """
        Args:
            x: (B, L, D) — input sequence

        Returns:
            out: (B, L, D) — output sequence
        """
        B, L, D = x.shape

        # 1. in_proj: x → (z, x')
        zx = self.in_proj(x)  # (B, L, 2*d_inner)
        z = zx[:, :, :self.d_inner]          # (B, L, d_inner)
        xp = zx[:, :, self.d_inner:]          # (B, L, d_inner)

        # 2. Depthwise causal conv on x'
        xp = xp.transpose(1, 2)              # (B, d_inner, L)
        xp = self.conv1d(xp)[:, :, :L]       # causal: trim right padding
        xp = xp.transpose(1, 2)              # (B, L, d_inner)

        # 3. Activation
        xp_act = self.act_fn(xp)             # (B, L, d_inner)
        z_act = self.act_fn(z)               # (B, L, d_inner) — no conv for z in v2

        # 4. x_proj: x' → (B, C, dt)
        x_proj_out = self.x_proj(xp_act)     # (B, L, dt_rank + 2*d_state)
        B_proj = x_proj_out[:, :, :self.d_state]                # (B, L, d_state)
        C_proj = x_proj_out[:, :, self.d_state:2*self.d_state]  # (B, L, d_state)
        dt = x_proj_out[:, :, 2*self.d_state:]                  # (B, L, dt_rank)

        # 5. dt projection + softplus
        dt = self.dt_proj(dt)                # (B, L, d_inner)
        dt = F.softplus(dt)

        # 6. Selective scan
        y = self._selective_scan(xp_act, dt, self.A_log, B_proj, C_proj, self.D)

        # 7. Gate
        out = y * z_act
        out = self.out_proj(out)             # (B, L, d_model)

        return out


def transfer_mamba_weights(mamba_py: MambaPy, mamba_ssm_module) -> MambaPy:
    """
    将 mamba_ssm.Mamba v2 的权重转移到 MambaPy。

    mamba_ssm 2.2.6 v2 内部结构:
        in_proj.weight:   (2*d_inner, d_model)          → (384, 96)
        conv1d.weight:    (d_inner, 1, d_conv)          → (192, 1, 4)
        conv1d.bias:      (d_inner,)                     → (192,)
        x_proj.weight:    (dt_rank + 2*d_state, d_inner) → (38, 192)
        dt_proj.weight:   (d_inner, dt_rank)             → (192, 6)
        dt_proj.bias:     (d_inner,)                     → (192,)
        out_proj.weight:  (d_model, d_inner)             → (96, 192)
        A_log:           (d_inner, d_state)             → (192, 16)
        D:               (d_inner,)                     → (192,)
    """
    src_state = mamba_ssm_module.state_dict()
    dst_state = mamba_py.state_dict()

    # 1-to-1 mappings (same key names)
    direct_map = [
        'in_proj.weight', 'conv1d.weight', 'conv1d.bias',
        'x_proj.weight', 'dt_proj.weight', 'dt_proj.bias',
        'out_proj.weight', 'A_log', 'D',
    ]

    matched = 0
    for key in direct_map:
        if key in src_state:
            src_shape = tuple(src_state[key].shape)
            dst_shape = tuple(dst_state[key].shape)
            if src_shape == dst_shape:
                dst_state[key] = src_state[key]
                matched += 1
            else:
                print(f"  ⚠ Shape mismatch '{key}': src {src_shape} vs dst {dst_shape}")
        else:
            print(f"  ⚠ Key '{key}' not found in source module")

    # Load matched weights
    missing, unexpected = mamba_py.load_state_dict(dst_state, strict=True)

    if missing:
        print(f"  ⚠ Missing keys (uninitialized): {missing}")
    if unexpected:
        print(f"  ⚠ Unexpected keys (not loaded): {unexpected}")

    print(f"  ✅ Transferred {matched}/{len(direct_map)} parameter groups")
    return mamba_py


# ===========================================================================
# ONNX-exportable Backbone (replaces MambaLayer with a pure PyTorch version)
# ===========================================================================

class MambaLayerPy(nn.Module):
    """Mamba layer that uses MambaPy instead of mamba_ssm.Mamba."""

    def __init__(self, d_model, dropout=0.0, d_state=16, d_conv=4, expand=2,
                 use_adaLN=False, cond_dim=None):
        super().__init__()
        self.use_adaLN = use_adaLN

        self.mamba = MambaPy(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        if use_adaLN:
            # Reuse AdaLN from original backbone
            from models.FlowMatching.RegimeFlow.arch.backbone import AdaLN
            self.norm = AdaLN(d_model, cond_dim)
        else:
            from models.FlowMatching.RegimeFlow.RegimeFlow_base import RMSNorm
            self.norm = RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x, cond=None):
        z = x.transpose(-1, -2)  # (B, H, L) -> (B, L, H)
        if self.use_adaLN:
            z = self.norm(z, cond)
        else:
            z = self.norm(z)
        z = self.mamba(z)
        z = z.transpose(-1, -2)  # (B, L, H) -> (B, H, L)
        z = self.dropout(z)
        return x + z


def replace_mamba_layers(backbone_model):
    """
    将 BackboneModel 中的 mamba_ssm.Mamba 层替换为 MambaLayerPy。

    原地修改模型并转移权重。
    """
    for i, block in enumerate(backbone_model.residual_blocks):
        if block.block_type == 'mamba':
            old_mamba_layer = block.mamba_layer
            d_model = old_mamba_layer.mamba.d_model
            d_state = old_mamba_layer.mamba.d_state
            d_conv = old_mamba_layer.mamba.d_conv
            expand = old_mamba_layer.mamba.expand

            # 创建新的纯 PyTorch 层
            new_mamba_layer = MambaLayerPy(
                d_model=d_model,
                dropout=old_mamba_layer.dropout.p if hasattr(old_mamba_layer.dropout, 'p') else 0.0,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                use_adaLN=old_mamba_layer.use_adaLN,
                cond_dim=old_mamba_layer.norm.adaLN_modulation.in_features
                    if old_mamba_layer.use_adaLN else None,
            )

            # 转移 Mamba 权重
            transfer_mamba_weights(new_mamba_layer.mamba, old_mamba_layer.mamba)

            # 转移 AdaLN/RMSNorm 权重
            new_mamba_layer.norm.load_state_dict(old_mamba_layer.norm.state_dict())

            # 替换
            block.mamba_layer = new_mamba_layer
            print(f"  ✅ Block {i}: Mamba replaced with PureMamba")

    return backbone_model


# ===========================================================================
# ONNX Export
# ===========================================================================

class BackboneForONNX(nn.Module):
    """
    Wrapper that makes the backbone ONNX-friendly.

    Input:
        t: (B,) — time step
        x_in: (B, L, 1) — input sequence
        cond_emb: (B, cond_dim) — condition embedding

    Output:
        velocity: (B, L, 1) — predicted velocity field
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, t, x_in, cond_emb):
        # 确保正确的 shape
        if t.dim() == 0:
            t = t.unsqueeze(0)
        return self.backbone(t, x_in, cond_emb)


def export_backbone_to_onnx(backbone, onnx_path, device='cpu'):
    """
    将 backbone 导出为 ONNX 格式。
    """
    hidden_dim = backbone.residual_blocks[0].mamba_layer.mamba.d_model \
        if hasattr(backbone.residual_blocks[0], 'mamba_layer') else 96
    cond_dim = backbone.cond_dim if hasattr(backbone, 'cond_dim') and backbone.cond_dim > 0 else 0

    L = 352  # context(96) + prediction(256)

    # 创建 wrapper
    wrapper = BackboneForONNX(backbone)
    wrapper.eval()
    wrapper.to(device)

    # Dummy inputs
    dummy_t = torch.rand(1, device=device)
    dummy_x = torch.randn(1, L, 1, device=device)
    dummy_cond = torch.randn(1, cond_dim, device=device) if cond_dim > 0 else None

    print(f"\n{'='*60}")
    print(f"Exporting backbone to ONNX...")
    print(f"{'='*60}")
    print(f"   Input shapes:")
    print(f"     t:        {dummy_t.shape}")
    print(f"     x_in:     {dummy_x.shape}")
    print(f"     cond_emb: {dummy_cond.shape if dummy_cond is not None else 'None'}")
    print(f"   Output path: {onnx_path}")

    # Forward pass to verify
    with torch.no_grad():
        if cond_dim > 0:
            out = wrapper(dummy_t, dummy_x, dummy_cond)
        else:
            out = wrapper(dummy_t, dummy_x, torch.zeros(1, 0, device=device))
    print(f"   Test output shape: {out.shape}")

    # Export
    if cond_dim > 0:
        input_names = ['t', 'x_in', 'cond_emb']
        dynamic_axes = {
            't': {0: 'batch'},
            'x_in': {0: 'batch'},
            'cond_emb': {0: 'batch'},
            'velocity': {0: 'batch'},
        }
        args = (dummy_t, dummy_x, dummy_cond)
    else:
        input_names = ['t', 'x_in']
        dynamic_axes = {
            't': {0: 'batch'},
            'x_in': {0: 'batch'},
            'velocity': {0: 'batch'},
        }
        args = (dummy_t, dummy_x)

    t0 = time.time()
    torch.onnx.export(
        wrapper,
        args,
        onnx_path,
        input_names=input_names,
        output_names=['velocity'],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    elapsed = time.time() - t0
    print(f"\n✅ ONNX export completed in {elapsed:.1f}s")
    print(f"   File size: {Path(onnx_path).stat().st_size / 1024 / 1024:.1f} MB")

    # 验证
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"   ONNX model check: ✅ valid")
        print(f"   IR version: {onnx_model.ir_version}")
        print(f"   Opset: {onnx_model.opset_import[0].version}")
    except ImportError:
        print("   (Install 'onnx' package for model validation)")
    except Exception as e:
        print(f"   ONNX check warning: {e}")

    return onnx_path


# ===========================================================================
# Full CPU Inference Pipeline (with ONNX Runtime)
# ===========================================================================

class RegimeFlowCPUInference:
    """
    CPU-compatible RegimeFlow inference using ONNX Runtime.

    Only the backbone is exported to ONNX; the BLR prior, condition encoder,
    and normalization/denormalization remain in PyTorch (all standard ops).
    """
    def __init__(self, model, backbone_onnx_path=None):
        self.model = model
        self.device = 'cpu'

        # 提取子模块
        self.condition_encoder = model.condition_encoder
        self.scaler = model.scaler
        self.prior = model.prior

        self.context_length = model.context_length
        self.prediction_length = model.prediction_length
        self.prior_context_length = model.prior_context_length
        self.num_samples = model.num_samples
        self.sigmax = model.sigmax

        # 加载 ONNX backbone
        self.use_onnx = backbone_onnx_path is not None
        if self.use_onnx:
            try:
                import onnxruntime as ort
                # CPU execution provider
                self.ort_session = ort.InferenceSession(
                    backbone_onnx_path,
                    providers=['CPUExecutionProvider']
                )
                print(f"✅ ONNX Runtime session created")
                print(f"   Inputs: {[i.name for i in self.ort_session.get_inputs()]}")
                print(f"   Outputs: {[o.name for o in self.ort_session.get_outputs()]}")
            except ImportError:
                print("⚠ onnxruntime not installed, falling back to PyTorch")
                self.use_onnx = False
        else:
            self.backbone = model.backbone

    def predict(self, past_target, traj_pattern=0, period=0.0):
        """
        Run prediction for a single species.

        Args:
            past_target: (B, L, 1) tensor
            traj_pattern: int
            period: float

        Returns:
            predictions: (B, num_samples, prediction_length, 1)
        """
        B, L, C = past_target.shape
        assert C == 1 and L == self.context_length

        # 重复 num_samples 次
        past_rep = past_target.repeat_interleave(self.num_samples, dim=0)
        past_obs = torch.ones_like(past_rep)
        B_rep = past_rep.shape[0]

        # Condition
        pattern_t = torch.full((B_rep,), traj_pattern, dtype=torch.long)
        period_t = torch.full((B_rep,), period, dtype=torch.float32)

        # Condition encoder
        if self.condition_encoder is not None:
            cond_emb = self.condition_encoder(
                batch_size=B_rep, device='cpu',
                traj_pattern=pattern_t, period=period_t,
            )
        else:
            cond_emb = None

        # Normalization
        _, loc, scale = self.scaler(past_rep, past_obs)
        scaled_prior = (past_rep[:, -self.prior_context_length:] - loc) / scale

        # BLR prior: generate x0
        past_times = torch.arange(self.prior_context_length, dtype=torch.float32).unsqueeze(0).repeat(B_rep, 1)
        future_times = torch.arange(self.prior_context_length, self.prior_context_length + self.prediction_length,
                                     dtype=torch.float32).unsqueeze(0).repeat(B_rep, 1)

        # Use default pattern=0, period=0 for BLR prior (since we pass them separately)
        blr_pattern = torch.zeros(B_rep, dtype=torch.long)
        blr_period = torch.zeros(B_rep, dtype=torch.float32)

        fut_list = []
        for ch in range(C):
            x0_ch, _, _ = self.prior(
                past_times=past_times,
                past_values=scaled_prior[:, :, ch],
                future_times=future_times,
                pattern_ids=blr_pattern,
                period=blr_period,
            )
            fut_list.append(x0_ch.unsqueeze(-1))
        fut = torch.cat(fut_list, dim=-1)  # (B_rep, pred_len, C)

        # Context (scaled)
        scaled_context = past_rep[:, -self.context_length:] / scale
        x0 = torch.cat([scaled_context, fut], dim=1)  # (B_rep, L_total, 1)
        x0 = x0 + self.sigmax * torch.randn_like(x0)

        # Backbone: predict velocity
        t_zero = torch.zeros(B_rep)

        if self.use_onnx:
            # ONNX Runtime inference
            ort_inputs = {
                't': t_zero.numpy().astype(np.float32),
                'x_in': x0.numpy().astype(np.float32),
            }
            if cond_emb is not None:
                ort_inputs['cond_emb'] = cond_emb.numpy().astype(np.float32)
            velocity = self.ort_session.run(None, ort_inputs)[0]
            velocity = torch.from_numpy(velocity)
        else:
            with torch.no_grad():
                velocity = self.backbone(t_zero, x0, cond_emb)

        # Euler step: x1 = x0 + velocity
        pred = x0 + velocity
        pred = pred[:, -self.prediction_length:]  # take future part

        # Denormalize
        pred = pred * scale + loc
        pred = pred.reshape(B, self.num_samples, self.prediction_length, 1)

        return pred


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="RegimeFlow ONNX Export & CPU Inference")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--export", action="store_true", help="Export backbone to ONNX")
    parser.add_argument("--test", action="store_true", help="Test CPU inference")
    parser.add_argument("--onnx-path", type=str, default=None,
                        help="Output path for ONNX model")
    parser.add_argument("--compare", action="store_true",
                        help="Compare GPU vs CPU predictions")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        candidates = [
            PROJECT_ROOT / "ckpt" / "RegimeFlow" / "seed53_best.ckpt",
            PROJECT_ROOT / "seed53_best.ckpt",
        ]
        for c in candidates:
            if c.exists():
                checkpoint_path = str(c)
                break
    if checkpoint_path is None:
        print("❌ Checkpoint not found. Specify with --checkpoint")
        sys.exit(1)

    onnx_path = args.onnx_path or str(PROJECT_ROOT / "ckpt" / "RegimeFlow" / "backbone.onnx")

    print(f"🚀 RegimeFlow ONNX Export & CPU Inference")
    print(f"   Checkpoint: {checkpoint_path}")

    # ── Step 1: Load model (on GPU if available) ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Loading on:  {device}")

    import pytorch_lightning as pl
    from models.FlowMatching.RegimeFlow.RegimeFlow import RegimeFlowCond

    model = RegimeFlowCond.load_from_checkpoint(
        checkpoint_path, map_location=device, strict=False
    )
    model.eval()
    print(f"✅ Model loaded")

    # ── Step 2: Replace Mamba with Pure PyTorch version ──
    print(f"\n📦 Replacing Mamba layers with Pure PyTorch...")
    backbone_cpu = replace_mamba_layers(model.backbone)
    backbone_cpu = backbone_cpu.cpu()
    backbone_cpu.eval()

    # ── Step 3: Move condition encoder to CPU too ──
    if model.condition_encoder is not None:
        model.condition_encoder = model.condition_encoder.cpu()
        model.condition_encoder.eval()

    # ── Step 4: Export to ONNX ──
    if args.export:
        export_backbone_to_onnx(backbone_cpu, onnx_path, device='cpu')

    # ── Step 5: Test CPU Inference ──
    if args.test:
        print(f"\n{'='*60}")
        print(f"Testing CPU Inference...")
        print(f"{'='*60}")

        # Create CPU inference pipeline
        model_cpu = model.cpu()
        engine = RegimeFlowCPUInference(model_cpu, onnx_path if args.export else None)

        # Generate test data
        t = np.arange(96, dtype=np.float32)
        context = 0.5 + 0.3 * np.sin(2 * np.pi * t / 12.5) + np.random.randn(96) * 0.05
        past = torch.from_numpy(context.astype(np.float32)).view(1, 96, 1)

        print(f"\n   Pattern: oscillation (3)")
        print(f"   Period:  12.5")

        t0 = time.time()
        with torch.no_grad():
            pred = engine.predict(past, traj_pattern=3, period=12.5)
        elapsed = time.time() - t0

        pred_np = pred.squeeze(-1).squeeze(0).numpy()
        print(f"\n✅ CPU inference completed in {elapsed*1000:.1f} ms")
        print(f"   Output shape: {pred_np.shape}")
        print(f"   Mean: {pred_np.mean():.4f}")
        print(f"   Std:  {pred_np.std():.4f}")

        # ── Step 6: Compare GPU vs CPU (if GPU available) ──
        if args.compare and torch.cuda.is_available():
            print(f"\n{'='*60}")
            print(f"Comparing GPU vs CPU predictions...")
            print(f"{'='*60}")

            model_gpu = model.cuda()
            past_gpu = past.cuda()

            t0 = time.time()
            with torch.no_grad():
                pred_gpu = model_gpu(
                    past_target=past_gpu,
                    past_observed_values=torch.ones_like(past_gpu),
                    mean=None,
                    condition_dict={
                        'traj_pattern': torch.tensor([3], device='cuda'),
                        'period': torch.tensor([12.5], device='cuda'),
                    },
                )
            gpu_time = (time.time() - t0) * 1000

            pred_gpu_np = pred_gpu.squeeze(-1).squeeze(0).cpu().numpy()

            # Compare
            diff = np.abs(pred_np.mean(0) - pred_gpu_np.mean(0))
            print(f"   GPU time:     {gpu_time:.1f} ms")
            print(f"   CPU time:     {elapsed*1000:.1f} ms")
            print(f"   Max diff:     {diff.max():.6f}")
            print(f"   Mean diff:    {diff.mean():.6f}")
            if diff.max() < 0.1:
                print(f"   ✅ GPU vs CPU predictions are consistent")
            else:
                print(f"   ⚠ Significant difference — may need debugging")

    print(f"\n{'='*60}")
    print(f"✅ Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
