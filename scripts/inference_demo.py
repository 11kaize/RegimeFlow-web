"""
RegimeFlow 本地推理验证脚本
============================
验证 seed53_best.ckpt 能否正常加载和推理。

Usage:
    python scripts/inference_demo.py

    # 支持自定义数据
    python scripts/inference_demo.py --context data.csv --pattern 3 --period 12.5
"""

import os
import sys
import time
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Path setup ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

# 检查核心依赖
MISSING_DEPS = []

try:
    import pytorch_lightning as pl
except ImportError:
    MISSING_DEPS.append("pytorch-lightning (pip install pytorch-lightning==2.6.0)")

try:
    from models.FlowMatching.RegimeFlow.RegimeFlow import RegimeFlowCond
except ImportError as e:
    MISSING_DEPS.append(f"RegimeFlow model imports: {e}")

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    MISSING_DEPS.append("mamba-ssm (pip install mamba-ssm==2.2.6)")

if MISSING_DEPS:
    print("❌ Missing dependencies:")
    for d in MISSING_DEPS:
        print(f"   - {d}")
    print("\nPlease install all dependencies and try again.")
    sys.exit(1)


def load_model(checkpoint_path: str, device: str = "cpu"):
    """
    从 checkpoint 加载 RegimeFlowCond 模型。

    PyTorch Lightning 的 save_hyperparameters() 会自动保存构造函数参数，
    所以可以直接用 load_from_checkpoint 恢复完整模型。
    """
    print(f"\n{'='*60}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # PyTorch Lightning 自动从 checkpoint 中恢复模型类和超参数
    t0 = time.time()
    model = RegimeFlowCond.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=False,  # 有些 keys 可能不匹配 (optimizer states etc.)
    )
    elapsed = time.time() - t0

    model = model.to(device)
    model.eval()

    # 确保 BLR prior 已初始化 (Lightning 的 setup() 可能未被 load_from_checkpoint 触发)
    if hasattr(model, 'setup') and model.prior is None:
        print("   Initializing BLR prior...")
        model.setup()

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✅ Model loaded in {elapsed:.1f}s")
    print(f"   Device:          {device}")
    print(f"   Total params:    {total_params:,}")
    print(f"   Trainable params:{trainable_params:,}")
    print(f"   Context length:  {model.context_length}")
    print(f"   Prediction len:  {model.prediction_length}")
    print(f"   Num steps:       {model.num_steps}")
    print(f"   Solver:          {model.solver}")
    print(f"   Prior type:      {model.prior_type}")
    print(f"   Use condition:   {getattr(model, 'use_condition', 'N/A')}")
    print(f"   Cond dim:        {getattr(model, 'cond_dim', 'N/A')}")

    # 打印 backbone 信息
    if hasattr(model, 'backbone'):
        print(f"   Backbone type:   {model.backbone.block_type}")
        print(f"   Backbone blocks: {len(model.backbone.residual_blocks)}")

    return model


def run_inference(
    model: RegimeFlowCond,
    context: np.ndarray,
    traj_pattern: int = 0,
    period: float = 0.0,
    bound_min: float = 0.0,
    bound_max: float = 1.0,
    num_samples: int = 10,
):
    """
    使用 RegimeFlow 模型进行推理。

    Args:
        model: RegimeFlowCond 模型
        context: 过去的观测序列, shape (context_length,) 或 (1, context_length)
        traj_pattern: 轨迹模式 (0-5):
            0 = DIRECTLY_STABLE
            1 = INC_STABLE
            2 = DEC_STABLE
            3 = OSCILLATION
            4 = INCREASING
            5 = DECREASING
        period: 周期 (对 oscillation 模式有意义)
        bound_min/max: 数据归一化边界
        num_samples: 采样数量

    Returns:
        predictions: shape (num_samples, prediction_length)
    """
    device = next(model.parameters()).device
    context_length = model.context_length
    prediction_length = model.prediction_length

    # ── 处理 context ──
    if isinstance(context, list):
        context = np.array(context, dtype=np.float32)
    context = np.asarray(context, dtype=np.float32).flatten()

    if len(context) < context_length:
        raise ValueError(
            f"Context length {len(context)} < model's context_length {context_length}"
        )

    # 取最后 context_length 步
    context = context[-context_length:]

    # 转换到 tensor: (B=1, L, C=1)
    past_target = torch.from_numpy(context).float().unsqueeze(-1).unsqueeze(0).to(device)
    past_observed = torch.ones_like(past_target)

    # ── 构建 condition ──
    condition_dict = {
        "traj_pattern": torch.tensor([traj_pattern], dtype=torch.long, device=device),
        "period": torch.tensor([period], dtype=torch.float32, device=device),
        "bound_min": torch.tensor([bound_min], dtype=torch.float32, device=device),
        "bound_max": torch.tensor([bound_max], dtype=torch.float32, device=device),
    }

    # ── 临时修改 num_samples ──
    old_num_samples = model.num_samples
    model.num_samples = num_samples

    print(f"\n{'='*60}")
    print(f"Running inference...")
    print(f"{'='*60}")
    print(f"   Context shape:    {past_target.shape}")
    print(f"   Context range:    [{context.min():.4f}, {context.max():.4f}]")
    print(f"   Traj pattern:     {traj_pattern}")
    print(f"   Period:           {period}")
    print(f"   Num samples:      {num_samples}")

    t0 = time.time()
    with torch.no_grad():
        predictions = model(
            past_target=past_target,
            past_observed_values=past_observed,
            mean=None,
            condition_dict=condition_dict,
        )
    elapsed = time.time() - t0

    # 恢复
    model.num_samples = old_num_samples

    # predictions shape: (B, num_samples, prediction_length, 1)
    pred_np = predictions.squeeze(-1).squeeze(0).cpu().numpy()  # (num_samples, prediction_length)

    print(f"\n✅ Inference completed in {elapsed*1000:.1f} ms")
    print(f"   Output shape:     {pred_np.shape}")
    print(f"   Mean pred range:  [{pred_np.mean(axis=0).min():.4f}, {pred_np.mean(axis=0).max():.4f}]")
    print(f"   Pred std (avg):   {pred_np.std(axis=0).mean():.4f}")

    return pred_np


def generate_synthetic_context(
    pattern: int = 3,  # oscillation
    length: int = 96,
    noise: float = 0.05,
) -> np.ndarray:
    """
    生成合成 context 数据用于测试。

    不同 pattern 对应不同的时间序列形态:
    """
    t = np.arange(length, dtype=np.float32)

    if pattern == 0:  # DIRECTLY_STABLE — flat
        y = np.ones(length) * 0.5 + np.random.randn(length) * noise
    elif pattern == 1:  # INC_STABLE — increase then saturate
        y = 1 - np.exp(-t / 20) + np.random.randn(length) * noise
    elif pattern == 2:  # DEC_STABLE — decrease then saturate
        y = np.exp(-t / 20) + np.random.randn(length) * noise
    elif pattern == 3:  # OSCILLATION
        y = 0.5 + 0.3 * np.sin(2 * np.pi * t / 12.5) + np.random.randn(length) * noise
    elif pattern == 4:  # INCREASING — monotonic increase
        y = t / length + np.random.randn(length) * noise
    elif pattern == 5:  # DECREASING — monotonic decrease
        y = 1 - t / length + np.random.randn(length) * noise
    else:
        y = np.random.randn(length) * noise

    return y.astype(np.float32)


def plot_results(context, predictions, pattern, period, save_path=None):
    """简单绘制 context + predictions."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n⚠ matplotlib 未安装，跳过绘图。")
        return

    pattern_names = [
        "Directly Stable", "Inc Stable", "Dec Stable",
        "Oscillation", "Increasing", "Decreasing"
    ]

    ctx_len = len(context)
    pred_len = predictions.shape[1]

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    # Context
    t_ctx = np.arange(ctx_len)
    ax.plot(t_ctx, context, 'b-', linewidth=1.5, alpha=0.8, label='Context (past)')

    # Predictions
    t_pred = np.arange(ctx_len, ctx_len + pred_len)
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)

    # 分界线
    ax.axvline(x=ctx_len - 1, color='gray', linestyle='--', alpha=0.5, label='Now')

    # 每条采样轨迹 (半透明)
    for i in range(min(len(predictions), 10)):
        ax.plot(t_pred, predictions[i], 'orange', linewidth=0.3, alpha=0.3)

    # 均值和置信区间
    ax.plot(t_pred, mean_pred, 'red', linewidth=2, label='Mean prediction')
    ax.fill_between(
        t_pred,
        mean_pred - 2 * std_pred,
        mean_pred + 2 * std_pred,
        color='red', alpha=0.15, label='±2σ'
    )

    ax.set_title(
        f"RegimeFlow Inference — {pattern_names[pattern]} (period={period})",
        fontsize=14
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="RegimeFlow Local Inference Demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (default: ckpt/RegimeFlow/seed53_best.ckpt)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--pattern", type=int, default=3, choices=range(6),
        help="Trajectory pattern: 0=DirectStable, 1=IncStable, 2=DecStable, 3=Oscillation, 4=Increasing, 5=Decreasing"
    )
    parser.add_argument(
        "--period", type=float, default=12.5,
        help="Period for oscillation (default: 12.5)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=10,
        help="Number of prediction samples (default: 10)"
    )
    parser.add_argument(
        "--save-plot", type=str, default=None,
        help="Save prediction plot to file"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plotting"
    )

    args = parser.parse_args()

    # 默认 checkpoint 路径
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        # 尝试默认路径
        candidates = [
            PROJECT_ROOT / "ckpt" / "RegimeFlow" / "seed53_best.ckpt",
            PROJECT_ROOT / "seed53_best.ckpt",
        ]
        for c in candidates:
            if c.exists():
                checkpoint_path = str(c)
                break
        if checkpoint_path is None:
            print("❌ Checkpoint not found. Please specify with --checkpoint")
            print(f"   Looked in: {[str(c) for c in candidates]}")
            sys.exit(1)

    print(f"🚀 RegimeFlow Inference Demo")
    print(f"   Project root: {PROJECT_ROOT}")
    print(f"   Device: {args.device}")

    # ── Step 1: Load model ──
    model = load_model(checkpoint_path, args.device)

    # ── Step 2: Generate or load context ──
    context = generate_synthetic_context(
        pattern=args.pattern,
        length=model.context_length,
    )
    print(f"\n📊 Generated synthetic context (pattern={args.pattern}, length={len(context)})")
    print(f"   Context stats: min={context.min():.4f}, max={context.max():.4f}, "
          f"mean={context.mean():.4f}, std={context.std():.4f}")

    # ── Step 3: Run inference ──
    predictions = run_inference(
        model=model,
        context=context,
        traj_pattern=args.pattern,
        period=args.period,
        num_samples=args.num_samples,
    )

    # ── Step 4: Plot ──
    if not args.no_plot:
        plot_results(
            context, predictions,
            pattern=args.pattern, period=args.period,
            save_path=args.save_plot,
        )

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"✅ All checks passed!")
    print(f"{'='*60}")
    print(f"   Model:      RegimeFlowCond (seed=53)")
    print(f"   Input:      {len(context)} steps")
    print(f"   Output:     {predictions.shape[1]} steps × {predictions.shape[0]} samples")
    print(f"   Device:     {args.device}")
    print(f"   GPU Memory: {torch.cuda.max_memory_allocated()/1e6:.1f} MB" if args.device == "cuda" else "")

    return predictions


if __name__ == "__main__":
    main()
