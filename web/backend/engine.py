"""
RegimeFlow CPU Inference Engine
================================
Lightweight inference for biological trajectory forecasting.
Uses pure PyTorch Mamba implementation — no CUDA required.

Architecture:
    Context (96 steps) + Conditions (pattern, period)
        -> BLR Prior -> x0
        -> Condition Encoder -> cond_emb
        -> Backbone (MambaPy x4 blocks) -> velocity (4 denoise steps)
        -> Denormalize -> Prediction (256 steps)
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

# Ensure project root is on path (needed for models.* imports)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pure PyTorch Mamba v2 — CPU compatible
# ---------------------------------------------------------------------------

class MambaPy(nn.Module):
    """Pure PyTorch Mamba v2 — matches mamba_ssm==2.2.6 weight layout."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(d_model * expand)
        self.dt_rank = max(1, d_model // 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1)
        self.act_fn = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _selective_scan(self, u, delta, A, B, C, D):
        B_dim, L, D_dim = u.shape
        N = A.shape[1]
        device, dtype = u.device, u.dtype

        A_real = -torch.exp(A.float())
        delta_A = delta.unsqueeze(-1) * A_real.unsqueeze(0).unsqueeze(1)
        A_bar = torch.exp(delta_A)
        B_bar = delta.unsqueeze(-1) * B.unsqueeze(2)
        h = torch.zeros(B_dim, D_dim, N, device=device, dtype=dtype)
        ys = []

        for t in range(L):
            h = A_bar[:, t] * h + B_bar[:, t] * u[:, t].unsqueeze(-1)
            ys.append((C[:, t].unsqueeze(1) * h).sum(-1))

        y = torch.stack(ys, dim=1)
        if D is not None:
            y = y + u * D.unsqueeze(0).unsqueeze(1)
        return y

    def forward(self, x):
        B, L, D = x.shape
        zx = self.in_proj(x)
        z, xp = zx[:, :, :self.d_inner], zx[:, :, self.d_inner:]

        xp = self.conv1d(xp.transpose(1, 2))[:, :, :L].transpose(1, 2)
        xp_act, z_act = self.act_fn(xp), self.act_fn(z)

        x_out = self.x_proj(xp_act)
        B_p = x_out[:, :, :self.d_state]
        C_p = x_out[:, :, self.d_state:2 * self.d_state]
        dt = F.softplus(self.dt_proj(x_out[:, :, 2 * self.d_state:]))

        y = self._selective_scan(xp_act, dt, self.A_log, B_p, C_p, self.D)
        return self.out_proj(y * z_act)


# ---------------------------------------------------------------------------
# Monkey-patch mamba_ssm BEFORE importing model modules
# ---------------------------------------------------------------------------

class _FakeMambaSSM:
    Mamba = MambaPy
    __version__ = "2.2.6-py"


sys.modules['mamba_ssm'] = _FakeMambaSSM()

# Now safe to import
from models.FlowMatching.RegimeFlow.RegimeFlow_base import RMSNorm
from models.FlowMatching.RegimeFlow.arch._base import StdScaler
from models.FlowMatching.RegimeFlow.arch.bio_cond_layers import ConditionEncoder
from models.FlowMatching.RegimeFlow.arch.source_BLR import (
    BLRTemplatePriorGenerator, BLRPriorConfig
)
from models.FlowMatching.RegimeFlow.arch.backbone import (
    BackboneModel, MambaLayer, AdaLN
)

# Patch MambaLayer to use MambaPy
_original_mamba_init = MambaLayer.__init__


def _patched_init(self, d_model, dropout=0.0, d_state=16, d_conv=4,
                  expand=2, use_adaLN=False, cond_dim=None, **kwargs):
    nn.Module.__init__(self)
    self.use_adaLN = use_adaLN
    self.mamba = MambaPy(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    if use_adaLN:
        self.norm = AdaLN(d_model, cond_dim)
    else:
        self.norm = RMSNorm(d_model)
    self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()


MambaLayer.__init__ = _patched_init


# ---------------------------------------------------------------------------
# RegimeFlow Inference Engine
# ---------------------------------------------------------------------------

class RegimeFlowEngine:
    """
    CPU-compatible RegimeFlow inference engine.

    Usage:
        engine = RegimeFlowEngine(checkpoint_path)
        pred = engine.predict(context=[...], traj_pattern=3, period=12.5)
    """

    def __init__(self, checkpoint_path: str, denoise_steps: int = 4):
        self.denoise_steps = denoise_steps

        logger.info(f"Loading RegimeFlow from {checkpoint_path} ...")
        t0 = time.time()

        # Load checkpoint
        import omegaconf, collections, typing
        torch.serialization.add_safe_globals([
            omegaconf.DictConfig, omegaconf.ListConfig,
            omegaconf.base.ContainerMetadata, typing.Any,
            dict, collections.defaultdict, omegaconf.nodes.AnyNode,
            omegaconf.base.Metadata, list, int,
        ])

        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        hp = ckpt['hyper_parameters']
        state_dict = ckpt['state_dict']
        ema_dict = ckpt.get('ema_state_dict')

        # Use EMA weights if available (better inference quality)
        weights = ema_dict if ema_dict else state_dict

        self.context_length = hp['context_length']
        self.prediction_length = hp['prediction_length']
        self.backbone_params = hp['backbone_params']

        # --- Build components ---

        # Scaler
        self.scaler = StdScaler(dim=1, keepdim=True, minimum_scale=1)

        # Condition Encoder
        self.use_condition = hp.get('use_condition', True)
        self.cond_dim = hp.get('cond_dim', 128)
        if self.use_condition:
            self.condition_encoder = ConditionEncoder(
                d_model=self.cond_dim,
                num_patterns=hp.get('num_patterns', 6),
                num_freqs=hp.get('num_freqs', 128),
            )
        else:
            self.condition_encoder = None

        # Backbone
        self.backbone = BackboneModel(
            input_dim=self.backbone_params['input_dim'],
            hidden_dim=self.backbone_params['hidden_dim'],
            output_dim=self.backbone_params['output_dim'],
            step_emb=self.backbone_params['step_emb'],
            num_residual_blocks=self.backbone_params['num_residual_blocks'],
            init_skip=self.backbone_params.get('init_skip', True),
            block_type=self.backbone_params.get('block_type', 'mamba'),
            d_state=self.backbone_params.get('d_state', 16),
            d_conv=self.backbone_params.get('d_conv', 4),
            expand=self.backbone_params.get('expand', 2),
            ffn_dim_multiplier=self.backbone_params.get('ffn_dim_multiplier', 4.0),
            ffn_dropout=self.backbone_params.get('ffn_dropout', 0.0),
            dropout=self.backbone_params.get('dropout', 0.0),
            use_adaLN=self.use_condition,
            cond_dim=self.cond_dim,
        )

        # BLR Prior
        prior_params = dict(hp['prior_params'])
        prior_type = prior_params.pop('name', 'BLRTemplatePriorGenerator')
        blr_config = BLRPriorConfig(
            alpha=prior_params.get('alpha', 1.0),
            beta=prior_params.get('beta', 20.0),
            noise_scale=prior_params.get('noise_scale', 0.1),
            saturation_rate=prior_params.get('saturation_rate', 3.0),
            slope_window=prior_params.get('slope_window', 10),
            min_variance=prior_params.get('min_variance', 1e-6),
            num_harmonics=prior_params.get('num_harmonics', 4),
            saturation_scales=tuple(prior_params.get('saturation_scales', (0.2, 0.5, 1.0, 2.0, 5.0))),
        )
        self.prior = BLRTemplatePriorGenerator(blr_config)

        # --- Load weights ---
        bb_weights = {k.replace('backbone.', ''): v for k, v in weights.items()
                      if k.startswith('backbone.')}
        self.backbone.load_state_dict(bb_weights, strict=False)

        if self.condition_encoder is not None:
            ce_weights = {k.replace('condition_encoder.', ''): v for k, v in weights.items()
                          if k.startswith('condition_encoder.')}
            self.condition_encoder.load_state_dict(ce_weights, strict=True)

        # Eval mode
        self.backbone.eval()
        if self.condition_encoder is not None:
            self.condition_encoder.eval()

        total_params = sum(p.numel() for p in self.backbone.parameters())
        elapsed = time.time() - t0
        logger.info(f"RegimeFlow loaded ({total_params:,} params) in {elapsed:.1f}s")
        logger.info(f"  context={self.context_length}, pred={self.prediction_length}, "
                    f"denoise_steps={denoise_steps}, condition={self.use_condition}")

    def predict(
        self,
        context: list[float],
        traj_pattern: int = 0,
        period: float = 0.0,
    ) -> np.ndarray:
        """
        Predict future trajectory.

        Args:
            context: Past observations (at least context_length values).
                     If longer, the last context_length values are used.
            traj_pattern: Trajectory regime (0-5):
                0=DIRECTLY_STABLE, 1=INC_STABLE, 2=DEC_STABLE,
                3=OSCILLATION, 4=INCREASING, 5=DECREASING
            period: Period for oscillation regime (ignored for others).

        Returns:
            Prediction array of shape (prediction_length,).
        """
        # Prepare context
        ctx = np.asarray(context, dtype=np.float32).flatten()
        if len(ctx) < self.context_length:
            # Pad with replication at the beginning
            pad = self.context_length - len(ctx)
            ctx = np.pad(ctx, (pad, 0), mode='edge')
        ctx = ctx[-self.context_length:]

        past_target = torch.from_numpy(ctx).float().view(1, self.context_length, 1)

        # Condition encoding
        if self.condition_encoder is not None:
            cond_emb = self.condition_encoder(
                batch_size=1, device='cpu',
                traj_pattern=torch.tensor([traj_pattern], dtype=torch.long),
                period=torch.tensor([period], dtype=torch.float32),
            )
        else:
            cond_emb = None

        # Normalize
        past_observed = torch.ones_like(past_target)
        _, loc, scale = self.scaler(past_target, past_observed)

        # BLR prior -> x0 future
        past_times = torch.arange(self.context_length, dtype=torch.float32).unsqueeze(0)
        fut_times = torch.arange(
            self.context_length,
            self.context_length + self.prediction_length,
            dtype=torch.float32
        ).unsqueeze(0)

        scaled_prior = (past_target[:, -self.context_length:] - loc) / scale
        x0_fut, _, _ = self.prior(
            past_times=past_times,
            past_values=scaled_prior[:, :, 0],
            future_times=fut_times,
            pattern_ids=torch.zeros(1, dtype=torch.long),
            period=torch.zeros(1),
        )
        x0_fut = x0_fut.unsqueeze(-1)

        scaled_context = past_target[:, -self.context_length:] / scale
        x0 = torch.cat([scaled_context, x0_fut], dim=1)

        # Add noise (sigma_max = 1)
        x0 = x0 + torch.randn_like(x0)

        # Flow Matching ODE (denoise_steps Euler steps)
        dt_ode = 1.0 / self.denoise_steps
        xt = x0

        with torch.no_grad():
            for step in range(self.denoise_steps):
                t_step = step * dt_ode
                t_tensor = torch.full((1,), t_step, dtype=torch.float32)
                vt = self.backbone(t_tensor, xt, cond_emb)
                xt = xt + dt_ode * vt

        # Denormalize and extract future
        pred = (xt[:, self.context_length:] * scale + loc).squeeze().numpy()
        return pred.astype(np.float64)


# ---------------------------------------------------------------------------
# Singleton loader
# ---------------------------------------------------------------------------

_engine: Optional[RegimeFlowEngine] = None


def get_engine() -> Optional[RegimeFlowEngine]:
    return _engine


def init_engine(checkpoint_path: str, denoise_steps: int = 4) -> RegimeFlowEngine:
    global _engine
    if _engine is not None:
        return _engine
    _engine = RegimeFlowEngine(checkpoint_path, denoise_steps=denoise_steps)
    return _engine
