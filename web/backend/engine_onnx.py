"""
RegimeFlow Lightweight Inference Engine (ONNX Runtime + NumPy)
==============================================================
Zero PyTorch dependency. Uses ONNX Runtime for neural components,
pure NumPy for BLR prior and data processing.

Components:
  - ConditionEncoder  → ONNX (cond_encoder.onnx, ~4KB)
  - Backbone           → ONNX (backbone.onnx, 34MB)
  - BLR Prior          → NumPy implementation
  - StdScaler          → NumPy implementation

Memory: ~150MB RSS (vs 400MB+ for PyTorch)
"""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from web.backend.inference_config import load_config, resolve_condition, transform_context

logger = logging.getLogger(__name__)

# ── ONNX Runtime imports ──────────────────────────────────────
try:
    import onnxruntime as ort
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False
    logger.warning("onnxruntime not installed; ONNX inference unavailable")


# ===========================================================================
# NumPy: StdScaler (masked z-score normalization)
# ===========================================================================

class StdScalerNP:
    """Masked z-score normalizer (pure NumPy)."""

    def __init__(self, minimum_scale=1e-10):
        self.minimum_scale = minimum_scale

    def fit_transform(self, x, observed=None):
        """Compute loc/scale and return normalized x."""
        if observed is None:
            observed = np.ones_like(x)
        # Model tensors are (batch, time, channels): match StdScaler(dim=1).
        num_obs = np.sum(observed, axis=1, keepdims=True)
        num_obs = np.maximum(num_obs, 1)
        loc = np.sum(x * observed, axis=1, keepdims=True) / num_obs
        centered = (x - loc) * observed
        variance = np.sum(centered ** 2, axis=1, keepdims=True) / num_obs
        scale = np.maximum(np.sqrt(variance), self.minimum_scale)
        return loc, scale


# ===========================================================================
# NumPy: RMSNorm
# ===========================================================================

class RMSNormNP:
    """Root Mean Square Layer Normalization (pure NumPy)."""

    def __init__(self, weight: np.ndarray, eps=1e-6):
        self.weight = weight  # (dim,)
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (..., dim), normalize along last axis."""
        rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight


# ===========================================================================
# NumPy: BLR Prior Generator
# ===========================================================================

class BLRPriorNP:
    """Bayesian Linear Regression prior for flow matching trajectory generation."""

    def __init__(self, config: Optional[dict] = None):
        settings = load_config()
        settings.update(config or {})
        for key in ('alpha', 'beta', 'noise_scale', 'saturation_scales',
                    'poly_degree', 'num_harmonics', 'min_variance'):
            setattr(self, key, settings[key])
        if isinstance(self.saturation_scales, str):
            import json
            self.saturation_scales = json.loads(self.saturation_scales)
        if not isinstance(self.saturation_scales, (list, tuple)) or any(
                not isinstance(x, (int, float)) or not np.isfinite(x) or x <= 0
                for x in self.saturation_scales):
            raise ValueError('Invalid saturation_scales')

    def basis(self, times, past_times, pattern_id, period):
        times = np.asarray(times, dtype=np.float64)
        past_times = np.asarray(past_times, dtype=np.float64)
        delta = times - past_times[-1]
        t_range = past_times[-1] - past_times[0]
        if pattern_id in (1, 2):
            rate = 1.0 / (t_range + 1e-6)
            return np.column_stack([np.ones(len(times))] +
                                   [1 - np.exp(-rate * v * delta) for v in self.saturation_scales])
        if pattern_id == 3:
            period_time = max(period * t_range / (len(past_times) - 1), t_range * .05)
            omega = 2 * np.pi / period_time
            return np.column_stack([np.ones(len(times))] +
                                   [f(h * omega * times) for h in range(1, self.num_harmonics + 1)
                                    for f in (np.sin, np.cos)])
        if pattern_id in (4, 5):
            # Reference normalizes past and future basis separately.
            normalized = delta / max(float(np.abs(delta).max()), 1e-6)
            return np.column_stack([normalized ** d for d in range(self.poly_degree + 1)])
        return np.ones((len(times), 1))

    @staticmethod
    def _blr_solve(Phi_past, y_past, Phi_future, alpha, beta, min_var=1e-6):
        precision = alpha * np.eye(Phi_past.shape[1]) + beta * Phi_past.T @ Phi_past
        rhs = beta * Phi_past.T @ np.asarray(y_past).reshape(-1)
        try:
            L = np.linalg.cholesky(precision)
            mean_weights = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
            projected = np.linalg.solve(L, Phi_future.T)
            var = 1 / beta + np.sum(projected ** 2, axis=0)
        except np.linalg.LinAlgError:
            covariance = np.linalg.pinv(precision)
            mean_weights = covariance @ rhs
            var = 1 / beta + np.sum((Phi_future @ covariance) * Phi_future, axis=1)
        return np.sum(Phi_future * mean_weights[None, :], axis=1), np.maximum(var, min_var)

    def mean(self, past_times, past_values, future_times, pattern_id=0, period=0.0):
        pattern_id, period = resolve_condition(pattern_id, period)
        pp = self.basis(past_times, past_times, pattern_id, period)
        pf = self.basis(future_times, past_times, pattern_id, period)
        mu, var = self._blr_solve(pp, past_values, pf, self.alpha, self.beta, self.min_variance)
        mu += past_values[-1] - mu[0]
        if not np.isfinite(mu).all():
            raise ValueError('Nonfinite BLR mean')
        return mu, var

    def generate(self, past_times, past_values, future_times, pattern_id=0, period=0.0, noise=None):
        mu, _ = self.mean(past_times, past_values, future_times, pattern_id, period)
        epsilon = np.random.randn(len(future_times)) if noise is None else np.asarray(noise)
        if epsilon.shape != mu.shape or not np.isfinite(epsilon).all():
            raise ValueError('Invalid BLR noise')
        return (mu + epsilon * self.noise_scale).astype(np.float32)


# ===========================================================================
# ONNX-based RegimeFlow Engine
# ===========================================================================

class RegimeFlowEngineONNX:
    """Torch-free RegimeFlow inference engine using ONNX Runtime."""

    def __init__(
        self,
        backbone_onnx: str,
        cond_encoder_onnx: str,
        checkpoint_path: str,  # still needed for hyperparameters
        denoise_steps: int = 4,
    ):
        if not _HAS_ONNX:
            raise RuntimeError("onnxruntime not installed")

        self.denoise_steps = denoise_steps

        hp = load_config()
        self.config = hp
        self.context_length = hp['context_length']
        self.prediction_length = hp['prediction_length']
        self.use_condition = hp['use_condition']
        self.cond_dim = hp['cond_dim']
        self.flow_sigma = hp['flow_sigma']
        if not isinstance(denoise_steps, int) or denoise_steps <= 0:
            raise ValueError('denoise_steps must be positive')

        # ── Load ONNX models ──
        # Disable graph optimization: on this 36MB Mamba backbone, ORT_ENABLE_ALL
        # spends ~6.5s fusing/rewriting the graph at load time with zero inference
        # gain (measured ~identical step time with it off). On Render's free tier
        # every cold start re-runs this load, so this is ~4x faster startup.
        logger.info(f"Loading ONNX backbone from {backbone_onnx} ...")
        _so = ort.SessionOptions()
        _so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.backbone_session = ort.InferenceSession(
            backbone_onnx, sess_options=_so, providers=['CPUExecutionProvider']
        )
        metadata = self.backbone_session.get_modelmeta().custom_metadata_map
        if metadata.get('regimeflow.mamba_projection_layout') != 'x,z;dt,B,C':
            raise RuntimeError('Legacy/unknown Mamba layout: repair or re-export the backbone')
        logger.info(f"Backbone loaded: inputs={[i.name for i in self.backbone_session.get_inputs()]}")

        logger.info(f"Loading ONNX condition encoder from {cond_encoder_onnx} ...")
        self.cond_encoder_session = ort.InferenceSession(
            cond_encoder_onnx, providers=['CPUExecutionProvider']
        )
        logger.info(f"Condition encoder loaded")

        # ── BLR Prior (NumPy) ──
        prior_params = hp
        self.prior = BLRPriorNP(prior_params)

        # ── Scaler (NumPy) ──
        self.scaler = StdScalerNP(minimum_scale=hp['min_scale'])

        logger.info(f"RegimeFlow ONNX engine ready: ctx={self.context_length}, "
                    f"pred={self.prediction_length}, steps={denoise_steps}")

    def _encode_condition(self, traj_pattern: int, period: float) -> np.ndarray:
        """Encode trajectory condition via ONNX condition encoder."""
        pattern_arr = np.array([traj_pattern], dtype=np.int64)
        period_arr = np.array([period], dtype=np.float32)
        out = self.cond_encoder_session.run(
            None,
            {'traj_pattern': pattern_arr, 'period': period_arr}
        )
        return out[0]  # (1, cond_dim)

    def _backbone_velocity(self, t: float, x: np.ndarray, cond_emb: np.ndarray) -> np.ndarray:
        """Compute velocity via ONNX backbone."""
        t_arr = np.array([t], dtype=np.float32)
        x_arr = x.astype(np.float32)
        cond_arr = cond_emb.astype(np.float32)
        out = self.backbone_session.run(
            None,
            {'t': t_arr, 'x_in': x_arr, 'cond_emb': cond_arr}
        )
        return out[0]  # (1, L, 1)

    def predict(
        self,
        context: list[float],
        traj_pattern: int = 0,
        period: float = 0.0,
        *, bounds=None, noise=None, trace=None,
    ) -> np.ndarray:
        """
        Predict future trajectory.

        Args:
            context: Past observations (at least context_length values).
            traj_pattern: 0-5 trajectory regime.
            period: Oscillation period.

        Returns:
            Prediction array of shape (prediction_length,).
        """
        # Prepare context
        ctx, bound_min, bound_span = transform_context(context, bounds)
        traj_pattern, period = resolve_condition(traj_pattern, period)
        noise = noise or {}
        if len(ctx) < self.context_length:
            pad = self.context_length - len(ctx)
            ctx = np.pad(ctx, (pad, 0), mode='edge')
        ctx = ctx[-self.context_length:]

        # Shape for model: (1, context_length, 1)
        past_target = ctx.reshape(1, self.context_length, 1)

        # ── Normalize ──
        past_obs = np.ones_like(past_target)
        loc, scale = self.scaler.fit_transform(past_target, past_obs)

        # ── Condition encoding ──
        if self.use_condition:
            cond_emb = self._encode_condition(traj_pattern, period)
        else:
            cond_emb = np.zeros((1, self.cond_dim), dtype=np.float32)

        # ── BLR Prior → x0 future ──
        past_times = np.arange(self.context_length, dtype=np.float32)
        fut_times = np.arange(self.context_length, self.context_length + self.prediction_length, dtype=np.float32)

        # Use last prior_context_length for BLR (same as context_length for simplicity)
        prior_ctx = past_target[0, :, 0]  # (context_length,)
        scaled_prior = (prior_ctx - loc[0, 0]) / scale[0, 0]

        x0_fut = self.prior.generate(
            past_times=past_times,
            past_values=scaled_prior,
            future_times=fut_times,
            pattern_id=traj_pattern, period=period, noise=noise.get("blr"),
        )
        x0_fut = x0_fut.reshape(1, self.prediction_length, 1)

        # ── Build x0: context + future prior ──
        scaled_context = past_target / scale
        x0 = np.concatenate([scaled_context, x0_fut], axis=1)  # (1, L_total, 1)

        # Additional flow noise, matching RegimeFlowCond's sigmax = sigmin.
        flow_noise = noise.get('flow')
        if flow_noise is None:
            flow_noise = np.random.randn(*x0.shape)
        flow_noise = np.asarray(flow_noise, dtype=np.float32)
        if flow_noise.size != x0.size or not np.isfinite(flow_noise).all():
            raise ValueError('Invalid flow noise')
        x0 = x0 + self.flow_sigma * flow_noise.reshape(x0.shape)
        if trace is not None:
            mu, _ = self.prior.mean(past_times, scaled_prior, fut_times, traj_pattern, period)
            trace.update(transformed_context=ctx.tolist(), loc=loc.tolist(), scale=scale.tolist(),
                         prior_mean=mu.tolist(), prior=x0_fut.reshape(-1).tolist(),
                         condition=cond_emb.tolist(), x0=x0.tolist(), velocities=[], states=[])


        # ── Flow Matching ODE (Euler steps) ──
        dt_ode = 1.0 / self.denoise_steps
        xt = x0

        for step in range(self.denoise_steps):
            t_step = step * dt_ode
            vt = self._backbone_velocity(t_step, xt, cond_emb)
            xt = xt + dt_ode * vt
            if trace is not None:
                trace['velocities'].append(vt.tolist())
                trace['states'].append(xt.tolist())

        # ── Denormalize and extract future ──
        pred = (xt[0, self.context_length:, 0] * scale[0, 0] + loc[0, 0])

        pred = pred.astype(np.float64) * bound_span + bound_min
        if not np.isfinite(pred).all():
            raise ValueError('Nonfinite prediction')
        return pred


# ── Singleton ──────────────────────────────────────────────────

_engine: Optional[RegimeFlowEngineONNX] = None


def init_engine(
    checkpoint_path: str,
    backbone_onnx: Optional[str] = None,
    cond_encoder_onnx: Optional[str] = None,
    denoise_steps: int = 4,
) -> RegimeFlowEngineONNX:
    global _engine
    if _engine is not None:
        return _engine

    if backbone_onnx is None:
        backbone_onnx = str(Path(checkpoint_path).parent / "backbone.onnx")
    if cond_encoder_onnx is None:
        cond_encoder_onnx = str(Path(checkpoint_path).parent / "cond_encoder.onnx")

    _engine = RegimeFlowEngineONNX(
        backbone_onnx=backbone_onnx,
        cond_encoder_onnx=cond_encoder_onnx,
        checkpoint_path=checkpoint_path,
        denoise_steps=denoise_steps,
    )
    return _engine
