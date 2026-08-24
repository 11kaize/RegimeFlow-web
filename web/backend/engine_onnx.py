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

import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np

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
        num_obs = np.sum(observed, axis=-1, keepdims=True)
        num_obs = np.maximum(num_obs, 1)
        loc = np.sum(x * observed, axis=-1, keepdims=True) / num_obs
        variance = np.sum(((x - loc) ** 2) * observed, axis=-1, keepdims=True) / num_obs
        scale = np.sqrt(np.maximum(variance, self.minimum_scale))
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
        if config is None:
            config = {}
        self.alpha = config.get('alpha', 1.0)
        self.beta = config.get('beta', 20.0)
        self.noise_scale = config.get('noise_scale', 0.1)
        self.saturation_scales = config.get('saturation_scales', (0.2, 0.5, 1.0, 2.0, 5.0))
        self.poly_degree = config.get('poly_degree', 2)
        self.num_harmonics = config.get('num_harmonics', 4)
        self.min_variance = config.get('min_variance', 1e-6)

    # ── Basis functions ──

    @staticmethod
    def _constant_basis(times, **_kw):
        return np.ones((len(times), 1))

    @staticmethod
    def _linear_basis(times, t_range, **_kw):
        return np.linspace(-1, 1, len(times)).reshape(-1, 1)

    @staticmethod
    def _poly_basis(times, t_range, degree=2, **_kw):
        t_norm = np.linspace(-1, 1, len(times))
        return np.column_stack([t_norm ** d for d in range(degree + 1)])

    @staticmethod
    def _saturation_basis(times, t_range, scales=(0.2, 0.5, 1.0, 2.0, 5.0), **_kw):
        t_norm = np.linspace(-1, 1, len(times))
        bases = [np.ones(len(times))]
        for lam in scales:
            bases.append(1 - np.exp(-lam * (t_norm + 1) / 2))
        return np.column_stack(bases)

    @staticmethod
    def _fourier_basis(times, t_range, num_harmonics=4, omega=1.0, **_kw):
        t_rad = omega * times.reshape(-1, 1)
        comps = [np.ones(len(times))]
        for k in range(1, num_harmonics + 1):
            comps.append(np.sin(k * t_rad).flatten())
            comps.append(np.cos(k * t_rad).flatten())
        return np.column_stack(comps)

    # ── BLR Solver ──

    @staticmethod
    def _blr_solve(Phi_past, y_past, Phi_future, alpha, beta, min_var=1e-6):
        """Closed-form BLR: posterior → predictive mean & variance."""
        K = Phi_past.shape[-1]
        # Posterior precision: S^-1 = alpha*I + beta * Phi^T * Phi
        S_inv = alpha * np.eye(K) + beta * Phi_past.T @ Phi_past

        try:
            L = np.linalg.cholesky(S_inv)
            # S = L^-T @ L^-1
            S = np.linalg.inv(L.T) @ np.linalg.inv(L)
        except np.linalg.LinAlgError:
            S = np.linalg.pinv(S_inv)

        # Posterior mean: m = beta * S @ Phi^T @ y
        m = beta * S @ Phi_past.T @ y_past.reshape(-1, 1)

        # Predictive mean: mu = Phi_future @ m
        mu = (Phi_future @ m).flatten()

        # Predictive variance
        var_diag = 1.0 / beta + np.diag(Phi_future @ S @ Phi_future.T)
        var = np.maximum(var_diag, min_var)

        return mu, var

    # ── Generate prior ──

    def generate(self, past_times, past_values, future_times, pattern_id=0, period=0.0):
        """
        Generate BLR prior x0 for flow matching.

        NOTE: The prior in RegimeFlow always uses pattern=0 (constant basis),
        regardless of the actual trajectory condition. The condition encoder
        handles the pattern-specific shaping through the backbone.

        Args:
            past_times: (N,) past time indices
            past_values: (N,) ALREADY SCALED past observation values (z-scored)
            future_times: (F,) future time indices
            pattern_id: always 0 for RegimeFlow (constant basis)
            period: always 0 for RegimeFlow

        Returns:
            x0: (F,) prior sample
        """
        # RegimeFlow always uses constant basis pattern for the prior
        pattern_id = 0
        t_all = np.concatenate([past_times, future_times])

        # Constant basis
        Phi_all = np.ones((len(t_all), 1))
        Phi_past = Phi_all[:len(past_times)]
        Phi_future = Phi_all[len(past_times):]

        # Values are already z-scored → no additional normalization needed
        mu, var = self._blr_solve(
            Phi_past, past_values, Phi_future,
            self.alpha, self.beta, self.min_variance
        )

        # Sample: x0 = mu + epsilon * noise_scale
        std = np.sqrt(np.maximum(var, self.min_variance))
        epsilon = np.random.randn(len(future_times))
        x0 = mu + epsilon * self.noise_scale

        return x0.astype(np.float32)


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

        # Load hyperparameters from JSON (torch-free!)
        import json
        hp_path = str(Path(checkpoint_path).parent / "hyperparams.json")
        with open(hp_path) as f:
            hp = json.load(f)

        self.context_length = hp['context_length']
        self.prediction_length = hp['prediction_length']
        self.use_condition = hp.get('use_condition', True)
        self.cond_dim = hp.get('cond_dim', 128)

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
        logger.info(f"Backbone loaded: inputs={[i.name for i in self.backbone_session.get_inputs()]}")

        logger.info(f"Loading ONNX condition encoder from {cond_encoder_onnx} ...")
        self.cond_encoder_session = ort.InferenceSession(
            cond_encoder_onnx, providers=['CPUExecutionProvider']
        )
        logger.info(f"Condition encoder loaded")

        # ── BLR Prior (NumPy) ──
        prior_params = dict(hp['prior_params'])
        prior_params.pop('name', None)
        self.prior = BLRPriorNP(prior_params)

        # ── Scaler (NumPy) ──
        self.scaler = StdScalerNP(minimum_scale=1.0)

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
        ctx = np.asarray(context, dtype=np.float32).flatten()
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
            # RegimeFlow always uses pattern=0 for BLR prior
        )
        x0_fut = x0_fut.reshape(1, self.prediction_length, 1)

        # ── Build x0: context + future prior ──
        scaled_context = past_target / scale
        x0 = np.concatenate([scaled_context, x0_fut], axis=1)  # (1, L_total, 1)

        # Add noise (sigma_max = 1 for flow matching)
        x0 = x0 + np.random.randn(*x0.shape).astype(np.float32)

        # ── Flow Matching ODE (Euler steps) ──
        dt_ode = 1.0 / self.denoise_steps
        xt = x0

        for step in range(self.denoise_steps):
            t_step = step * dt_ode
            vt = self._backbone_velocity(t_step, xt, cond_emb)
            xt = xt + dt_ode * vt

        # ── Denormalize and extract future ──
        pred = (xt[0, self.context_length:, 0] * scale[0, 0] + loc[0, 0])

        return pred.astype(np.float64)


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
