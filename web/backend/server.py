"""
RegimeFlow Backend API Server (Render 轻量版)
==============================
无需 GPU，使用轻量级合成预测。适合 Render 免费套餐部署。

Usage:
    python server.py
    # or: uvicorn server:app --host 0.0.0.0 --port 8000
"""

import os
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 默认禁用 Chronos（Render 免费版无 GPU，内存有限）
LOAD_CHRONOS = os.environ.get("LOAD_CHRONOS", "false").lower() != "false"

# ---------------------------------------------------------------------------
# Global model handles
# ---------------------------------------------------------------------------
chronos_pipeline: Optional[object] = None
model_info: dict = {"chronos_loaded": False, "device": "cpu", "error": None}


# ---------------------------------------------------------------------------
# Lifespan: 仅当 LOAD_CHRONOS=true 时加载模型
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load Chronos-Bolt on startup only if enabled."""
    global chronos_pipeline, model_info

    if LOAD_CHRONOS:
        logger.info("Loading Chronos-Bolt-Base from HuggingFace …")
        try:
            import torch
            from chronos import BaseChronosPipeline

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_info["device"] = device

            chronos_pipeline = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-bolt-base",
                device_map=device,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            )
            model_info["chronos_loaded"] = True
            logger.info(f"Chronos-Bolt-Base loaded successfully on {device}")
        except Exception as exc:
            model_info["error"] = str(exc)
            logger.error(f"Failed to load Chronos model: {exc}")
            logger.warning("Server will start, but /api/predict will use fallback.")
    else:
        model_info["device"] = "cpu"
        model_info["error"] = "Chronos disabled (low-memory mode)"
        logger.info("Chronos disabled — using lightweight synthetic forecast for demo")

    yield

    chronos_pipeline = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RegimeFlow Prediction API",
    description="Backend API for biological trajectory forecasting (demo mode).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 托管静态前端文件
# ---------------------------------------------------------------------------
static_dir = Path(__file__).parent.parent  # serve from web/ root
app.mount("/css", StaticFiles(directory=str(static_dir / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(static_dir / "js")), name="js")
app.mount("/data", StaticFiles(directory=str(static_dir / "data")), name="data")


@app.get("/")
async def serve_frontend():
    return FileResponse(str(static_dir / "index.html"))


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    context: list[float] = Field(
        ..., description="1-D array of past observation values (e.g. 96 time steps)"
    )
    prediction_length: int = Field(
        default=256, ge=1, le=512, description="Number of future steps to predict"
    )


class PredictResponse(BaseModel):
    predictions: list[float] = Field(..., description="Mean forecast (prediction_length,)")
    median: list[float] = Field(..., description="Median forecast")
    lower: list[float] = Field(..., description="10th percentile (lower bound)")
    upper: list[float] = Field(..., description="90th percentile (upper bound)")
    samples: int = Field(..., description="Number of samples used for quantiles")
    model: str = Field(default="chronos-bolt-base")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class PredictMultiRequest(BaseModel):
    """Predict multiple species at once (same context length, batched)."""
    contexts: list[list[float]] = Field(
        ..., description="List of 1-D context arrays, one per species"
    )
    prediction_length: int = Field(default=256, ge=1, le=512)


class PredictMultiResponse(BaseModel):
    results: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    chronos_loaded: bool
    device: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper: Chronos inference
# ---------------------------------------------------------------------------
def run_chronos_predict(
    context: list[float], prediction_length: int
) -> PredictResponse:
    """Run Chronos pipeline on a single 1-D context."""
    import torch

    if chronos_pipeline is None:
        raise RuntimeError("Chronos model is not loaded. Check server logs.")

    ctx_tensor = torch.tensor(context, dtype=torch.float32)

    t0 = time.perf_counter()
    with torch.no_grad():
        # Chronos returns [batch=1, num_quantiles=9, prediction_length]
        forecast = chronos_pipeline.predict(
            ctx_tensor,
            prediction_length=prediction_length,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # forecast shape: [1, 9, prediction_length] — 9 quantile levels
    arr = forecast.cpu().numpy().squeeze(0)  # -> [9, prediction_length]
    samples = arr.shape[0]

    return PredictResponse(
        predictions=arr.mean(axis=0).tolist(),
        median=arr[4].tolist(),  # 5th of 9 quantiles = median
        lower=arr[0].tolist(),   # lowest quantile (~10th percentile)
        upper=arr[-1].tolist(),  # highest quantile (~90th percentile)
        samples=samples,
        model="chronos-bolt-base",
        inference_time_ms=round(elapsed_ms, 1),
    )


# ---------------------------------------------------------------------------
# Fallback: BLR-like synthetic prior (when Chronos is unavailable)
# ---------------------------------------------------------------------------
def _synthetic_forecast(
    context: list[float], prediction_length: int
) -> PredictResponse:
    """
    Lightweight fallback forecast using simple trend + noise extrapolation.
    Used when Chronos model fails to load.  Not scientifically meaningful,
    but keeps the UI functional for demo purposes.
    """
    ctx = np.array(context, dtype=np.float64)
    n = len(ctx)

    # Simple linear extrapolation with damped oscillation
    t_ctx = np.arange(n)
    t_pred = np.arange(n, n + prediction_length)

    # Linear trend from last 20% of context
    tail = max(4, n // 5)
    slope = (ctx[-1] - ctx[-tail]) / tail
    intercept = ctx[-1] - slope * (n - 1)

    trend = intercept + slope * t_pred

    # Damped sinusoidal residual
    residuals = ctx - (intercept + slope * t_ctx)
    amplitude = np.std(residuals) * 0.5
    oscillation = amplitude * np.sin(2 * np.pi * 0.05 * t_pred) * np.exp(
        -0.005 * (t_pred - n)
    )

    noise = np.random.default_rng(42).normal(0, amplitude * 0.3, prediction_length)

    mean = (trend + oscillation + noise).tolist()

    return PredictResponse(
        predictions=mean,
        median=mean,
        lower=[v - amplitude * 0.5 for v in mean],
        upper=[v + amplitude * 0.5 for v in mean],
        samples=1,
        model="fallback-synthetic",
        inference_time_ms=0.0,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if model_info["chronos_loaded"] else "degraded",
        chronos_loaded=model_info["chronos_loaded"],
        device=model_info["device"],
        error=model_info.get("error"),
    )


@app.post("/api/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Predict future trajectory from a single-species context."""
    if len(req.context) < 4:
        raise HTTPException(400, "Context must have at least 4 observations.")

    try:
        if chronos_pipeline is not None:
            return run_chronos_predict(req.context, req.prediction_length)
        else:
            logger.warning("Chronos not available — using fallback.")
            return _synthetic_forecast(req.context, req.prediction_length)
    except Exception as exc:
        logger.error(f"Prediction failed: {exc}")
        raise HTTPException(500, f"Prediction error: {exc}")


@app.post("/api/predict/multi", response_model=PredictMultiResponse)
async def predict_multi(req: PredictMultiRequest):
    """Predict for multiple species (e.g. all species in one biological system)."""
    results = []
    for ctx in req.contexts:
        try:
            if chronos_pipeline is not None:
                r = run_chronos_predict(ctx, req.prediction_length)
            else:
                r = _synthetic_forecast(ctx, req.prediction_length)
            results.append(r)
        except Exception as exc:
            raise HTTPException(500, f"Prediction error for species: {exc}")
    return PredictMultiResponse(results=results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
