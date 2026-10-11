"""
RegimeFlow Backend API Server
==============================
Serves biological trajectory predictions using trained RegimeFlow model.

Prediction endpoints return real RegimeFlow results or explicit errors.
Chronos may optionally load for development, but is never substituted for
a RegimeFlow prediction.

Usage:
    python server.py
    # or: uvicorn server:app --host 0.0.0.0 --port 8000

Environment variables:
    REGIMEFLOW_CKPT    — path to seed53_best.ckpt (default: auto-detect)
    REGIMEFLOW_STEPS   — denoising steps, 4 by default
    LOAD_CHRONOS       — set to "true" to load Chronos-Bolt (heavy)
    PORT               — server port (default: 8000)
"""

from __future__ import annotations

import os
import sys
import time
import threading
import logging
import json
import mimetypes
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

# Windows' mimetypes registry maps .mjs → text/plain, which breaks ES module
# dynamic import() (onnxruntime-web loads vendor/ort-wasm-simd-threaded.mjs).
# Register the correct JS MIME types before StaticFiles guesses them.
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("text/javascript", ".js")

# Add project root for model imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────
REGIMEFLOW_CKPT = os.environ.get("REGIMEFLOW_CKPT", "")
REGIMEFLOW_STEPS = int(os.environ.get("REGIMEFLOW_STEPS", "4"))
LOAD_CHRONOS = os.environ.get("LOAD_CHRONOS", "false").lower() != "false"
LOAD_REGIMEFLOW = os.environ.get("LOAD_REGIMEFLOW", "true").lower() != "false"

# ── Auto-detect / download checkpoint ────────────────────────────
def _ensure_checkpoint():
    """Find or download the RegimeFlow checkpoint."""
    global REGIMEFLOW_CKPT

    if REGIMEFLOW_CKPT and Path(REGIMEFLOW_CKPT).exists():
        return

    # Check if it's a URL → download
    if REGIMEFLOW_CKPT and (REGIMEFLOW_CKPT.startswith("http://") or
                             REGIMEFLOW_CKPT.startswith("https://")):
        dest = _PROJECT_ROOT / "ckpt" / "RegimeFlow" / "seed53_best.ckpt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            logger.info(f"Downloading checkpoint from {REGIMEFLOW_CKPT} ...")
            try:
                import urllib.request
                urllib.request.urlretrieve(REGIMEFLOW_CKPT, str(dest))
                logger.info(f"Downloaded to {dest}")
            except Exception as e:
                logger.warning(f"Download failed: {e}")
                return
        REGIMEFLOW_CKPT = str(dest)
        return

    # Search local filesystem — try multiple base paths (Render vs local)
    cwd = Path.cwd()
    search_bases = [_PROJECT_ROOT]
    if cwd != _PROJECT_ROOT:
        search_bases.append(cwd)

    candidates = []
    for base in search_bases:
        candidates.extend([
            base / "ckpt" / "RegimeFlow" / "seed53_best.ckpt",
            base / "seed53_best.ckpt",
            base / "web" / "backend" / "ckpt" / "RegimeFlow" / "seed53_best.ckpt",
        ])

    logger.info(f"Searching checkpoint: _PROJECT_ROOT={_PROJECT_ROOT}, cwd={cwd}")
    for c in candidates:
        logger.info(f"  Checking: {c} (exists={c.exists()})")
        if c.exists():
            REGIMEFLOW_CKPT = str(c)
            logger.info(f"Found checkpoint at {REGIMEFLOW_CKPT}")
            return

    logger.warning(f"Checkpoint not found in any of {len(candidates)} candidate paths")

if LOAD_REGIMEFLOW:
    _ensure_checkpoint()


# ── Global handles ─────────────────────────────────────────────
regimeflow_engine: Optional[object] = None
chronos_pipeline: Optional[object] = None
model_info: dict = {
    "regimeflow_loaded": False,
    "chronos_loaded": False,
    "device": "cpu",
    "error": None,
}


# ── Lifespan ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Models are loaded lazily on the first /api/predict (see
    _load_regimeflow_engine) so /api/health and /api/csv answer instantly on a
    cold start instead of waiting for the ~2s ONNX load at startup."""
    yield
    global regimeflow_engine, chronos_pipeline
    regimeflow_engine = None
    chronos_pipeline = None


_engine_lock = threading.Lock()


def _load_regimeflow_engine():
    """Return the RegimeFlow engine, loading it once on first call.

    Thread-safe and idempotent. Kept out of the startup lifespan so a free-tier
    cold start only pays the model-load cost when a prediction is actually
    requested — the CSV proxy and /api/health stay fast.
    """
    global regimeflow_engine, model_info
    if regimeflow_engine is not None:
        return regimeflow_engine
    with _engine_lock:
        if regimeflow_engine is not None:
            return regimeflow_engine
        if not (LOAD_REGIMEFLOW and REGIMEFLOW_CKPT):
            logger.info("RegimeFlow disabled (LOAD_REGIMEFLOW=false or no checkpoint found)")
            return None
        logger.info(f"Loading RegimeFlow ONNX from {REGIMEFLOW_CKPT} ...")
        try:
            from web.backend.engine_onnx import init_engine as init_engine_onnx
            regimeflow_engine = init_engine_onnx(REGIMEFLOW_CKPT, denoise_steps=REGIMEFLOW_STEPS)
            model_info["regimeflow_loaded"] = True
            model_info["device"] = "cpu"
            logger.info("RegimeFlow ONNX loaded successfully")
        except Exception as exc:
            model_info["error"] = f"RegimeFlow: {exc}"
            logger.error(f"Failed to load RegimeFlow ONNX: {exc}")
        return regimeflow_engine


# ── FastAPI app ────────────────────────────────────────────────
app = FastAPI(
    title="RegimeFlow Prediction API",
    description="Backend API for biological trajectory forecasting.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ───────────────────────────────────────────────
static_dir = Path(__file__).parent.parent
app.mount("/css", StaticFiles(directory=str(static_dir / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(static_dir / "js")), name="js")
app.mount("/data", StaticFiles(directory=str(static_dir / "data")), name="data")
app.mount("/models", StaticFiles(directory=str(static_dir / "models")), name="models")
app.mount("/vendor", StaticFiles(directory=str(static_dir / "vendor")), name="vendor")


@app.get("/")
async def serve_frontend():
    return FileResponse(str(static_dir / "index.html"))


# ── Schemas ────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    context: list[float] = Field(
        ..., description="1-D array of past observation values (at least 96 steps)"
    )
    prediction_length: int = Field(
        default=256, ge=1, le=512, description="Number of future steps to predict"
    )
    # RegimeFlow-specific conditions (ignored by other backends)
    traj_pattern: int = Field(
        default=0, ge=0, le=5,
        description="Trajectory regime: 0=DirectStable, 1=IncStable, 2=DecStable, "
                    "3=Oscillation, 4=Increasing, 5=Decreasing"
    )
    period: float = Field(
        default=0.0,
        description="Oscillation period in sample intervals (only used for type 3)"
    )
    bounds: Optional[list[float]] = Field(default=None, min_length=2, max_length=2)


class PredictResponse(BaseModel):
    predictions: list[float] = Field(..., description="Mean forecast")
    median: list[float] = Field(..., description="Median forecast (equals mean when only one sample is drawn)")
    lower: list[float] = Field(..., description="Lower edge of the uncertainty band")
    upper: list[float] = Field(..., description="Upper edge of the uncertainty band")
    samples: int = Field(..., description="Number of samples drawn (1 = single trajectory, band is synthetic)")
    model: str = Field(default="regimeflow")
    inference_time_ms: float = Field(..., description="Inference time in ms")


class PredictMultiRequest(BaseModel):
    contexts: list[list[float]] = Field(..., description="List of context arrays")
    prediction_length: int = Field(default=256, ge=1, le=512)
    traj_pattern: int = Field(default=0, ge=0, le=5)
    period: float = Field(default=0.0)
    # Optional per-species values; existing scalar fields remain compatible.
    traj_patterns: Optional[list[int]] = None
    periods: Optional[list[float]] = None
    bounds: Optional[list[Optional[list[float]]]] = None


class PredictMultiResponse(BaseModel):
    results: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    regimeflow_loaded: bool
    chronos_loaded: bool
    device: str
    denoise_steps: int = 0
    error: Optional[str] = None


# ── Prediction backends ────────────────────────────────────────

def _predict_regimeflow(
    context: list[float],
    prediction_length: int,
    traj_pattern: int = 0,
    period: float = 0.0,
    bounds=None,
) -> PredictResponse:
    """RegimeFlow inference (CPU, ~400ms with 4 denoise steps)."""
    if regimeflow_engine is None:
        raise RuntimeError("RegimeFlow engine not loaded")

    t0 = time.perf_counter()
    pred = regimeflow_engine.predict(
        context=context,
        traj_pattern=traj_pattern,
        period=period, bounds=bounds,
    )
    pred = pred[:prediction_length]
    elapsed = (time.perf_counter() - t0) * 1000

    # RegimeFlow returns a single sampled trajectory (one noise draw).
    # The band below is NOT a statistical prediction interval — it is a fixed
    # ±20% reference band scaled by the signal magnitude.
    mean = pred.tolist()
    std_est = float(np.abs(pred).mean() * 0.1)  # 10% of mean magnitude → band = ±20%

    return PredictResponse(
        predictions=mean,
        median=mean,
        lower=[v - 2 * std_est for v in mean],
        upper=[v + 2 * std_est for v in mean],
        samples=1,
        model="regimeflow",
        inference_time_ms=round(elapsed, 1),
    )


def _predict_chronos(
    context: list[float],
    prediction_length: int,
) -> PredictResponse:
    """Chronos-Bolt inference."""
    import torch as _torch
    if chronos_pipeline is None:
        raise RuntimeError("Chronos model not loaded")

    ctx_tensor = _torch.tensor(context, dtype=_torch.float32)
    t0 = time.perf_counter()
    with _torch.no_grad():
        forecast = chronos_pipeline.predict(ctx_tensor, prediction_length=prediction_length)
    elapsed = (time.perf_counter() - t0) * 1000

    arr = forecast.cpu().numpy().squeeze(0)
    samples = arr.shape[0]

    return PredictResponse(
        predictions=arr.mean(axis=0).tolist(),
        median=arr[4].tolist(),
        lower=arr[0].tolist(),
        upper=arr[-1].tolist(),
        samples=samples,
        model="chronos-bolt-base",
        inference_time_ms=round(elapsed, 1),
    )



# ── Routes ─────────────────────────────────────────────────────

@app.get("/api/debug/ckpt")
async def debug_ckpt():
    """Diagnostic endpoint: show filesystem state for checkpoint discovery."""
    import os as _os
    cwd = str(Path.cwd())
    proj = str(_PROJECT_ROOT)
    candidates = {}
    for base_name, base in [("_PROJECT_ROOT", _PROJECT_ROOT), ("cwd", Path.cwd())]:
        for rel in ["ckpt/RegimeFlow/seed53_best.ckpt", "seed53_best.ckpt",
                     "web/backend/ckpt/RegimeFlow/seed53_best.ckpt"]:
            p = base / rel
            candidates[f"{base_name}/{rel}"] = {"path": str(p), "exists": p.exists()}
    # List repo root top-level
    root_contents = sorted(os.listdir(str(_PROJECT_ROOT))) if _PROJECT_ROOT.exists() else []
    return {
        "cwd": cwd,
        "_PROJECT_ROOT": proj,
        "REGIMEFLOW_CKPT": REGIMEFLOW_CKPT or "(empty)",
        "LOAD_REGIMEFLOW": LOAD_REGIMEFLOW,
        "candidates": candidates,
        "root_top_level": root_contents[:50],
    }


@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if model_info["regimeflow_loaded"] else "degraded",
        regimeflow_loaded=model_info["regimeflow_loaded"],
        chronos_loaded=model_info["chronos_loaded"],
        device=model_info["device"],
        denoise_steps=REGIMEFLOW_STEPS if model_info["regimeflow_loaded"] else 0,
        error=model_info.get("error"),
    )


def _require_regimeflow():
    _load_regimeflow_engine()
    if regimeflow_engine is None:
        logger.error('Real RegimeFlow inference unavailable: %s', model_info.get('error'))
        raise HTTPException(503, 'RegimeFlow inference unavailable')


def _batch_arguments(req):
    count = len(req.contexts)
    if any(len(context) < 4 for context in req.contexts):
        raise HTTPException(400, 'Each context must have at least 4 observations')
    for key in ('traj_patterns', 'periods', 'bounds'):
        values = getattr(req, key)
        if values is not None and len(values) != count:
            raise HTTPException(400, f'{key} must match the number of contexts')
    return [dict(context=ctx, prediction_length=req.prediction_length,
                 traj_pattern=req.traj_patterns[i] if req.traj_patterns is not None else req.traj_pattern,
                 period=req.periods[i] if req.periods is not None else req.period,
                 bounds=req.bounds[i] if req.bounds is not None else None)
            for i, ctx in enumerate(req.contexts)]


@app.post('/api/predict', response_model=PredictResponse)
async def predict(req: PredictRequest):
    if len(req.context) < 4:
        raise HTTPException(400, 'Context must have at least 4 observations')
    _require_regimeflow()
    try:
        return _predict_regimeflow(req.context, req.prediction_length,
                                  traj_pattern=req.traj_pattern, period=req.period, bounds=req.bounds)
    except ValueError as exc:
        logger.warning('Invalid prediction input: %s', exc)
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception('RegimeFlow prediction failed')
        raise HTTPException(500, 'RegimeFlow prediction failed') from exc


@app.post('/api/predict/multi', response_model=PredictMultiResponse)
async def predict_multi(req: PredictMultiRequest):
    arguments = _batch_arguments(req)
    _require_regimeflow()
    try:
        return PredictMultiResponse(results=[_predict_regimeflow(**args) for args in arguments])
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception('RegimeFlow batch prediction failed')
        raise HTTPException(500, 'RegimeFlow batch prediction failed') from exc


@app.post('/api/predict/multi/stream')
def predict_multi_stream(req: PredictMultiRequest):
    arguments = _batch_arguments(req)
    _require_regimeflow()

    def results():
        for i, args in enumerate(arguments):
            try:
                prediction = _predict_regimeflow(**args)
                yield json.dumps({'index': i, 'result': prediction.model_dump()}, allow_nan=False) + '\n'
            except Exception:
                logger.exception('RegimeFlow stream prediction failed for species %s', i)
                yield json.dumps({'index': i, 'error': 'RegimeFlow prediction failed'}) + '\n'

    return StreamingResponse(results(), media_type='application/x-ndjson')


# Both proxies use the same dataset revision. No condition UI is added.
def _fetch_dataset_file(model_id, model_name, suffix):
    import urllib.request
    import urllib.parse
    revision = os.environ.get('SYSBIO_DATASET_REVISION', 'main')
    safe_id = urllib.parse.quote(model_id, safe='')
    safe_name = urllib.parse.quote(model_name, safe='')
    url = ('https://huggingface.co/datasets/HengRao/SysBio-Traj/resolve/'
           + urllib.parse.quote(revision, safe='') + '/Data/' + safe_id + '/' + safe_name + suffix)
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'RegimeFlow-web/2.0'})
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()
    except Exception as exc:
        logger.warning('Dataset fetch failed for %s/%s%s: %s', model_id, model_name, suffix, exc)
        raise HTTPException(404, f'Dataset file unavailable for {model_id}/{model_name}{suffix}') from exc


@app.get('/api/csv/{model_id}/{model_name}')
def proxy_csv(model_id: str, model_name: str):
    return Response(content=_fetch_dataset_file(model_id, model_name, '.csv'), media_type='text/csv')


@app.get('/api/conditions/{model_id}/{model_name}')
def proxy_conditions(model_id: str, model_name: str):
    data = _fetch_dataset_file(model_id, model_name, '_conditions.json')
    try:
        records = json.loads(data)
        if not isinstance(records, dict):
            raise ValueError('Expected an object keyed by species')
        encoded = json.dumps(records, allow_nan=False)
    except (ValueError, UnicodeDecodeError) as exc:
        logger.warning('Invalid conditions JSON for %s/%s: %s', model_id, model_name, exc)
        raise HTTPException(502, 'Invalid conditions JSON') from exc
    return Response(content=encoded, media_type='application/json')


# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting RegimeFlow server on port {port}")
    logger.info(f"  RegimeFlow: {model_info['regimeflow_loaded']}")
    logger.info(f"  Chronos:    {model_info['chronos_loaded']}")
    logger.info(f"  Checkpoint: {REGIMEFLOW_CKPT or 'N/A'}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
