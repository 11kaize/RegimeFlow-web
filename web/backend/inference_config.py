"""Shared inference configuration and raw-unit preprocessing (no torch)."""
import json
import logging
from pathlib import Path

import numpy as np

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'models' / 'inference-config.json'


def load_config(path=CONFIG_PATH):
    config = json.loads(Path(path).read_text())
    positive = ('context_length', 'prediction_length', 'cond_dim', 'denoise_steps',
                'alpha', 'beta', 'min_scale', 'min_variance')
    for key in positive:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid inference parameter: {key}')
    for key in ('context_length', 'prediction_length', 'cond_dim', 'denoise_steps', 'num_harmonics', 'poly_degree'):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f'Invalid integer inference parameter: {key}')
    for key in ('noise_scale', 'flow_sigma'):
        if not np.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f'Invalid inference parameter: {key}')
    scales = config['saturation_scales']
    if not isinstance(scales, list) or not scales or any(type(x) not in (int, float) or not np.isfinite(x) or x <= 0 for x in scales):
        raise ValueError('saturation_scales must be a positive numeric array')
    if config['transform'] != 'minmax':
        raise ValueError('Unsupported data transform')
    return config


def resolve_condition(pattern=0, period=0):
    valid_pattern = type(pattern) in (int, float) and np.isfinite(pattern) and int(pattern) == pattern and 0 <= pattern <= 5
    if not valid_pattern:
        logging.getLogger(__name__).warning('Default condition (0,0): invalid trajectory type %r', pattern)
        return 0, 0.0
    valid_period = type(period) in (int, float) and np.isfinite(period) and period >= 0
    if int(pattern) == 3 and (not valid_period or period <= 0):
        logging.getLogger(__name__).warning('Default condition (0,0): invalid oscillation period %r', period)
        return 0, 0.0
    return int(pattern), float(period) if valid_period else 0.0


def transform_context(context, bounds=None):
    """Dataset bounds cover the filtered full CSV; custom bounds use history only.

    A supplied dataset bound triggers the reference nonnegative clipping. Custom
    data may be signed. Zero-range inputs stay constant after inverse transform.
    """
    values = np.asarray(context, dtype=np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError('Context must be nonempty and finite')
    if bounds is None:
        lo, hi = float(values.min()), float(values.max())
    else:
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[1] < bounds[0]:
            raise ValueError('Invalid minmax bounds')
        lo, hi = map(float, bounds)
        config = load_config()
        values = np.clip(values.astype(np.float32), *config['dataset_clip']).astype(np.float64)
        if np.any(values < lo) or np.any(values > hi):
            raise ValueError('Context outside supplied dataset bounds')
    span = hi - lo
    return ((values - lo) / span if span else np.zeros_like(values)).astype(np.float32), lo, span
