# Inference repair validation

Recorded on 2026-10-11 against baseline commit
`179c04300ecd2a2349c464a5cc4bcee64f923239`.

The repaired demo keeps four Euler steps, the existing checkpoint/EMA weights,
and the existing website controls. The checkpoint's 32-step setting is retained
for offline comparison; four-step results are not a reproduction of the paper's
evaluation protocol.

## Checks completed

- 16 Python tests passed across the scientific and API environments. Scientific
  discovery ran 12 tests and skipped the API module because FastAPI was absent;
  the four API tests passed separately in the API environment.
- Nine JavaScript tests passed for preprocessing, species conditions, backend
  result validation, stale requests, and predictions before browser warm-up.
- Six regimes passed an independent upstream Mamba CPU semantic comparison:
  maximum ONNX velocity error `2.02e-6`, maximum four-step prediction error
  `5.97e-7`. Upstream sources and their license are in `tests/reference/`.
- Fresh backbone and condition encoder exports passed actual ONNX Runtime
  comparisons, confirming that re-export does not restore the old slice layout.
- The vendored ONNX Runtime Web 1.20.1 WASM build ran under Node on six real
  species fixtures. Maximum final prediction error divided by species range
  was `1.61e-6`. This did not use native onnxruntime-node inference.
- Fixed-noise diagnostics covered 15 SysBio-Traj systems / 34 species. Median
  species nMAE changed from `0.388584` to `0.054475`. This is a diagnostic result,
  not a confirmed held-out test score.
- Both deployed backbone graphs passed ONNX structural checks. All legacy
  initializer arrays and the external backbone weight file were unchanged.

## Pending checks and limitations

Actual browser WASM and production page interactions have not passed acceptance:
the browser automation tool failed during startup with a filesystem glob
configuration error. Node WASM and mocked frontend tests do not replace a real
browser run. Check system selection, species switching, multi-species display,
custom input, and failure handling before release.

The checkpoint's historical training configuration, split seed, and the cached
dataset's HF revision have not been linked conclusively. The frozen samples
must not be called the official OOD test split. The dataset proxies support a
shared `SYSBIO_DATASET_REVISION`; the default remains `main` until a trusted
revision is established.

Missing conditions deliberately use explicit `(pattern, period)=(0,0)`, which
differs from the training null embedding. Missing oscillation periods can impair
oscillatory forecasts. Single sampled trajectories can still be inaccurate or
noisy; no smoothing or output nonnegativity constraint was added.

## Reproduction

Run these commands from the repository root after installing
`scripts/requirements_validation.txt`:

```bash
python -m unittest discover -s tests -p 'test_*.py'
node --test tests/test_web_preprocessing.cjs tests/test_prediction_contracts.cjs
git diff --check
```

The full frozen diagnostics bundle is kept in the shared project directory
`Materials/Validation/InferenceParity/`, **outside this Git checkout**. It is not
included in a clone. Obtain that bundle from the project maintainer/shared Drive
and place it at `../Materials/Validation/InferenceParity/` relative to the clone.
It contains the report, baseline predictions, exact noise arrays, sample/source
manifest, CSV/condition snapshots, legacy Python source/config, stage metrics,
browser fixtures, and test logs. Do not substitute freshly sampled noise or
overwrite the frozen baseline.

With that bundle in place:

```bash
python scripts/export_onnx_v2.py --output-root /private/tmp/regimeflow-reexport
python scripts/validate_inference.py --only reference --reexport /private/tmp/regimeflow-reexport
python scripts/validate_stages.py
python scripts/validate_inference.py --only real
node scripts/cross-check/check.cjs
node scripts/plot_inference_comparison.cjs
```

`validate_inference.py` optionally checks initializer identity against the old
model at `/private/tmp/regimeflow-before/backbone.onnx` and its `.data` file.
Restore those from the baseline commit to repeat that structural comparison;
without them the script skips that comparison. Numerical checks still run.

For real browser numerical acceptance, serve the parent project directory with
`python -m http.server 8765`, then open
`http://localhost:8765/RegimeFlow-web/scripts/browser_validation.html`.
Check the displayed output and `window.validationResult`, then separately test
the production UI. The standalone harness is not loaded by the production page.

## Deployed asset hashes

| Asset | SHA-256 |
|---|---|
| `ckpt/RegimeFlow/backbone.onnx` | `132b37d153be347a522bd264abed322e74aa8214f529302c421014a70c39f557` |
| `ckpt/RegimeFlow/backbone.onnx.data` | `5d1023c4bc14d233f6fb32770466f8c0e96895f2850f37550123a33e80502ea0` |
| `web/models/backbone.onnx` | `d1122bbf2062313d32939cd4ad58a1c12a83393771e0fdc86a237570ebab6bcf` |
| `web/models/inference-config.json` | `f3ea11ddbf9fd0772c4cacdf40052338143b74edcfb2827be2306134490445f6` |

Both backbones carry `regimeflow.mamba_projection_layout=x,z;dt,B,C`.
