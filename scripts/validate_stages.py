"""Reproduce the first three stages using frozen legacy source/config and noise.

The original baseline predictions are read, never recomputed/overwritten.
Run this before validate_inference.py --only real.
"""
import sys,json,csv,time,importlib.util
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];out=root.parent/'Materials/Validation/InferenceParity'; baseline=json.loads((out/'baseline_predictions.json').read_text())['samples']; noise=json.loads((out/'fixed_noise_arrays.json').read_text())
spec=importlib.util.spec_from_file_location('legacy_engine',str(out/'sources/legacy/engine_onnx.py')); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
e=m.RegimeFlowEngineONNX(str(root/'ckpt/RegimeFlow/backbone.onnx'),str(root/'ckpt/RegimeFlow/cond_encoder.onnx'),str(out/'sources/legacy/seed53_best.ckpt'))
rows=[]
for stage in ['baseline','mamba','preprocessing']:
 for s in baseline:
  b=s['bounds']; span=b[1]-b[0]; ctx=s['context']; c=s['baseline_condition']; n=noise[s['id']]
  if stage=='preprocessing': ctx=((np.clip(np.array(ctx,dtype=np.float32),0,1e9)-b[0])/span).tolist()
  start=time.perf_counter()
  if stage=='baseline': p=np.array(s['prediction']); ms=s['elapsed_ms']
  else:
   draws=iter([np.array(n['blr']),np.array(n['flow']).reshape(1,352,1)]); orig=np.random.randn; np.random.randn=lambda *shape:next(draws)
   try: p=e.predict(ctx,c['traj_pattern'],c['period'])
   finally: np.random.randn=orig
   if stage=='preprocessing':p=p*span+b[0]
   ms=(time.perf_counter()-start)*1000
  err=p-np.clip(np.array(s['target'],dtype=np.float32),0,1e9); rows.append(dict(stage=stage,system=s['system'],species=s['species'],mae=float(np.abs(err).mean()),mse=float((err**2).mean()),nmae=float(np.abs(err).mean()/span),points=len(err),elapsed_ms=ms))
with (out/'stage_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Stage metrics saved',len(rows))
