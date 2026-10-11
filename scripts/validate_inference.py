"""Actual-model parity and fixed real-trajectory diagnostics.

Run after freezing the legacy baseline. Outputs go outside the Git checkout to
Materials/Validation/InferenceParity; the baseline is never overwritten here.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests/reference'))
import numpy as np
import onnx
import onnxruntime as ort
import torch
from web.backend.engine import RegimeFlowEngine, MambaPy
from web.backend.engine_onnx import RegimeFlowEngineONNX
from web.backend.inference_config import load_config
from models.FlowMatching.RegimeFlow.arch.source_BLR import BLRPriorConfig, BLRTemplatePriorGenerator
from upstream_cpu import upstream_mamba

OUT = ROOT.parent/'Materials/Validation/InferenceParity'

def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def validate_reference(reexport):
    torch.set_num_threads(1)
    torch.manual_seed(20261011)
    cpu = RegimeFlowEngine(str(ROOT/'ckpt/RegimeFlow/seed53_best.ckpt'))
    upstream = upstream_mamba()
    reference = copy.deepcopy(cpu.backbone)
    report = {'reference': 'upstream Mamba + selective_scan_ref from v2.2.6, CPU only',
              'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'tests/reference').glob('*v2_2_6.py')},
              'versions': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__, 'onnx': onnx.__version__, 'onnxruntime': ort.__version__},
              'weight_source': 'ema_state_dict', 'strict_weight_loading': True, 'mamba_blocks': []}
    for block, source in zip(reference.residual_blocks, cpu.backbone.residual_blocks):
        current = source.mamba_layer.mamba
        module = upstream(d_model=current.d_model, d_state=current.d_state,
                          d_conv=current.conv1d.kernel_size[0], expand=current.d_inner/current.d_model,
                          use_fast_path=False)
        module.load_state_dict(current.state_dict(), strict=True)
        module.eval()
        block.mamba_layer.mamba = module
        x = torch.randn(2, 352, current.d_model)
        with torch.no_grad():
            expected = module(x).numpy(); actual = current(x).numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        report['mamba_blocks'].append(float(np.max(np.abs(actual-expected))))
    reference.eval()
    engine = RegimeFlowEngineONNX(str(ROOT/'ckpt/RegimeFlow/backbone.onnx'), str(ROOT/'ckpt/RegimeFlow/cond_encoder.onnx'), str(ROOT/'ckpt/RegimeFlow/seed53_best.ckpt'))
    export_session = None
    if reexport:
        opts=ort.SessionOptions();opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL;opts.intra_op_num_threads=1
        export_session=ort.InferenceSession(str(Path(reexport)/'ckpt/RegimeFlow/backbone.onnx'), opts, providers=['CPUExecutionProvider'])
    cases=[]
    for pattern in range(6):
        period=24.0 if pattern==3 else 0.0
        context=(.5 + .3*np.sin(np.arange(96)*2*np.pi/24) + np.arange(96)*.001).tolist()
        noise={'blr':np.random.default_rng(1).normal(size=256).tolist(), 'flow':np.random.default_rng(2).normal(size=352).tolist()}
        trace={}; prediction=engine.predict(context,pattern,period,bounds=[0,1],noise=noise,trace=trace)
        cond=cpu.condition_encoder(batch_size=1,device='cpu',traj_pattern=torch.tensor([pattern]),period=torch.tensor([period]))
        np.testing.assert_allclose(np.array(trace['condition']),cond.detach().numpy(),atol=2e-6,rtol=2e-5)
        x=torch.tensor(trace['x0'],dtype=torch.float32)
        pt=torch.arange(96,dtype=torch.float64)[None];ft=torch.arange(96,352,dtype=torch.float64)[None]
        vals=torch.tensor((np.array(trace['transformed_context'])-np.array(trace['loc']).item())/np.array(trace['scale']).item(),dtype=torch.float64)[None]
        # Validate the source BLR in float64 to isolate solver rounding from the
        # implementation semantics. The original mean container is float32.
        config=engine.config
        prior=BLRTemplatePriorGenerator(BLRPriorConfig(**{key: config[key] for key in ('alpha','beta','noise_scale','min_variance','num_harmonics','saturation_scales','poly_degree')}))
        ref=prior._extract_reference(pt,vals)
        omega=prior._compute_omega(torch.tensor([period],dtype=torch.float64),ref['t_range'],96)
        pp=prior._build_basis(pt,pattern,ref,omega if pattern==3 else None).double()
        pf=prior._build_basis(ft,pattern,ref,omega if pattern==3 else None).double()
        from models.FlowMatching.RegimeFlow.arch.source_BLR import BLRSolver
        mu,_=BLRSolver.solve(pp,vals,pf,prior.config.alpha,prior.config.beta,prior.config.min_variance)
        mu=prior._apply_continuity_constraint(mu,ref['y_ref'])
        prior_error=float(np.max(np.abs(mu.numpy().reshape(-1)-trace['prior_mean'])))
        np.testing.assert_allclose(mu.numpy().reshape(-1),trace['prior_mean'],atol=2e-6,rtol=2e-5)
        errors=[]; exported=[]
        with torch.no_grad():
            for k in range(4):
                timepoint=k/4
                # Same input to each implementation prevents integration drift
                # from obscuring a velocity-field error.
                state=np.array(trace['x0'] if k==0 else trace['states'][k-1],np.float32)
                expected=reference(torch.tensor([timepoint]),torch.from_numpy(state),cond).numpy()
                actual=np.array(trace['velocities'][k])
                np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=2e-4)
                errors.append(float(np.max(np.abs(actual-expected))))
                if export_session:
                    value=export_session.run(None,{'t':np.array([timepoint],np.float32),'x_in':state,'cond_emb':cond.numpy()})[0]
                    np.testing.assert_allclose(value,expected,atol=1e-4,rtol=2e-4)
                    exported.append(float(np.max(np.abs(value-expected))))
                x=x+.25*reference(torch.tensor([timepoint]),x,cond)
        reference_pred=(x.numpy()[0,96:,0]*np.array(trace['scale']).item()+np.array(trace['loc']).item())
        np.testing.assert_allclose(prediction,reference_pred,atol=1e-4,rtol=2e-4)
        # The original training prior solves in float32. Also measure that
        # rounding separately from the float64 mathematical BLR check above.
        _,mu32,_=prior(pt.float(),vals.float(),ft.float(),torch.tensor([pattern]),torch.tensor([period]))
        prior_float32_error=float(np.max(np.abs(mu32.numpy().reshape(-1)-trace['prior_mean'])))
        cpu_prediction=cpu.predict(context,pattern,period,bounds=[0,1],noise=noise)
        cpu_prediction_error=float(np.max(np.abs(cpu_prediction-prediction)))
        np.testing.assert_allclose(cpu_prediction,prediction,atol=5e-4,rtol=5e-4)
        cases.append({'source_float32_prior_max_abs_error':prior_float32_error,
                      'cpu_engine_prediction_max_abs_error':cpu_prediction_error,'pattern':pattern,'prior_max_abs_error':prior_error,'velocity_max_abs_errors':errors,'reexport_max_abs_errors':exported,'prediction_max_abs_error':float(np.max(np.abs(prediction-reference_pred)))})
    report['cases']=cases
    report['model_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'ckpt/RegimeFlow/backbone.onnx',ROOT/'ckpt/RegimeFlow/backbone.onnx.data',ROOT/'web/models/backbone.onnx',ROOT/'web/models/inference-config.json']}
    # Structural checks and learned-tensor identity against the frozen legacy.
    oldpath=Path('/private/tmp/regimeflow-before/backbone.onnx')
    if oldpath.exists():
        old=onnx.load(oldpath);new=onnx.load(ROOT/'ckpt/RegimeFlow/backbone.onnx')
        tensors={t.name:onnx.numpy_helper.to_array(t) for t in new.graph.initializer}
        for tensor in old.graph.initializer:
            np.testing.assert_array_equal(onnx.numpy_helper.to_array(tensor),tensors[tensor.name])
        report['all_legacy_initializers_unchanged']=True
        report['external_data_sha256_unchanged']=hashlib.sha256(oldpath.with_suffix('.onnx.data').read_bytes()).hexdigest()==report['model_hashes']['ckpt/RegimeFlow/backbone.onnx.data']
    for path in [ROOT/'ckpt/RegimeFlow/backbone.onnx',ROOT/'web/models/backbone.onnx']:
        onnx.checker.check_model(str(path))
        metadata={p.key:p.value for p in onnx.load(path,load_external_data=False).metadata_props}
        assert metadata['regimeflow.mamba_projection_layout']=='x,z;dt,B,C'
    save('reference_validation.json',report)
    print('REFERENCE PASS',json.dumps(cases),flush=True)


def validate_real():
    samples=json.loads((OUT/'baseline_predictions.json').read_text())['samples']
    noises=json.loads((OUT/'fixed_noise_arrays.json').read_text())
    engine=RegimeFlowEngineONNX(str(ROOT/'ckpt/RegimeFlow/backbone.onnx'),str(ROOT/'ckpt/RegimeFlow/cond_encoder.onnx'),str(ROOT/'ckpt/RegimeFlow/seed53_best.ckpt'))
    with (OUT/'stage_metrics.csv').open() as f: rows=[row for row in csv.DictReader(f) if row['stage'] in ('baseline','mamba','preprocessing')]
    comparisons=[];predictions=[];fixtures=[]
    for sample in samples:
        record=sample['condition'];pattern=record['trajectory_type'];period=record.get('period',0)
        values=[];noise=noises[sample['id']]
        for steps in (4,32):
            engine.denoise_steps=steps;trace={};start=time.perf_counter()
            pred=engine.predict(sample['context'],pattern,period,bounds=sample['bounds'],noise=noise,trace=trace)
            elapsed=(time.perf_counter()-start)*1000;err=pred-np.clip(np.array(sample['target'],dtype=np.float32),*engine.config['dataset_clip']);span=sample['bounds'][1]-sample['bounds'][0]
            metrics=dict(system=sample['system'],species=sample['species'],mae=float(np.abs(err).mean()),mse=float(np.mean(err**2)),nmae=float(np.abs(err).mean()/span),points=len(err),elapsed_ms=elapsed)
            values.append(metrics)
            if steps==4:
                for stage in ('conditions_blr','final'): rows.append(dict(stage=stage,**metrics))
                predictions.append(dict(id=sample['id'],prediction=pred.tolist(),condition={'traj_pattern':pattern,'period':period}))
                if len(fixtures)<6 and pattern not in [f['pattern'] for f in fixtures]:
                    fixtures.append(dict(id=sample['id'],context=sample['context'],bounds=sample['bounds'],pattern=pattern,period=period,noise=noise,trace=trace,prediction=pred.tolist()))
        comparisons.append(dict(system=sample['system'],species=sample['species'],nmae_4=values[0]['nmae'],nmae_32=values[1]['nmae'],delta_nmae_4_minus_32=values[0]['nmae']-values[1]['nmae'],elapsed_ms_4=values[0]['elapsed_ms'],elapsed_ms_32=values[1]['elapsed_ms']))
        print('REAL',sample['id'],values[0]['nmae'],values[1]['nmae'],flush=True)
    for filename,data in [('stage_metrics.csv',rows),('steps_comparison.csv',comparisons)]:
        with (OUT/filename).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    save('final_predictions.json',predictions);save('browser_fixtures.json',fixtures)
    summary={}
    for stage in ('baseline','mamba','preprocessing','conditions_blr','final'):
        systems={}
        for row in rows:
            if row['stage']==stage:systems.setdefault(row['system'],[]).append(float(row['nmae']))
        summary[stage]={'system_mean_nmae':float(np.mean([np.mean(v) for v in systems.values()])), 'systems':len(systems),'species':sum(map(len,systems.values()))}
    save('metrics_summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reexport',type=Path);parser.add_argument('--only',choices=['reference','real'])
    args=parser.parse_args()
    if args.only!='real':validate_reference(args.reexport)
    if args.only!='reference':validate_real()
