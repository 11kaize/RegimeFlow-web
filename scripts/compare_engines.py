"""Compare PyTorch and ONNX engine velocities."""
import sys; sys.path.insert(0, '.')
import numpy as np
np.random.seed(42)

import torch; torch.set_num_threads(1); torch.manual_seed(42)

from web.backend.engine import init_engine as init_pt
from web.backend.engine_onnx import init_engine as init_onnx

ctx = np.array(0.5 + 0.3 * np.sin(2*np.pi*np.arange(96)/12.5), dtype=np.float32)

# === PyTorch ===
e_pt = init_pt('ckpt/RegimeFlow/seed53_best.ckpt', denoise_steps=4)
past_target = torch.from_numpy(ctx).float().view(1, 96, 1)
past_observed = torch.ones_like(past_target)
_, loc_pt, scale_pt = e_pt.scaler(past_target, past_observed)
print(f'PT: loc={loc_pt.item():.4f}, scale={scale_pt.item():.4f}')

scaled_context_pt = past_target / scale_pt
scaled_prior_pt = (past_target[:, -96:] - loc_pt) / scale_pt
past_t = torch.arange(96, dtype=torch.float32).unsqueeze(0)
fut_t = torch.arange(96, 352, dtype=torch.float32).unsqueeze(0)
x0_fut_pt, _, _ = e_pt.prior(past_t, scaled_prior_pt[:,:,0], fut_t,
    torch.zeros(1, dtype=torch.long), torch.zeros(1))
x0_pt = torch.cat([scaled_context_pt, x0_fut_pt.unsqueeze(-1)], dim=1)

cond_pt = e_pt.condition_encoder(batch_size=1, device='cpu',
    traj_pattern=torch.tensor([3], dtype=torch.long),
    period=torch.tensor([12.5], dtype=torch.float32))

# === ONNX ===
np.random.seed(42)
e_onx = init_onnx('ckpt/RegimeFlow/seed53_best.ckpt')
past_target_np = ctx.reshape(1, 96, 1)
past_obs_np = np.ones_like(past_target_np)
loc_np, scale_np = e_onx.scaler.fit_transform(past_target_np, past_obs_np)
print(f'ONNX: loc={loc_np[0,0,0]:.4f}, scale={scale_np[0,0,0]:.4f}')

scaled_ctx_np = past_target_np / scale_np
scaled_prior_np = (past_target_np[0,:,0] - loc_np[0,0,0]) / scale_np[0,0,0]
x0_fut_np = e_onx.prior.generate(
    np.arange(96, dtype=np.float32), scaled_prior_np,
    np.arange(96, 352, dtype=np.float32))
x0_np = np.concatenate([scaled_ctx_np, x0_fut_np.reshape(1,256,1)], axis=1)

cond_np = e_onx._encode_condition(3, 12.5)

# Compare x0
print(f'\nx0 PT future range: [{x0_pt[0,96:,0].min():.4f}, {x0_pt[0,96:,0].max():.4f}]')
print(f'x0 ONNX future range: [{x0_np[0,96:,0].min():.4f}, {x0_np[0,96:,0].max():.4f}]')

# Compare cond_emb
print(f'\ncond PT range: [{cond_pt.min():.4f}, {cond_pt.max():.4f}]')
print(f'cond ONNX range: [{cond_np.min():.4f}, {cond_np.max():.4f}]')

cond_diff = np.abs(cond_pt.detach().numpy() - cond_np).max()
print(f'cond max diff: {cond_diff:.6f}')

# Compare backbone velocities at t=0
with torch.no_grad():
    vt_pt = e_pt.backbone(torch.zeros(1), x0_pt, cond_pt)
vt_np = e_onx._backbone_velocity(0.0, x0_np, cond_np)

print(f'\nvt PT range: [{vt_pt.min():.4f}, {vt_pt.max():.4f}]')
print(f'vt ONNX range: [{vt_np.min():.4f}, {vt_np.max():.4f}]')

# Compare final predictions
pt_pred = e_pt.predict(ctx.tolist(), traj_pattern=3, period=12.5)
onx_pred = e_onx.predict(ctx.tolist(), traj_pattern=3, period=12.5)
print(f'\nFinal PT: mean={pt_pred.mean():.4f}, std={pt_pred.std():.4f}')
print(f'Final ONNX: mean={onx_pred.mean():.4f}, std={onx_pred.std():.4f}')
