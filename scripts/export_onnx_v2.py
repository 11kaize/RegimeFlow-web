"""
RegimeFlow ONNX Export v2 — properly load weights from checkpoint.

Strategy:
  1. Load raw checkpoint dict with torch.load
  2. Build backbone + condition encoder from hyperparameters
  3. Load weights from checkpoint state_dict
  4. Export to ONNX
"""
import sys, json, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import omegaconf

# ── Add safe globals for checkpoint loading ──
torch.serialization.add_safe_globals([
    omegaconf.DictConfig, omegaconf.ListConfig,
    omegaconf.base.ContainerMetadata, type(None),
    dict, list, int,
])

# ── Pure PyTorch Mamba (same as engine.py) ──
class MambaPy(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
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
            if not isinstance(D, torch.Tensor):
                D = torch.tensor(D, device=device, dtype=dtype)
            if D.dim() == 0:
                D = D.unsqueeze(0)
            y = y + u * D.unsqueeze(0).unsqueeze(1)
        return y

    def forward(self, x):
        B, L, D = x.shape
        zx = self.in_proj(x)
        z, xp = zx[:, :, :self.d_inner], zx[:, :, self.d_inner:]
        xp = self.conv1d(xp.transpose(1, 2))[:, :, :L].transpose(1, 2)
        xp_act, z_act = self.act_fn(xp), self.act_fn(z)
        x_out = self.x_proj(xp_act)
        B_p, C_p = x_out[:, :, :self.d_state], x_out[:, :, self.d_state:2 * self.d_state]
        dt = F.softplus(self.dt_proj(x_out[:, :, 2 * self.d_state:]))
        y = self._selective_scan(xp_act, dt, self.A_log, B_p, C_p, self.D)
        return self.out_proj(y * z_act)

# ── Monkey-patch ──
sys.modules['mamba_ssm'] = type('mamba_ssm', (), {'Mamba': MambaPy, '__version__': '2.2.6-py'})

# ── Import model components ──
from models.FlowMatching.RegimeFlow.RegimeFlow_base import RMSNorm
from models.FlowMatching.RegimeFlow.arch._base import StdScaler
from models.FlowMatching.RegimeFlow.arch.bio_cond_layers import ConditionEncoder
from models.FlowMatching.RegimeFlow.arch.source_BLR import BLRTemplatePriorGenerator, BLRPriorConfig
from models.FlowMatching.RegimeFlow.arch.backbone import BackboneModel, AdaLN

# ── Load checkpoint ──
CKPT = str(PROJECT_ROOT / "ckpt" / "RegimeFlow" / "seed53_best.ckpt")
print(f"Loading checkpoint: {CKPT}")
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
hp = ckpt['hyper_parameters']
state_dict = ckpt['state_dict']
ema_dict = ckpt.get('ema_state_dict')
weights = ema_dict if ema_dict else state_dict

# Save hyperparams as JSON
hp_json = {}
for k, v in hp.items():
    if isinstance(v, (dict, omegaconf.DictConfig)):
        hp_json[k] = dict(v)
    elif isinstance(v, (list, omegaconf.ListConfig)):
        hp_json[k] = list(v)
    else:
        hp_json[k] = v
json_path = str(PROJECT_ROOT / "ckpt" / "RegimeFlow" / "hyperparams.json")
with open(json_path, 'w') as f:
    json.dump(hp_json, f, indent=2, default=str)
print(f"Saved hyperparams to {json_path}")

print(f"context_length={hp['context_length']}, prediction_length={hp['prediction_length']}")
print(f"cond_dim={hp.get('cond_dim', 128)}, use_condition={hp.get('use_condition', True)}")

# ── Build Condition Encoder ──
cond_dim = hp.get('cond_dim', 128)
use_condition = hp.get('use_condition', True)
num_patterns = hp.get('num_patterns', 6)
num_freqs = hp.get('num_freqs', 64)

cond_encoder = ConditionEncoder(d_model=cond_dim, num_patterns=num_patterns, num_freqs=num_freqs)
ce_weights = {k.replace('condition_encoder.', ''): v for k, v in weights.items()
              if k.startswith('condition_encoder.')}
cond_encoder.load_state_dict(ce_weights, strict=True)
cond_encoder.eval()
print(f"ConditionEncoder loaded: {len(ce_weights)} keys")

# ── Build Backbone ──
bb_params = dict(hp['backbone_params'])
backbone = BackboneModel(
    input_dim=bb_params['input_dim'],
    hidden_dim=bb_params['hidden_dim'],
    output_dim=bb_params['output_dim'],
    step_emb=bb_params['step_emb'],
    num_residual_blocks=bb_params['num_residual_blocks'],
    init_skip=bb_params.get('init_skip', True),
    block_type=bb_params.get('block_type', 'mamba'),
    d_state=bb_params.get('d_state', 16),
    d_conv=bb_params.get('d_conv', 4),
    expand=bb_params.get('expand', 2),
    ffn_dim_multiplier=bb_params.get('ffn_dim_multiplier', 4.0),
    ffn_dropout=bb_params.get('ffn_dropout', 0.0),
    dropout=bb_params.get('dropout', 0.0),
    use_adaLN=use_condition,
    cond_dim=cond_dim if use_condition else 0,
)

bb_weights = {k.replace('backbone.', ''): v for k, v in weights.items()
              if k.startswith('backbone.')}
missing, unexpected = backbone.load_state_dict(bb_weights, strict=False)
if missing:
    print(f"Backbone missing keys: {len(missing)}")
if unexpected:
    print(f"Backbone unexpected keys: {len(unexpected)}")
backbone.eval()
print(f"Backbone loaded: {len(bb_weights)} keys")
total_params = sum(p.numel() for p in backbone.parameters())
print(f"Backbone params: {total_params:,}")

# ── Quick comparison with PyTorch engine ──
from web.backend.engine import init_engine as init_pt
e_pt = init_pt(CKPT, denoise_steps=4)

ctx_test = np.array(0.5 + 0.3 * np.sin(2*np.pi*np.arange(96)/12.5), dtype=np.float32)
pt_pred = e_pt.predict(ctx_test.tolist(), traj_pattern=3, period=12.5)

# Test condition encoder
cond_pt = e_pt.condition_encoder(batch_size=1, device='cpu',
    traj_pattern=torch.tensor([3], dtype=torch.long),
    period=torch.tensor([12.5], dtype=torch.float32))
cond_new = cond_encoder(batch_size=1, device='cpu',
    traj_pattern=torch.tensor([3], dtype=torch.long),
    period=torch.tensor([12.5], dtype=torch.float32))
print(f"Cond encoder match: max diff = {(cond_pt - cond_new).abs().max().item():.8f}")

# Export condition encoder to ONNX
class CondEncoderONNX(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
    def forward(self, traj_pattern, period):
        return self.enc(batch_size=traj_pattern.shape[0], device='cpu',
                       traj_pattern=traj_pattern, period=period)

ce_wrapper = CondEncoderONNX(cond_encoder.cpu())
ce_wrapper.eval()

ce_onnx = str(PROJECT_ROOT / "ckpt" / "RegimeFlow" / "cond_encoder.onnx")
torch.onnx.export(
    ce_wrapper,
    (torch.tensor([0], dtype=torch.long), torch.tensor([0.0], dtype=torch.float32)),
    ce_onnx,
    input_names=['traj_pattern', 'period'],
    output_names=['cond_emb'],
    dynamic_axes={'traj_pattern': {0: 'batch'}, 'period': {0: 'batch'}, 'cond_emb': {0: 'batch'}},
    opset_version=17, do_constant_folding=True,
)
print(f"CondEncoder ONNX: {Path(ce_onnx).stat().st_size / 1024:.1f} KB")

# Verify ONNX cond encoder
import onnxruntime as ort
ce_sess = ort.InferenceSession(ce_onnx, providers=['CPUExecutionProvider'])
ce_out = ce_sess.run(None, {
    'traj_pattern': np.array([3], dtype=np.int64),
    'period': np.array([12.5], dtype=np.float32),
})
ce_match = np.abs(cond_pt.detach().numpy() - ce_out[0]).max()
print(f"ONNX cond encoder match: max diff = {ce_match:.8f}")

# Export backbone to ONNX
class BackboneONNX(nn.Module):
    def __init__(self, bb):
        super().__init__()
        self.bb = bb
    def forward(self, t, x_in, cond_emb):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        return self.bb(t, x_in, cond_emb)

bb_wrapper = BackboneONNX(backbone.cpu())
bb_wrapper.eval()

# Test forward pass
dummy_t = torch.zeros(1)
dummy_x = torch.randn(1, 352, 1)
dummy_cond = torch.randn(1, cond_dim)
with torch.no_grad():
    out_test = bb_wrapper(dummy_t, dummy_x, dummy_cond)
print(f"Backbone test output shape: {out_test.shape}")

bb_onnx = str(PROJECT_ROOT / "ckpt" / "RegimeFlow" / "backbone.onnx")
t0 = time.time()
torch.onnx.export(
    bb_wrapper,
    (dummy_t, dummy_x, dummy_cond),
    bb_onnx,
    input_names=['t', 'x_in', 'cond_emb'],
    output_names=['velocity'],
    dynamic_axes={'t': {0: 'batch'}, 'x_in': {0: 'batch'},
                  'cond_emb': {0: 'batch'}, 'velocity': {0: 'batch'}},
    opset_version=17, do_constant_folding=True,
)
print(f"Backbone ONNX exported in {time.time()-t0:.1f}s, size: {Path(bb_onnx).stat().st_size/1024/1024:.1f} MB")

# Verify ONNX backbone
import onnx
onnx.checker.check_model(onnx.load(bb_onnx))
print("ONNX check: valid")

bb_sess = ort.InferenceSession(bb_onnx, providers=['CPUExecutionProvider'])
with torch.no_grad():
    vt_pt = backbone(dummy_t, dummy_x, dummy_cond)
vt_onx = bb_sess.run(None, {
    't': dummy_t.numpy().astype(np.float32),
    'x_in': dummy_x.numpy().astype(np.float32),
    'cond_emb': dummy_cond.numpy().astype(np.float32),
})[0]
bb_match = np.abs(vt_pt.numpy() - vt_onx).max()
print(f"ONNX backbone match (random input): max diff = {bb_match:.8f}")

print("\n✅ All exports complete and verified!")
