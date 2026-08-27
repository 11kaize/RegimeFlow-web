/* Deterministic JS reference from engine-web.js, for cross-checking the Python
   engine. Loads the merged single-file ONNX via onnxruntime-node (native, the
   same C++ runtime the Python backend uses), so any output difference is a real
   logic difference in the port, not a runtime difference.
*/
'use strict';
const path = require('path');
const ort = require('onnxruntime-node');

// Stub `window` so the browser IIFE in engine-web.js binds to the global object.
global.window = global;
global.ort = ort;

require('../../web/js/engine-web.js');
const RF = global.RegimeFlowWeb;

(async () => {
  // Reproduce the exact synthetic context used by cross_check_py.py.
  const N = 120;
  const context = [];
  for (let t = 0; t < N; t++) {
    context.push(0.5 * t + 2.0 * Math.sin((2 * Math.PI * t) / 40.0));
  }
  const trajPattern = 3;
  const period = 12.5;

  const ok = await RF.load({
    backbone: path.resolve(__dirname, '../../web/models/backbone.onnx'),
    condEncoder: path.resolve(__dirname, '../../web/models/cond_encoder.onnx'),
  }, { executionProviders: ['cpu'] });
  if (!ok) {
    console.error('load failed:', RF.getError());
    process.exit(1);
  }

  const pred = await RF.predict(context, trajPattern, period, { deterministic: true });

  // Recompute intermediates the same way for the comparison dump.
  // 引擎恒等归一化：loc=ctx[0]、scale=1.0（见 engine_onnx.py StdScaler 的 axis 细节）。
  const ctx = context.slice(-96);
  const L = 96;
  const loc = ctx[0];
  const scale = 1.0;
  let ssum = 0;
  for (let i = 0; i < L; i++) ssum += (ctx[i] - loc) / scale;
  const mu = (10.0 / (1.0 + 10.0 * L)) * ssum;

  const out = {
    context_head: context.slice(0, 3),
    traj_pattern: trajPattern,
    period: period,
    loc: loc,
    scale: scale,
    mu: mu,
    x0_ctx_head: [ctx[0] / scale, ctx[1] / scale, ctx[2] / scale],
    x0_ctx_tail: [ctx[93] / scale, ctx[94] / scale, ctx[95] / scale],
    x0_fut_head: [mu, mu, mu],
    pred_len: pred.length,
    pred_head: pred.slice(0, 6),
    pred_tail: pred.slice(-6),
  };
  console.log(JSON.stringify(out, null, 2));
})().catch((e) => { console.error(e); process.exit(1); });
