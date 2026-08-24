/* ================================================================
   RegimeFlow Web — 浏览器端推理引擎（onnxruntime-web / WASM）
   ================================================================
   这是 web/backend/engine_onnx.py 里 RegimeFlowEngineONNX.predict() 的
   JavaScript 移植。把 backbone + condition encoder + BLR 先验 + StdScaler
   + 欧拉 ODE 全部放到访问者浏览器里跑，预测速度取决于访问者的机器，
   而不是共享的免费层 CPU（Render 0.1 CPU 才是「云端慢」的根因）。

   依赖：onnxruntime-web（需在本文件之前加载，暴露全局 `ort`）。

   模型文件必须是「合并单文件」.onnx（外置 .data 已合并进去）：
     - backbone.onnx      (~43MB)
     - cond_encoder.onnx  (~400KB)

   对外暴露 window.RegimeFlowWeb：
     load({backbone, condEncoder}) → Promise<boolean>  启动加载
     ready()                       → Promise<boolean>  加载完成后 resolve
     isLoaded()                    → boolean
     getError()                    → string|null
     predict(context, pattern, period, opts?) → Promise<number[]>
     HP                            → 超参数（供调试/交叉验证）
   ================================================================ */
window.RegimeFlowWeb = (function () {
  'use strict';

  // ── 超参数（来自 ckpt/RegimeFlow/hyperparams.json）──
  var HP = {
    context_length: 96,
    prediction_length: 256,
    denoise_steps: 4,
    cond_dim: 128,
    // BLR 先验（常数基 → 闭式解），见 engine_onnx.py BLRPriorNP
    alpha: 1.0,
    beta: 10.0,
    noise_scale: 0.1,
    min_variance: 1e-6,
    // StdScaler minimum_scale（engine_onnx.py 用 StdScalerNP(minimum_scale=1.0)；
    // 但因 axis=-1 是特征维，variance 恒为 0 → scale 恒等于 1.0，见 predict()）
    min_scale: 1.0,
  };

  var _sessions = { backbone: null, condEncoder: null };
  var _loading = null;   // Promise<boolean> | null
  var _error = null;

  // ── 高斯随机数（Box–Muller，标准正态）──
  function randn() {
    var u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  // ── 加载两个 ONNX session（合并单文件模型）──
  function load(modelUrls, opts) {
    if (_loading) return _loading;
    opts = opts || {};
    _loading = (async function () {
      try {
        if (!window.ort || !ort.InferenceSession) {
          throw new Error('onnxruntime-web 未加载');
        }
        // 浏览器用 wasm；Node 侧（onnxruntime-node）交叉验证时传 ['cpu']
        var sessionOpts = { executionProviders: opts.executionProviders || ['wasm'] };
        _sessions.condEncoder = await ort.InferenceSession.create(modelUrls.condEncoder, sessionOpts);
        _sessions.backbone = await ort.InferenceSession.create(modelUrls.backbone, sessionOpts);
        _error = null;
        return true;
      } catch (e) {
        _error = (e && e.message) || String(e);
        console.error('[RegimeFlowWeb] load failed:', _error);
        return false;
      }
    })();
    return _loading;
  }

  function isLoaded() { return !!_sessions.backbone && !!_sessions.condEncoder; }
  function getError() { return _error; }

  function ready() {
    if (_loading) return _loading;
    return Promise.resolve(isLoaded());
  }

  // ── 一次 backbone 前向：速度场 v(t, x) ──
  async function _backboneVelocity(t, x, condEmb) {
    var feeds = {
      t: new ort.Tensor('float32', new Float32Array([t]), [1]),
      x_in: new ort.Tensor('float32', x, [1, HP.context_length + HP.prediction_length, 1]),
      cond_emb: new ort.Tensor('float32', condEmb, [1, HP.cond_dim]),
    };
    var out = await _sessions.backbone.run(feeds);
    // 复制一份，避免 onnxruntime-web 复用输出缓冲区的潜在风险
    return Float32Array.from(out[_sessions.backbone.outputNames[0]].data);
  }

  async function _encodeCondition(trajPattern, period) {
    var feeds = {
      traj_pattern: new ort.Tensor('int64', BigInt64Array.from([BigInt(trajPattern | 0)]), [1]),
      period: new ort.Tensor('float32', new Float32Array([period]), [1]),
    };
    var out = await _sessions.condEncoder.run(feeds);
    return Float32Array.from(out[_sessions.condEncoder.outputNames[0]].data);
  }

  // ── 预测 ──
  async function predict(context, trajPattern, period, opts) {
    if (!isLoaded()) throw new Error('引擎未加载');
    opts = opts || {};

    var L = HP.context_length;
    var F = HP.prediction_length;
    var total = L + F;

    // 1) 准备上下文：edge 填充 / 截断到 L
    var src = context;
    var ctx = new Float32Array(L);
    if (src.length < L) {
      var pad = L - src.length;
      for (var i = 0; i < pad; i++) ctx[i] = src[0];
      for (var j = 0; j < src.length; j++) ctx[pad + j] = src[j];
    } else {
      for (var k = 0; k < L; k++) ctx[k] = src[src.length - L + k];
    }

    // 2) StdScaler —— 注意：引擎把 context reshape 成 (1,96,1) 后对
    //    axis=-1（特征维，尺寸=1）做归一化，因此是「恒等」：
    //    loc = x（逐元素）、scale = 1.0；最终只用到 loc[0,0]=ctx[0]、scale[0,0]=1.0。
    //    这里 bug-for-bug 复刻，保证浏览器输出与部署后端（engine_onnx.py）一致。
    var loc = ctx[0];
    var scale = 1.0;

    // 3) 条件嵌入 (1,128)
    var condEmb = await _encodeCondition(trajPattern, period);

    // 4) BLR 先验（常数基）→ mu = beta/(alpha + beta*N) * Σ(scaled_prior)
    //    scaled_prior = (prior_ctx - loc) / scale = ctx - ctx[0]
    var scaledSum = 0;
    for (var i2 = 0; i2 < L; i2++) scaledSum += (ctx[i2] - loc) / scale;
    var mu = (HP.beta / (HP.alpha + HP.beta * L)) * scaledSum;

    // 5) 构建 x0：上下文 = past_target/scale = ctx（原始值，scale=1.0）
    //           未来先验 = mu + 噪声
    var x0 = new Float32Array(total);
    for (var i3 = 0; i3 < L; i3++) x0[i3] = ctx[i3] / scale;
    for (var i4 = 0; i4 < F; i4++) {
      var priorNoise = opts.deterministic ? 0 : HP.noise_scale * randn();
      x0[L + i4] = mu + priorNoise;
    }
    // flow-matching sigma_max=1 噪声
    for (var i5 = 0; i5 < total; i5++) {
      x0[i5] += opts.deterministic ? 0 : randn();
    }

    // 6) 欧拉 ODE（denoise_steps 步）
    //    匹配 Python：x0 是 float32，但欧拉累加用 float64（Python 里
    //    `xt = xt + dt*vt` 因 dt 是 python float 会把 xt 提升成 float64）。
    //    backbone 每次仍收 float32（等价 Python 的 x.astype(np.float32)）。
    var dt = 1.0 / HP.denoise_steps;
    var xt = Float64Array.from(x0);
    for (var step = 0; step < HP.denoise_steps; step++) {
      var t = step * dt;
      var vt = await _backboneVelocity(t, Float32Array.from(xt), condEmb);
      for (var i6 = 0; i6 < total; i6++) xt[i6] += dt * vt[i6];
    }

    // 7) 反归一化 + 取出未来段
    var pred = new Float64Array(F);
    for (var i7 = 0; i7 < F; i7++) pred[i7] = xt[L + i7] * scale + loc;
    return Array.from(pred);
  }

  return {
    load: load,
    ready: ready,
    isLoaded: isLoaded,
    getError: getError,
    predict: predict,
    HP: HP,
  };
})();
