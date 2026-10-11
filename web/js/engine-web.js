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

  // Generated from checkpoint; shared with the Python engine.
  var HP = {};
  var _configLoading = null;

  function validateConfig(config) {
    ['context_length', 'prediction_length', 'cond_dim', 'denoise_steps', 'alpha', 'beta', 'min_scale', 'min_variance'].forEach(function(key) {
      if (typeof config[key] !== 'number' || !Number.isFinite(config[key]) || config[key] <= 0) throw new Error('Invalid inference parameter: ' + key);
    });
    ['context_length', 'prediction_length', 'cond_dim', 'denoise_steps', 'num_harmonics', 'poly_degree'].forEach(function(key) {
      if (!Number.isInteger(config[key]) || config[key] < 0) throw new Error('Invalid integer parameter: ' + key);
    });
    ['noise_scale', 'flow_sigma'].forEach(function(key) {
      if (!Number.isFinite(config[key]) || config[key] < 0) throw new Error('Invalid noise parameter: ' + key);
    });
    if (!Array.isArray(config.saturation_scales) || !config.saturation_scales.length || config.saturation_scales.some(function(x) { return typeof x !== 'number' || !Number.isFinite(x) || x <= 0; })) throw new Error('Invalid saturation_scales');
    if (config.transform !== 'minmax') throw new Error('Unsupported transform');
    Object.assign(HP, config);
  }

  function configReady() {
    return _configLoading || Promise.reject(new Error('Inference config not loaded'));
  }

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
    _configLoading = opts.config ? Promise.resolve(opts.config) : fetch(modelUrls.config || '/models/inference-config.json').then(function(resp) {
      if (!resp.ok) throw new Error('Inference config HTTP ' + resp.status);
      return resp.json();
    });
    _configLoading = _configLoading.then(function(config) { validateConfig(config); return HP; });
    _loading = (async function () {
      try {
        await _configLoading;
        if (!window.ort || !ort.InferenceSession) {
          throw new Error('onnxruntime-web 未加载');
        }
        // 浏览器用 wasm；Node 侧（onnxruntime-node）交叉验证时传 ['cpu']
        var sessionOpts = { executionProviders: opts.executionProviders || ['wasm'] };
        if (opts.onProgress) {
          // 浏览器侧：先 fetch 成 ArrayBuffer 再建 session，以便 UI 显示
          // 43MB 模型的实时下载进度（onnxruntime 内部 fetch 不暴露进度）。
          var condBuf = await _fetchBuffer(modelUrls.condEncoder, null);
          _sessions.condEncoder = await ort.InferenceSession.create(condBuf, sessionOpts);
          var bbBuf = await _fetchBuffer(modelUrls.backbone, opts.onProgress);
          _sessions.backbone = await ort.InferenceSession.create(bbBuf, sessionOpts);
        } else {
          _sessions.condEncoder = await ort.InferenceSession.create(modelUrls.condEncoder, sessionOpts);
          _sessions.backbone = await ort.InferenceSession.create(modelUrls.backbone, sessionOpts);
        }
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

  // 下载 URL → ArrayBuffer；可选 onProgress(receivedBytes, totalBytes)。
  async function _fetchBuffer(url, onProgress) {
    var resp = await fetch(url);
    if (!resp.ok) throw new Error('HTTP ' + resp.status + ' for ' + url);
    var total = Number(resp.headers.get('Content-Length')) || 0;
    if (!resp.body || !onProgress || !total) return resp.arrayBuffer();
    var reader = resp.body.getReader();
    var received = 0, chunks = [];
    for (;;) {
      var r = await reader.read();
      if (r.done) break;
      chunks.push(r.value);
      received += r.value.length;
      onProgress(received, total);
    }
    var buf = new Uint8Array(received);
    var off = 0;
    for (var i = 0; i < chunks.length; i++) { buf.set(chunks[i], off); off += chunks[i].length; }
    return buf.buffer;
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

  function condition(pattern, period) {
    if (!Number.isInteger(pattern) || pattern < 0 || pattern > 5) {
      console.warn('[RegimeFlow] default condition: invalid type', pattern);
      return [0, 0];
    }
    var validPeriod = typeof period === 'number' && Number.isFinite(period) && period >= 0;
    if (pattern === 3 && (!validPeriod || period <= 0)) {
      console.warn('[RegimeFlow] default condition: invalid oscillation period', period);
      return [0, 0];
    }
    return [pattern, validPeriod ? period : 0];
  }

  function datasetBounds(values) {
    if (!values.length) throw new Error('Empty dataset series');
    var lo = Infinity, hi = -Infinity;
    values.forEach(function(x) {
      x = Math.fround(x);
      if (!Number.isFinite(x) || Math.abs(x) > HP.dataset_clip[1]) throw new Error('Invalid dataset series');
      x = Math.max(HP.dataset_clip[0], x);
      lo = Math.min(lo, x); hi = Math.max(hi, x);
    });
    return [lo, hi];
  }

  function transform(context, bounds) {
    var values = Array.from(context);
    if (!values.length || values.some(function(x) { return typeof x !== 'number' || !Number.isFinite(x); })) throw new Error('Context must be nonempty and finite');
    var lo = Infinity, hi = -Infinity;
    if (bounds) {
      if (bounds.length !== 2 || !bounds.every(Number.isFinite) || bounds[1] < bounds[0]) throw new Error('Invalid minmax bounds');
      lo = bounds[0]; hi = bounds[1];
      values = values.map(function(x) { return Math.max(HP.dataset_clip[0], Math.min(HP.dataset_clip[1], Math.fround(x))); });
      if (values.some(function(x) { return x < lo || x > hi; })) throw new Error('Context outside supplied dataset bounds');
    } else {
      values.forEach(function(x) { lo = Math.min(lo, x); hi = Math.max(hi, x); });
    }
    var span = hi - lo;
    return { values: values.map(function(x) { return Math.fround(span ? (x - lo) / span : 0); }), lo: lo, span: span };
  }

  function basis(times, pastTimes, pattern, period) {
    var ref = pastTimes[pastTimes.length - 1], range = ref - pastTimes[0];
    var scale = Math.max.apply(null, times.map(function(t) { return Math.abs(t - ref); }));
    scale = Math.max(scale, 1e-6);
    var omega = pattern === 3 ? 2 * Math.PI / Math.max(period * range / (pastTimes.length - 1), range * 0.05) : 0;
    return times.map(function(t) {
      var row = [1], delta = t - ref;
      if (pattern === 1 || pattern === 2) {
        HP.saturation_scales.forEach(function(s) { row.push(1 - Math.exp(-s * delta / (range + 1e-6))); });
      } else if (pattern === 3) {
        for (var h = 1; h <= HP.num_harmonics; h++) { row.push(Math.sin(h * omega * t), Math.cos(h * omega * t)); }
      } else if (pattern === 4 || pattern === 5) {
        for (var d = 1; d <= HP.poly_degree; d++) row.push(Math.pow(delta / scale, d));
      }
      return row;
    });
  }

  // alpha > 0 makes the precision positive definite. Solve with Cholesky in
  // float64; failure is explicit, never replaced by a synthetic prior.
  function priorMean(values, pattern, period) {
    var past = values.map(function(_, i) { return i; });
    var future = Array.from({ length: HP.prediction_length }, function(_, i) { return values.length + i; });
    var pp = basis(past, past, pattern, period), pf = basis(future, past, pattern, period);
    var K = pp[0].length;
    var precision = Array.from({ length: K }, function() { return new Float64Array(K); });
    var rhs = new Float64Array(K);
    for (var i = 0; i < K; i++) {
      precision[i][i] = HP.alpha;
      for (var n = 0; n < values.length; n++) {
        rhs[i] += HP.beta * pp[n][i] * values[n];
        for (var j = 0; j < K; j++) precision[i][j] += HP.beta * pp[n][i] * pp[n][j];
      }
    }
    var chol = Array.from({ length: K }, function() { return new Float64Array(K); });
    for (var r = 0; r < K; r++) {
      for (var c = 0; c <= r; c++) {
        var sum = precision[r][c];
        for (var k = 0; k < c; k++) sum -= chol[r][k] * chol[c][k];
        if (r === c) {
          if (!(sum > 0) || !Number.isFinite(sum)) throw new Error('BLR precision is not positive definite');
          chol[r][c] = Math.sqrt(sum);
        } else chol[r][c] = sum / chol[c][c];
      }
    }
    var temp = new Float64Array(K), weights = new Float64Array(K);
    for (var a = 0; a < K; a++) {
      var v = rhs[a];
      for (var b = 0; b < a; b++) v -= chol[a][b] * temp[b];
      temp[a] = v / chol[a][a];
    }
    for (var a2 = K - 1; a2 >= 0; a2--) {
      var v2 = temp[a2];
      for (var b2 = a2 + 1; b2 < K; b2++) v2 -= chol[b2][a2] * weights[b2];
      weights[a2] = v2 / chol[a2][a2];
    }
    var mu = pf.map(function(row) { return row.reduce(function(sum, x, i) { return sum + x * weights[i]; }, 0); });
    var offset = values[values.length - 1] - mu[0];
    return mu.map(function(x) { return x + offset; });
  }

  function drawNoise(provided, index, deterministic, size) {
    if (provided) {
      if (provided.length !== size || !Number.isFinite(provided[index])) throw new Error('Invalid fixed noise');
      return provided[index];
    }
    return deterministic ? 0 : randn();
  }

  async function predict(context, trajPattern, period, opts) {
    if (!isLoaded()) throw new Error('引擎未加载');
    opts = opts || {};

    var L = HP.context_length;
    var F = HP.prediction_length;
    var total = L + F;

    // 1) 准备上下文：edge 填充 / 截断到 L
    var transformed = transform(context, opts.bounds);
    var pair = condition(trajPattern === undefined ? 0 : trajPattern, period === undefined ? 0 : period);
    trajPattern = pair[0]; period = pair[1];
    var src = transformed.values;
    var ctx = new Float32Array(L);
    if (src.length < L) {
      var pad = L - src.length;
      for (var i = 0; i < pad; i++) ctx[i] = src[0];
      for (var j = 0; j < src.length; j++) ctx[pad + j] = src[j];
    } else {
      for (var k = 0; k < L; k++) ctx[k] = src[src.length - L + k];
    }

    // 2) StdScaler: temporal statistics, matching StdScaler(dim=1) on (1,L,1).
    var loc = 0;
    for (var i1 = 0; i1 < L; i1++) loc += ctx[i1];
    loc /= L;
    var variance = 0;
    for (var v1 = 0; v1 < L; v1++) variance += (ctx[v1] - loc) ** 2;
    var scale = Math.max(Math.sqrt(variance / L), HP.min_scale);

    // 3) 条件嵌入 (1,128)
    var condEmb = HP.use_condition ? await _encodeCondition(trajPattern, period) : new Float32Array(HP.cond_dim);

    var scaledPrior = Array.from(ctx, function(x) { return (x - loc) / scale; });
    var mu = priorMean(scaledPrior, trajPattern, period);
    var noise = opts.noise || {};

    // 5) 构建 x0：上下文 = past_target/scale（与原模型一致，不减 loc）
    //           未来先验 = mu + 噪声
    var x0 = new Float32Array(total);
    for (var i3 = 0; i3 < L; i3++) x0[i3] = ctx[i3] / scale;
    for (var i4 = 0; i4 < F; i4++) {
      var priorNoise = HP.noise_scale * drawNoise(noise.blr, i4, opts.deterministic, F);
      x0[L + i4] = mu[i4] + priorNoise;
    }
    // Additional flow noise uses sigmax = sigmin; BLR noise above stays separate.
    for (var i5 = 0; i5 < total; i5++) {
      x0[i5] = Math.fround(x0[i5] + Math.fround(HP.flow_sigma * Math.fround(drawNoise(noise.flow, i5, opts.deterministic, total))));
    }

    // 6) 欧拉 ODE（denoise_steps 步）
    //    Float32 Euler states match the ONNX Python pipeline.
    var dt = 1.0 / HP.denoise_steps;
    var xt = Float32Array.from(x0);
    if (opts.trace) Object.assign(opts.trace, { transformed_context: Array.from(ctx), loc: loc, scale: scale,
      prior_mean: mu, condition: Array.from(condEmb), x0: Array.from(x0), velocities: [], states: [] });
    for (var step = 0; step < HP.denoise_steps; step++) {
      var t = step * dt;
      var vt = await _backboneVelocity(t, Float32Array.from(xt), condEmb);
      for (var i6 = 0; i6 < total; i6++) xt[i6] = Math.fround(xt[i6] + Math.fround(dt * vt[i6]));
      if (opts.trace) { opts.trace.velocities.push(Array.from(vt)); opts.trace.states.push(Array.from(xt)); }
    }

    // 7) 反归一化 + 取出未来段
    var pred = new Float64Array(F);
    for (var i7 = 0; i7 < F; i7++) pred[i7] = Math.fround(Math.fround(xt[L + i7] * scale) + loc) * transformed.span + transformed.lo;
    if (Array.from(pred).some(function(x) { return !Number.isFinite(x); })) throw new Error('Nonfinite prediction');
    return Array.from(pred);
  }

  return {
    load: load,
    ready: ready,
    isLoaded: isLoaded,
    getError: getError,
    predict: predict,
    HP: HP,
    configReady: configReady,
    datasetBounds: datasetBounds,
    priorMean: priorMean,
  };
})();
