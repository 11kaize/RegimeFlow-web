/* Real ONNX Runtime Web WASM verification, also runnable offline under Node.
   This checks the browser runtime build, but does not replace a browser UI run.
   Run from any directory: node scripts/cross-check/check.cjs */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { pathToFileURL } = require('node:url');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '../..');
const output = path.resolve(root, '../Materials/Validation/InferenceParity');
const ort = require(path.join(root, 'web/vendor/ort.min.js'));
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = {
  mjs: pathToFileURL(path.join(root, 'web/vendor/ort-wasm-simd-threaded.mjs')).href,
  wasm: pathToFileURL(path.join(root, 'web/vendor/ort-wasm-simd-threaded.wasm')).href,
};
ort.env.wasm.wasmBinary = fs.readFileSync(path.join(root, 'web/vendor/ort-wasm-simd-threaded.wasm'));
global.window = global;
global.ort = ort;
vm.runInThisContext(fs.readFileSync(path.join(root, 'web/js/engine-web.js'), 'utf8'));
const config = JSON.parse(fs.readFileSync(path.join(root, 'web/models/inference-config.json')));
function flat(value) { return Array.isArray(value) ? value.flat(Infinity) : Array.from(value); }
function maxDiff(a, b) {
  a = flat(a); b = flat(b);
  assert.equal(a.length, b.length);
  return Math.max(...a.map((x, i) => Math.abs(x - b[i])));
}
(async () => {
  assert.equal(await RegimeFlowWeb.load({
    backbone: fs.readFileSync(path.join(root, 'web/models/backbone.onnx')),
    condEncoder: fs.readFileSync(path.join(root, 'web/models/cond_encoder.onnx')),
  }, { config }), true, RegimeFlowWeb.getError());
  const fixtures = JSON.parse(fs.readFileSync(path.join(output, 'browser_fixtures.json')));
  const cases = [];
  for (const sample of fixtures) {
    const trace = {};
    const prediction = await RegimeFlowWeb.predict(sample.context, sample.pattern, sample.period,
      { bounds: sample.bounds, noise: sample.noise, trace });
    const span = sample.bounds[1] - sample.bounds[0];
    const result = {
      id: sample.id, pattern: sample.pattern,
      prior_max_abs_error: maxDiff(trace.prior_mean, sample.trace.prior_mean),
      condition_max_abs_error: maxDiff(trace.condition, sample.trace.condition),
      x0_max_abs_error: maxDiff(trace.x0, sample.trace.x0),
      velocity_max_abs_error: maxDiff(trace.velocities, sample.trace.velocities),
      state_max_abs_error: maxDiff(trace.states, sample.trace.states),
      prediction_max_abs_error_in_range_units: maxDiff(prediction, sample.prediction) / span,
    };
    for (const key of Object.keys(result).filter(key => key.includes('error'))) {
      assert.ok(result[key] < 1e-4, `${sample.id}: ${key}=${result[key]}`);
    }
    cases.push(result);
    console.log(result);
  }
  const report = { runtime: 'onnxruntime-web WASM under Node (not a browser UI)', versions: ort.env.versions, cases };
  fs.writeFileSync(path.join(output, 'wasm_validation.json'), JSON.stringify(report, null, 2) + '\n');
})().catch(error => { console.error(error); process.exitCode = 1; });
