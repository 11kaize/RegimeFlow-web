/* Run: node --test tests/test_web_preprocessing.cjs
   Zero-velocity ONNX stubs isolate source initialization and denormalization. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

async function makeEngine() {
  const captured = [];
  class Tensor {
    constructor(type, data, dims) { Object.assign(this, { type, data, dims }); }
  }
  const ort = {
    Tensor,
    InferenceSession: {
      async create(url) {
        return url === 'condition' ? {
          outputNames: ['condition'],
          async run() { return { condition: { data: new Float32Array(128) } }; },
        } : {
          outputNames: ['velocity'],
          async run(feeds) {
            captured.push(Float32Array.from(feeds.x_in.data));
            return { velocity: { data: new Float32Array(352) } };
          },
        };
      },
    },
  };
  let randomIndex = 0;
  const controlledMath = Object.create(Math);
  // Box-Muller returns +1: sqrt(2) * cos(pi/4), with both uniforms in (0,1).
  controlledMath.random = () => randomIndex++ % 2 === 0 ? Math.exp(-1) : .125;
  const sandbox = { ort, Math: controlledMath, console };
  sandbox.window = sandbox;
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../web/js/engine-web.js'), 'utf8'), sandbox);
  const engine = sandbox.RegimeFlowWeb;
  assert.equal(await engine.load({ backbone: 'backbone', condEncoder: 'condition' }, { config: JSON.parse(fs.readFileSync(path.join(__dirname, '../web/models/inference-config.json'), 'utf8')) }), true);
  return { engine, captured };
}

function near(actual, expected, tolerance = 1e-6) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
}

test('temporal statistics preserve uncentered context and denormalize forecast', async () => {
  const { engine, captured } = await makeEngine();
  const ctx = Array.from({ length: 96 }, (_, i) => i % 2 ? 4 : 0);
  const pred = await engine.predict(ctx, 0, 0, { deterministic: true });
  near(captured[0][0], 0);
  near(captured[0][1], 1); // minmax first, then loc=.5 and scale=1.
  near(captured[0][96], .5);
  assert.equal(pred.length, 256);
  pred.forEach(value => near(value, 4));
});

test('temporal standard deviation is clamped for small-amplitude contexts', async () => {
  const { engine, captured } = await makeEngine();
  const ctx = Array.from({ length: 96 }, (_, i) => i % 2 ? 3.5 : 3);
  const pred = await engine.predict(ctx, 0, 0, { deterministic: true });
  near(captured[0][0], 0); // minmax [0,1], temporal std=.5 is clamped to 1.
  near(captured[0][1], 1);
  pred.forEach(value => near(value, 3.5));
});

test('additional flow noise is .001 while BLR noise remains .1', async () => {
  const { engine, captured } = await makeEngine();
  const pred = await engine.predict(Array(96).fill(5), 0, 0, { bounds: [0, 10] });
  near(captured[0][0], 0.501);
  near(captured[0][96], .101);
  pred.forEach(value => near(value, 6.01));
});
