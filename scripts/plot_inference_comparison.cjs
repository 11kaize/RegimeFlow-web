/* Standalone diagnostic plot; never loaded by the production website. */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const d3 = require('../web/vendor/d3.min.js');
const out = path.resolve(__dirname, '../../Materials/Validation/InferenceParity');
const baseline = JSON.parse(fs.readFileSync(path.join(out, 'baseline_predictions.json'))).samples;
const fixed = new Map(JSON.parse(fs.readFileSync(path.join(out, 'final_predictions.json'))).map(s => [s.id, s]));
const metrics = d3.csvParse(fs.readFileSync(path.join(out, 'stage_metrics.csv'), 'utf8'));
const ids = ['Calzone2007/MPFc', 'Calzone2007/Xp', 'Alvarez2019/T_1', 'Oxhamre2005/var_1'];
const esc = s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;');
const width = 1100, height = 700;
const svg = [`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
  '<rect width="100%" height="100%" fill="white"/>',
  '<g font-family="Arial,sans-serif" font-size="12" fill="#222">',
  '<text x="35" y="28" font-size="19">RegimeFlow: fixed-noise trajectory diagnostics (4 steps)</text>',
  '<text x="35" y="49">15 systems / 34 species; four illustrative species shown. Diagnostic samples; test membership unverified.</text>',
  '<text x="35" y="76" fill="#666">History</text><text x="110" y="76">Target</text>',
  '<text x="175" y="76" fill="#c54c4c">Before repair</text><text x="285" y="76" fill="#1768bc">After repair</text>'];
for (const [i, id] of ids.entries()) {
  const sample = baseline.find(s => s.id === id);
  if (!sample || !fixed.has(id)) throw new Error('Missing plot sample: ' + id);
  const span = sample.bounds[1] - sample.bounds[0];
  const normalize = value => (value - sample.bounds[0]) / span;
  const history = sample.context.map(normalize);
  const target = sample.target.map(v => normalize(Math.max(0, Math.min(1e9, Math.fround(v)))));
  const before = sample.prediction.map(normalize);
  const after = fixed.get(id).prediction.map(normalize);
  const left = 65 + (i % 2) * 540, top = 126 + Math.floor(i / 2) * 280;
  const x = d3.scaleLinear().domain([0, history.length + target.length - 1]).range([0, 460]);
  const y = d3.scaleLinear().domain(d3.extent([...history, ...target, ...before, ...after])).nice().range([200, 0]);
  const nmae = stage => Number(metrics.find(m => m.system === sample.system && m.species === sample.species && m.stage === stage).nmae).toFixed(4);
  svg.push(`<g transform="translate(${left},${top})"><text y="-22" font-size="15">${esc(id)}</text><text y="-6" fill="#555">nMAE: ${nmae('baseline')} → ${nmae('final')}</text>`);
  for (const tick of y.ticks(5)) svg.push(`<path d="M0,${y(tick)}H460" stroke="#e6e6e6"/><text x="-8" y="${y(tick)+4}" text-anchor="end">${d3.format('.2~g')(tick)}</text>`);
  for (const tick of x.ticks(5)) svg.push(`<text x="${x(tick)}" y="220" text-anchor="middle">${tick}</text>`);
  svg.push(`<path d="M${x(96)},0V200" stroke="#aaa" stroke-dasharray="4,4"/>`);
  const line = (values, offset, color, strokeWidth) => svg.push(`<path d="${d3.line().x((_, j) => x(j+offset)).y(v => y(v))(values)}" fill="none" stroke="${color}" stroke-width="${strokeWidth}"/>`);
  line(before, 96, '#c54c4c', 1.1); line(after, 96, '#1768bc', 1.3); line(target, 96, '#111', 1.5); line(history, 0, '#777', 1.5);
  svg.push('<text x="230" y="243" text-anchor="middle">Sample index</text><text transform="translate(-45,100) rotate(-90)" text-anchor="middle">Value / full-series range</text></g>');
}
svg.push('<text x="35" y="684" fill="#555">Same checkpoint and noise arrays before/after. No smoothing or post-hoc nonnegativity constraint.</text></g></svg>');
const content = svg.join('\n');
if (/NaN|Infinity/.test(content)) throw new Error('Nonfinite plot coordinate');
fs.writeFileSync(path.join(out, 'trajectory_comparison.svg'), content);
console.log('Wrote trajectory_comparison.svg');
