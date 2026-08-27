/* ================================================================
   RegimeFlow — Figure 3 radar chart (About page)
   MSE comparison between Ours and five competitive baselines across
   the ten most frequent biological system taxonomies.

   Values are reconstructed from the paper's Figure 3 vector geometry
   (radar polygon vertex radii → MSE via the printed y-axis ticks).
   Lower is better; the radius is proportional to MSE, so "Ours" is
   the innermost (smallest) polygon.
   ================================================================ */

var FIG3_TAX = [
  "Rattus norvegicus", "Mus musculus", "Unspecified", "Mammalia",
  "Arabidopsis thaliana", "Homo sapiens", "Trypanosoma brucei",
  "Xenopus laevis", "cellular organisms", "Saccharomyces cerevisiae"
];

var FIG3_METHODS = [
  { name: "Ours",     color: "#D05860", data: [0.014, 0.015, 0.006, 0.009, 0.020, 0.006, 0.005, 0.006, 0.013, 0.014] },
  { name: "TSFlow",   color: "#4078C8", data: [0.017, 0.019, 0.009, 0.012, 0.026, 0.011, 0.009, 0.013, 0.018, 0.017] },
  { name: "TSDiff",   color: "#70C868", data: [0.036, 0.042, 0.018, 0.028, 0.073, 0.023, 0.013, 0.031, 0.031, 0.036] },
  { name: "NSformer", color: "#B078C0", data: [0.030, 0.032, 0.014, 0.021, 0.034, 0.024, 0.027, 0.022, 0.041, 0.030] },
  { name: "PatchTST", color: "#D6C696", data: [0.033, 0.049, 0.022, 0.033, 0.058, 0.039, 0.037, 0.048, 0.060, 0.033] },
  { name: "DLinear",  color: "#70B8D8", data: [0.054, 0.071, 0.041, 0.050, 0.072, 0.060, 0.053, 0.073, 0.081, 0.054] }
];

/**
 * Initialize (or resize) the Figure 3 radar chart. Safe to call repeatedly;
 * only builds the chart on first call so it works when the About view is
 * hidden (display:none) at page load.
 */
function initFigure3Chart() {
  var el = document.getElementById('figure3-chart');
  if (!el || typeof echarts === 'undefined') return;

  if (window._figure3Chart) {
    window._figure3Chart.resize();
    return;
  }

  var chart = echarts.init(el);
  window._figure3Chart = chart;

  // Draw outer polygons first so the innermost "Ours" stays on top.
  var drawOrder = ["DLinear", "PatchTST", "TSDiff", "NSformer", "TSFlow", "Ours"];
  var byName = {};
  FIG3_METHODS.forEach(function (m) { byName[m.name] = m; });

  chart.setOption({
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      backgroundColor: '#1a2a3a', borderColor: '#2a4057', textStyle: { color: '#e0e6ed' }
    },
    legend: {
      bottom: 0,
      textStyle: { color: '#c0d0e0', fontSize: 11 },
      itemWidth: 14, itemHeight: 10
    },
    radar: {
      indicator: FIG3_TAX.map(function (t) { return { name: t, max: 0.09 }; }),
      center: ['50%', '50%'],
      radius: '58%',
      splitNumber: 3,
      axisName: { color: '#c0d0e0', fontSize: 10 },
      splitLine: { lineStyle: { color: '#2a4057' } },
      splitArea: { show: false },
      axisLine: { lineStyle: { color: '#2a4057' } }
    },
    series: [{
      type: 'radar',
      symbol: 'circle',
      symbolSize: 3,
      data: drawOrder.map(function (nm) {
        var m = byName[nm];
        var isOurs = m.name === 'Ours';
        return {
          value: m.data,
          name: m.name,
          itemStyle: { color: m.color },
          lineStyle: { color: m.color, width: isOurs ? 3 : 1.5 },
          areaStyle: { color: m.color, opacity: isOurs ? 0.16 : 0.05 }
        };
      })
    }]
  });
}
