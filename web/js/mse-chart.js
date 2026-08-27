/* ================================================================
   RegimeFlow — MSE benchmark bar chart (About page)
   Full Table 1 Overall MSE. Point & Zero-Shot methods have a single
   bar each; probabilistic methods have two bars (w/o vs w/ regime).
   The legend lives in index.html as static HTML (so its full-length
   labels and swatches always render, independent of this script).
   ================================================================ */

var MSE_CHART_CATS = [
  "DLinear", "iTransformer", "PatchTST", "TimeMixer", "TimeXer", "SMamba", "BiMamba4TS", "NSformer",
  "TimeMoE", "Sundial", "TimesFM", "Chronos",
  "CSDI", "TSDiff", "TSFlow", "RegimeFlow"
];

var MSE_CHART_SERIES = [
  { name: "Point",      fullName: "Point Forecasting",          color: "#4A90D9",
    data: [0.061, 0.045, 0.041, 0.051, 0.043, 0.045, 0.045, 0.028, null, null, null, null, null, null, null, null] },
  { name: "Zero-Shot",  fullName: "Zero-Shot TSFM",             color: "#9B59B6",
    data: [null, null, null, null, null, null, null, null, 0.120, 0.111, 0.099, 0.097, null, null, null, null] },
  { name: "w/o Regime", fullName: "Probabilistic w/o Regime",   color: "#6a8299",
    data: [null, null, null, null, null, null, null, null, null, null, null, null, 0.099, 0.042, 0.026, 0.023] },
  { name: "w/ Regime",  fullName: "Probabilistic w/ Regime",    color: "#F39C12", highlight: true,
    data: [null, null, null, null, null, null, null, null, null, null, null, null, 0.076, 0.034, 0.016, 0.012] }
];

var MSE_FULL_NAMES = {};
MSE_CHART_SERIES.forEach(function (s) { MSE_FULL_NAMES[s.name] = s.fullName; });

/**
 * Initialize (or resize) the MSE benchmark chart. Safe to call repeatedly;
 * only builds the chart on first call so it works when the About view is
 * hidden (display:none) at page load.
 */
function initMseChart() {
  var el = document.getElementById('mse-chart');
  if (!el || typeof echarts === 'undefined') return;

  if (window._mseChart) {
    window._mseChart.resize();
    return;
  }

  var chart = echarts.init(el);
  window._mseChart = chart;

  chart.setOption({
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: function (p) {
        var s = '<b>' + MSE_CHART_CATS[p[0].dataIndex] + '</b>';
        p.forEach(function (item) {
          if (item.value != null) {
            s += '<br/>' + item.marker + (MSE_FULL_NAMES[item.seriesName] || item.seriesName) + ': <b>' + item.value.toFixed(3) + '</b>';
          }
        });
        return s;
      },
      backgroundColor: '#1a2a3a', borderColor: '#2a4057', textStyle: { color: '#e0e6ed' }
    },
    grid: { left: 55, right: 20, top: 32, bottom: 72 },
    xAxis: {
      type: 'category',
      data: MSE_CHART_CATS,
      axisLabel: { color: '#c0d0e0', rotate: 35, fontSize: 12, interval: 0 },
      axisLine: { lineStyle: { color: '#2a4057' } }
    },
    yAxis: {
      type: 'value',
      name: 'MSE',
      nameLocation: 'middle',
      nameGap: 46,
      nameTextStyle: { color: '#8899aa' },
      axisLabel: { color: '#8899aa' },
      splitLine: { lineStyle: { color: '#1e3045' } }
    },
    series: MSE_CHART_SERIES.map(function (s) {
      var out = {
        name: s.name,
        type: 'bar',
        barMaxWidth: 34,
        data: s.data,
        itemStyle: { color: s.color, borderRadius: [3, 3, 0, 0] }
      };
      if (s.highlight) {
        out.label = {
          show: true, position: 'top', color: s.color, fontSize: 11,
          formatter: function (p) { return p.value != null ? p.value.toFixed(3) : ''; }
        };
      }
      return out;
    })
  });
}
