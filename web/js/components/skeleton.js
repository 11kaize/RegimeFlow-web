/* ================================================================
   RegimeFlow — Chart Skeleton Screen Component
   Lightweight loading placeholder shown while CSV trajectory
   data is fetched from HuggingFace. Replaces ECharts content
   with an animated shimmer skeleton.

   Public API:
     ChartSkeleton.show(model)   — show skeleton (model: {name, id, species})
     ChartSkeleton.hide()        — fade out & remove skeleton
     ChartSkeleton.ensureMin(cb) — resolve cb after minimum display time
     ChartSkeleton.clear()       — remove skeleton immediately, no animation

   Uses:
     - window._predictChart (ECharts instance, if present)
     - #predict-chart DOM element
   ================================================================ */

var ChartSkeleton = (function() {
  'use strict';

  var timer    = null;
  var start    = 0;
  var minMs    = 350;
  var chartEl  = null;

  // ── Internal: build skeleton DOM ──────────────────────────────
  function buildDom(model) {
    var waves = '';
    for (var w = 1; w <= 5; w++) {
      waves += '<div class="chart-skeleton-wave"></div>';
    }
    for (var b = 1; b <= 4; b++) {
      waves += '<div class="chart-skeleton-bump"></div>';
    }

    var modelName = model ? model.name : '';
    var modelMeta = model ? (model.id + ' · ' + model.species + ' species') : '';

    return '' +
      '<div class="chart-skeleton-loading-text">' +
        '<div class="sk-name">🔄 Loading ' + modelName + '…</div>' +
        (modelMeta ? '<div class="sk-meta">' + modelMeta + '</div>' : '') +
      '</div>' +
      '<div class="chart-skeleton-waves">' +
        waves +
        '<div class="chart-skeleton-axis"></div>' +
      '</div>';
  }

  // ── Public ────────────────────────────────────────────────────

  function show(model) {
    if (!chartEl) chartEl = document.getElementById('predict-chart');
    if (!chartEl) return;

    clear();           // remove any existing skeleton
    start = Date.now();

    var skel = document.createElement('div');
    skel.className = 'chart-skeleton';
    skel.id = 'chart-skeleton';
    skel.innerHTML = buildDom(model);
    chartEl.appendChild(skel);

    // Clear ECharts so no stale content shows through
    if (window._predictChart) {
      window._predictChart.setOption({
        backgroundColor: 'transparent',
        title:  { text: '' },
        xAxis:  { show: false },
        yAxis:  { show: false },
        series: [],
        animation: false
      }, true);
    }
  }

  function hide() {
    var skel = document.getElementById('chart-skeleton');
    if (!skel) return;
    clearTimeout(timer);

    skel.classList.add('fading-out');
    timer = setTimeout(function() {
      if (skel.parentNode) skel.remove();
    }, 350);
  }

  /**
   * Calls callback after at least `minMs` ms from when skeleton was shown.
   * Resolves immediately if min display time already elapsed.
   */
  function ensureMin(callback) {
    var elapsed   = Date.now() - start;
    var remaining = Math.max(0, minMs - elapsed);
    if (remaining <= 0) {
      callback();
    } else {
      timer = setTimeout(callback, remaining);
    }
  }

  function clear() {
    var skel = document.getElementById('chart-skeleton');
    if (skel) skel.remove();
    clearTimeout(timer);
  }

  // Expose public API
  return {
    show:      show,
    hide:      hide,
    ensureMin: ensureMin,
    clear:     clear
  };
})();
