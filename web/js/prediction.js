/* ================================================================
   RegimeFlow — Prediction View
   Pathway-organised bio model trajectory browser + Chronos-Bolt
   prediction (placeholder until AI weights arrive).

   Modes:
     Pathway Models tab — browse real BIOMD models by pathway,
        load HuggingFace CSV, view context / ground-truth split,
        prediction placeholder.
     Custom Data tab    — paste own time-series, run prediction.
   ================================================================ */

// ── Backend config ─────────────────────────────────────────────
// 三种模式自动切换:
//   1. Render 部署   — 前后端同源，API_BASE = ''（空字符串 = 相对路径）
//   2. 本地 dev 服务器 — python server.py 启动，同源访问，API_BASE = ''
//   3. 本地文件打开   — file:// 协议，连接到本地 localhost:8000
// 如需连接远程 Render 后端，将 RENDER_URL 设为你的 Render 服务地址:
//   例如: var RENDER_URL = 'https://regimeflow-web.onrender.com';
var RENDER_URL = 'https://regimeflow-web.onrender.com';
var API_BASE = (function() {
  if (RENDER_URL) return RENDER_URL;
  // 同源部署（Render 生产 / 本地 server.py）→ 使用相对路径
  if (window.location.protocol === 'http:' || window.location.protocol === 'https:') return '';
  // file:// 协议打开 → 默认连接本地后端
  return 'http://localhost:8000';
})();

// ── State ──────────────────────────────────────────────────────
var _pwState = {
  pathwayIdx:    -1,      // expanded pathway index, -1 = none
  modelIdx:      -1,      // selected model within pathway, -1 = none
  spIdx:         0,       // species index
  spNames:       [],      // species names from loaded CSV
  ctxLen:        96,      // context length
  ctxData:       null,    // context data array (current species)
  gtData:        null,    // ground-truth data array (current species)
  gtTime:        null,    // ground-truth time array
  ctxTime:       null,    // context time array
  fullColumns:   null,    // all parsed CSV columns (cached for species switching)
  fullTime:      null,    // full time array from CSV
  currentModel:  null,    // currently loaded pathway model
  loading:       false
};

var _backendStatus = { regimeflow_loaded: false, chronos_loaded: false, device: 'cpu', denoise_steps: 0 };
var _bioTrajectoryLoaded = false;  // legacy flag for backward compat
var _failedModels = {};  // key: 'pi-mi' → true for models that returned 404/500
var _activeViewMode = 'pathway';  // 'pathway' | 'bio' — which view currently owns the chart

// ================================================================
// ChartError — clean error-state overlay for the prediction chart
// ================================================================
var ChartError = (function() {
  'use strict';

  function _clearDOM() {
    var old = document.getElementById('chart-error-state');
    if (old) old.remove();
  }

  function _clearECharts() {
    if (!window._predictChart) return;
    window._predictChart.setOption({
      backgroundColor: 'transparent',
      title: { text: '' },
      xAxis: { show: false },
      yAxis: { show: false },
      series: [],
      animation: false
    }, true);
  }

  function show(model, errMessage) {
    var el = document.getElementById('predict-chart');
    if (!el) return;

    _clearDOM();
    _clearECharts();
    // Hide legend guide on error
    var guide = document.getElementById('chart-legend-guide');
    if (guide) guide.style.display = 'none';

    var overlay = document.createElement('div');
    overlay.className = 'chart-error-state';
    overlay.id = 'chart-error-state';

    var detail = errMessage || 'Unknown error';
    var is404 = detail.indexOf('404') !== -1;
    var icon = is404 ? '📭' : '⚠';
    var title = is404
      ? 'Data not available for ' + model.name
      : 'Failed to load ' + model.name;
    var hint = is404
      ? 'This model\'s trajectory data is not yet available<br>in the HuggingFace dataset.'
      : 'An unexpected error occurred while fetching<br>the trajectory data.';

    overlay.innerHTML =
      '<div class="chart-error-inner">' +
        '<div class="chart-error-icon">' + icon + '</div>' +
        '<div class="chart-error-title">' + title + '</div>' +
        '<div class="chart-error-detail">' +
          hint +
          '<br><code>' + detail.replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</code>' +
        '</div>' +
        '<div class="chart-error-retry" onclick="retryCurrentModel()">↻ Try again</div>' +
      '</div>';

    el.appendChild(overlay);
  }

  function clear() {
    _clearDOM();
  }

  return { show: show, clear: clear };
})();

// ── Retry helper (exposed globally for onclick) ──────────────────
function retryCurrentModel() {
  if (_pwState.currentModel) {
    ChartError.clear();
    loadPathwayTrajectory(_pwState.currentModel);
  }
}

// ── i18n helpers ────────────────────────────────────────────────
function _pwSysName(m) { return m.name; }

// ── Error badge helpers ──────────────────────────────────────────
function markModelFailed(pi, mi) {
  var key = pi + '-' + mi;
  _failedModels[key] = true;
  var badge = document.getElementById('pw-errbadge-' + pi + '-' + mi);
  if (badge) badge.classList.add('visible');
  var item = document.getElementById('pw-model-' + pi + '-' + mi);
  if (item) item.classList.add('has-error');
}

function markModelOk(pi, mi) {
  var key = pi + '-' + mi;
  delete _failedModels[key];
  var badge = document.getElementById('pw-errbadge-' + pi + '-' + mi);
  if (badge) badge.classList.remove('visible');
  var item = document.getElementById('pw-model-' + pi + '-' + mi);
  if (item) item.classList.remove('has-error');
}

// ================================================================
// initPrediction — entry point
// ================================================================
function initPrediction() {
  if (typeof PATHWAY_MODELS === 'undefined') return;

  // Check backend
  fetch(API_BASE + '/api/health').then(function(r) { return r.json(); })
    .then(function(s) { _backendStatus = s; updateSidebarNote(); })
    .catch(function() {});

  // ── Render pathway list ──
  var listEl = document.getElementById('system-list');
  if (!listEl) return;

  listEl.innerHTML = '<div style="font-size:11px;color:var(--color-text-muted);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px;">Biological Pathways</div>';

  PATHWAY_MODELS.forEach(function(pw, pi) {
    var item = document.createElement('div');
    item.className = 'pw-category';
    item.innerHTML =
      '<div class="pw-cat-header">' +
        '<span class="pw-cat-icon">' + (pw.icon || '📁') + '</span>' +
        '<span class="pw-cat-name">' + pw.pathway + '</span>' +
        '<span class="pw-cat-count">' + pw.models.length + '</span>' +
      '</div>' +
      '<div class="pw-cat-desc">' + (pw.desc || '') + '</div>';
    item.addEventListener('click', function() { togglePathway(pi); });
    listEl.appendChild(item);

    // Model list (hidden until expanded)
    var modelList = document.createElement('div');
    modelList.className = 'pw-model-list';
    modelList.id = 'pw-models-' + pi;
    pw.models.forEach(function(m, mi) {
      var mel = document.createElement('div');
      mel.className = 'pw-model-item';
      mel.id = 'pw-model-' + pi + '-' + mi;
      var l1 = REGIME_TO_L1[m.regime] || 'stable';
      var l1def = REGIME_L1_DEFS[l1] || {};
      mel.innerHTML =
        '<span class="pw-model-name">' + m.name + '</span>' +
        '<span class="pw-model-error-badge" id="pw-errbadge-' + pi + '-' + mi + '">⚠</span>' +
        '<span class="pw-model-id">' + m.id + '</span>' +
        '<span class="pw-model-tag" style="color:' + (l1def.color||'#8899aa') + '">' + (l1def.icon||'') + ' ' + (l1def.label||m.regime) + '</span>' +
        (m.note ? '<span class="pw-model-note" title="' + m.note.replace(/"/g,'&quot;') + '">' + m.note + '</span>' : '');
      mel.addEventListener('click', function(e) {
        e.stopPropagation();
        selectPathwayModel(pi, mi);
      });
      modelList.appendChild(mel);
    });
    item.appendChild(modelList);
  });

  // Total count
  var totalDiv = document.createElement('div');
  totalDiv.style.cssText = 'font-size:10px;color:var(--color-text-muted);text-align:center;padding:8px 0;';
  totalDiv.textContent = PATHWAY_TOTAL_MODELS + ' pathway models';
  listEl.appendChild(totalDiv);

  // ── Species selector ──
  var selEl = document.getElementById('species-select');
  if (selEl) {
    selEl.innerHTML = '<option value="0">— Select a model first —</option>';
    selEl.addEventListener('change', function() {
      var spIdx = parseInt(selEl.value) || 0;
      if (_activeViewMode === 'bio') {
        // Multi-line view: highlight one species, dim others
        _bioChartState.selSpIdx = spIdx;
        highlightBioSpecies(spIdx);
      } else {
        // Single-species pathway view: switch to that species
        _pwState.spIdx = spIdx;
        redrawPathwayChart();
      }
    });
  }

  // ── Chart ──
  if (!window._predictChart) {
    window._predictChart = echarts.init(document.getElementById('predict-chart'));
  }
  window.addEventListener('resize', debounce(function() {
    if (window._predictChart) window._predictChart.resize();
  }, 200));

  // Show initial prompt
  if (window._predictChart) {
    window._predictChart.setOption({
      backgroundColor: 'transparent',
      title: { text: 'Select a pathway model', subtext: 'to view its trajectory', left: 'center', top: '38%',
        textStyle: { color: '#6a8299', fontSize: 16 }, subtextStyle: { color: '#4a5f73', fontSize: 12 } },
      xAxis: { show: false }, yAxis: { show: false }, series: []
    });
  }

  // ── Tab switching ──
  document.querySelectorAll('.input-tab').forEach(function(tab) {
    tab.addEventListener('click', function() {
      document.querySelectorAll('.input-tab').forEach(function(t) { t.classList.remove('active'); });
      tab.classList.add('active');
      var showPathway = tab.dataset.tab === 'examples';
      document.getElementById('system-list').style.display = showPathway ? '' : 'none';
      document.getElementById('custom-input-panel').style.display = showPathway ? 'none' : '';
      if (showPathway) {
        setTimeout(function() { if (window._predictChart) window._predictChart.resize(); }, 100);
      }
    });
  });

  // ── Custom predict button ──
  var cpb = document.getElementById('btn-custom-predict');
  if (cpb) cpb.addEventListener('click', handleCustomPredict);

  // ── Sidebar note ──
  updateSidebarNote();

  // ── Lang change ──
  window.addEventListener('langchange', function() { updateSidebarNote(); });
}

// ================================================================
// Pathway browsing
// ================================================================
function togglePathway(pi) {
  var wasOpen = (_pwState.pathwayIdx === pi);
  // Close all
  document.querySelectorAll('.pw-category').forEach(function(el) { el.classList.remove('open'); });
  document.querySelectorAll('.pw-model-list').forEach(function(el) { el.classList.remove('open'); });
  document.querySelectorAll('.pw-model-item').forEach(function(el) { el.classList.remove('active'); });

  if (wasOpen) {
    _pwState.pathwayIdx = -1;
    return;
  }

  _pwState.pathwayIdx = pi;
  var catEl = document.querySelectorAll('.pw-category')[pi];
  if (catEl) catEl.classList.add('open');
  var listEl = document.getElementById('pw-models-' + pi);
  if (listEl) listEl.classList.add('open');
}

function selectPathwayModel(pi, mi) {
  _pwState.pathwayIdx = pi;
  _pwState.modelIdx   = mi;
  _pwState.spIdx      = 0;
  _bioTrajectoryLoaded = false;

  var model = PATHWAY_MODELS[pi].models[mi];
  _pwState.currentModel = model;

  // Clear any previous error state before loading new model
  ChartError.clear();

  // Highlight
  document.querySelectorAll('.pw-model-item').forEach(function(el) { el.classList.remove('active'); });
  var mel = document.getElementById('pw-model-' + pi + '-' + mi);
  if (mel) mel.classList.add('active');

  // Update regime tag
  var tagEl = document.getElementById('regime-tag');
  if (tagEl) {
    var l1 = REGIME_TO_L1[model.regime] || 'stable';
    var l1def = REGIME_L1_DEFS[l1] || {};
    tagEl.textContent = (l1def.icon||'') + ' ' + (l1def.label||model.regime);
    tagEl.className = 'regime-tag regime-' + model.regime.replace(/_/g, '-');
  }

  // Load trajectory
  loadPathwayTrajectory(model);
}

// ================================================================
// Load pathway trajectory from HuggingFace
// ================================================================
function loadPathwayTrajectory(model) {
  if (_pwState.loading) return;
  _pwState.loading = true;

  _activeViewMode = 'pathway';

  // Clear any stale error overlay before loading
  ChartError.clear();

  // Update sidebar note
  var noteEl = document.querySelector('.sidebar-note');
  if (noteEl) {
    noteEl.innerHTML = '🔄 Loading ' + model.name + '…';
    noteEl.style.background = 'var(--color-tag-yellow, #1a1a0a)';
    noteEl.style.borderColor = '#332e15';
    noteEl.style.color = '#998844';
  }

  ChartSkeleton.show(model);

  var csvUrl = 'https://huggingface.co/datasets/HengRao/SysBio-Traj/resolve/main/Data/'
    + model.id + '/' + model.name + '.csv';

  fetch(csvUrl).then(function(resp) {
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    return resp.text();
  }).then(function(csvText) {
    // Parse CSV
    var lines = csvText.trim().split('\n');
    var headers = lines[0].split(',').map(function(h) { return h.trim(); });
    var columns = {};
    headers.forEach(function(h) { columns[h] = []; });

    for (var i = 1; i < lines.length; i++) {
      var vals = lines[i].split(',');
      headers.forEach(function(h, j) { columns[h].push(parseFloat(vals[j])); });
    }

    var time    = columns['time'] || [];
    var timeLen = time.length;
    var spNames = headers.filter(function(h) {
      return h !== 'time' && !/_(max|min|norm|mean|std|sum|avg)$/.test(h);
    });

    _pwState.spNames = spNames;
    _pwState.ctxLen  = Math.min(96, Math.floor(timeLen * 0.25));
    _pwState.fullColumns = columns;  // cache for species switching
    _pwState.fullTime    = time;

    // Split: first ctxLen points = context, rest = ground truth
    var ctxLen  = _pwState.ctxLen;
    var spIdx   = _pwState.spIdx;
    if (spIdx >= spNames.length) spIdx = 0;

    var spName  = spNames[spIdx];
    var spData  = columns[spName];
    var ctxTime = time.slice(0, ctxLen);
    var ctxData = spData.slice(0, ctxLen);
    var gtTime  = time.slice(ctxLen);
    var gtData  = spData.slice(ctxLen);

    _pwState.ctxTime = ctxTime;
    _pwState.ctxData = ctxData;
    _pwState.gtTime  = gtTime;
    _pwState.gtData  = gtData;
    _pwState.loading = false;

    // Clear error badge on success
    markModelOk(_pwState.pathwayIdx, _pwState.modelIdx);
    ChartError.clear();

    // Populate species selector
    var selEl = document.getElementById('species-select');
    if (selEl) {
      selEl.innerHTML = spNames.map(function(n, i) {
        return '<option value="' + i + '"' + (i === spIdx ? ' selected' : '') + '>' + n + '</option>';
      }).join('');
    }

    // Update sidebar note
    if (noteEl) {
      noteEl.innerHTML = '🧬 <b>' + model.name + '</b><br>' +
        '<span style="font-size:10px;color:#8899aa;">' + model.id + ' · ' + spNames.length +
        ' species · ' + timeLen + ' steps</span><br>' +
        '<span style="color:var(--color-text-highlight, #c0d0e0);">' +
        '📊 Context: ' + ctxLen + ' pts · GT: ' + gtData.length + ' pts</span>';
      noteEl.style.background = 'var(--color-tag-green, #1a3020)';
      noteEl.style.borderColor = '#2a4a30';
      noteEl.style.color = '#88aa88';
    }

    // Ensure skeleton shows for at least _skelMinMs, then render
    ChartSkeleton.ensureMin(function() {
      ChartSkeleton.hide();  // fade out skeleton
      // Small delay so fade starts before chart renders
      setTimeout(function() {
        drawChart(ctxTime, ctxData, gtTime, gtData, null, spName, model.name);
        // Attempt prediction (placeholder until weights arrive)
        attemptPrediction(ctxData, ctxTime, gtTime, gtData, spName, model.name, model.regime);
      }, 100);
    });

  }).catch(function(err) {
    _pwState.loading = false;
    console.error('Failed to load trajectory:', err);

    // Mark model as failed in sidebar
    markModelFailed(_pwState.pathwayIdx, _pwState.modelIdx);

    // Let skeleton show for minimum time, then transition to error state
    ChartSkeleton.ensureMin(function() {
      ChartSkeleton.hide();
      setTimeout(function() {
        ChartError.show(model, err.message);
      }, 200);
    });

    if (noteEl) {
      noteEl.innerHTML = '❌ Failed to load ' + model.name;
      noteEl.style.background = 'var(--color-tag-red, #301a1a)';
      noteEl.style.borderColor = '#4a2a30';
      noteEl.style.color = '#E74C3C';
    }
  });
}

// ── Regime pattern mapping (for RegimeFlow backend) ────────────
var REGIME_TO_PATTERN = {
  directly_stable: 0,
  inc_stable:      1,
  dec_stable:      2,
  oscillation:     3,
  increasing:      4,
  decreasing:      5
};

// ================================================================
// Prediction attempt
// ================================================================
function attemptPrediction(ctxData, ctxTime, gtTime, gtData, spName, modelName, modelRegime) {
  // Determine regime conditions from model metadata
  var pattern = REGIME_TO_PATTERN[modelRegime] || 0;
  var period = (modelRegime === 'oscillation') ? 12.5 : 0.0;

  var reqBody = {
    context: ctxData,
    prediction_length: Math.min(256, gtData.length),
    traj_pattern: pattern,
    period: period
  };

  fetch(API_BASE + '/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(reqBody)
  }).then(function(r) {
    if (!r.ok) throw new Error('API error');
    return r.json();
  }).then(function(result) {
    drawChart(ctxTime, ctxData, gtTime, gtData, result, spName, modelName);
  }).catch(function() {
    drawChart(ctxTime, ctxData, gtTime, gtData, { _pending: true }, spName, modelName);
  });
}

// ================================================================
// Redraw chart on species change
// ================================================================
function redrawPathwayChart() {
  if (!_pwState.currentModel || !_pwState.fullColumns) return;

  var model   = _pwState.currentModel;
  var spIdx   = _pwState.spIdx;
  var spNames = _pwState.spNames;
  if (spIdx >= spNames.length) spIdx = 0;

  var spName  = spNames[spIdx];
  var columns = _pwState.fullColumns;
  var time    = _pwState.fullTime;
  var ctxLen  = _pwState.ctxLen;

  // Slice from cached columns — no network re-fetch
  var spData  = columns[spName];
  if (!spData) return;

  var ctxTime = time.slice(0, ctxLen);
  var ctxData = spData.slice(0, ctxLen);
  var gtTime  = time.slice(ctxLen);
  var gtData  = spData.slice(ctxLen);

  _pwState.ctxTime = ctxTime;
  _pwState.ctxData = ctxData;
  _pwState.gtTime  = gtTime;
  _pwState.gtData  = gtData;

  // Update selector to reflect current species (without triggering re-render)
  var selEl = document.getElementById('species-select');
  if (selEl) selEl.value = spIdx;

  // Update chart immediately
  drawChart(ctxTime, ctxData, gtTime, gtData, null, spName, model.name);
  attemptPrediction(ctxData, ctxTime, gtTime, gtData, spName, model.name, model.regime);
}

// Skeleton Screen: see js/components/skeleton.js (ChartSkeleton API)
// drawChart — shared chart renderer
// ================================================================
function drawChart(ctxTime, ctxData, gtTime, gtData, apiResult, spName, sysName) {
  if (!window._predictChart) return;
  var series = [];

  // 1) Context (blue solid)
  series.push({
    name: 'Context',
    type: 'line',
    data: ctxTime.map(function(t, i) { return [t, ctxData[i]]; }),
    lineStyle:  { color: '#4A90D9', width: 2.5 },
    itemStyle:  { color: '#4A90D9' },
    symbol: 'none', smooth: true,
    markArea: {
      silent: true,
      itemStyle: { color: 'rgba(74,144,217,0.08)' },
      data: [[{ xAxis: ctxTime[0] }, { xAxis: ctxTime[ctxTime.length-1] }]]
    }
  });

  // 2) Ground truth (gray dashed)
  series.push({
    name: 'Ground Truth',
    type: 'line',
    data: gtTime.map(function(t, i) { return [t, gtData[i]]; }),
    lineStyle:  { color: '#6a8299', width: 1.5, type: 'dashed' },
    itemStyle:  { color: '#6a8299' },
    symbol: 'none', smooth: true
  });

  // 3) Prediction — real or placeholder
  if (apiResult && !apiResult._pending && apiResult.predictions) {
    var lastCtxTime = ctxTime[ctxTime.length-1];
    var dt = ctxTime[1] - ctxTime[0];
    var predLen = apiResult.predictions.length;
    var predTime = [];
    for (var pt = 0; pt < predLen; pt++) {
      predTime.push(+(lastCtxTime + (pt+1) * dt).toFixed(2));
    }
    series.push({
      name: 'Prediction',
      type: 'line',
      data: predTime.map(function(t, i) { return [t, apiResult.predictions[i]]; }),
      lineStyle:  { color: '#F39C12', width: 2.5 },
      itemStyle:  { color: '#F39C12' },
      symbol: 'none', smooth: true,
      markArea: {
        silent: true,
        itemStyle: { color: 'rgba(243,156,18,0.06)' },
        data: [[{ xAxis: predTime[0] }, { xAxis: predTime[predLen-1] }]]
      }
    });
    if (apiResult.lower && apiResult.upper) {
      var lo = predTime.map(function(t, i) { return [t, apiResult.lower[i]]; });
      var hi = predTime.map(function(t, i) { return [t, apiResult.upper[i]]; }).reverse();
      series.push({
        name: 'Confidence',
        type: 'line',
        data: lo.concat(hi),
        lineStyle: { color: 'transparent', width: 0 },
        areaStyle: { color: 'rgba(243,156,18,0.12)' },
        symbol: 'none', silent: true, stack: 'confidence'
      });
    }
  } else if (apiResult && apiResult._pending) {
    // Placeholder — dashed orange line at same position as GT, with note
    var lastCtxTime2 = ctxTime[ctxTime.length-1];
    var dt2 = ctxTime[1] - ctxTime[0];
    var predLen2 = Math.min(256, gtData.length);
    var predTime2 = [];
    for (var pt2 = 0; pt2 < predLen2; pt2++) {
      predTime2.push(+(lastCtxTime2 + (pt2+1) * dt2).toFixed(2));
    }
    series.push({
      name: 'Prediction (pending)',
      type: 'line',
      data: predTime2.map(function(t) { return [t, null]; }),
      lineStyle:  { color: '#F39C12', width: 1.5, type: 'dotted' },
      itemStyle:  { color: '#F39C12' },
      symbol: 'none',
      markArea: {
        silent: true,
        itemStyle: { color: 'rgba(243,156,18,0.04)' },
        data: [[{ xAxis: predTime2[0] }, { xAxis: predTime2[predLen2-1] }]]
      }
    });
  }

  var subtext = (apiResult && apiResult._pending)
    ? '⚠ Prediction pending — waiting for AI model weights'
    : 'Context: ' + ctxData.length + ' pts  ·  Ground Truth: ' + gtData.length + ' pts';

  // Show legend guide strip
  var guide = document.getElementById('chart-legend-guide');
  if (guide) guide.style.display = '';

  window._predictChart.setOption({
    backgroundColor: 'transparent',
    animationDuration: 600,
    animationEasing: 'cubicOut',
    title: {
      text: sysName,
      subtext: subtext,
      left: 'center', top: 8,
      textStyle:    { color: '#e0e6ed', fontSize: 16 },
      subtextStyle: { color: (apiResult && apiResult._pending ? '#F39C12' : '#6a8299'), fontSize: 11 }
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#1a2a3a',
      borderColor: '#2a4057',
      textStyle: { color: '#e0e6ed', fontSize: 12 }
    },
    legend: {
      bottom: 10,
      icon: 'circle', itemWidth: 10, itemHeight: 10,
      textStyle: { color: '#8899aa', fontSize: 11 },
      emphasis: { focus: 'series' },
      data: (apiResult && apiResult._pending)
        ? ['Context', 'Ground Truth', 'Prediction (pending)']
        : ['Context', 'Ground Truth']
    },
    grid: { left: 55, right: 30, top: 70, bottom: 55 },
    xAxis: {
      type: 'value',
      axisLine:  { lineStyle: { color: '#2a4057' } },
      axisLabel: { color: '#6a8299', fontSize: 10 },
      splitLine: { lineStyle: { color: '#1a2a3a' } }
    },
    yAxis: {
      type: 'value',
      axisLine:  { lineStyle: { color: '#2a4057' } },
      axisLabel: { color: '#6a8299', fontSize: 10 },
      splitLine: { lineStyle: { color: '#1a2a3a' } }
    },
    series: series
  }, true);  // notMerge — clear any stale series from previous renders
}

// ================================================================
// Custom data prediction (unchanged from original)
// ================================================================
function parseCustomData(rawText) {
  if (!rawText || !rawText.trim()) return { error: 'No data provided' };
  var lines = rawText.trim().split('\n').filter(function(l) { return l.trim(); });
  if (lines.length < 10) return { error: 'Need at least 10 data points' };

  var values = [];
  lines.forEach(function(line) {
    var parts = line.split(/[,\t\s]+/);
    var v = parseFloat(parts[parts.length-1]);
    if (!isNaN(v)) values.push(v);
  });
  if (values.length < 10) return { error: 'Need at least 10 valid numeric values' };
  return { values: values };
}

function handleCustomPredict() {
  var inputEl  = document.getElementById('custom-data-input');
  var statusEl = document.getElementById('custom-status');
  var ctxLenEl = document.getElementById('custom-ctx-len');
  if (!inputEl || !statusEl) return;

  var rawText = inputEl.value;
  var parsed  = parseCustomData(rawText);
  if (parsed.error) {
    statusEl.textContent = parsed.error;
    statusEl.className = 'custom-status error';
    return;
  }

  var ctxLen = parseInt(ctxLenEl.value) || 96;
  if (ctxLen >= parsed.values.length) {
    statusEl.textContent = 'Context length must be less than total data points';
    statusEl.className = 'custom-status error';
    return;
  }

  statusEl.textContent = 'Predicting…';
  statusEl.className = 'custom-status loading';

  var allValues = parsed.values;
  var ctxData = allValues.slice(0, ctxLen);
  var gtData  = allValues.slice(ctxLen);
  var dt = 0.1;
  var ctxTime = ctxData.map(function(_, i) { return +(i * dt).toFixed(2); });
  var gtTime  = gtData.map(function(_, i) { return +((ctxLen + i) * dt).toFixed(2); });

  if (!window._predictChart) {
    window._predictChart = echarts.init(document.getElementById('predict-chart'));
  }
  drawChart(ctxTime, ctxData, gtTime, gtData, null, 'Custom', 'Custom Data');

  fetch(API_BASE + '/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ context: ctxData, prediction_length: Math.min(256, gtData.length) })
  }).then(function(r) {
    if (!r.ok) throw new Error('API error');
    return r.json();
  }).then(function(result) {
    drawChart(ctxTime, ctxData, gtTime, gtData, result, 'Custom', 'Custom Data');
    statusEl.textContent = 'Prediction complete';
    statusEl.className = 'custom-status success';
  }).catch(function(e) {
    statusEl.textContent = 'Backend unavailable · showing ground truth only';
    statusEl.className = 'custom-status error';
  });
}

// ================================================================
// highlightBioSpecies — dim all series except the selected one
// ================================================================
function highlightBioSpecies(spIdx) {
  var state = window._bioChartState;
  if (!state || !window._predictChart) return;

  // If species is outside visible top-N, force-expand first
  if (!state.expanded && state.needsExpand && spIdx >= state.topNames.length) {
    state.selSpIdx = spIdx;
    renderBioChart(true);
    return;
  }

  state.selSpIdx = spIdx;
  var allSeries = state.allSeries;
  var palette = state.palette;
  var topNames = state.topNames;
  var expanded = state.expanded;

  // Only include currently visible series (respect expand/collapse)
  var visibleIndices = [];
  for (var i = 0; i < allSeries.length; i++) {
    if (expanded || !state.needsExpand || i < topNames.length) {
      visibleIndices.push(i);
    }
  }

  var highlighted = [];
  for (var vi = 0; vi < visibleIndices.length; vi++) {
    var i = visibleIndices[vi];
    var s = allSeries[i];
    var isSel = (i === spIdx);
    var clone = {
      name: s.name,
      type: s.type,
      data: s.data,
      symbol: s.symbol,
      smooth: s.smooth,
      z: isSel ? 10 : 0,
      lineStyle: {
        color: palette[i % palette.length],
        width: isSel ? 3 : 0.8,
        opacity: isSel ? 1 : 0.06
      },
      itemStyle: {
        color: palette[i % palette.length],
        opacity: isSel ? 1 : 0.06
      }
    };
    highlighted.push(clone);
  }

  window._predictChart.setOption({
    series: highlighted
  });  // merge mode — only updates series
}

// ================================================================
// Bio chart renderer (used by loadBioTrajectory)
// ================================================================
function renderBioChart(expanded) {
  var state = window._bioChartState;
  if (!state) return;
  state.expanded = expanded;

  // Decide which series to show
  var showAll = expanded || !state.needsExpand;
  var series = showAll ? state.allSeries.slice() : state.allSeries.slice(0, state.topNames.length);
  var shownCount = showAll ? state.totalSpecies : state.topNames.length;
  var hiddenCount = state.totalSpecies - shownCount;

  // Build legend data from the actual visible series
  var legendData = [];
  for (var i = 0; i < shownCount; i++) {
    legendData.push(state.speciesMeta[i].name);
  }
  // Expand / collapse toggle entry
  if (!showAll && hiddenCount > 0) {
    legendData.push('▸ +' + hiddenCount + ' others');
  }
  if (showAll && state.needsExpand) {
    legendData.push('◂ collapse');
  }

  // Add dummy toggle series
  var chartSeries = series.slice();
  if (!showAll && hiddenCount > 0) {
    chartSeries.push({ name: '▸ +' + hiddenCount + ' others', type: 'line', data: [],
      lineStyle: { opacity: 0 }, itemStyle: { opacity: 0 } });
  }
  if (showAll && state.needsExpand) {
    chartSeries.push({ name: '◂ collapse', type: 'line', data: [],
      lineStyle: { opacity: 0 }, itemStyle: { opacity: 0 } });
  }

  // Subtitle
  var subtext = state.modelId + ' · ' + state.totalSpecies + ' species · ' + state.timeLen + ' steps';
  if (!showAll && hiddenCount > 0) {
    subtext += '  |  Top ' + shownCount + ' by range — click ▸ to show all ' + state.totalSpecies;
  }

  // Y range always from all data (consistent axis, no jump on expand)
  var yMin = state.yMin, yMax = state.yMax;

  var useScroll = state.totalSpecies > 8;
  var legendHeight = useScroll ? 55 : 22;
  var gridBottom = useScroll ? 82 : 40;

  ChartSkeleton.ensureMin(function() {
    ChartSkeleton.hide();
    setTimeout(function() {
      window._predictChart.setOption({
        backgroundColor: 'transparent',
        animationDuration: 600,
        animationEasing: 'cubicOut',
        title: {
          text: state.modelName,
          subtext: subtext,
          left: 'center', top: 8,
          textStyle: { color: '#e0e6ed', fontSize: 16 },
          subtextStyle: { color: '#6a8299', fontSize: 11 }
        },
        tooltip: { trigger: 'axis', backgroundColor: '#1a2a3a', borderColor: '#2a4057',
          textStyle: { color: '#e0e6ed', fontSize: 12 } },
        legend: {
          type: useScroll ? 'scroll' : 'plain',
          orient: 'horizontal',
          bottom: useScroll ? 10 : 6,
          left: 5, right: 5,
          height: legendHeight,
          width: 'auto',
          itemGap: 6,
          itemWidth: 10, itemHeight: 10,
          icon: 'circle',
          textStyle: { color: '#8899aa', fontSize: 10, padding: [0, 4, 0, 0] },
          pageTextStyle: { color: '#6a8299' },
          pageIconSize: 11,
          pageButtonItemGap: 4,
          pageButtonGap: 8,
          emphasis: { focus: 'series' },
          padding: [6, 8],
          data: legendData
        },
        grid: { left: 55, right: 30, top: 75, bottom: gridBottom },
        xAxis: { type: 'value', axisLine: { lineStyle: { color: '#2a4057' } },
          axisLabel: { color: '#6a8299', fontSize: 10 },
          splitLine: { lineStyle: { color: '#1a2a3a' } } },
        yAxis: { type: 'value', min: yMin, max: yMax,
          axisLine: { lineStyle: { color: '#2a4057' } },
          axisLabel: { color: '#6a8299', fontSize: 10 },
          splitLine: { lineStyle: { color: '#1a2a3a' } } },
        dataZoom: [
          { type: 'inside', start: 0, end: 100 },
          { type: 'slider', start: 0, end: 100, height: 20, bottom: 32,
            borderColor: '#1e3045', backgroundColor: '#162231',
            fillerColor: 'rgba(74,144,217,0.15)',
            textStyle: { color: '#6a8299' } }
        ],
        series: chartSeries
      }, true);  // notMerge — clean replace for expand/collapse
      // Re-apply species highlight if one is selected
      if (state.selSpIdx >= 0) {
        highlightBioSpecies(state.selSpIdx);
      }
      _bindLegendToggle();
    }, 100);
  });
}

// Legend toggle handler — intercept clicks on "+N others" / "collapse"
// Bound once, delegates via ECharts legendselectchanged event
if (!window._bioLegendBound) {
  window._bioLegendBound = true;
  // We attach the handler lazily after the chart is initialized.
  // renderBioChart re-attaches each time (ECharts off/on is idempotent).
}
function _bindLegendToggle() {
  if (!window._predictChart) return;
  window._predictChart.off('legendselectchanged');
  window._predictChart.on('legendselectchanged', function(params) {
    if (!window._bioChartState) return;
    var n = params.name || '';
    if (n.indexOf('+') === 0 && n.indexOf('others') > 0) {
      renderBioChart(true);
    } else if (n.indexOf('collapse') >= 0) {
      renderBioChart(false);
    }
  });
}

// ================================================================
// Legacy: loadBioTrajectory (called from detail panel button)
// Shows ALL species as multi-line chart — backward compatible
// ================================================================
function loadBioTrajectory(modelId, modelName, speciesCount) {
  _bioTrajectoryLoaded = true;
  _activeViewMode = 'bio';

  // Switch to prediction view
  document.querySelectorAll('.nav-btn').forEach(function(b) { b.classList.remove('active'); });
  var predBtn = document.querySelector('.nav-btn[data-view="predict"]');
  if (predBtn) predBtn.classList.add('active');
  document.querySelectorAll('.view').forEach(function(v) { v.classList.remove('active'); });
  var predView = document.getElementById('view-predict');
  if (predView) predView.classList.add('active');

  // Hide pathway list + custom panel, show legacy view
  var listEl = document.getElementById('system-list');
  var customPanel = document.getElementById('custom-input-panel');
  if (listEl) listEl.style.display = 'none';
  if (customPanel) customPanel.style.display = 'none';
  document.querySelectorAll('.input-tab').forEach(function(t) { t.classList.remove('active'); });

  var noteEl = document.querySelector('.sidebar-note');
  if (noteEl) {
    noteEl.innerHTML = '🧬 <b>' + modelName + '</b><br>' +
      '<span style="font-size:10px;color:#8899aa;">' + modelId + ' · ' + speciesCount + ' species · 512 steps</span><br>' +
      '<span style="color:#F39C12;">📊 Full trajectory from HuggingFace</span>';
    noteEl.style.background = '#1a2a20';
    noteEl.style.borderColor = '#2a4a30';
    noteEl.style.color = '#88aa88';
  }

  if (!window._predictChart) {
    window._predictChart = echarts.init(document.getElementById('predict-chart'));
  }
  // Chart was initialized while hidden — force resize after view switch
  setTimeout(function() { if (window._predictChart) window._predictChart.resize(); }, 150);

  ChartSkeleton.show({ name: modelName, id: modelId, species: speciesCount });

  var csvUrl = 'https://huggingface.co/datasets/HengRao/SysBio-Traj/resolve/main/Data/'
    + modelId + '/' + modelName + '.csv';

  fetch(csvUrl).then(function(resp) {
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    return resp.text();
  }).then(function(csvText) {
    var lines = csvText.trim().split('\n');
    var headers = lines[0].split(',').map(function(h) { return h.trim(); });
    var columns = {};
    headers.forEach(function(h) { columns[h] = []; });

    for (var i = 1; i < lines.length; i++) {
      var vals = lines[i].split(',');
      headers.forEach(function(h, j) { columns[h].push(parseFloat(vals[j])); });
    }

    var time = columns['time'] || [];
    var timeLen = time.length;

    // ── Filter: exclude metric suffixes (_max, _min, _norm) ──
    function isMetricSuffix(name) {
      return /_(max|min|norm|mean|std|sum|avg)$/.test(name);
    }
    var speciesNames = headers.filter(function(h) {
      return h !== 'time' && !isMetricSuffix(h);
    });

    var palette = ['#4A90D9','#F39C12','#2ECC71','#E74C3C','#9B59B6','#1ABC9C','#F1C40F',
      '#E67E22','#3498DB','#8E44AD','#2C3E50','#16A085','#C0392B','#2980B9','#D35400'];

    // ── Rank species by variance (most dynamic first) ──
    var speciesMeta = speciesNames.map(function(sp) {
      var data = columns[sp];
      if (!data || !data.length) return null;
      var min = Infinity, max = -Infinity;
      for (var d = 0; d < data.length; d++) {
        if (data[d] < min) min = data[d];
        if (data[d] > max) max = data[d];
      }
      return { name: sp, range: max - min, min: min, max: max, data: data };
    }).filter(Boolean);
    speciesMeta.sort(function(a, b) { return b.range - a.range; });

    var totalSpecies = speciesMeta.length;

    // ── Smart threshold: ≤30 species → show all; >30 → top 15 + expand ──
    var MAX_VISIBLE = 30;
    var TOP_N = Math.min(totalSpecies, totalSpecies > MAX_VISIBLE ? 15 : totalSpecies);
    var needsExpand = totalSpecies > MAX_VISIBLE;
    var topNames = speciesMeta.slice(0, TOP_N).map(function(s) { return s.name; });

    // ── Build ALL series ──
    var allSeries = speciesMeta.map(function(sm, idx) {
      if (!sm.data || !sm.data.length) return null;
      return {
        name: sm.name,
        type: 'line',
        data: time.map(function(t, i) { return [t, sm.data[i]]; }),
        lineStyle: { color: palette[idx % palette.length], width: 1.5 },
        itemStyle: { color: palette[idx % palette.length] },
        symbol: 'none', smooth: true
      };
    }).filter(Boolean);

    // ── Y-axis range from all data (not just visible) ──
    var allMin = Infinity, allMax = -Infinity;
    speciesMeta.forEach(function(sm) {
      if (sm.min < allMin) allMin = sm.min;
      if (sm.max > allMax) allMax = sm.max;
    });
    var yPad = Math.max((allMax - allMin) * 0.05, 0.01);

    // Store for expand/collapse toggle
    window._bioChartState = {
      modelName: modelName, modelId: modelId,
      time: time, timeLen: timeLen, speciesNames: speciesNames,
      allSeries: allSeries,
      topNames: topNames, totalSpecies: totalSpecies,
      needsExpand: needsExpand, expanded: false,
      yMin: allMin - yPad, yMax: allMax + yPad,
      speciesMeta: speciesMeta, palette: palette,
      selSpIdx: -1  // -1 = show all, otherwise highlight this species
    };

    // If small model → show all immediately; if large → show top N first
    renderBioChart(needsExpand ? false : true);

  var selEl = document.getElementById('species-select');
  if (selEl) {
    var prevVal = selEl.value;
    selEl.innerHTML = '<option value="-1">— All species —</option>' +
      speciesNames.map(function(n, i) {
        return '<option value="' + i + '">' + n + '</option>';
      }).join('');
    // Restore previous selection or default to "All species"
    selEl.value = (prevVal && parseInt(prevVal) >= 0) ? prevVal : '-1';
  }

  var tagEl = document.getElementById('regime-tag');
  if (tagEl) { tagEl.textContent = 'BioModels'; tagEl.className = 'regime-tag regime-oscillation'; }
  }).catch(function(err) {
    console.error('Failed to load bio trajectory:', err);
    ChartSkeleton.clear();
    window._predictChart.setOption({
      title: { text: 'Failed to load ' + modelName, left: 'center', top: '40%',
        textStyle: { color: '#E74C3C', fontSize: 16 },
        subtext: err.message, subtextStyle: { color: '#6a8299', fontSize: 12 } },
      backgroundColor: 'transparent'
    });
  });
}

// ================================================================
// Sidebar note
// ================================================================
function updateSidebarNote() {
  var noteEl = document.querySelector('.sidebar-note');
  if (!noteEl) return;
  if (_backendStatus.regimeflow_loaded) {
    noteEl.innerHTML = '🧠 RegimeFlow ready' +
      '<br><span style="font-size:10px;">' + _backendStatus.device +
      ' · ' + _backendStatus.denoise_steps + ' denoise steps</span>';
    noteEl.style.background = '#1a2a20';
    noteEl.style.borderColor = '#2a4a30';
    noteEl.style.color = '#88aa88';
  } else if (_backendStatus.chronos_loaded) {
    noteEl.innerHTML = '✅ Chronos-Bolt ready on ' + _backendStatus.device;
    noteEl.style.background = '#1a2a20';
    noteEl.style.borderColor = '#2a4a30';
    noteEl.style.color = '#88aa88';
  } else {
    noteEl.innerHTML = '🔮 No model loaded<br><span style="font-size:10px;">Trajectory browsing available</span>';
    noteEl.style.background = 'var(--color-tag-yellow, #1a1a0a)';
    noteEl.style.borderColor = '#332e15';
    noteEl.style.color = '#998844';
  }
}
