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
// 如需让 file:// 直连远程后端，把 RENDER_URL 设为远程地址（默认留空）:
//   例如: var RENDER_URL = 'https://regimeflow-web.onrender.com';
var RENDER_URL = '';
var API_BASE = (function() {
  if (RENDER_URL) return RENDER_URL;
  // 同源部署（Render 生产 / 本地 server.py）→ 使用相对路径（本地加载 RegimeFlow 模型）
  if (window.location.protocol === 'http:' || window.location.protocol === 'https:') return '';
  // file:// 协议打开 → 默认连接本地后端
  return 'http://localhost:8000';
})();

// ── State ──────────────────────────────────────────────────────
var _pwState = {
  modelIdx:      -1,      // selected model index within currentModelList, -1 = none
  spIdx:         0,       // species index
  spNames:       [],      // species names from loaded CSV
  ctxLen:        96,      // context length
  ctxData:       null,    // context data array (current species)
  gtData:        null,    // ground-truth data array (current species)
  gtTime:        null,    // ground-truth time array
  ctxTime:       null,    // context time array
  fullColumns:   null,    // all parsed CSV columns (cached for species switching)
  fullTime:      null,    // full time array from CSV
  currentModel:  null,    // currently loaded model
  loading:       false,
  overlayAll:    true,    // true = overlay all species; false = single species
  dim:           'taxonomy', // active classification dimension: 'taxonomy' | 'process'
  selKey:        '',         // selected category key in current dim ('' = all)
  currentModelList: []       // flat model list matching the current single-dim filter
};

var _backendStatus = { regimeflow_loaded: false, chronos_loaded: false, device: 'cpu', denoise_steps: 0 };
var _failedModels = {};  // key: model index → true for models that returned 404/500
var _activeViewMode = 'pathway';  // 'pathway' | 'compare' — which view currently owns the chart

// Multi-species overlay palette + cap (one colour per species)
var SPECIES_COLORS = ['#4A90D9','#F39C12','#2ECC71','#E74C3C','#9B59B6','#1ABC9C','#F1C40F','#E67E22','#3498DB','#E84393','#00CEC9','#A29BFE','#55EFC4','#FD79A8','#6C5CE7','#FF7675'];
var MAX_OVERLAY_SPECIES = 15;

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
      ? 'Data not available for ' + getDisplayName(model)
      : 'Failed to load ' + getDisplayName(model);
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
function markModelFailed(i) {
  _failedModels[i] = true;
  var badge = document.getElementById('pw-errbadge-' + i);
  if (badge) badge.classList.add('visible');
  var item = document.getElementById('pw-model-' + i);
  if (item) item.classList.add('has-error');
}

function markModelOk(i) {
  delete _failedModels[i];
  var badge = document.getElementById('pw-errbadge-' + i);
  if (badge) badge.classList.remove('visible');
  var item = document.getElementById('pw-model-' + i);
  if (item) item.classList.remove('has-error');
}

// ================================================================
// Category comparison — render all models in a selected category
// ================================================================
var CAT_REGIME_COLOR = {
  oscillation:     '#F39C12',
  inc_stable:      '#2ECC71',
  dec_stable:      '#E74C3C',
  directly_stable: '#4A90D9',
  increasing:      '#2ECC71',
  decreasing:      '#E74C3C'
};

var _catCompareState = { seq: 0 };

// Flat model list filtered by the active dimension only (single-select, no AND).
function currentFilteredModels() {
  if (!_pwState.selKey) return BIO_MODELS_DATA.slice();
  var entries = (_pwState.dim === 'taxonomy')
    ? (BIO_DOMAIN_DATA[_pwState.selKey] || [])
    : (BIO_PROCESSES_DATA[_pwState.selKey] || []);
  var ids = {};
  entries.forEach(function(e) { ids[e.id] = true; });
  return BIO_MODELS_DATA.filter(function(m) { return ids[m.id]; });
}

function categoryLabel() {
  if (!_pwState.selKey) return '';
  return (_pwState.dim === 'taxonomy' ? '🧬 ' : '🧪 ') + _pwState.selKey;
}

function parseCsv(text) {
  var lines = text.trim().split('\n');
  var headers = lines[0].split(',').map(function(h) { return h.trim(); });
  var columns = {};
  headers.forEach(function(h) { columns[h] = []; });
  for (var i = 1; i < lines.length; i++) {
    var vals = lines[i].split(',');
    headers.forEach(function(h, j) { columns[h].push(parseFloat(vals[j])); });
  }
  var time = columns['time'] || [];
  var spNames = headers.filter(function(h) {
    return h !== 'time' && !/_(max|min|norm|mean|std|sum|avg)$/.test(h);
  });
  return { time: time, spNames: spNames, columns: columns };
}

function pickRepresentativeSpecies(parsed) {
  var bestData = null, bestRange = -1;
  parsed.spNames.forEach(function(sp) {
    var d = parsed.columns[sp];
    if (!d || !d.length) return;
    var min = Infinity, max = -Infinity;
    for (var i = 0; i < d.length; i++) {
      if (d[i] < min) min = d[i];
      if (d[i] > max) max = d[i];
    }
    var range = max - min;
    if (range > bestRange) { bestRange = range; bestData = d; }
  });
  return { time: parsed.time, data: bestData || [] };
}

function renderCategoryComparison() {
  if (!window._predictChart) return;

  _activeViewMode = 'compare';
  updateBackButton();

  // Hide single-model legend guide in comparison mode
  var guide = document.getElementById('chart-legend-guide');
  if (guide) guide.style.display = 'none';

  var models = currentFilteredModels();
  var seq = ++_catCompareState.seq;

  // Nothing selected — prompt
  if (_pwState.selKey === '') {
    _activeViewMode = 'compare';
    window._predictChart.setOption({
      backgroundColor: 'transparent',
      title: { text: 'Select a category', subtext: 'pick a category above, or a model from the list', left: 'center', top: '38%',
        textStyle: { color: '#6a8299', fontSize: 16 }, subtextStyle: { color: '#4a5f73', fontSize: 12 } },
      xAxis: { show: false }, yAxis: { show: false }, series: []
    }, true);
    return;
  }

  // Empty intersection
  if (models.length === 0) {
    _activeViewMode = 'compare';
    window._predictChart.setOption({
      backgroundColor: 'transparent',
      title: { text: 'No models match', subtext: categoryLabel(), left: 'center', top: '38%',
        textStyle: { color: '#E74C3C', fontSize: 16 }, subtextStyle: { color: '#6a8299', fontSize: 12 } },
      xAxis: { show: false }, yAxis: { show: false }, series: []
    }, true);
    return;
  }

  _activeViewMode = 'compare';

  // Loading state
  window._predictChart.setOption({
    backgroundColor: 'transparent',
    title: { text: categoryLabel(), subtext: 'Loading ' + models.length + ' models…', left: 'center', top: '38%',
      textStyle: { color: '#e0e6ed', fontSize: 16 }, subtextStyle: { color: '#F39C12', fontSize: 12 } },
    xAxis: { show: false }, yAxis: { show: false }, series: []
  }, true);

  var collected = [];
  var cursor = 0;
  var CONCURRENCY = 8;

  function renderSoFar() {
    if (seq !== _catCompareState.seq) return;
    var series = collected.map(function(c) {
      return {
        name: c.shortName,
        type: 'line',
        data: c.time.map(function(t, i) { return [t, c.data[i]]; }),
        lineStyle: { color: c.color, width: 1.2 },
        itemStyle: { color: c.color },
        symbol: 'none', smooth: true
      };
    });
    window._predictChart.setOption({
      backgroundColor: 'transparent',
      animationDuration: 200,
      animationEasing: 'cubicOut',
      title: { text: categoryLabel(), subtext: collected.length + ' / ' + models.length + ' models', left: 'center', top: 8,
        textStyle: { color: '#e0e6ed', fontSize: 16 }, subtextStyle: { color: '#6a8299', fontSize: 11 } },
      tooltip: { trigger: 'axis', backgroundColor: '#1a2a3a', borderColor: '#2a4057', textStyle: { color: '#e0e6ed', fontSize: 12 } },
      legend: { type: 'scroll', orient: 'horizontal', bottom: 10, left: 5, right: 5, itemWidth: 10, itemHeight: 10,
        icon: 'circle', textStyle: { color: '#8899aa', fontSize: 10 } },
      grid: { left: 55, right: 30, top: 70, bottom: 55 },
      xAxis: { type: 'value', axisLine: { lineStyle: { color: '#2a4057' } }, axisLabel: { color: '#6a8299', fontSize: 10 },
        splitLine: { lineStyle: { color: '#1a2a3a' } } },
      yAxis: { type: 'value', axisLine: { lineStyle: { color: '#2a4057' } }, axisLabel: { color: '#6a8299', fontSize: 10 },
        splitLine: { lineStyle: { color: '#1a2a3a' } } },
      series: series
    }, true);
  }

  function loadNext() {
    if (seq !== _catCompareState.seq) return;
    if (cursor >= models.length) return;
    var m = models[cursor++];
    var csvUrl = 'https://huggingface.co/datasets/HengRao/SysBio-Traj/resolve/main/Data/'
      + m.id + '/' + m.name + '.csv';
    fetch(csvUrl).then(function(r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.text();
    }).then(function(text) {
      if (seq !== _catCompareState.seq) return;
      var parsed = parseCsv(text);
      var rep = pickRepresentativeSpecies(parsed);
      if (rep.data.length) {
        collected.push({
          shortName: getLabelName(m, 28),
          color: CAT_REGIME_COLOR[m.regime] || '#8899aa',
          time: rep.time,
          data: rep.data
        });
        renderSoFar();
      }
      loadNext();
    }).catch(function() {
      if (seq !== _catCompareState.seq) return;
      loadNext();
    });
  }

  for (var c = 0; c < CONCURRENCY; c++) loadNext();
}

// Show the "back to list" button only while viewing a single model.
function updateBackButton() {
  var btn = document.getElementById('btn-back-to-list');
  if (!btn) return;
  btn.style.display = (_activeViewMode === 'pathway') ? '' : 'none';
}

// Exit the single-model view and return to the category comparison / list.
function backToList() {
  _pwState.modelIdx = -1;
  _pwState.currentModel = null;
  _pwState.spIdx = 0;

  // Clear list highlight
  document.querySelectorAll('.pw-model-item').forEach(function(el) { el.classList.remove('active'); });

  // Reset regime tag
  var tagEl = document.getElementById('regime-tag');
  if (tagEl) { tagEl.textContent = ''; tagEl.className = 'regime-tag'; }

  // Reset species selector
  var selEl = document.getElementById('species-select');
  if (selEl) selEl.innerHTML = '<option value="0">— Select a model first —</option>';

  // Reset sidebar note to backend status
  updateSidebarNote();

  // Return to category comparison (or the select-a-category prompt)
  renderCategoryComparison();
}

// ================================================================
// initPrediction — entry point
// ================================================================
function initPrediction() {
  if (typeof BIO_MODELS_DATA === 'undefined' ||
      typeof BIO_DOMAIN_DATA === 'undefined' ||
      typeof BIO_PROCESSES_DATA === 'undefined') return;

  // Check backend
  fetch(API_BASE + '/api/health').then(function(r) { return r.json(); })
    .then(function(s) { _backendStatus = s; updateSidebarNote(); })
    .catch(function() {});

  // ── Dimension toggle + single dropdown + model list ──
  var filterEl = document.getElementById('category-filter');
  var listEl   = document.getElementById('pw-model-list');
  var dimTabs  = document.querySelectorAll('.dim-tab');
  if (!filterEl || !listEl || !dimTabs.length) return;

  // Populate the single dropdown from the active dimension's categories.
  function buildDropdown() {
    var keys, allLabel, icon;
    if (_pwState.dim === 'taxonomy') {
      keys = Object.keys(BIO_DOMAIN_DATA).sort();
      allLabel = '— All domains —';
      icon = '🧬';
    } else {
      keys = Object.keys(BIO_PROCESSES_DATA).sort();
      allLabel = '— All processes —';
      icon = '🧪';
    }
    filterEl.innerHTML = '<option value="">' + allLabel + '</option>' +
      keys.map(function(k) {
        var len = _pwState.dim === 'taxonomy' ? BIO_DOMAIN_DATA[k].length : BIO_PROCESSES_DATA[k].length;
        return '<option value="' + escapeHtml(k) + '">' + icon + ' ' + escapeHtml(k) + ' (' + len + ')</option>';
      }).join('');
    filterEl.value = _pwState.selKey || '';
  }

  // Render the flat model list for the current filter.
  function renderModelList(models) {
    listEl.innerHTML = '';
    models.forEach(function(m, i) {
      var mel = document.createElement('div');
      mel.className = 'pw-model-item';
      mel.id = 'pw-model-' + i;
      var l1 = REGIME_TO_L1[m.regime] || 'stable';
      var l1def = REGIME_L1_DEFS[l1] || {};
      mel.innerHTML =
        '<span class="pw-model-name">' + escapeHtml(getDisplayName(m)) + '</span>' +
        '<span class="pw-model-error-badge" id="pw-errbadge-' + i + '">⚠</span>' +
        '<span class="pw-model-id">' + escapeHtml(m.id) + '</span>' +
        '<span class="pw-model-tag" style="color:' + (l1def.color || '#8899aa') + '">' + (l1def.icon || '') + ' ' + (l1def.label || m.regime) + '</span>';
      mel.addEventListener('click', function() { selectModel(i); });
      listEl.appendChild(mel);
    });
    var total = document.createElement('div');
    total.className = 'pw-total';
    total.textContent = models.length + ' / ' + BIO_MODELS_DATA.length + ' models';
    listEl.appendChild(total);
  }

  function refresh() {
    _pwState.currentModelList = currentFilteredModels();
    renderModelList(_pwState.currentModelList);
    renderCategoryComparison();
  }

  // Mode switch: reset the prior dimension's filter, rebuild dropdown + list.
  dimTabs.forEach(function(tab) {
    tab.addEventListener('click', function() {
      var dim = tab.dataset.dim;
      if (_pwState.dim === dim) return;
      _pwState.dim = dim;
      _pwState.selKey = '';
      _pwState.modelIdx = -1;
      dimTabs.forEach(function(t) { t.classList.remove('active'); });
      tab.classList.add('active');
      buildDropdown();
      refresh();
    });
  });

  filterEl.addEventListener('change', function() {
    _pwState.selKey = filterEl.value;
    _pwState.modelIdx = -1;
    refresh();
  });

  // Initial list (chart prompt is rendered after echarts.init below).
  buildDropdown();
  _pwState.currentModelList = currentFilteredModels();
  renderModelList(_pwState.currentModelList);

  // ── Species selector ──
  var selEl = document.getElementById('species-select');
  if (selEl) {
    selEl.innerHTML = '<option value="0">— Select a model first —</option>';
    selEl.addEventListener('change', function() {
      var val = selEl.value;
      if (val === '-1') {
        // Overlay all species
        renderAllSpecies();
      } else {
        // Single-species pathway view: switch to that species
        _pwState.spIdx = parseInt(val) || 0;
        redrawPathwayChart();
      }
    });
  }

  // ── Back to list button ──
  var backBtn = document.getElementById('btn-back-to-list');
  if (backBtn) backBtn.addEventListener('click', backToList);

  // ── Chart ──
  if (!window._predictChart) {
    window._predictChart = echarts.init(document.getElementById('predict-chart'));
  }
  window.addEventListener('resize', debounce(function() {
    if (window._predictChart) window._predictChart.resize();
  }, 200));

  // Show initial prompt (default category view)
  renderCategoryComparison();

  // ── Tab switching ──
  document.querySelectorAll('.input-tab').forEach(function(tab) {
    tab.addEventListener('click', function() {
      document.querySelectorAll('.input-tab').forEach(function(t) { t.classList.remove('active'); });
      tab.classList.add('active');
      var showPathway = tab.dataset.tab === 'models';
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
function selectModel(i) {
  var models = _pwState.currentModelList;
  var model  = models[i];
  if (!model) return;

  _pwState.modelIdx       = i;
  _pwState.spIdx          = 0;
  _pwState.currentModel   = model;

  // Clear any previous error state before loading new model
  ChartError.clear();

  // Highlight
  document.querySelectorAll('.pw-model-item').forEach(function(el) { el.classList.remove('active'); });
  var mel = document.getElementById('pw-model-' + i);
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
  updateBackButton();
  var displayName = getDisplayName(model);

  // Clear any stale error overlay before loading
  ChartError.clear();

  // Update sidebar note
  var noteEl = document.querySelector('.sidebar-note');
  if (noteEl) {
    noteEl.innerHTML = '🔄 Loading ' + escapeHtml(displayName) + '…';
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
    markModelOk(_pwState.modelIdx);
    ChartError.clear();

    // Populate species selector (with "All species" overlay option)
    var selEl = document.getElementById('species-select');
    if (selEl) {
      selEl.innerHTML = '<option value="-1" selected>— All species —</option>' +
        spNames.map(function(n, i) {
          return '<option value="' + i + '">' + n + '</option>';
        }).join('');
    }

    // Update sidebar note
    if (noteEl) {
      noteEl.innerHTML = '🧬 <b>' + escapeHtml(displayName) + '</b><br>' +
        '<span style="font-size:10px;color:#8899aa;">' + escapeHtml(model.id) + ' · ' + spNames.length +
        ' species · ' + timeLen + ' steps</span><br>' +
        '<span style="color:var(--color-text-highlight, #c0d0e0);">' +
        '📊 ' + spNames.length + ' species · Input ' + ctxLen + ' pts</span>';
      noteEl.style.background = 'var(--color-tag-green, #1a3020)';
      noteEl.style.borderColor = '#2a4a30';
      noteEl.style.color = '#88aa88';
    }

    // Ensure skeleton shows for at least _skelMinMs, then render all species overlaid
    ChartSkeleton.ensureMin(function() {
      ChartSkeleton.hide();  // fade out skeleton
      // Small delay so fade starts before chart renders
      setTimeout(function() {
        renderAllSpecies();
      }, 100);
    });

  }).catch(function(err) {
    _pwState.loading = false;
    console.error('Failed to load trajectory:', err);

    // Mark model as failed in sidebar
    markModelFailed(_pwState.modelIdx);

    // Let skeleton show for minimum time, then transition to error state
    ChartSkeleton.ensureMin(function() {
      ChartSkeleton.hide();
      setTimeout(function() {
        ChartError.show(model, err.message);
      }, 200);
    });

    if (noteEl) {
      noteEl.innerHTML = '❌ Failed to load ' + escapeHtml(displayName);
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
  _pwState.overlayAll = false;

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
  var displayName = getDisplayName(model);
  drawChart(ctxTime, ctxData, gtTime, gtData, null, spName, displayName);
  attemptPrediction(ctxData, ctxTime, gtTime, gtData, spName, displayName, model.regime);
}

// ================================================================
// Multi-species overlay — all species of one model on one chart
// ================================================================
function renderAllSpecies() {
  if (!_pwState.currentModel || !_pwState.fullColumns) return;
  _pwState.overlayAll = true;

  var model       = _pwState.currentModel;
  var spNames     = _pwState.spNames;
  var columns     = _pwState.fullColumns;
  var time        = _pwState.fullTime;
  var ctxLen      = _pwState.ctxLen;
  var displayName = getDisplayName(model);

  // Cap overlay to keep the chart readable for large models
  var truncated = spNames.length > MAX_OVERLAY_SPECIES;
  var shown     = truncated ? spNames.slice(0, MAX_OVERLAY_SPECIES) : spNames;

  var ctxTime = time.slice(0, ctxLen);
  var gtTime  = time.slice(ctxLen);
  var speciesCtx = {}, speciesGt = {};
  shown.forEach(function(sp) {
    var d = columns[sp] || [];
    speciesCtx[sp] = d.slice(0, ctxLen);
    speciesGt[sp] = d.slice(ctxLen);
  });

  // Draw actual (ground truth) lines immediately; predictions fill in async
  drawAllSpeciesChart(shown, ctxTime, speciesCtx, speciesGt, gtTime, {}, displayName, truncated, spNames.length);

  // Fire predictions for every species in parallel
  var predictions = {};
  var pending = shown.length;
  function done() {
    pending--;
    if (pending === 0) {
      drawAllSpeciesChart(shown, ctxTime, speciesCtx, speciesGt, gtTime, predictions, displayName, truncated, spNames.length);
    }
  }
  shown.forEach(function(sp) {
    var ctxData = speciesCtx[sp];
    var gtLen   = (speciesGt[sp] || []).length;
    var reqBody = {
      context: ctxData,
      prediction_length: Math.min(256, gtLen),
      traj_pattern: REGIME_TO_PATTERN[model.regime] || 0,
      period: (model.regime === 'oscillation') ? 12.5 : 0.0
    };
    fetch(API_BASE + '/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(reqBody)
    }).then(function(r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    }).then(function(result) {
      predictions[sp] = result.predictions || [];
      done();
    }).catch(function() {
      predictions[sp] = null;
      done();
    });
  });
}

function drawAllSpeciesChart(spNames, ctxTime, speciesCtx, speciesGt, gtTime, predictions, sysName, truncated, totalCount) {
  if (!window._predictChart) return;

  var series = [];
  var lastCtxTime = ctxTime[ctxTime.length - 1];
  var dt = ctxTime[1] - ctxTime[0];

  spNames.forEach(function(sp, i) {
    var color = SPECIES_COLORS[i % SPECIES_COLORS.length];
    var ctxData = speciesCtx[sp] || [];
    var gtData  = speciesGt[sp] || [];

    // Actual trajectory (dashed, dim) — context + ground truth
    var actualTime = ctxTime.concat(gtTime);
    var actualData = ctxData.concat(gtData);
    series.push({
      name: sp, type: 'line',
      data: actualTime.map(function(t, j) { return [t, actualData[j]]; }),
      lineStyle: { color: color, width: 1.2, type: 'dashed', opacity: 0.4 },
      itemStyle: { color: color, opacity: 0.4 },
      symbol: 'none', smooth: true
    });

    // Prediction trajectory (solid) — context + forecast
    var pred = predictions[sp];
    var predData;
    if (pred && pred.length) {
      var predTime = [];
      for (var pt = 0; pt < pred.length; pt++) {
        predTime.push(+(lastCtxTime + (pt + 1) * dt).toFixed(2));
      }
      predData = ctxTime.concat(predTime).map(function(t, j) {
        return [t, j < ctxData.length ? ctxData[j] : pred[j - ctxData.length]];
      });
    } else {
      // Pending — show observed context only
      predData = ctxTime.map(function(t, j) { return [t, ctxData[j]]; });
    }
    series.push({
      name: sp, type: 'line',
      data: predData,
      lineStyle: { color: color, width: 2 },
      itemStyle: { color: color },
      symbol: 'none', smooth: true
    });
  });

  // "now" divider on the first series
  if (series.length) {
    series[0].markLine = {
      silent: true, symbol: 'none',
      lineStyle: { color: '#8899aa', type: 'dashed', width: 1 },
      label: {
        formatter: 'now', color: '#c0d0e0', fontSize: 10, fontWeight: 600,
        backgroundColor: '#162231', padding: [2, 6], borderRadius: 3,
        borderColor: '#2a4057', borderWidth: 1
      },
      data: [{ xAxis: lastCtxTime }]
    };
  }

  var firstPred = predictions[spNames[0]];
  var forecastSteps = (firstPred && firstPred.length) ? firstPred.length : '…';
  var subtext = spNames.length + ' species · dashed = actual · solid = prediction · observed ' +
    ctxTime.length + ' → forecast ' + forecastSteps + ' steps';
  if (truncated) subtext += ' · showing first ' + spNames.length + ' of ' + totalCount;

  // Hide the single-model legend guide; the ECharts legend carries species colours
  var guide = document.getElementById('chart-legend-guide');
  if (guide) guide.style.display = 'none';

  window._predictChart.setOption({
    backgroundColor: 'transparent',
    animationDuration: 600,
    animationEasing: 'cubicOut',
    title: {
      text: sysName,
      subtext: subtext,
      left: 'center', top: 8,
      textStyle:    { color: '#e0e6ed', fontSize: 16, fontWeight: 600 },
      subtextStyle: { color: '#6a8299', fontSize: 11, fontWeight: 500 }
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#1a2a3a',
      borderColor: '#2a4057',
      textStyle: { color: '#e0e6ed', fontSize: 12 }
    },
    legend: {
      bottom: 10,
      type: 'scroll',
      icon: 'circle', itemWidth: 10, itemHeight: 10,
      textStyle: { color: '#8899aa', fontSize: 11 },
      data: spNames
    },
    grid: { left: 55, right: 28, top: 64, bottom: 44 },
    xAxis: {
      type: 'value',
      axisLine:  { lineStyle: { color: '#2a4057' } },
      axisLabel: { color: '#6a8299', fontSize: 11, margin: 8 },
      splitLine: { lineStyle: { color: '#1a2a3a' } }
    },
    yAxis: {
      type: 'value',
      axisLine:  { lineStyle: { color: '#2a4057' } },
      axisLabel: { color: '#6a8299', fontSize: 11, margin: 8 },
      splitLine: { lineStyle: { color: '#1a2a3a' } }
    },
    series: series
  }, true);  // notMerge — clear any stale series
}

// Skeleton Screen: see js/components/skeleton.js (ChartSkeleton API)
// drawChart — shared chart renderer
// ================================================================
function drawChart(ctxTime, ctxData, gtTime, gtData, apiResult, spName, sysName) {
  if (!window._predictChart) return;
  var series = [];

  var lastCtxTime = ctxTime[ctxTime.length - 1];
  var dt = ctxTime[1] - ctxTime[0];

  // 1) Actual trajectory (dashed gray) — context + ground truth as one continuous line
  var actualTime = ctxTime.concat(gtTime);
  var actualData = ctxData.concat(gtData);
  series.push({
    name: 'Actual',
    type: 'line',
    data: actualTime.map(function(t, i) { return [t, actualData[i]]; }),
    lineStyle:  { color: '#6a8299', width: 1.6, type: 'dashed' },
    itemStyle:  { color: '#6a8299' },
    symbol: 'none', smooth: true,
    markLine: {
      silent: true, symbol: 'none',
      lineStyle: { color: '#8899aa', type: 'dashed', width: 1 },
      label: {
        formatter: 'now', color: '#c0d0e0', fontSize: 10, fontWeight: 600,
        backgroundColor: '#162231', padding: [2, 6], borderRadius: 3,
        borderColor: '#2a4057', borderWidth: 1
      },
      data: [{ xAxis: lastCtxTime }]
    }
  });

  // 2) Prediction trajectory (solid orange) — context + predicted future
  if (apiResult && !apiResult._pending && apiResult.predictions) {
    var predLen = apiResult.predictions.length;
    var predTime = [];
    for (var pt = 0; pt < predLen; pt++) {
      predTime.push(+(lastCtxTime + (pt + 1) * dt).toFixed(2));
    }
    series.push({
      name: 'Prediction',
      type: 'line',
      data: ctxTime.concat(predTime).map(function(t, i) {
        return [t, i < ctxData.length ? ctxData[i] : apiResult.predictions[i - ctxData.length]];
      }),
      lineStyle:  { color: '#F39C12', width: 2.5, shadowBlur: 12, shadowColor: 'rgba(243,156,18,0.4)' },
      itemStyle:  { color: '#F39C12' },
      symbol: 'none', smooth: true
    });
  } else if (apiResult && apiResult._pending) {
    // Placeholder — dotted orange line spanning the forecast horizon, no data
    var predLen2 = Math.min(256, gtData.length);
    var predTime2 = [];
    for (var pt2 = 0; pt2 < predLen2; pt2++) {
      predTime2.push(+(lastCtxTime + (pt2 + 1) * dt).toFixed(2));
    }
    series.push({
      name: 'Prediction (pending)',
      type: 'line',
      data: predTime2.map(function(t) { return [t, null]; }),
      lineStyle:  { color: '#F39C12', width: 1.5, type: 'dotted' },
      itemStyle:  { color: '#F39C12' },
      symbol: 'none'
    });
  }

  var subtext;
  if (apiResult && apiResult._pending) {
    subtext = '⚠ Prediction pending — waiting for AI model weights';
  } else if (apiResult && apiResult.predictions) {
    subtext = 'Observed ' + ctxData.length + '  →  forecast ' + apiResult.predictions.length + ' steps';
  } else {
    subtext = 'Observed ' + ctxData.length + '  ·  actual ' + (ctxData.length + gtData.length) + ' steps';
  }

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
      textStyle:    { color: '#e0e6ed', fontSize: 16, fontWeight: 600 },
      subtextStyle: { color: (apiResult && apiResult._pending ? '#F39C12' : '#6a8299'), fontSize: 11, fontWeight: 500 }
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#1a2a3a',
      borderColor: '#2a4057',
      textStyle: { color: '#e0e6ed', fontSize: 12 }
    },
    legend: { show: false },
    grid: { left: 55, right: 28, top: 64, bottom: 30 },
    xAxis: {
      type: 'value',
      axisLine:  { lineStyle: { color: '#2a4057' } },
      axisLabel: { color: '#6a8299', fontSize: 11, margin: 8 },
      splitLine: { lineStyle: { color: '#1a2a3a' } }
    },
    yAxis: {
      type: 'value',
      axisLine:  { lineStyle: { color: '#2a4057' } },
      axisLabel: { color: '#6a8299', fontSize: 11, margin: 8 },
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
// Unified entry point — bubble chart "Load Trajectory to Prediction"
// Renders the single-model actual+prediction view (same as clicking a
// model in the pathway list), replacing the legacy multi-line view.
// ================================================================
function openModelPrediction(model) {
  if (!model) return;

  // Reset to a clean single-model state (model not in the pathway list).
  _pwState.currentModel = model;
  _pwState.modelIdx     = -1;   // no list item to highlight
  _pwState.spIdx        = 0;

  // Switch to the prediction view.
  document.querySelectorAll('.nav-btn').forEach(function(b) {
    b.classList.toggle('active', b.dataset.view === 'predict');
  });
  document.querySelectorAll('.view').forEach(function(v) {
    v.classList.toggle('active', v.id === 'view-predict');
  });

  var backBubbles = document.getElementById('btn-back-to-bubbles');
  if (backBubbles) backBubbles.style.display = '';

  // Show the pathway sidebar (single-model view uses the species selector + list).
  var listEl = document.getElementById('system-list');
  var customPanel = document.getElementById('custom-input-panel');
  if (listEl) listEl.style.display = '';
  if (customPanel) customPanel.style.display = 'none';
  document.querySelectorAll('.input-tab').forEach(function(t) {
    t.classList.toggle('active', t.dataset.tab === 'models');
  });

  // Chart may have been initialized while hidden — force resize after view switch.
  if (!window._predictChart) {
    window._predictChart = echarts.init(document.getElementById('predict-chart'));
  }
  setTimeout(function() { if (window._predictChart) window._predictChart.resize(); }, 150);

  loadPathwayTrajectory(model);
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
