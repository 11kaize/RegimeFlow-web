// 轨迹预测演示 — 自动检测本地/生产环境
const API_BASE = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
  ? 'http://localhost:8000'
  : '';  // same origin when deployed (FastAPI serves both frontend + API)
let selectedSystem = null;
let selectedSpeciesIdx = 0;
let _bioTrajectoryLoaded = false;  // true when viewing a bio model's full CSV trajectory

// regime → i18n key 映射（系统名称/描述）
const REGIME_I18N = {
  'oscillation':          { name: 'sys.oscillation',          desc: 'sys.oscillation.desc' },
  'increasing-stable':    { name: 'sys.increasingStable',     desc: 'sys.increasingStable.desc' },
  'decreasing-stable':    { name: 'sys.decreasingStable',     desc: 'sys.decreasingStable.desc' },
  'monotonic increasing': { name: 'sys.monotonicInc',         desc: 'sys.monotonicInc.desc' },
  'monotonic decreasing': { name: 'sys.monotonicDec',         desc: 'sys.monotonicDec.desc' },
  'complex':              { name: 'sys.complex',              desc: 'sys.complex.desc' },
};

function getSysName(sys) { var k = REGIME_I18N[sys.regime]; return k ? t(k.name) : sys.name; }
function getSysDesc(sys) { var k = REGIME_I18N[sys.regime]; return k ? t(k.desc) : sys.description; }

// 后端模型状态缓存
let _backendStatus = { chronos_loaded: false, device: 'cpu' };

async function initPrediction() {
  if (typeof SAMPLE_SYSTEMS === 'undefined') return;

  // 检查后端状态
  try {
    const resp = await fetch(API_BASE + '/api/health');
    _backendStatus = await resp.json();
    console.log('Backend status:', _backendStatus);
  } catch (e) {
    console.warn('Backend not reachable, using fallback mode:', e.message);
  }

  // 渲染系统列表
  const listEl = document.getElementById('system-list');

  SAMPLE_SYSTEMS.forEach((sys, i) => {
    const div = document.createElement('div');
    div.className = 'system-item' + (i === 0 ? ' active' : '');
    div.innerHTML =
      '<div class="sys-name">' + getSysName(sys) + '</div>' +
      '<div class="sys-regime">' + t('regime.' + sys.regime) + ' · ' + t('sys.speciesCount', {n: sys.speciesCount}) + '</div>';
    div.addEventListener('click', function() { selectSystem(i); });
    listEl.appendChild(div);
  });

  // 物种选择器
  const selEl = document.getElementById('species-select');
  selEl.addEventListener('change', () => {
    selectedSpeciesIdx = parseInt(selEl.value);
    renderPrediction();
  });

  // 更新 sidebar 提示
  updateSidebarNote();

  // 初始化图表
  selectSystem(0);
  window._predictChart = echarts.init(document.getElementById('predict-chart'));
  window.addEventListener('resize', () => window._predictChart && window._predictChart.resize());

  // Tab 切换（示例系统 / 自定义数据）
  document.querySelectorAll('.input-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.input-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const showExamples = tab.dataset.tab === 'examples';
      document.getElementById('system-list').style.display = showExamples ? '' : 'none';
      document.getElementById('custom-input-panel').style.display = showExamples ? 'none' : '';
      // 切换回示例时 resize 图表
      if (showExamples) {
        setTimeout(() => window._predictChart && window._predictChart.resize(), 100);
      }
    });
  });

  // 自定义数据预测按钮
  document.getElementById('btn-custom-predict').addEventListener('click', handleCustomPredict);

  // 语言切换时重建系统列表和图表
  window.addEventListener('langchange', function() {
    // 重建 sidebar 系统列表
    listEl.innerHTML = '';
    var selIdx = selectedSystem ? SAMPLE_SYSTEMS.indexOf(selectedSystem) : 0;
    SAMPLE_SYSTEMS.forEach(function(sys, i) {
      var div = document.createElement('div');
      div.className = 'system-item' + (i === selIdx ? ' active' : '');
      div.innerHTML =
        '<div class="sys-name">' + getSysName(sys) + '</div>' +
        '<div class="sys-regime">' + t('regime.' + sys.regime) + ' · ' + t('sys.speciesCount', {n: sys.speciesCount}) + '</div>';
      div.addEventListener('click', (function(idx) { return function() { selectSystem(idx); }; })(i));
      listEl.appendChild(div);
    });
    // 刷新 regime tag 和 sidebar note
    if (selectedSystem) {
      var tagEl = document.getElementById('regime-tag');
      if (tagEl) tagEl.textContent = t('regime.' + selectedSystem.regime);
    }
    updateSidebarNote();
    // 重绘当前图表
    if (window._predictChart) renderPrediction();
  });
}

function updateSidebarNote() {
  const noteEl = document.querySelector('.sidebar-note');
  if (!noteEl) return;
  if (_backendStatus.chronos_loaded) {
    noteEl.innerHTML = t('predict.sidebarReady', {device: _backendStatus.device}).replace(/\n/g, '<br>');
    noteEl.style.background = '#1a2a20';
    noteEl.style.borderColor = '#2a4a30';
    noteEl.style.color = '#88aa88';
  } else if (_backendStatus.error) {
    noteEl.innerHTML = t('predict.sidebarError', {error: _backendStatus.error}).replace(/\n/g, '<br>');
  }
}

function selectSystem(idx) {
  _bioTrajectoryLoaded = false;  // exit bio trajectory mode
  selectedSystem = SAMPLE_SYSTEMS[idx];
  selectedSpeciesIdx = 0;

  // Restore sidebar visibility
  var listEl = document.getElementById('system-list');
  var customPanel = document.getElementById('custom-input-panel');
  if (listEl) listEl.style.display = '';
  if (customPanel) customPanel.style.display = 'none';
  document.querySelectorAll('.input-tab').forEach(function(t) { t.classList.remove('active'); });
  var examplesTab = document.querySelector('.input-tab[data-tab="examples"]');
  if (examplesTab) examplesTab.classList.add('active');

  document.querySelectorAll('.system-item').forEach((el, i) => {
    el.classList.toggle('active', i === idx);
  });

  const selEl = document.getElementById('species-select');
  selEl.innerHTML = selectedSystem.speciesNames.map((n, i) =>
    `<option value="${i}">${n}</option>`
  ).join('');

  const tagEl = document.getElementById('regime-tag');
  tagEl.textContent = t('regime.' + selectedSystem.regime);
  tagEl.className = 'regime-tag regime-' + selectedSystem.regime;

  renderPrediction();
}

async function renderPrediction() {
  // If viewing a bio model trajectory, don't overwrite with API prediction
  if (_bioTrajectoryLoaded) return;
  if (!selectedSystem || !window._predictChart) return;

  const sys = selectedSystem;
  const spName = sys.speciesNames[selectedSpeciesIdx];
  const spData = sys.species[spName];
  const time = sys.time;

  // 分割 context (0-95) 和 ground-truth prediction (96+)
  const ctxTime = time.slice(0, 96);
  const ctxData = spData.slice(0, 96);
  const gtTime = time.slice(96);          // ground-truth time
  const gtData = spData.slice(96);        // ground-truth values

  // 先绘制基础图表（context + loading 提示）
  drawChart(ctxTime, ctxData, gtTime, gtData, null, spName, getSysName(sys));

  // 调用后端 API 获取真实预测
  try {
    const resp = await fetch(API_BASE + '/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        context: ctxData,
        prediction_length: 256
      })
    });

    if (resp.ok) {
      const result = await resp.json();
      const predTime = Array.from(
        { length: result.predictions.length },
        (_, i) => ctxTime[ctxTime.length - 1] + (i + 1) * (ctxTime[1] - ctxTime[0])
      );

      drawChart(ctxTime, ctxData, gtTime, gtData, result, spName, getSysName(sys));
    } else {
      // API 返回错误，保持只显示 ground truth
      const err = await resp.text();
      console.error('API error:', err);
      drawChart(ctxTime, ctxData, gtTime, gtData, null, spName, getSysName(sys));
    }
  } catch (e) {
    // 后端不可达 — 保持 ground truth 显示
    console.warn('Backend unreachable, showing ground truth only:', e.message);
    drawChart(ctxTime, ctxData, gtTime, gtData, null, spName, getSysName(sys));
  }
}

function drawChart(ctxTime, ctxData, gtTime, gtData, apiResult, spName, sysName) {
  const series = [];

  // 1) 输入上下文 (蓝色实线)
  series.push({
    name: t('chart.context'),
    type: 'line',
    data: ctxTime.map((t, i) => [t, ctxData[i]]),
    lineStyle: { color: '#4A90D9', width: 2.5 },
    itemStyle: { color: '#4A90D9' },
    symbol: 'none',
    smooth: true,
    markArea: {
      silent: true,
      itemStyle: { color: 'rgba(74,144,217,0.08)' },
      data: [[{ xAxis: ctxTime[0] }, { xAxis: ctxTime[ctxTime.length - 1] }]]
    }
  });

  // 2) Ground truth 预测区域 (灰色虚线 = 合成数据原始值)
  series.push({
    name: t('chart.groundTruth'),
    type: 'line',
    data: gtTime.map((t, i) => [t, gtData[i]]),
    lineStyle: { color: '#6a8299', width: 1.5, type: 'dashed' },
    itemStyle: { color: '#6a8299' },
    symbol: 'none',
    smooth: true,
  });

  if (apiResult && apiResult.predictions) {
    const predLen = apiResult.predictions.length;
    const lastCtxTime = ctxTime[ctxTime.length - 1];
    const dt = ctxTime[1] - ctxTime[0];
    const predTime = Array.from({ length: predLen }, (_, i) => +(lastCtxTime + (i + 1) * dt).toFixed(2));

    // 3) RegimeFlow 预测 (橙色实线)
    series.push({
      name: t('chart.chronos'),
      type: 'line',
      data: predTime.map((t, i) => [t, apiResult.predictions[i]]),
      lineStyle: { color: '#F39C12', width: 2.5 },
      itemStyle: { color: '#F39C12' },
      symbol: 'none',
      smooth: true,
      markArea: {
        silent: true,
        itemStyle: { color: 'rgba(243,156,18,0.06)' },
        data: [[{ xAxis: predTime[0] }, { xAxis: predTime[predLen - 1] }]]
      }
    });

    // 4) 不确定度区间
    if (apiResult.lower && apiResult.upper) {
      const bandDataLo = predTime.map((t, i) => [t, apiResult.lower[i]]);
      const bandDataHi = predTime.map((t, i) => [t, apiResult.upper[i]]).reverse();
      series.push({
        name: t('chart.confidence'),
        type: 'line',
        data: [...bandDataLo, ...bandDataHi],
        lineStyle: { color: 'transparent', width: 0 },
        areaStyle: { color: 'rgba(243,156,18,0.12)' },
        symbol: 'none',
        silent: true,
        stack: 'confidence',
      });
    }
  }

  window._predictChart.setOption({
    backgroundColor: 'transparent',
    title: {
      text: sysName,
      left: 'center', top: 12,
      textStyle: { color: '#e0e6ed', fontSize: 16 }
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#1a2a3a',
      borderColor: '#2a4057',
      textStyle: { color: '#e0e6ed', fontSize: 12 }
    },
    legend: {
      data: series.map(s => s.name),
      bottom: 10,
      textStyle: { color: '#8899aa', fontSize: 11 }
    },
    grid: { left: 55, right: 30, top: 65, bottom: 55 },
    xAxis: {
      type: 'value', name: 'Time',
      nameTextStyle: { color: '#6a8299' },
      axisLabel: { color: '#6a8299' },
      axisLine: { lineStyle: { color: '#2a4057' } },
      splitLine: { lineStyle: { color: '#1a2a3a', type: 'dashed' } }
    },
    yAxis: {
      type: 'value', name: spName,
      nameTextStyle: { color: '#6a8299' },
      axisLabel: { color: '#6a8299' },
      axisLine: { lineStyle: { color: '#2a4057' } },
      splitLine: { lineStyle: { color: '#1a2a3a', type: 'dashed' } }
    },
    series: series
  }, true);
}

// 解析用户输入的时序数据
// 支持格式：每行一个值、逗号/空格/Tab分隔的 time,value 两列
function parseCustomData(rawText) {
  if (!rawText || !rawText.trim()) {
    return { error: t('status.noData') };
  }

  const lines = rawText.trim().split(/\r?\n/);
  const values = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    const parts = line.split(/[,\t\s]+/);
    let val;

    if (parts.length >= 2) {
      val = parseFloat(parts[parts.length - 1]);
    } else {
      val = parseFloat(parts[0]);
    }

    if (!isNaN(val)) {
      values.push(val);
    }
  }

  if (values.length < 10) {
    return { error: t('status.tooFew', {n: values.length}) };
  }

  return { values: values };
}

// 处理自定义数据预测
async function handleCustomPredict() {
  const btn = document.getElementById('btn-custom-predict');
  const statusEl = document.getElementById('custom-status');
  const textarea = document.getElementById('custom-data-input');
  const ctxLenInput = document.getElementById('custom-ctx-len');

  // 解析输入
  const parsed = parseCustomData(textarea.value);
  if (parsed.error) {
    statusEl.textContent = t('status.parseError') + parsed.error;
    statusEl.className = 'custom-status error';
    return;
  }

  const allValues = parsed.values;
  const ctxLen = Math.min(parseInt(ctxLenInput.value) || 96, allValues.length - 1);
  const ctxData = allValues.slice(0, ctxLen);
  const gtData = allValues.slice(ctxLen);

  // UI 反馈
  btn.disabled = true;
  btn.textContent = t('predict.btnPredicting');
  statusEl.textContent = t('status.predicting');
  statusEl.className = 'custom-status loading';

  // 构造时间轴（假设均匀间隔 0.1）
  const dt = 0.1;
  const ctxTime = Array.from({ length: ctxLen }, (_, i) => +(i * dt).toFixed(2));
  const gtTime = gtData.length > 0
    ? Array.from({ length: gtData.length }, (_, i) => +((ctxLen + i) * dt).toFixed(2))
    : [];

  const customLabel = t('predict.customData');
  // 先绘制上下文 + 基准
  drawChart(ctxTime, ctxData, gtTime, gtData, null, customLabel, customLabel);

  // 调用后端
  try {
    const resp = await fetch(API_BASE + '/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        context: ctxData,
        prediction_length: 256
      })
    });

    if (resp.ok) {
      const result = await resp.json();
      const predTime = Array.from(
        { length: result.predictions.length },
        (_, i) => +(ctxTime[ctxTime.length - 1] + (i + 1) * dt).toFixed(2)
      );
      drawChart(ctxTime, ctxData, gtTime, gtData, result, customLabel, customLabel);

      statusEl.textContent = t('status.done', {model: result.model, time: result.inference_time_ms});
      statusEl.className = 'custom-status success';
    } else {
      const err = await resp.text();
      statusEl.textContent = t('status.apiError') + err;
      statusEl.className = 'custom-status error';
    }
  } catch (e) {
    statusEl.textContent = t('status.offline');
    statusEl.className = 'custom-status error';
    console.warn('Backend unreachable for custom predict:', e.message);
  }

  btn.disabled = false;
  btn.textContent = t('predict.btnPredict');
}

// ===========================================================================
// Load bio-model trajectory CSV from HuggingFace and plot in prediction chart
// ===========================================================================
async function loadBioTrajectory(modelId, modelName, speciesCount) {
  _bioTrajectoryLoaded = true;  // prevent API prediction from overwriting

  // Switch to prediction view
  document.querySelectorAll('.nav-btn').forEach(function(b) { b.classList.remove('active'); });
  var predBtn = document.querySelector('.nav-btn[data-view="predict"]');
  if (predBtn) predBtn.classList.add('active');
  document.querySelectorAll('.view').forEach(function(v) { v.classList.remove('active'); });
  var predView = document.getElementById('view-predict');
  if (predView) predView.classList.add('active');

  // Update sidebar to show bio model state
  var listEl = document.getElementById('system-list');
  var customPanel = document.getElementById('custom-input-panel');
  if (listEl) listEl.style.display = 'none';
  if (customPanel) customPanel.style.display = 'none';
  // Reset tabs
  document.querySelectorAll('.input-tab').forEach(function(t) { t.classList.remove('active'); });

  // Update sidebar note
  var noteEl = document.querySelector('.sidebar-note');
  if (noteEl) {
    noteEl.innerHTML = '🧬 <b>' + modelName + '</b><br><span style="font-size:10px;color:#8899aa;">' + modelId + ' · ' + speciesCount + ' species · 512 steps</span><br><span style="color:#F39C12;">📊 Viewing full trajectory from HuggingFace</span>';
    noteEl.style.background = '#1a2a20';
    noteEl.style.borderColor = '#2a4a30';
    noteEl.style.color = '#88aa88';
  }

  // Init chart if needed
  if (!window._predictChart) {
    window._predictChart = echarts.init(document.getElementById('predict-chart'));
  }

  window._predictChart.setOption({
    title: { text: 'Loading ' + modelName + '...', left: 'center', top: '40%',
      textStyle: { color: '#8899aa', fontSize: 16 } },
    backgroundColor: 'transparent'
  });

  // Fetch CSV from HuggingFace
  var csvUrl = 'https://huggingface.co/datasets/HengRao/SysBio-Traj/resolve/main/Data/'
    + modelId + '/' + modelName + '.csv';

  try {
    var resp = await fetch(csvUrl);
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    var csvText = await resp.text();

    // Parse CSV
    var lines = csvText.trim().split('\n');
    var headers = lines[0].split(',');
    var columns = {};
    headers.forEach(function(h) { columns[h.trim()] = []; });

    for (var i = 1; i < lines.length; i++) {
      var vals = lines[i].split(',');
      headers.forEach(function(h, j) {
        columns[h.trim()].push(parseFloat(vals[j]));
      });
    }

    var time = columns['time'] || [];
    var timeLen = time.length;

    // Build series — all species columns except 'time'
    var series = [];
    var speciesNames = headers.filter(function(h) { return h.trim() !== 'time'; });

    // Color palette
    var palette = ['#4A90D9','#F39C12','#2ECC71','#E74C3C','#9B59B6','#1ABC9C','#F1C40F',
      '#E67E22','#3498DB','#8E44AD','#2C3E50','#16A085','#C0392B','#2980B9','#D35400'];

    speciesNames.forEach(function(sp, idx) {
      var data = columns[sp.trim()];
      if (data.length === 0) return;
      series.push({
        name: sp.trim(),
        type: 'line',
        data: time.map(function(t, i) { return [t, data[i]]; }),
        lineStyle: { color: palette[idx % palette.length], width: 1.5 },
        itemStyle: { color: palette[idx % palette.length] },
        symbol: 'none',
        smooth: true,
      });
    });

    // Also add legend-selected state so users can toggle species
    window._predictChart.setOption({
      backgroundColor: 'transparent',
      title: {
        text: modelName,
        subtext: modelId + ' · ' + speciesCount + ' species · ' + timeLen + ' time steps',
        left: 'center', top: 8,
        textStyle: { color: '#e0e6ed', fontSize: 16 },
        subtextStyle: { color: '#6a8299', fontSize: 11 }
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#1a2a3a',
        borderColor: '#2a4057',
        textStyle: { color: '#e0e6ed', fontSize: 12 }
      },
      legend: {
        type: 'scroll',
        bottom: 10,
        textStyle: { color: '#8899aa', fontSize: 10 },
        pageTextStyle: { color: '#6a8299' }
      },
      grid: { left: 55, right: 30, top: 75, bottom: 60 },
      xAxis: {
        type: 'value', name: 'Time',
        nameTextStyle: { color: '#6a8299' },
        axisLabel: { color: '#6a8299' },
        axisLine: { lineStyle: { color: '#2a4057' } },
        splitLine: { lineStyle: { color: '#1a2a3a', type: 'dashed' } }
      },
      yAxis: {
        type: 'value', name: 'Value',
        nameTextStyle: { color: '#6a8299' },
        axisLabel: { color: '#6a8299' },
        axisLine: { lineStyle: { color: '#2a4057' } },
        splitLine: { lineStyle: { color: '#1a2a3a', type: 'dashed' } }
      },
      dataZoom: [
        { type: 'inside', start: 0, end: 100 },
        { type: 'slider', bottom: 35, height: 20,
          borderColor: '#2a4057', backgroundColor: '#1a2a3a',
          dataBackground: { lineStyle: { color: '#4a90d9' }, areaStyle: { color: 'rgba(74,144,217,0.1)' } },
          textStyle: { color: '#8899aa' },
          fillerColor: 'rgba(74,144,217,0.15)' }
      ],
      series: series
    }, true);

    // Update regime tag
    var tagEl = document.getElementById('regime-tag');
    if (tagEl) { tagEl.textContent = 'BioModels'; tagEl.className = 'regime-tag regime-oscillation'; }
    var selEl = document.getElementById('species-select');
    if (selEl) {
      selEl.innerHTML = speciesNames.map(function(n, i) {
        return '<option value="' + i + '">' + n.trim() + '</option>';
      }).join('');
    }

  } catch (e) {
    console.error('Failed to load trajectory:', e);
    window._predictChart.setOption({
      title: { text: 'Failed to load: ' + modelName, left: 'center', top: '40%',
        textStyle: { color: '#E74C3C', fontSize: 14 } },
      backgroundColor: 'transparent'
    });
  }
}
