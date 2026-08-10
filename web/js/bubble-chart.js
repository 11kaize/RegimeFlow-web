// 气泡图 — D3 Circle Packing + dark theme
// v38: 响应式字体 + 自动换行 + 全节点Tooltip + 最小半径阈值
function initBubbleChart(models, mode) {
  mode = mode || 'ml';
  var container = document.getElementById('bubble-chart');
  if (!container) return;
  container.innerHTML = '';
  // 安全回退 t() 函数
  if (typeof t !== 'function') { window.t = function(k) { return k; }; }

  var containerWidth = container.clientWidth || 1400;
  var W = Math.max(1200, containerWidth), H = Math.round(W * 0.58);

  // ========== ML 模式配置 ==========
  var mlFamilyDefs = [
    { key: 'Flow Matching', color: '#7BA8C8', icon: '🌊' },
    { key: 'Diffusion',     color: '#8DBFB4', icon: '🌫️' },
    { key: 'Transformer',   color: '#9DA8CC', icon: '⚡' },
    { key: 'MLP-Mixer',     color: '#BFA0C4', icon: '🧪' },
    { key: 'Linear',        color: '#D4AD8A', icon: '📏' },
    { key: 'Mamba',         color: '#8CB88C', icon: '🐍' },
    { key: 'Zero-shot',     color: '#CCB882', icon: '🎯' }
  ];

  // ========== Bio 三级浏览结构 ==========
  // Level 1: 4 大动态类别
  var bioL1Defs = [
    { key: 'stable',      color: '#50B86C', icon: '⚖',  label: 'Stable',        desc: 'Direct-to-steady-state' },
    { key: 'oscillation', color: '#9B59B6', icon: '🔄', label: 'Oscillation',    desc: 'Sustained periodic' },
    { key: 'growth',      color: '#5B9BD5', icon: '📈', label: 'Growth',         desc: 'Increase-then-stable' },
    { key: 'decay',       color: '#E8913A', icon: '📉', label: 'Decay',          desc: 'Decay-then-stable' }
  ];
  // Level 2: species-count tiers
  var sizeLabels = [
    { key: 'micro',  label: 'Micro',  range: [1, 10] },
    { key: 'small',  label: 'Small',  range: [11, 30] },
    { key: 'medium', label: 'Medium', range: [31, 100] },
    { key: 'large',  label: 'Large',  range: [101, 9999] }
  ];
  // Level 1 颜色 → Level 2 变体 (略浅) — see data/regime-mappings.js

  // 原始 regime 合并映射

  // 构建三级 hierarchy
  function buildBioHierarchy(models) {
    // L1 分组
    var l1Groups = {};
    models.forEach(function(m) {
      var l1 = REGIME_TO_L1[m.regime] || 'stable';
      if (!l1Groups[l1]) l1Groups[l1] = [];
      l1Groups[l1].push(m);
    });

    var children = bioL1Defs.map(function(l1def) {
      var modelsInL1 = l1Groups[l1def.key] || [];
      // L2: 按物种数量分组
      var l2Groups = {};
      modelsInL1.forEach(function(m) {
        var sp = m.species || 1;
        var sizeKey = 'medium';
        for (var s = 0; s < sizeLabels.length; s++) {
          if (sp >= sizeLabels[s].range[0] && sp <= sizeLabels[s].range[1]) {
            sizeKey = sizeLabels[s].key; break;
          }
        }
        if (!l2Groups[sizeKey]) l2Groups[sizeKey] = [];
        l2Groups[sizeKey].push(m);
      });

      var l2Children = sizeLabels.map(function(sz, si) {
        var modelsInL2 = l2Groups[sz.key] || [];
        var groups = [];
        var chunkSize = 18;
        for (var i = 0; i < modelsInL2.length; i += chunkSize) {
          var chunk = modelsInL2.slice(i, i + chunkSize);
          groups.push({
            name: sz.label + (modelsInL2.length > chunkSize ? ' (' + (i+1) + '-' + Math.min(i+chunkSize, modelsInL2.length) + ')' : ''),
            models: chunk,
            isModelGroup: true
          });
        }
        return {
          name: sz.label,
          sizeKey: sz.key,
          models: modelsInL2,
          groups: groups,
          l1key: l1def.key,
          colorIdx: si,
          value: Math.max(120, modelsInL2.length * 8)
        };
      }).filter(function(l2) { return l2.models.length > 0; });

      return {
        name: l1def.label,
        l1key: l1def.key,
        icon: l1def.icon,
        desc: l1def.desc,
        l2Children: l2Children,
        modelCount: modelsInL1.length,
        value: Math.max(240, modelsInL1.length * 6)
      };
    }).filter(function(l1) { return l1.modelCount > 0; });

    return { name: 'SysBio-Traj', children: children, isRoot: true };
  }

  var shortNames = {
    'Non-stationary Transformer': 'NSformer', 'Chronos (Bolt-Base)': 'Chronos',
    'TSFlow + PE + Regime': 'TSFlow+Rgm', 'CSDI + Regime': 'CSDI',
    'TSDiff + Regime': 'TSDiff', 'BiMamba4TS': 'BiMamba', 'iTransformer': 'iTrans',
    'PatchTST': 'PatchTST', 'TimeMixer': 'TMixer', 'DLinear': 'DLinear',
    'S-Mamba': 'S-Mamba', 'TimeXer': 'TimeXer', 'RegimeFlow': 'RegimeFlow',
    'TSFlow + PE': 'TSFlow+PE'
  };

  var groupDefs, groupMeta, groups, color, hierarchyData;

  if (mode === 'bio') {
    var bioTree = buildBioHierarchy(models);
    hierarchyData = {
      name: 'SysBio-Traj',
      children: bioTree.children.map(function(l1) {
        return {
          name: (l1.icon || '') + ' ' + l1.name,
          l1key: l1.l1key,
          level: 1,
          modelCount: l1.modelCount,
          value: l1.value,
          children: l1.l2Children.map(function(l2) {
            var allModels = l2.models.slice();
            var sampleModels = allModels;
            if (allModels.length > 6) {
              allModels.sort(function(a, b) { return (a.species || 0) - (b.species || 0); });
              var step = Math.floor(allModels.length / 6);
              sampleModels = [];
              for (var s = 0; s < 6; s++) {
                sampleModels.push(allModels[Math.min(s * step, allModels.length - 1)]);
              }
            }
            return {
              name: l2.name,
              l1key: l2.l1key,
              level: 2,
              sizeKey: l2.sizeKey,
              modelCount: l2.models.length,
              models: l2.models,
              value: 800,
              children: sampleModels.map(function(m) {
                return {
                  name: m.name,
                  fullName: m.id,
                  value: Math.max(55, (m.species || 5) * 2.5),
                  model: m,
                  l1key: l2.l1key,
                  level: 3
                };
              })
            };
          }).filter(function(l2) { return l2.modelCount > 0; })
        };
      }).filter(function(l1) { return l1.modelCount > 0; })
    };

    // 颜色映射
    var l1ColorMap = {};
    bioL1Defs.forEach(function(d) { l1ColorMap[d.key] = d.color; });
    var l2ColorMap = {};
    Object.keys(REGIME_L2_COLOR_VARIANTS).forEach(function(k) {
      l2ColorMap[k] = {};
      sizeLabels.forEach(function(sz, i) {
        l2ColorMap[k][sz.key] = REGIME_L2_COLOR_VARIANTS[k][i];
      });
    });

    window._bioColor = function(d) {
      if (!d || !d.data) return '#555';
      var l1 = d.data.l1key;
      if (!l1) {
        if (d.parent && d.parent.data && d.parent.data.l1key) l1 = d.parent.data.l1key;
        else if (d.parent && d.parent.parent && d.parent.parent.data && d.parent.parent.data.l1key) l1 = d.parent.parent.data.l1key;
      }
      if (d.data.level === 1) return l1ColorMap[l1] || '#555';
      if (d.data.level === 2) {
        var cm = l2ColorMap[l1] || {};
        return cm[d.data.sizeKey] || l1ColorMap[l1] || '#555';
      }
      if (d.data.level === 3) {
        var cm2 = l2ColorMap[l1] || {};
        return cm2[d.data.sizeKey] || l1ColorMap[l1] || '#555';
      }
      var l1k = d.data.l1key;
      if (d.parent && d.parent.data && d.parent.data.sizeKey) {
        var cm3 = l2ColorMap[l1k] || {};
        return cm3[d.parent.data.sizeKey] || l1ColorMap[l1k] || '#555';
      }
      return l1ColorMap[l1k] || '#555';
    };

    color = { toString: function() { return '#555'; } };

    groupMeta = {};
    bioL1Defs.forEach(function(d) { groupMeta[d.key] = d; });
    sizeLabels.forEach(function(sz) { groupMeta[sz.key] = sz; });
    groups = bioL1Defs.map(function(d) { return d.key; });

    var _shortNames = shortNames;
    shortNames = {};
  } else {
    groupDefs = mlFamilyDefs;
    groupMeta = {}; groupDefs.forEach(function(d) { groupMeta[d.key] = d; });
    groups = groupDefs.map(function(d) { return d.key; });
    color = d3.scaleOrdinal().domain(groups).range(groupDefs.map(function(d) { return d.color; }));

    hierarchyData = {
      name: 'ML Models',
      children: groups.map(function(fam) {
        var children = models.filter(function(m) { return m.family === fam; }).map(function(m) {
          return {
            name: shortNames[m.name] || m.name,
            fullName: m.name,
            value: m.hidden_dim > 0 ? Math.max(300, m.hidden_dim * 1.3) : 350,
            model: m
          };
        });
        return { name: fam, children: children };
      }).filter(function(f) { return f.children.length > 0; })
    };
  }

  var pack = d3.pack().size([W, H]).padding(28);
  var root = pack(d3.hierarchy(hierarchyData).sum(function(d) { return d.value; })
    .sort(function(a, b) { return b.value - a.value; }));
  var focus = root;

  // ---- SVG ----
  var svg = d3.select(container).append('svg')
    .attr('viewBox', '0 0 ' + W + ' ' + H)
    .attr('width', W)
    .attr('height', H)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .attr('style', 'width:100%;height:100%;display:block;background:transparent;cursor:pointer;');

  // ================================================================
  // 通用工具函数
  // ================================================================

  // 最小半径阈值 — 小于此值的节点不渲染文字
  var MIN_RADIUS_FOR_TEXT = 14;

  // 获取节点的 L1 key（向上遍历追溯）
  function getL1key(d) {
    if (d.data && d.data.l1key) return d.data.l1key;
    if (d.parent) return getL1key(d.parent);
    return null;
  }

  // 获取节点的 sizeKey（向上遍历追溯）
  function getSizeKey(d) {
    if (d.data && d.data.sizeKey) return d.data.sizeKey;
    if (d.parent) return getSizeKey(d.parent);
    return null;
  }

  // 构建 Tooltip HTML
  function buildTooltipHTML(d) {
    var html = '';

    if (!d.children && d.data.model) {
      // ===== 叶子节点：模型详情 =====
      var m = d.data.model;
      if (mode === 'bio') {
        var l1k = REGIME_TO_L1[m.regime] || 'stable';
        var l1info = groupMeta[l1k] || {};
        html +=
          '<div style="font-size:15px;font-weight:700;color:#ffffff;margin-bottom:4px;word-break:break-word;">' +
          (m.name || '') + '</div>' +
          '<div style="color:#8899aa;font-size:12px;margin-bottom:6px;">' +
          (l1info.icon || '') + ' ' + (l1info.label || '') + ' <span style="color:#4a5f73;">·</span> ' +
          (m.regime || '') + '</div>' +
          '<div style="display:flex;gap:14px;flex-wrap:wrap;font-size:12px;">' +
          '<span>🧬 Species: <b style="color:#e0e6ed;">' + (m.species || '—') + '</b></span>' +
          '</div>' +
          '<div style="color:#4a5f73;font-size:11px;margin-top:4px;font-family:Consolas,Monaco,monospace;">' +
          (m.id || '') + '</div>';
      } else {
        var gi = groupMeta[m.family] || {};
        html +=
          '<div style="font-size:15px;font-weight:700;color:#ffffff;margin-bottom:4px;word-break:break-word;">' +
          (m.fullName || m.name || '') + '</div>' +
          '<div style="color:#8899aa;font-size:12px;margin-bottom:6px;">' +
          (gi.icon || '') + ' ' + (t('family.' + m.family) || m.family) +
          ' <span style="color:#4a5f73;">·</span> ' +
          (t('type.' + m.type) || m.type) + '</div>';
        if (m.hidden_dim > 0) {
          html += '<div style="font-size:12px;color:#8899aa;">' +
            '📐 hidden_dim=' + m.hidden_dim + ', layers=' + m.layers + '</div>';
        } else {
          html += '<div style="font-size:12px;color:#8899aa;">🧠 ' + (t('tooltip.pretrained') || 'Pretrained') + '</div>';
        }
        html += '<div style="font-size:12px;color:#8899aa;">' +
          '⚙ lr=' + m.lr + ', ctx=' + m.context_len + '→pred=' + m.pred_len + '</div>';
        if (m.paper_metrics && Object.keys(m.paper_metrics).length) {
          html += '<div style="font-size:11px;color:#6a8299;margin-top:3px;">' +
            Object.entries(m.paper_metrics).map(function(kv) {
              return '📊 ' + kv[0] + ': <b style="color:#c0d0e0;">' + kv[1] + '</b>';
            }).join('&nbsp;&nbsp;') + '</div>';
        }
      }
    } else if (d.children) {
      // ===== 父节点：分组信息 =====
      var name = d.data.name || '';
      var mcount = d.data.modelCount;
      var level = d.data.level;
      var l1k = getL1key(d);
      var szKey = getSizeKey(d);

      html += '<div style="font-size:14px;font-weight:700;color:#ffffff;margin-bottom:4px;">' +
        name + '</div>';

      if (mode === 'bio') {
        if (level === 1) {
          var l1def = groupMeta[l1k] || {};
          html += '<div style="color:#8899aa;font-size:12px;margin-bottom:2px;">' +
            (l1def.icon || '') + ' ' + (l1def.desc || '') + '</div>';
          html += '<div style="color:#c0d0e0;font-size:12px;">' +
            '📦 <b>' + mcount + '</b> models in ' + d.children.length + ' size groups</div>';
        } else if (level === 2) {
          var l1def2 = groupMeta[l1k] || {};
          var szdef = groupMeta[szKey] || {};
          html += '<div style="color:#8899aa;font-size:12px;margin-bottom:2px;">' +
            (l1def2.icon || '') + ' ' + (l1def2.label || '') + ' · Species: ' +
            (szdef.range ? szdef.range[0] + '–' + szdef.range[1] : szKey) + '</div>';
          html += '<div style="color:#c0d0e0;font-size:12px;">' +
            '📦 <b>' + mcount + '</b> models</div>';
        } else {
          html += '<div style="color:#c0d0e0;font-size:12px;">' +
            '📦 <b>' + d.children.length + '</b> items</div>';
        }
        html += '<div style="color:#F39C12;font-size:11px;margin-top:4px;">' +
          '🖱 Click to view details →</div>';
      } else {
        html += '<div style="color:#c0d0e0;font-size:12px;">' +
          '📦 <b>' + d.children.length + '</b> models</div>';
        html += '<div style="color:#F39C12;font-size:11px;margin-top:4px;">' +
          '🖱 Click to zoom in →</div>';
      }
    }

    return html;
  }

  // ================================================================
  // Tooltip 元素
  // ================================================================
  var tooltip = d3.select(container).append('div').attr('class','bubble-tooltip')
    .style('position','absolute').style('pointer-events','none').style('opacity',0)
    .style('background','rgba(22,34,49,0.97)').style('color','#e0e6ed')
    .style('padding','12px 18px').style('border-radius','12px')
    .style('font-size','13px').style('line-height','1.65')
    .style('border','1px solid #3a5570')
    .style('box-shadow','0 6px 28px rgba(0,0,0,0.55)')
    .style('max-width','420px').style('z-index','10')
    .style('backdrop-filter','blur(12px)').style('-webkit-backdrop-filter','blur(12px)')
    .style('transition','opacity 0.10s');

  // ============================
  // 图层: 圆圈 → 标签
  // ============================
  var gNode  = svg.append('g');
  var gLabel = svg.append('g').attr('pointer-events','none').attr('text-anchor','middle');

  var descendants = root.descendants().slice(1);

  window._bubbleSelected = null;
  var hoverTimer = null;

  // ============================
  // 圆圈节点
  // ============================
  var node = gNode.selectAll('circle').data(descendants).join('circle')
    .attr('cx', function(d) { return d.x; }).attr('cy', function(d) { return d.y; })
    .attr('r',  function(d) { return d.r; })
    .attr('fill', function(d) {
      if (mode === 'bio') {
        var c = window._bioColor ? window._bioColor(d) : '#555';
        if (d.children) {
          var fc = d3.color(c);
          if (fc) { fc.opacity = d.data.level === 1 ? 0.22 : d.data.level === 2 ? 0.20 : 0.16; return fc + ''; }
          return 'rgba(180,190,200,0.18)';
        }
        return c;
      }
      if (d.children) {
        var fc = d3.color(color(d.data.name));
        if (fc) { fc.opacity = 0.22; return fc + ''; }
        return 'rgba(180,190,200,0.20)';
      }
      return color(d.parent.data.name);
    })
    .attr('stroke', function(d) {
      if (mode === 'bio') {
        if (d.children) {
          var c2 = window._bioColor ? window._bioColor(d) : '#555';
          var fc2 = d3.color(c2);
          if (fc2) { fc2.opacity = 0.45; return fc2 + ''; }
          return 'rgba(160,175,185,0.40)';
        }
        return '#ffffff';
      }
      if (d.children) {
        var fc = d3.color(color(d.data.name));
        if (fc) { fc.opacity = 0.40; return fc + ''; }
        return 'rgba(160,175,185,0.35)';
      }
      return '#ffffff';
    })
    .attr('stroke-width', function(d) { return d.children ? 1.2 : 2.5; })
    .attr('cursor', 'pointer')
    // ---- 悬停：全节点 Tooltip ----
    .on('mouseenter', function(event, d) {
      clearTimeout(hoverTimer);
      var self = d3.select(this);
      self.attr('stroke', 'rgba(255,255,255,0.75)')
        .attr('stroke-width', d.children ? 2.5 : 3.5);

      // 所有节点都显示 tooltip，无延迟
      tooltip.style('opacity', 1).html(buildTooltipHTML(d));
    })
    .on('mousemove', function(event) {
      var r = container.getBoundingClientRect();
      var tx = event.clientX - r.left + 16, ty = event.clientY - r.top - 10;
      var tw = 430;
      if (tx + tw > r.width) tx = event.clientX - r.left - tw;
      if (ty < 4) ty = 4;
      tooltip.style('left',tx+'px').style('top',ty+'px');
    })
    .on('mouseleave', function() {
      clearTimeout(hoverTimer);
      hoverTimer = setTimeout(function() { tooltip.style('opacity', 0); }, 100);
      d3.select(this)
        .attr('stroke', function(d) {
          if (mode === 'bio') {
            var c3 = window._bioColor ? window._bioColor(d) : '#555';
            if (d.children) { var fc3 = d3.color(c3); if (fc3) { fc3.opacity = 0.45; return fc3 + ''; } }
          }
          if (d.children) { var fc = d3.color(color(d.data.name)); if (fc) { fc.opacity = 0.40; return fc + ''; } }
          return '#ffffff';
        })
        .attr('stroke-width', function(d) { return d.children ? 1.2 : 2.5; });
    })
    // ---- 点击 ----
    .on('click', function(event, d) {
      event.stopPropagation();
      tooltip.style('opacity', 0);  // 点击后隐藏 tooltip
      if (!d.children && d.data.model) {
        if (window._bubbleSelected) {
          window._bubbleSelected
            .attr('stroke', '#ffffff')
            .attr('stroke-width', 2.5)
            .attr('filter', null);
        }
        window._bubbleSelected = d3.select(this);
        window._bubbleSelected
          .attr('stroke', '#F39C12')
          .attr('stroke-width', 3.5)
          .attr('filter', 'drop-shadow(0 0 6px rgba(243,156,18,0.6))');

        openDetailPanel(d.data.model, mode);
      }
      else if (d.children) {
        if (mode === 'bio' && d.children[0] && d.children[0].data && d.children[0].data.model) {
          showCardGrid(d);
        } else {
          zoom(event, d);
        }
      }
    });

  // ============================
  // 家族标签 — 圆圈内部居中
  // ============================
  var familyLabels = gLabel.selectAll('text.family-label').data(
    descendants.filter(function(d) { return d.children; })
  ).join('text').attr('class','family-label')
    .attr('x', function(d) { return d.x; }).attr('y', function(d) { return d.y; })
    .attr('text-anchor','middle')
    .attr('fill','#e8e8f0')
    .style('font-weight','700')
    .style('font-size', function(d) {
      if (mode === 'bio' && d.data.level === 1) return '20px';
      if (d.r < 60) return '13px';
      return Math.max(13, Math.min(20, d.r * 0.25)) + 'px';
    })
    .style('font-family','-apple-system,BlinkMacSystemFont,"Segoe UI","Inter","PingFang SC","Microsoft YaHei",sans-serif')
    .style('paint-order','stroke').style('stroke','rgba(15,25,35,0.70)')
    .style('stroke-width','2.8px').style('stroke-linecap','round').style('stroke-linejoin','round')
    .style('pointer-events','none')
    .text(function(d) { return d.data.name || ''; });

  // ============================
  // 模型名称 — 响应式字体 + 自动换行 + 最小半径阈值
  // ============================
  var modelLabelData = descendants.filter(function(d) { return !d.children && d.data.model; });

  var modelTexts = gLabel.selectAll('text.model-center').data(modelLabelData, function(d) {
    return (d.data.model ? d.data.model.id : '') + '-' + d.data.name;
  })
    .join('text').attr('class','model-center')
      .attr('text-anchor','middle')
      .attr('fill','#ffffff')
      .style('font-weight','600')
      .style('font-family','-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif')
      .style('pointer-events','none')
      .style('paint-order','stroke')
      .style('stroke','rgba(0,0,0,0.45)')
      .style('stroke-width','1.5px')
      .style('stroke-linecap','round')
      .style('stroke-linejoin','round')
      .attr('visibility', function(d) {
        return d.r >= MIN_RADIUS_FOR_TEXT ? 'visible' : 'hidden';
      })
      .each(function(d) {
        var el = d3.select(this);
        var r = d.r;
        if (r < MIN_RADIUS_FOR_TEXT) return;

        // 动态字号：半径越大字越大，clamp 在 10~17px
        var fontSize = Math.max(10, Math.min(17, r * 0.40));
        el.style('font-size', fontSize + 'px');

        // 圆圈内可用宽度 ≈ r * 1.6，西文字符宽 ≈ fontSize * 0.55
        var maxCharsPerLine = Math.max(4, Math.floor(r * 1.55 / (fontSize * 0.56)));

        var name = d.data.name || '';
        var lines = wrapText(name, maxCharsPerLine, 2);

        // 行高偏移
        var lineHeight = fontSize * 1.30;
        var totalHeight = (lines.length - 1) * lineHeight;
        var startY = -totalHeight / 2;
        var cx = d.x;

        el.selectAll('tspan').remove();
        el.attr('y', null);  // 移除单行时的 y 绑定，改用 tspan dy

        lines.forEach(function(line, i) {
          el.append('tspan')
            .attr('x', cx)
            .attr('dy', i === 0 ? startY + 'px' : lineHeight + 'px')
            .text(line);
        });
      });

  // ---- 底部交互提示 ----
  svg.append('text').attr('x',W/2).attr('y',H-14).attr('text-anchor','middle')
    .attr('fill','#556678').style('font-size','11px').style('pointer-events','none')
    .style('font-family','-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif')
    .style('letter-spacing','0.3px')
    .text(t('legend.hint'));

  // ---- 点击空白：关闭面板或返回上一级 ----
  svg.on('click', function(event) {
    if (event.target === svg.node() || event.target.tagName === 'rect') {
      closeDetailPanel();
      if (focus !== root) zoomBack();
    }
  });

  // ============================
  // 缩放 & 返回
  // ============================
  var view = { x: W / 2, y: H / 2, k: 1 };

  function applyTransform() {
    var tx = W / 2 - view.x * view.k, ty = H / 2 - view.y * view.k;
    var t = 'translate(' + tx + ',' + ty + ') scale(' + view.k + ')';
    gNode.attr('transform', t);
    gLabel.attr('transform', t);
  }
  applyTransform();

  var familyNav = document.getElementById('family-nav');
  var familyNameEl = document.getElementById('family-name');
  var backBtn = document.getElementById('btn-back-all');

  if (backBtn) {
    backBtn.addEventListener('click', function() {
      if (focus !== root) zoomBack();
    });
  }

  function zoomBack() {
    if (focus === root) return;
    var parent = focus.parent || root;
    modelTexts.attr('visibility','hidden');
    if (parent === root) {
      if (familyNav) familyNav.style.display = 'none';
    }
    var targetK = parent === root ? 1 : Math.min(W, H) / (parent.r * 2.2);
    var startX = view.x, startY = view.y, startK = view.k;
    focus = parent;
    svg.transition().duration(400).tween('zoom', function() {
      var ix = d3.interpolate(startX, parent.x), iy = d3.interpolate(startY, parent.y), ik = d3.interpolate(startK, targetK);
      return function(t) { view.x = ix(t); view.y = iy(t); view.k = ik(t); applyTransform(); };
    });
    if (parent !== root) updateFamilyNav(parent);
  }

  // ============================
  // 卡片网格视图
  // ============================
  var cardGridView = document.getElementById('card-grid-view');
  var cgGrid = document.getElementById('cg-grid');
  var cgTitle = document.getElementById('cg-title');
  var cgSearch = document.getElementById('cg-search');
  var cgSort = document.getElementById('cg-sort');
  var cgFilter = document.getElementById('cg-filter');
  var cgPagination = document.getElementById('cg-pagination');
  var cgBack = document.getElementById('cg-back');

  // Guard: if required DOM elements don't exist, skip card-grid setup
  if (!cardGridView || !cgGrid) {
    // Card grid view not available in this page — functions that use
    // these elements (showCardGrid, renderCards) will return early.
  }

  var currentCardModels = [];
  var currentCardPage = 0;
  var currentCardParent = null;
  var CARDS_PER_PAGE = 20;

  function extractYear(name) {
    var m = name.match(/(\d{4})/);
    return m ? parseInt(m[1]) : 0;
  }

  function showCardGrid(node) {
    currentCardParent = node;
    currentCardModels = [];
    if (node.data && node.data.models) {
      currentCardModels = node.data.models;
    } else if (node.children) {
      node.children.forEach(function(child) {
        if (child.data && child.data.model) {
          currentCardModels.push(child.data.model);
        } else if (child.data && child.data.models) {
          currentCardModels = currentCardModels.concat(child.data.models);
        }
      });
    }
    currentCardPage = 0;

    if (!cardGridView || !cgGrid) return;

    document.getElementById('bubble-chart').style.display = 'none';
    if (familyNav) familyNav.style.display = 'none';
    cardGridView.style.display = '';

    if (cgTitle) cgTitle.textContent = (node.data.name || '') + ' · ' + currentCardModels.length + ' models';

    if (cgSearch) {
      cgSearch.value = '';
      cgSearch.oninput = debounce(function() { currentCardPage = 0; renderCards(); }, 200);
    }
    if (cgSort) {
      cgSort.value = 'name';
      cgSort.onchange = function() { currentCardPage = 0; renderCards(); };
    }
    if (cgFilter) {
      cgFilter.value = 'all';
      cgFilter.onchange = function() { currentCardPage = 0; renderCards(); };
    }
    if (cgBack) {
      cgBack.onclick = function() {
        cardGridView.style.display = 'none';
        document.getElementById('bubble-chart').style.display = '';
        currentCardModels = [];
        currentCardParent = null;
        zoomBack();
      };
    }

    renderCards();
  }

  function getFilteredModels() {
    var models = currentCardModels.slice();
    var filter = cgFilter ? cgFilter.value : 'all';
    if (filter !== 'all') {
      var ranges = { micro: [1,10], small: [11,30], medium: [31,100], large: [101,9999] };
      var range = ranges[filter] || [1,9999];
      models = models.filter(function(m) {
        var sp = m.species || 1;
        return sp >= range[0] && sp <= range[1];
      });
    }
    var q = cgSearch ? cgSearch.value.toLowerCase().trim() : '';
    if (q) {
      models = models.filter(function(m) {
        return (m.name || '').toLowerCase().indexOf(q) >= 0 ||
               (m.id || '').toLowerCase().indexOf(q) >= 0;
      });
    }
    var sortBy = cgSort ? cgSort.value : 'name';
    if (sortBy === 'species') {
      models.sort(function(a, b) { return (b.species || 0) - (a.species || 0); });
    } else if (sortBy === 'year') {
      models.sort(function(a, b) { return extractYear(b.name) - extractYear(a.name); });
    } else {
      models.sort(function(a, b) { return (a.name || '').localeCompare(b.name || ''); });
    }
    return models;
  }

  function renderCards() {
    if (!cgGrid) return;
    var filtered = getFilteredModels();
    var totalPages = Math.ceil(filtered.length / CARDS_PER_PAGE);
    if (currentCardPage >= totalPages) currentCardPage = Math.max(0, totalPages - 1);
    var start = currentCardPage * CARDS_PER_PAGE;
    var pageModels = filtered.slice(start, start + CARDS_PER_PAGE);

    var regimeColorClass = {
      stable: 'regime-stable', oscillation: 'regime-osc',
      growth: 'regime-growth', decay: 'regime-decay'
    };
    var regimeLabelMap = { stable: 'Stable', oscillation: 'Osc', growth: 'Growth', decay: 'Decay' };

    var html = '';
    pageModels.forEach(function(m) {
      var l1 = REGIME_TO_L1[m.regime] || 'stable';
      var year = extractYear(m.name);
      var cardDisplay = getShortName(m) || getDisplayName(m) || (m.name || '');
      html += '<div class="model-card" data-id="' + m.id + '">' +
        '<div class="mc-name" title="' + (m.name||'') + '">' + cardDisplay + '</div>' +
        '<div class="mc-meta">🧬 Species: <b>' + (m.species || '?') + '</b></div>' +
        '<div class="mc-tags">' +
          '<span class="mc-tag ' + (regimeColorClass[l1] || '') + '">' + (regimeLabelMap[l1] || l1) + '</span>' +
          (year ? '<span class="mc-tag year">' + year + '</span>' : '') +
        '</div>' +
        '<div class="mc-id">' + (m.id || '') + '</div>' +
      '</div>';
    });
    cgGrid.innerHTML = html || '<div style="color:#556678;text-align:center;padding:40px;">No matching models</div>';

    cgGrid.querySelectorAll('.model-card').forEach(function(card) {
      card.addEventListener('click', function() {
        var mid = card.getAttribute('data-id');
        var model = currentCardModels.find(function(m) { return m.id === mid; });
        if (model) openDetailPanel(model, mode);
      });
    });

    if (cgPagination) {
      var ph = '';
      if (totalPages > 1) {
        ph += '<button class="cg-page-btn" ' + (currentCardPage === 0 ? 'disabled' : '') + ' data-pg="prev">←</button>';
        for (var p = 0; p < totalPages; p++) {
          if (totalPages <= 8 || p === 0 || p === totalPages-1 || Math.abs(p - currentCardPage) <= 2) {
            ph += '<button class="cg-page-btn' + (p === currentCardPage ? ' active' : '') + '" data-pg="' + p + '">' + (p+1) + '</button>';
          } else if (p === 1 && currentCardPage > 4) {
            ph += '<span class="cg-page-info">…</span>';
          } else if (p === totalPages-2 && currentCardPage < totalPages-5) {
            ph += '<span class="cg-page-info">…</span>';
          }
        }
        ph += '<button class="cg-page-btn" ' + (currentCardPage >= totalPages-1 ? 'disabled' : '') + ' data-pg="next">→</button>';
        ph += '<span class="cg-page-info">Page ' + (currentCardPage+1) + '/' + totalPages + '</span>';
      }
      cgPagination.innerHTML = ph;

      cgPagination.querySelectorAll('.cg-page-btn:not([disabled])').forEach(function(btn) {
        btn.addEventListener('click', function() {
          var pg = btn.getAttribute('data-pg');
          if (pg === 'prev') currentCardPage--;
          else if (pg === 'next') currentCardPage++;
          else currentCardPage = parseInt(pg);
          renderCards();
          cgGrid.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
      });
    }
  }

  function updateFamilyNav(node) {
    if (!familyNav || !familyNameEl) return;
    familyNav.style.display = 'flex';
    var level = node.data.level || 0;
    var name = node.data.name || '';
    var count = node.children ? node.children.length : 0;
    var mcount = node.data.modelCount || '';
    if (level === 2) {
      familyNameEl.textContent = name + ' · ' + count + ' sub-groups / ' + (mcount || '?') + ' models';
    } else if (level === 3) {
      familyNameEl.textContent = name + ' · ' + count + ' models';
    } else {
      familyNameEl.textContent = name + ' · ' + count + ' sub-groups' + (mcount ? ' / ' + mcount + ' models' : '');
    }
  }

  function zoom(event, d) {
    focus = d;
    var targetK = d === root ? 1 : Math.min(W, H) / (d.r * 2.2);
    var startX = view.x, startY = view.y, startK = view.k;
    var duration = event.altKey ? 7500 : 500;

    modelTexts.attr('visibility','hidden');

    svg.transition().duration(duration).tween('zoom', function() {
      var ix = d3.interpolate(startX, d.x), iy = d3.interpolate(startY, d.y), ik = d3.interpolate(startK, targetK);
      return function(t) { view.x = ix(t); view.y = iy(t); view.k = ik(t); applyTransform(); };
    });

    if (d.children && d !== root) {
      updateFamilyNav(d);
      // 缩放后：r < 阈值仍隐藏，r >= 阈值显示并渲染文字
      if (d.data.level === 4 || (d.data.isModelGroup)) {
        var leaves = d.children || [];
        var visibleLeaves = leaves.filter(function(leaf) { return leaf.r >= MIN_RADIUS_FOR_TEXT; });
        // 只显示半径达标的叶子节点
        modelTexts.filter(function(md) { return leaves.indexOf(md) >= 0 && md.r >= MIN_RADIUS_FOR_TEXT; })
          .attr('visibility','visible');
      }
    } else if (d === root) {
      if (familyNav) familyNav.style.display = 'none';
    }
  }

  window._bubbleChart = { redraw: function() { initBubbleChart(models, mode); } };
  window._bubbleResizeHandler = debounce(function() { initBubbleChart(models, mode); }, 250);
  window.addEventListener('resize', window._bubbleResizeHandler);
  window.addEventListener('langchange', function() { initBubbleChart(models, mode); });
}
