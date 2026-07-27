// 气泡图 — D3 Circle Packing + dark theme
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
  // Level 1 颜色 → Level 2 变体 (略浅)
  var l2ColorVariants = {
    stable:      ['#6DC88E','#82D49E','#97E0AE','#ACECBE'],
    oscillation: ['#AD6BC4','#BF7DD4','#D18FE4','#E3A1F4'],
    growth:      ['#6DABE5','#82BBF0','#97CBFB','#ACDBFF'],
    decay:       ['#F0A34A','#F5B560','#FAC776','#FFD98C']
  };

  // 原始 regime 合并映射
  var regimeToL1 = {
    directly_stable: 'stable',
    oscillation: 'oscillation',
    inc_stable: 'growth',
    increasing: 'growth',
    dec_stable: 'decay',
    decreasing: 'decay'
  };

  // 构建三级 hierarchy
  function buildBioHierarchy(models) {
    // L1 分组
    var l1Groups = {};
    models.forEach(function(m) {
      var l1 = regimeToL1[m.regime] || 'stable';
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
        // L3: 模型列表，>20 个则分组
        var groups = [];
        var chunkSize = 18; // 每组最多 18 个模型
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
    // 三级：L1 大类 → L2 物种规模(泡泡) → 卡片网格
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
            // 每个子类只展示 6 个代表性模型泡泡
            var allModels = l2.models.slice();
            var sampleModels = allModels;
            if (allModels.length > 6) {
              // 按物种数均匀抽样
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
              models: l2.models,  // 全量模型 → 卡片网格
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

    // 颜色: L1/L2/L3 使用 L1 变体系列, L4 模型用 L2 颜色
    var l1ColorMap = {};
    bioL1Defs.forEach(function(d) { l1ColorMap[d.key] = d.color; });
    var l2ColorMap = {};
    Object.keys(l2ColorVariants).forEach(function(k) {
      l2ColorMap[k] = {};
      sizeLabels.forEach(function(sz, i) {
        l2ColorMap[k][sz.key] = l2ColorVariants[k][i];
      });
    });

    // Color accessor — 根据 level 返回合适的颜色
    window._bioColor = function(d) {
      if (!d || !d.data) return '#555';
      var l1 = d.data.l1key;
      if (!l1) {
        // Try parent
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
      // Level 4: individual models — use L2 color
      var l1k = d.data.l1key;
      if (d.parent && d.parent.data && d.parent.data.sizeKey) {
        var cm3 = l2ColorMap[l1k] || {};
        return cm3[d.parent.data.sizeKey] || l1ColorMap[l1k] || '#555';
      }
      return l1ColorMap[l1k] || '#555';
    };

    color = { toString: function() { return '#555'; } }; // fallback, not used directly

    // groupMeta for navigation info
    groupMeta = {};
    bioL1Defs.forEach(function(d) { groupMeta[d.key] = d; });
    sizeLabels.forEach(function(sz) { groupMeta[sz.key] = sz; });
    groups = bioL1Defs.map(function(d) { return d.key; });

    // shortNames for truncation in Level 4
    var _shortNames = shortNames;
    shortNames = {}; // bio mode doesn't use shortNames for model truncation; handled inline
  } else {
    groupDefs = mlFamilyDefs;
    groupMeta = {}; groupDefs.forEach(function(d) { groupMeta[d.key] = d; });
    groups = groupDefs.map(function(d) { return d.key; });
    color = d3.scaleOrdinal().domain(groups).range(groupDefs.map(function(d) { return d.color; }));

    // ML: 按家族分组
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

  // ---- SVG: 透明背景 ----
  var svg = d3.select(container).append('svg')
    .attr('viewBox', '0 0 ' + W + ' ' + H)
    .attr('width', W)
    .attr('height', H)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .attr('style', 'width:100%;height:100%;display:block;background:transparent;cursor:pointer;');

  // ---- Tooltip: 深色主题 ----
  var tooltip = d3.select(container).append('div').attr('class','bubble-tooltip')
    .style('position','absolute').style('pointer-events','none').style('opacity',0)
    .style('background','#1a2a3a').style('color','#e0e6ed')
    .style('padding','10px 16px').style('border-radius','10px')
    .style('font-size','13px').style('line-height','1.6')
    .style('border','1px solid #2a4057')
    .style('box-shadow','0 4px 20px rgba(0,0,0,0.5)')
    .style('max-width','360px').style('z-index','10')
    .style('transition','opacity 0.12s');

  // ============================
  // 图层: 圆圈 → 标签
  // ============================
  var gNode  = svg.append('g');
  var gLabel = svg.append('g').attr('pointer-events','none').attr('text-anchor','middle');

  var descendants = root.descendants().slice(1);

  window._bubbleSelected = null;
  var hoverTimer = null;

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
    .on('mouseenter', function(event, d) {
      // 父级圆圈事件来自子元素 → 忽略，避免 L2/L3 tooltip 冲突
      if (d.children && event.target !== this) return;
      var self = d3.select(this);
      clearTimeout(hoverTimer);
      self.attr('stroke', d.children ? 'rgba(255,255,255,0.6)' : '#ffffff')
        .attr('stroke-width', d.children ? 2.5 : 3.5);

      hoverTimer = setTimeout(function() {
        if (!d.children && d.data.model) {
          var m = d.data.model, gi = groupMeta[mode==='bio'?m.regime:m.family] || {};
          if (mode === 'bio') {
            var l1 = regimeToL1[m.regime] || 'stable';
            var l1info = groupMeta[l1] || {};
            tooltip.html(
              '<b style="font-size:15px;color:#e0e6ed;">'+m.name+'</b><br/>'+
              '<span style="color:#8899aa;">'+(l1info.icon||'')+' '+l1info.label+' · '+m.regime+'</span><br/>'+
              '🧬 Species: <b>'+m.species+'</b><br/>'+
              '🆔 '+m.id);
          } else {
            var met = '';
            if (m.paper_metrics && Object.keys(m.paper_metrics).length)
              met = '<br/>' + Object.entries(m.paper_metrics).map(function(kv) { return '📊 '+kv[0]+': <b>'+kv[1]+'</b>'; }).join('&nbsp;&nbsp;');
            tooltip.html(
              '<b style="font-size:15px;color:#e0e6ed;">'+m.name+'</b><br/>'+
              '<span style="color:#8899aa;">'+(gi.icon||'')+' '+t('family.'+m.family)+' · '+t('type.'+m.type)+'</span><br/>'+
              (m.hidden_dim>0 ? '📐 hidden_dim='+m.hidden_dim+', layers='+m.layers+'<br/>' : '🧠 '+t('tooltip.pretrained')+'<br/>')+
              '⚙ lr='+m.lr+', ctx='+m.context_len+'→pred='+m.pred_len + met);
          }
          tooltip.style('opacity', 1);
        } else if (d.children) {
          var name = d.data.name || '';
          var mcount = d.data.modelCount || '';
          var showCount = d.children.length;
          tooltip.html(
            '<b style="font-size:14px;color:#e0e6ed;">'+name+'</b><br/>'+
            '<span style="color:#8899aa;font-size:12px;">Sample ' + showCount + ' / ' + (mcount || showCount) + ' total</span><br/>'+
            '<span style="color:#F39C12;font-size:12px;">View all models →</span>');
          tooltip.style('opacity', 1);
        }
      }, 150);
    })
    .on('mousemove', function(event) {
      var r = container.getBoundingClientRect();
      var tx = event.clientX - r.left + 16, ty = event.clientY - r.top - 10;
      if (tx + 370 > r.width) tx = event.clientX - r.left - 370;
      if (ty < 4) ty = 4;
      tooltip.style('left',tx+'px').style('top',ty+'px');
    })
    .on('mouseleave', function() {
      clearTimeout(hoverTimer);
      hoverTimer = setTimeout(function() { tooltip.style('opacity', 0); }, 150);
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
    .on('click', function(event, d) {
      event.stopPropagation();
      if (!d.children && d.data.model) {
        // 高亮选中气泡
        if (window._bubbleSelected) {
          window._bubbleSelected
            .attr('stroke', '#ffffff')
            .attr('stroke-width', 2.5);
        }
        window._bubbleSelected = d3.select(this);
        window._bubbleSelected
          .attr('stroke', '#F39C12')
          .attr('stroke-width', 3.5)
          .attr('filter', 'drop-shadow(0 0 6px rgba(243,156,18,0.6))');

        // 打开侧边栏
        openDetailPanel(d.data.model, mode, groupMeta);
      }
      else if (d.children) {
        // 子节点是具体模型 → 卡片网格（L1 大类直接进入）
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
    .style('font-size', function(d) { return (mode === 'bio' && d.data.level === 1) ? '20px' : '14px'; })
    .style('font-family','-apple-system,BlinkMacSystemFont,"Segoe UI","Inter","PingFang SC","Microsoft YaHei",sans-serif')
    .style('paint-order','stroke').style('stroke','rgba(15,25,35,0.70)')
    .style('stroke-width','2.8px').style('stroke-linecap','round').style('stroke-linejoin','round')
    .style('pointer-events','none')
    .text(function(d) { return d.data.name || ''; });

  // ============================
  // 模型名称 — 放入气泡内部正中心
  // ============================
  var modelLabelData = descendants.filter(function(d) { return !d.children && d.data.model; });

  var modelTexts = gLabel.selectAll('text.model-center').data(modelLabelData, function(d) { return d.data.name + '-' + d.data.id; })
    .join('text').attr('class','model-center')
      .attr('x', function(d) { return d.x; })
      .attr('y', function(d) { return d.y; })
      .attr('text-anchor','middle')
      .attr('dominant-baseline','central')
      .text(function(d) {
        var name = d.data.name;
        if (name.length > 18) name = (shortNames[name] || name.substring(0, 16) + '…');
        else if (name.length > 13) name = name.substring(0, 11) + '…';
        return name;
      })
      .attr('fill','#ffffff')
      .style('font-weight','600')
      .style('font-size', function(d) {
        var sz = Math.max(9, Math.min(13, d.r / 5.5));
        return sz + 'px';
      })
      .style('font-family','-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif')
      .style('pointer-events','none')
      .style('paint-order','stroke')
      .style('stroke','rgba(0,0,0,0.35)')
      .style('stroke-width','1.5px')
      .style('stroke-linecap','round')
      .style('stroke-linejoin','round')
      .attr('visibility','hidden');

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

  // 家族导航条
  var familyNav = document.getElementById('family-nav');
  var familyNameEl = document.getElementById('family-name');
  var backBtn = document.getElementById('btn-back-all');

  if (backBtn) {
    backBtn.addEventListener('click', function() {
      if (focus !== root) zoomBack();
    });
  }

  function zoomBack() {
    // 返回上一级
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
    // Show family nav at level 1+
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

  var currentCardModels = [];
  var currentCardPage = 0;
  var currentCardParent = null; // the Level 2 bubble node
  var CARDS_PER_PAGE = 20;

  function extractYear(name) {
    var m = name.match(/(\d{4})/);
    return m ? parseInt(m[1]) : 0;
  }

  function showCardGrid(node) {
    currentCardParent = node;
    // 收集模型：优先 node.data.models，否则从子节点提取 model
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

    // 切换视图
    document.getElementById('bubble-chart').style.display = 'none';
    if (familyNav) familyNav.style.display = 'none';
    cardGridView.style.display = '';

    // 标题
    if (cgTitle) cgTitle.textContent = (node.data.name || '') + ' · ' + currentCardModels.length + ' models';

    // 搜索 & 排序事件
    if (cgSearch) {
      cgSearch.value = '';
      cgSearch.oninput = function() { currentCardPage = 0; renderCards(); };
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
    // 规模筛选
    var filter = cgFilter ? cgFilter.value : 'all';
    if (filter !== 'all') {
      var ranges = { micro: [1,10], small: [11,30], medium: [31,100], large: [101,9999] };
      var range = ranges[filter] || [1,9999];
      models = models.filter(function(m) {
        var sp = m.species || 1;
        return sp >= range[0] && sp <= range[1];
      });
    }
    // 搜索
    var q = cgSearch ? cgSearch.value.toLowerCase().trim() : '';
    if (q) {
      models = models.filter(function(m) {
        return (m.name || '').toLowerCase().indexOf(q) >= 0 ||
               (m.id || '').toLowerCase().indexOf(q) >= 0;
      });
    }
    // 排序
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

    // 颜色映射
    var regimeColorClass = {
      stable: 'regime-stable', oscillation: 'regime-osc',
      growth: 'regime-growth', decay: 'regime-decay'
    };
    var regimeLabelMap = { stable: 'Stable', oscillation: 'Osc', growth: 'Growth', decay: 'Decay' };

    var html = '';
    pageModels.forEach(function(m) {
      var l1 = (typeof regimeToL1 !== 'undefined') ? (regimeToL1[m.regime] || 'stable') : 'stable';
      var year = extractYear(m.name);
      html += '<div class="model-card" data-id="' + m.id + '">' +
        '<div class="mc-name">' + (m.name || '') + '</div>' +
        '<div class="mc-meta">🧬 Species: <b>' + (m.species || '?') + '</b></div>' +
        '<div class="mc-tags">' +
          '<span class="mc-tag ' + (regimeColorClass[l1] || '') + '">' + (regimeLabelMap[l1] || l1) + '</span>' +
          (year ? '<span class="mc-tag year">' + year + '</span>' : '') +
        '</div>' +
        '<div class="mc-id">' + (m.id || '') + '</div>' +
      '</div>';
    });
    cgGrid.innerHTML = html || '<div style="color:#556678;text-align:center;padding:40px;">No matching models</div>';

    // 卡片点击 → 详情面板
    cgGrid.querySelectorAll('.model-card').forEach(function(card) {
      card.addEventListener('click', function() {
        var mid = card.getAttribute('data-id');
        var model = currentCardModels.find(function(m) { return m.id === mid; });
        if (model) openDetailPanel(model, mode, groupMeta);
      });
    });

    // 分页
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
      // Level 4 (individual models): show names
      if (d.data.level === 4 || (d.data.isModelGroup)) {
        var leaves = d.children || [];
        modelTexts.filter(function(md) { return leaves.indexOf(md) >= 0; })
          .attr('visibility','visible');
      }
    } else if (d === root) {
      if (familyNav) familyNav.style.display = 'none';
    }
  }

  window._bubbleChart = { redraw: function() { initBubbleChart(models, mode); } };
  var rt; window._bubbleResizeHandler = function() {
    clearTimeout(rt); rt = setTimeout(function() { initBubbleChart(models, mode); }, 300);
  };
  window.addEventListener('resize', window._bubbleResizeHandler);
  window.addEventListener('langchange', function() { initBubbleChart(models, mode); });
}

// =====================================================================
// 全局侧边栏详情面板
// =====================================================================
function openDetailPanel(model, mode, groupMeta) {
  var panel = document.getElementById('detail-panel');
  var overlay = document.getElementById('detail-overlay');
  var content = document.getElementById('detail-content');
  if (!panel || !content) return;

  var html = '';

  if (mode === 'bio') {
    var l1key = (typeof regimeToL1 !== 'undefined') ? (regimeToL1[model.regime] || 'stable') : 'stable';
    var l1info = groupMeta ? (groupMeta[l1key] || {}) : {};
    var hfBase = 'https://huggingface.co/datasets/HengRao/SysBio-Traj/resolve/main/Data/' + model.id + '/' + model.name;
    html +=
      '<div class="dp-header">' +
        '<div class="dp-name">🧬 ' + (model.name || '') + '</div>' +
        '<div class="dp-id">' + (model.id || '') + '</div>' +
      '</div>' +
      '<div class="dp-section">' +
        '<div class="dp-section-title">Basic Info</div>' +
        '<div class="dp-row"><span class="dp-label">Dynamic Type</span><span class="dp-value">' + (l1info.icon || '') + ' ' + (l1info.label || model.regime || '—') + '</span></div>' +
        '<div class="dp-row"><span class="dp-label">Original Regime</span><span class="dp-value">' + (model.regime || '—') + '</span></div>' +
        '<div class="dp-row"><span class="dp-label">Species</span><span class="dp-value">' + (model.species || '—') + '</span></div>' +
        '<div class="dp-row"><span class="dp-label">Data Source</span><span class="dp-value">SysBio-Traj</span></div>' +
      '</div>' +
      '<div class="dp-section">' +
        '<div class="dp-section-title">Data Files</div>' +
        '<div class="dp-links">' +
          '<a class="dp-link" href="' + hfBase + '.csv" target="_blank" title="Trajectory CSV (512 time steps)">📊 Trajectory CSV ↗</a>' +
          '<a class="dp-link" href="' + hfBase + '.xml" target="_blank" title="SBML source model">📄 SBML XML ↗</a>' +
          '<a class="dp-link" href="' + hfBase + '_conditions.json" target="_blank" title="Per-species regime labels">🏷 Conditions JSON ↗</a>' +
          '<a class="dp-link" href="' + hfBase.replace('/' + model.name, '') + '/initial_conditions.json" target="_blank" title="Initial conditions">🔬 Initial Conditions ↗</a>' +
        '</div>' +
      '</div>' +
      '<div class="dp-section">' +
        '<button class="btn-predict" onclick="loadBioTrajectory(\'' + model.id + '\',\'' + model.name + '\',' + model.species + ')">📈 Load Trajectory to Prediction</button>' +
      '</div>';
  } else {
    // ML 模式
    var typeLabel = model.type === 'probabilistic' ? 'Probabilistic' : model.type === 'point' ? 'Point' : 'Zero-shot';
    var familyName = (typeof t === 'function') ? t('family.' + model.family) : model.family;

    html +=
      '<div class="dp-header">' +
        '<div class="dp-name">📦 ' + (model.name || '') + '</div>' +
        '<div class="dp-id">' + (model.id || '') + '</div>' +
      '</div>' +
      '<div class="dp-section">' +
        '<div class="dp-section-title">Key Parameters</div>' +
        '<div class="dp-row"><span class="dp-label">Family</span><span class="dp-value">' + (familyName || model.family || '—') + '</span></div>' +
        '<div class="dp-row"><span class="dp-label">Type</span><span class="dp-value">' + typeLabel + '</span></div>';

    if (model.hidden_dim > 0) {
      html +=
        '<div class="dp-row"><span class="dp-label">Hidden Dim</span><span class="dp-value">' + model.hidden_dim + '</span></div>' +
        '<div class="dp-row"><span class="dp-label">Layers</span><span class="dp-value">' + model.layers + '</span></div>' +
        '<div class="dp-row"><span class="dp-label">Learning Rate</span><span class="dp-value">' + model.lr + '</span></div>';
    } else {
      html += '<div class="dp-row"><span class="dp-label">Pretrained</span><span class="dp-value">' + (model.pretrained || '—') + '</span></div>';
    }

    html +=
        '<div class="dp-row"><span class="dp-label">Context Length</span><span class="dp-value">' + model.context_len + '</span></div>' +
        '<div class="dp-row"><span class="dp-label">Prediction Length</span><span class="dp-value">' + model.pred_len + '</span></div>' +
      '</div>';

    // Metrics
    if (model.paper_metrics && Object.keys(model.paper_metrics).length) {
      html += '<div class="dp-section"><div class="dp-section-title">Metrics</div><div class="dp-tags">';
      Object.entries(model.paper_metrics).forEach(function(e) {
        html += '<span class="dp-tag regime">' + e[0] + ': ' + e[1] + '</span>';
      });
      html += '</div></div>';
    }

    // Features
    var features = model.features || [];
    if (features.length) {
      html += '<div class="dp-section"><div class="dp-section-title">Features</div><div class="dp-tags">';
      features.forEach(function(f) {
        html += '<span class="dp-tag feature">' + f + '</span>';
      });
      html += '</div></div>';
    }

    // Description
    if (model.description) {
      html += '<div class="dp-section"><div class="dp-section-title">Description</div><div class="dp-desc">' + model.description + '</div></div>';
    }
  }

  content.innerHTML = html;
  panel.classList.add('active');
  overlay.classList.add('active');
}

function closeDetailPanel() {
  var panel = document.getElementById('detail-panel');
  var overlay = document.getElementById('detail-overlay');
  if (panel) panel.classList.remove('active');
  if (overlay) overlay.classList.remove('active');

  // 取消高亮
  if (typeof window._bubbleSelected !== 'undefined' && window._bubbleSelected) {
    window._bubbleSelected
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 2.5)
      .attr('filter', null);
    window._bubbleSelected = null;
  }
}

// ESC 关闭
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeDetailPanel();
});
