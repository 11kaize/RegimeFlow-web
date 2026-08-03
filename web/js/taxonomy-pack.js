// Taxonomy Circle Packing — D3 + mixed-mode bubble/table
// v8: individual bio models as L2 bubbles; table view for domains >30 models
var _tp = { focus: null, root: null, W: 1280, H: 800, init: false };

// Config
var MAX_VISIBLE_BUBBLES = 20;  // max sample bubbles per domain at overview
var TABLE_THRESHOLD    = 30;  // domains with >30 models → table view on click

// ================================================================
// Build hierarchy from BIO_MODELS_DATA + BIO_DOMAIN_DATA
// ================================================================
function buildHierarchy() {
  if (typeof BIO_MODELS_DATA === 'undefined' || typeof BIO_DOMAIN_DATA === 'undefined') {
    return { name: 'No Data', children: [] };
  }

  // Index bio models by ID
  var modelById = {};
  BIO_MODELS_DATA.forEach(function(m) { modelById[m.id] = m; });

  var domainColors = {
    'Mammals': '#E74C3C', 'General Eukaryotes': '#607D8B', 'Fungi': '#F39C12',
    'Plants & Algae': '#2ECC71', 'Vertebrates': '#3498DB', 'Bacteria': '#1ABC9C',
    'Biological Processes': '#9B59B6', 'Protists': '#E67E22',
    'Invertebrates': '#00BCD4', 'Viruses': '#E91E63', 'Unclassified': '#8899aa'
  };

  var domainOrder = ['Mammals','General Eukaryotes','Fungi','Plants & Algae',
    'Vertebrates','Bacteria','Biological Processes','Protists',
    'Invertebrates','Viruses','Unclassified'];

  var domainModelMap = {};  // domain → [model objects]
  // Collect all bio models and assign domain
  BIO_MODELS_DATA.forEach(function(m) {
    var domain = 'Unclassified';
    // Look up domain from BIO_DOMAIN_DATA
    for (var d in BIO_DOMAIN_DATA) {
      var found = BIO_DOMAIN_DATA[d].some(function(dm) { return dm.id === m.id; });
      if (found) { domain = d; break; }
    }
    if (!domainModelMap[domain]) domainModelMap[domain] = [];
    domainModelMap[domain].push(m);
  });

  var children = domainOrder.map(function(domName) {
    var models = domainModelMap[domName] || [];
    if (!models.length) return null;

    // Sample models for visual bubbles at overview level
    var sampleCount = Math.min(MAX_VISIBLE_BUBBLES, models.length);
    var sampled = models.slice(0, sampleCount);

    // Build child nodes for pack layout
    var childNodes = sampled.map(function(m) {
      return {
        name: m.name,
        model: m,
        value: Math.max(20, (m.species || 2) * 4)
      };
    });

    return {
      name: domName,
      color: domainColors[domName] || '#8899aa',
      models: models,
      modelCount: models.length,
      sampleCount: sampleCount,
      needsTable: models.length > TABLE_THRESHOLD,
      value: Math.max(300, models.length * 15),
      children: childNodes
    };
  }).filter(Boolean);

  return {
    name: 'SysBio-Traj',
    children: children,
    totalModels: BIO_MODELS_DATA.length
  };
}

// ================================================================
// Domain table view
// ================================================================
var _tableDomain = null;
var _tablePage = 0;
var PAGE_SIZE = 20;

function showDomainTable(domainNode) {
  _tableDomain = domainNode;
  _tablePage = 0;

  document.getElementById('taxonomy-pack-chart').style.display = 'none';
  document.getElementById('taxonomy-hint').style.display = 'none';
  document.getElementById('taxonomy-table-view').style.display = '';

  var title = document.getElementById('tax-table-title');
  if (title) title.textContent = domainNode.data.name + ' · ' + domainNode.data.modelCount + ' models';

  var searchEl = document.getElementById('tax-table-search');
  var sortEl = document.getElementById('tax-table-sort');
  var filterEl = document.getElementById('tax-table-filter');
  if (searchEl) { searchEl.value = ''; searchEl.oninput = debounce(function() { _tablePage = 0; renderTaxTable(); }, 200); }
  if (sortEl) { sortEl.value = 'name'; sortEl.onchange = function() { _tablePage = 0; renderTaxTable(); }; }
  if (filterEl) { filterEl.value = 'all'; filterEl.onchange = function() { _tablePage = 0; renderTaxTable(); }; }

  var backBtn = document.getElementById('tax-table-back');
  if (backBtn) {
    backBtn.onclick = function() {
      document.getElementById('taxonomy-table-view').style.display = 'none';
      document.getElementById('taxonomy-pack-chart').style.display = '';
      document.getElementById('taxonomy-hint').style.display = '';
      _tableDomain = null;
    };
  }

  renderTaxTable();
}

function renderTaxTable() {
  var models = _tableDomain.data.models.slice();
  var filter = document.getElementById('tax-table-filter');
  var search = document.getElementById('tax-table-search');
  var sort = document.getElementById('tax-table-sort');

  // Filter by regime
  if (filter && filter.value !== 'all') {
    models = models.filter(function(m) { return m.regime === filter.value; });
  }
  // Search
  if (search) {
    var q = search.value.toLowerCase().trim();
    if (q) {
      models = models.filter(function(m) {
        return (m.name||'').toLowerCase().indexOf(q) >= 0 ||
               (m.id||'').toLowerCase().indexOf(q) >= 0;
      });
    }
  }
  // Sort
  var sortBy = sort ? sort.value : 'name';
  if (sortBy === 'species') {
    models.sort(function(a, b) { return (b.species||0) - (a.species||0); });
  } else if (sortBy === 'id') {
    models.sort(function(a, b) { return (a.id||'').localeCompare(b.id||''); });
  } else {
    models.sort(function(a, b) { return (a.name||'').localeCompare(b.name||''); });
  }

  // ---- Stats bar ----
  var statsText = document.getElementById('tax-stats-text');
  if (statsText) {
    statsText.textContent = 'Found: ' + models.length + ' models';
  }

  var domainName = _tableDomain.data.name || '';
  _tableSelected = {};

  // ---- Select all ----
  var selectAllBtn = document.getElementById('tax-select-all');
  if (selectAllBtn) {
    selectAllBtn.textContent = '☐ Select all';
    selectAllBtn.classList.remove('active');
    selectAllBtn.onclick = function() {
      var allSelected = Object.keys(_tableSelected).length >= models.length && models.length > 0;
      if (allSelected) {
        _tableSelected = {};
        selectAllBtn.textContent = '☐ Select all';
        selectAllBtn.classList.remove('active');
      } else {
        models.forEach(function(m) { _tableSelected[m.id] = true; });
        selectAllBtn.textContent = '☑ Deselect all';
        selectAllBtn.classList.add('active');
      }
      renderTaxRows(models);
    };
  }

  // ---- Download ----
  var downloadBtn = document.getElementById('tax-download');
  if (downloadBtn) {
    downloadBtn.onclick = function() {
      var selected = models.filter(function(m) { return _tableSelected[m.id]; });
      if (!selected.length) { alert('No rows selected.'); return; }
      var header = 'ID,Name,Regime,Species,Domain';
      var rows = selected.map(function(m) {
        return [m.id, '"' + (m.name||'') + '"', m.regime||'', m.species||0, domainName].join(',');
      });
      var csv = header + '\n' + rows.join('\n');
      var blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url; a.download = domainName.replace(/\s+/g,'_') + '_models.csv';
      a.click(); URL.revokeObjectURL(url);
    };
  }

  // ---- Pagination data ----
  var totalPages = Math.ceil(models.length / PAGE_SIZE);
  if (_tablePage >= totalPages) _tablePage = Math.max(0, totalPages - 1);
  var start = _tablePage * PAGE_SIZE;
  var pageModels = models.slice(start, start + PAGE_SIZE);

  // ---- Render rows ----
  renderTaxRows(models);

  // ---- Pagination ----
  var pag = document.getElementById('tax-table-pagination');
  if (pag) {
    var ph = '';
    if (totalPages > 1) {
      // First
      ph += '<button class="tax-page-btn" ' + (_tablePage === 0 ? 'disabled' : '') + ' data-pg="0">« First</button>';
      // Prev
      ph += '<button class="tax-page-btn" ' + (_tablePage === 0 ? 'disabled' : '') + ' data-pg="' + (_tablePage-1) + '">‹ Prev</button>';
      // Page numbers
      for (var p = 0; p < totalPages; p++) {
        if (totalPages <= 9 || p === 0 || p === totalPages-1 || Math.abs(p - _tablePage) <= 2) {
          ph += '<button class="tax-page-btn' + (p === _tablePage ? ' active' : '') + '" data-pg="' + p + '">' + (p+1) + '</button>';
        } else if (p === 1 && _tablePage > 4) {
          ph += '<span class="tax-page-gap">…</span>';
        } else if (p === totalPages-2 && _tablePage < totalPages-5) {
          ph += '<span class="tax-page-gap">…</span>';
        }
      }
      // Next
      ph += '<button class="tax-page-btn" ' + (_tablePage >= totalPages-1 ? 'disabled' : '') + ' data-pg="' + (_tablePage+1) + '">Next ›</button>';
      // Last
      ph += '<button class="tax-page-btn" ' + (_tablePage >= totalPages-1 ? 'disabled' : '') + ' data-pg="' + (totalPages-1) + '">Last »</button>';
    }
    ph += '<span class="tax-page-info">Page ' + (_tablePage+1) + ' of ' + totalPages + '</span>';
    pag.innerHTML = ph;
    pag.querySelectorAll('.tax-page-btn:not([disabled])').forEach(function(btn) {
      btn.addEventListener('click', function() {
        _tablePage = parseInt(btn.getAttribute('data-pg'));
        renderTaxTable();
        document.getElementById('tax-table-grid').scrollTop = 0;
      });
    });
  }
}

var _tableSelected = {};

function renderTaxRows(models) {
  var start = _tablePage * PAGE_SIZE;
  var pageModels = models.slice(start, start + PAGE_SIZE);

  var regimeLabelMap = {
    oscillation: 'Oscillation', inc_stable: 'Inc-Stable', dec_stable: 'Dec-Stable',
    directly_stable: 'Direct-Stable', increasing: 'Growth', decreasing: 'Decay'
  };
  var domainName = _tableDomain ? _tableDomain.data.name || '' : '';

  var grid = document.getElementById('tax-table-grid');
  var html = '';
  pageModels.forEach(function(m) {
    var regimeLabel = regimeLabelMap[m.regime] || m.regime || '—';
    var checked = _tableSelected[m.id] ? ' checked' : '';
    var selClass = _tableSelected[m.id] ? ' selected' : '';

    html += '<div class="tax-row' + selClass + '" data-id="' + m.id + '">' +
      '<div class="tax-row-main">' +
        '<span class="tax-row-title">' + (m.name || '') + '</span>' +
        '<span class="tax-row-meta">' +
          (m.id || '') +
          '<span class="tax-row-meta-sep">|</span>Regime: ' + regimeLabel +
          '<span class="tax-row-meta-sep">|</span>Species: ' + (m.species || '?') +
          '<span class="tax-row-meta-sep">|</span>Domain: ' + domainName +
        '</span>' +
      '</div>' +
      '<input type="checkbox" class="tax-row-check"' + checked + ' data-id="' + m.id + '">' +
    '</div>';
  });

  if (grid) {
    grid.innerHTML = html || '<div style="color:#556678;text-align:center;padding:60px;">No matching models</div>';

    // Title click → detail panel
    grid.querySelectorAll('.tax-row-title').forEach(function(el) {
      el.addEventListener('click', function(e) {
        e.stopPropagation();
        var mid = el.closest('.tax-row').getAttribute('data-id');
        var model = _tableDomain.data.models.find(function(m) { return m.id === mid; });
        if (model) openDetailPanel(model, 'bio');
      });
    });

    // Row click → detail panel (entire row is clickable, checkbox excluded)
    grid.querySelectorAll('.tax-row').forEach(function(row) {
      row.addEventListener('click', function(e) {
        // Don't open detail if clicking checkbox
        if (e.target.classList.contains('tax-row-check')) return;
        var mid = row.getAttribute('data-id');
        var model = _tableDomain.data.models.find(function(m) { return m.id === mid; });
        if (model) openDetailPanel(model, 'bio');
      });
    });

    // Checkbox toggle
    grid.querySelectorAll('.tax-row-check').forEach(function(cb) {
      cb.addEventListener('change', function(e) {
        e.stopPropagation();
        var mid = cb.getAttribute('data-id');
        if (cb.checked) _tableSelected[mid] = true;
        else delete _tableSelected[mid];

        // Update row highlight
        var row = cb.closest('.tax-row');
        if (row) row.classList.toggle('selected', cb.checked);

        // Update select-all button state
        var allSel = document.getElementById('tax-select-all');
        if (allSel) {
          var totalOnPage = grid.querySelectorAll('.tax-row-check').length;
          var checkedCount = Object.keys(_tableSelected).length;
          if (checkedCount >= models.length && models.length > 0) {
            allSel.textContent = '☑ Deselect all';
            allSel.classList.add('active');
          } else {
            allSel.textContent = '☐ Select all';
            allSel.classList.remove('active');
          }
        }
      });
    });
  }

  // Update stats
  var statsText = document.getElementById('tax-stats-text');
  if (statsText) {
    statsText.textContent = 'Found: ' + models.length + ' models';
  }
}

// ================================================================
// initTaxonomyPack
// ================================================================
function initTaxonomyPack(data, containerId) {
  if (_tp.init) return;
  _tp.init = true;

  var el = document.getElementById(containerId);
  if (!el) return;

  var rect = el.getBoundingClientRect();
  _tp.W = Math.max(1280, rect.width || 1280);
  _tp.H = Math.max(780, rect.height || 680);
  _tp.el = el;
  _tp.containerId = containerId;

  // Build hierarchy from BIO_MODELS_DATA + BIO_DOMAIN_DATA
  _tp.root = d3.hierarchy(buildHierarchy())
    .sum(function(d) { return d.value || 0; })
    .sort(function(a, b) { return (b.value||0) - (a.value||0); });

  // Pack layout — tight spacing between L1 domains
  _tp.pack = d3.pack()
    .size([_tp.W - 30, _tp.H - 50])
    .padding(function(d) { return d.depth === 1 ? 8 : 3; });

  _tp.pack(_tp.root);

  // Color scale
  _tp.color = function(d) {
    while (d && d.depth > 1 && d.parent) d = d.parent;
    if (d && d.data && d.data.color) return d.data.color;
    return '#607D8B';
  };

  // ---- Clear & SVG ----
  el.innerHTML = '';

  _tp.svg = d3.select('#' + containerId)
    .append('svg')
    .attr('viewBox', '0 0 ' + _tp.W + ' ' + _tp.H)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .attr('width', _tp.W).attr('height', _tp.H)
    .style('width','100%').style('height','100%')
    .style('display','block').style('background','#0f1923')
    .style('overflow','hidden').style('cursor','pointer');

  // ---- Tooltip ----
  _tp.tooltip = d3.select(el).append('div').attr('class','tp-tooltip')
    .style('position','absolute').style('pointer-events','none').style('opacity',0)
    .style('background','rgba(22,34,49,0.97)').style('color','#e0e6ed')
    .style('padding','12px 18px').style('border-radius','12px')
    .style('font-size','13px').style('line-height','1.65')
    .style('border','1px solid #3a5570')
    .style('box-shadow','0 6px 28px rgba(0,0,0,0.55)')
    .style('max-width','380px').style('z-index','10')
    .style('backdrop-filter','blur(12px)').style('-webkit-backdrop-filter','blur(12px)')
    .style('transition','opacity 0.10s');

  // Footer
  _tp.svg.append('text').attr('x',_tp.W/2).attr('y',_tp.H-10)
    .attr('text-anchor','middle').style('font-size','11px')
    .style('fill','#556678').style('font-family','sans-serif')
    .style('pointer-events','none')
    .text('SysBio-Traj · ' + (_tp.root.data.totalModels || '1,050') + ' Models · ' + _tp.root.children.length + ' Domains · Click bubble to explore');

  // Container group
  _tp.g = _tp.svg.append('g');
  _tp.focus = _tp.root;

  // Background click → zoom out
  _tp.svg.on('click', function(event) {
    if (event.target === _tp.svg.node() || event.target.tagName === 'rect' || event.target.tagName === 'svg') {
      if (_tp.focus !== _tp.root) { _tp.focus = _tp.root; drawTP(); }
    }
  });

  drawTP();

  // ---- Global search initialisation ----
  initGlobalSearch();
}

// ================================================================
// Global Search
// ================================================================
function initGlobalSearch() {
  var input = document.getElementById('tax-global-search');
  var results = document.getElementById('tax-global-results');
  if (!input || !results) return;

  // Build search index: all models with their domain
  var searchIndex = [];
  if (typeof BIO_MODELS_DATA !== 'undefined' && typeof BIO_DOMAIN_DATA !== 'undefined') {
    BIO_MODELS_DATA.forEach(function(m) {
      var domain = 'Unclassified';
      for (var d in BIO_DOMAIN_DATA) {
        var found = BIO_DOMAIN_DATA[d].some(function(dm) { return dm.id === m.id; });
        if (found) { domain = d; break; }
      }
      searchIndex.push({ id: m.id, name: m.name, domain: domain, model: m });
    });
  }

  var activeIdx = -1;

  var _doSearch = debounce(function() {
    var q = input.value.toLowerCase().trim();
    if (!q) { results.style.display = 'none'; return; }

    var matches = searchIndex.filter(function(item) {
      return item.name.toLowerCase().indexOf(q) >= 0 || item.id.toLowerCase().indexOf(q) >= 0;
    }).slice(0, 15);

    if (!matches.length) {
      results.innerHTML = '<div class="tax-search-empty">No models found for "' + q + '"</div>';
      results.style.display = '';
      return;
    }

    activeIdx = -1;
    results.innerHTML = '';
    matches.forEach(function(item, i) {
      var div = document.createElement('div');
      div.className = 'tax-search-item';
      div.innerHTML =
        '<span class="tax-search-name">' + item.name + '</span>' +
        '<span class="tax-search-domain">' + item.domain + '</span>' +
        '<span class="tax-search-id">' + item.id + '</span>';
      div.addEventListener('mousedown', function(e) {
        e.preventDefault();
        selectSearchResult(item);
      });
      results.appendChild(div);
    });
    results.style.display = '';
  }, 200);

  input.addEventListener('input', _doSearch);

  input.addEventListener('keydown', function(e) {
    var items = results.querySelectorAll('.tax-search-item');
    if (!items.length) return;

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      activeIdx = Math.min(activeIdx + 1, items.length - 1);
      updateActive(items);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      activeIdx = Math.max(activeIdx - 1, 0);
      updateActive(items);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (activeIdx >= 0 && activeIdx < items.length) {
        var activeEl = items[activeIdx];
        var idx = Array.prototype.indexOf.call(items, activeEl);
        var allMatches = getCurrentMatches();
        if (idx >= 0 && idx < allMatches.length) selectSearchResult(allMatches[idx]);
      }
    } else if (e.key === 'Escape') {
      results.style.display = 'none';
      input.blur();
    }
  });

  function updateActive(items) {
    items.forEach(function(item, i) {
      item.classList.toggle('active', i === activeIdx);
    });
  }

  function getCurrentMatches() {
    var q = input.value.toLowerCase().trim();
    if (!q) return [];
    return searchIndex.filter(function(item) {
      return item.name.toLowerCase().indexOf(q) >= 0 || item.id.toLowerCase().indexOf(q) >= 0;
    }).slice(0, 15);
  }

  function selectSearchResult(item) {
    results.style.display = 'none';
    input.value = item.name;
    // Open detail panel for the model
    openDetailPanel(item.model, 'bio', null);
  }

  // Click outside closes dropdown
  document.addEventListener('click', function(e) {
    if (!input.contains(e.target) && !results.contains(e.target)) {
      results.style.display = 'none';
    }
  });
}

// ================================================================
// drawTP — render circles + labels
// ================================================================
function drawTP() {
  var F = _tp.focus;
  var G = _tp.g;
  var W = _tp.W, H = _tp.H;
  var tip = _tp.tooltip;
  var el = _tp.el;
  var color = _tp.color;

  var minDim = Math.min(W, H);

  // ---- Zoom scale: use L1-domain bounding box instead of root-circle radius ----
  // The root circle is much larger than the actual domain bubbles → wastes space.
  // We compute the tight bounding box of all visible L1 (depth=1) nodes.
  var k, tx, ty;
  if (F === _tp.root && F.children && F.children.length) {
    var minX =  Infinity, minY =  Infinity;
    var maxX = -Infinity, maxY = -Infinity;
    F.children.forEach(function(d) {
      if (d.r > 0) {
        minX = Math.min(minX, d.x - d.r);
        minY = Math.min(minY, d.y - d.r);
        maxX = Math.max(maxX, d.x + d.r);
        maxY = Math.max(maxY, d.y + d.r);
      }
    });
    var bbW = maxX - minX || 1;
    var bbH = maxY - minY || 1;
    var bbCX = (minX + maxX) / 2;
    var bbCY = (minY + maxY) / 2;
    // Scale to fill viewport with 8% margin
    var scaleX = (W * 0.92) / bbW;
    var scaleY = (H * 0.92) / bbH;
    k = Math.min(scaleX, scaleY);
    tx = W/2 - bbCX * k;
    ty = H/2 - bbCY * k;
  } else {
    // Zoomed into a single domain: fit its enclosing circle
    k = (minDim * 0.90) / (F.r * 2 + 1);
    tx = W/2 - F.x * k;
    ty = H/2 - F.y * k;
  }

  var t = d3.transition().duration(550).ease(d3.easeCubicInOut);
  G.transition(t).attr('transform', 'translate(' + tx + ',' + ty + ') scale(' + k + ')');

  var nodes = F.descendants().filter(function(d) { return d.depth >= 1; });

  // JOIN
  var grp = G.selectAll('g.cn').data(nodes, function(d) {
    return (d.data.model ? d.data.model.id : d.data.name) + '@' + d.depth;
  });

  // EXIT
  grp.exit().transition(t)
    .attr('transform', 'translate(' + F.x + ',' + F.y + ') scale(0.01)')
    .style('opacity', 0).remove();

  // ENTER
  var enter = grp.enter().append('g').attr('class','cn')
    .attr('transform', 'translate(' + F.x + ',' + F.y + ') scale(0.01)')
    .style('opacity', 0);

  // Circle
  enter.append('circle')
    .attr('r', function(d) { return Math.max(1.8, d.r); })
    .attr('fill', function(d) {
      var c = d3.color(color(d));
      if (d.depth === 1) {
        if (c) { c.opacity = 0.78; return c + ''; }
        return 'rgba(180,180,200,0.70)';
      }
      // L2: lighter
      if (c) { c.opacity = 0.45; return c + ''; }
      return 'rgba(255,255,255,0.20)';
    })
    .attr('stroke', function(d) {
      if (d.depth === 1) {
        var sc = d3.color(color(d));
        if (sc) { sc.opacity = 0.9; return sc + ''; }
        return 'rgba(255,255,255,0.70)';
      }
      // L2: bright white stroke for visibility on dark background
      return 'rgba(255,255,255,0.65)';
    })
    .attr('stroke-width', function(d) { return d.depth === 1 ? 2.5 : 1.5; })
    .attr('cursor', 'pointer')
    .attr('visibility', function(d) { return d.r >= 2.5 ? 'visible' : 'hidden'; })
    // Interactions
    .on('mouseenter', function(event, d) {
      d3.select(this).attr('stroke','#F39C12').attr('stroke-width', 3);
      var html = '';
      if (d.depth === 1) {
        var mc = d.data.modelCount || 0;
        html = '<div style="font-size:15px;font-weight:700;color:#fff;">' + d.data.name + '</div>' +
          '<div style="font-size:16px;font-weight:700;color:#F39C12;margin:4px 0;">' + mc + ' models</div>';
        if (d.data.needsTable) {
          html += '<div style="color:#F39C12;font-size:12px;">🖱 Click to view full list →</div>';
        } else {
          html += '<div style="color:#8899aa;font-size:12px;">🖱 Click to zoom in (' + mc + ' bubbles)</div>';
        }
      } else if (d.data.model) {
        var m = d.data.model;
        var parentName = (d.parent && d.parent.data) ? d.parent.data.name : '';
        html = '<div style="font-size:14px;font-weight:700;color:#fff;">' + (m.name||'') + '</div>' +
          '<div style="color:#8899aa;font-size:11px;">' + (m.id||'') + '</div>' +
          '<div style="color:#6a8299;font-size:11px;margin-top:2px;">' + parentName + '</div>' +
          '<div style="margin-top:4px;">🧬 Species: <b>' + (m.species||'?') + '</b> · 🏷 ' + (m.regime||'?') + '</div>' +
          '<div style="color:#F39C12;font-size:11px;margin-top:2px;">🖱 Click to view details →</div>';
      }
      tip.style('opacity',1).html(html);
    })
    .on('mousemove', function(event) {
      var r2 = el.getBoundingClientRect();
      var ttx = event.clientX - r2.left + 16, tty = event.clientY - r2.top - 10;
      if (ttx + 390 > r2.width) ttx = event.clientX - r2.left - 390;
      if (tty < 4) tty = 4;
      tip.style('left',ttx+'px').style('top',tty+'px');
    })
    .on('mouseleave', function(event, d) {
      var self = d3.select(this);
      if (d.depth === 1) {
        var sc = d3.color(color(d)); if (sc) { sc.opacity = 0.9; self.attr('stroke', sc+''); }
        else self.attr('stroke', 'rgba(255,255,255,0.70)');
      } else {
        self.attr('stroke', 'rgba(255,255,255,0.65)');
      }
      self.attr('stroke-width', d.depth === 1 ? 2.5 : 1.5);
      tip.style('opacity',0);
    })
    .on('click', function(event, d) {
      event.stopPropagation();
      tip.style('opacity',0);
      if (d.depth === 1) {
        if (d.data.needsTable) {
          showDomainTable(d);
        } else {
          _tp.focus = d;
          drawTP();
        }
      } else if (d.data.model) {
        openDetailPanel(d.data.model, 'bio', null);
      }
    });

  // Name text (on circles)
  enter.append('text').attr('class','tp-name')
    .attr('text-anchor','middle')
    .style('font-family','-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif')
    .style('pointer-events','none')
    .style('font-weight','700')
    .style('paint-order','stroke')
    .style('stroke-linecap','round').style('stroke-linejoin','round');

  // MERGE
  grp.merge(enter).transition(t)
    .attr('transform', function(d) { return 'translate(' + d.x + ',' + d.y + ')'; })
    .style('opacity', 1);

  // Post-transition: update text
  setTimeout(function() {
    G.selectAll('text.tp-name').each(function(d) {
      var elText = d3.select(this);
      elText.selectAll('tspan').remove();
      if (d.r < 12) { elText.text(''); return; }

      var isDomain = (d.depth === 1);
      var fontSize = isDomain ? Math.max(13, Math.min(22, d.r * 0.13)) : Math.max(8, Math.min(12, d.r * 0.22));
      elText.style('font-size', fontSize + 'px');

      var name = d.data.name || (d.data.model ? d.data.model.name : '');
      // For leaf bubbles, truncate name to fit
      if (!isDomain && name.length > 15) name = name.substring(0, 13) + '…';

      var maxChars = isDomain ? Math.max(6, Math.floor(d.r * 1.6 / (fontSize * 0.55))) : Math.max(4, Math.floor(d.r * 1.4 / (fontSize * 0.55)));
      var lines = wrapText(name, maxChars, isDomain ? 2 : 1);

      elText.style('fill', isDomain ? '#1a1a2e' : '#e0e6ed');
      elText.style('stroke', isDomain ? 'rgba(255,255,255,0.25)' : 'rgba(0,0,0,0.40)');
      elText.style('stroke-width', isDomain ? '2px' : '1.2px');

      var lh = fontSize * 1.25;
      var totalH = (lines.length - 1) * lh;
      var startY = -totalH / 2;
      lines.forEach(function(line, i) {
        elText.append('tspan').attr('x', 0)
          .attr('dy', i === 0 ? startY + 'px' : lh + 'px').text(line);
      });
    });

    // "+N more" labels on large domains
    G.selectAll('g.cn').each(function(d) {
      if (d.depth === 1 && d.data.needsTable && d.data.sampleCount < d.data.modelCount) {
        var gNode = d3.select(this);
        gNode.selectAll('text.tp-more').remove();
        var moreN = d.data.modelCount - d.data.sampleCount;
        gNode.append('text').attr('class','tp-more')
          .attr('text-anchor','middle')
          .style('font-size', Math.max(9, Math.min(13, d.r * 0.08)) + 'px')
          .style('fill','#F39C12').style('font-weight','700')
          .style('font-family','sans-serif').style('pointer-events','none')
          .attr('y', d.r * 0.55 + 'px')
          .text('+' + moreN + ' more');
      }
    });
  }, 580);
}

// ================================================================
function resizeTaxonomyPack() {
  if (!_tp.init) return;
  var el = document.getElementById(_tp.containerId);
  if (!el) return;
  var rect = el.getBoundingClientRect();
  _tp.W = Math.max(1280, rect.width || 1280);
  _tp.H = Math.max(780, rect.height || 680);

  _tp.svg.attr('viewBox', '0 0 ' + _tp.W + ' ' + _tp.H)
    .attr('width', _tp.W).attr('height', _tp.H);
  _tp.pack.size([_tp.W - 30, _tp.H - 50]);
  _tp.pack(_tp.root);
  _tp.focus = _tp.root;
  drawTP();
}

// Debounced resize for taxonomy pack
window.addEventListener('resize', debounce(function() {
  if (_tp.init) resizeTaxonomyPack();
}, 250));
