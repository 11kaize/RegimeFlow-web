// Taxonomy Circle Packing — D3 + mixed-mode bubble/table
// v8: individual bio models as L2 bubbles; table view for domains >30 models
var _tp = { focus: null, root: null, W: 1280, H: 800, init: false, mode: 'taxonomy' };

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
    'Protists': '#E67E22', 'Invertebrates': '#00BCD4', 'Viruses': '#E91E63',
    'Unclassified': '#8899aa'
  };

  var domainOrder = ['Mammals','General Eukaryotes','Fungi','Plants & Algae',
    'Vertebrates','Bacteria','Protists',
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
// Build Biological Processes hierarchy — hierarchical like taxonomy
// but with boosted bubble sizes to fill the canvas
// ================================================================
function buildBPHierarchy() {
  if (typeof BIO_PROCESSES_DATA === 'undefined' || typeof BIO_MODELS_DATA === 'undefined') {
    return { name: 'No Data', children: [] };
  }

  var modelById = {};
  BIO_MODELS_DATA.forEach(function(m) { modelById[m.id] = m; });

  var catColors = {
    'Immune Response':      '#E74C3C',
    'Signal Transduction':  '#F39C12',
    'Metabolic Pathways':   '#27AE60',
    'Cell Cycle & Division':'#2980B9',
    'Gene Regulation':      '#8E44AD',
    'Protein & Folding':    '#16A085',
    'Apoptosis & Stress':   '#D35400',
    'Development':          '#2C3E50',
    'Muscle & Contraction': '#C0392B',
    'Drug & Transport':     '#7F8C8D',
    'Other Processes':      '#95A5A6'
  };

  var children = [];
  var totalBp = 0;
  for (var cat in BIO_PROCESSES_DATA) {
    var bpEntries = BIO_PROCESSES_DATA[cat];
    totalBp += bpEntries.length;
    var fullModels = [];
    bpEntries.forEach(function(entry) {
      var fm = modelById[entry.id];
      if (fm) fullModels.push(fm);
    });
    if (!fullModels.length) continue;

    var childNodes = fullModels.map(function(m) {
      return {
        name: m.name, model: m,
        // Large constant → each model bubble is visually substantial (r ≈ 40px)
        value: 5000
      };
    });

    var nModels = fullModels.length;
    // Parent padding: buffers for d3.pack's enclosing-circle requirement.
    // Small-n needs lots of buffer (2 circles can't fill a circle efficiently);
    // large-n packs more tightly so needs less.
    var padRatio;
    if (nModels <= 1)       padRatio = 0.55;
    else if (nModels <= 2)  padRatio = 1.60;
    else if (nModels <= 3)  padRatio = 0.85;
    else if (nModels <= 5)  padRatio = 0.45;
    else if (nModels <= 10) padRatio = 0.32;
    else                    padRatio = 0.24;
    var parentPad = Math.round(nModels * 5000 * padRatio);
    children.push({
      name: cat,
      color: catColors[cat] || '#95A5A6',
      models: fullModels,
      modelCount: nModels,
      sampleCount: nModels,
      needsTable: false,
      value: parentPad,
      children: childNodes
    });
  }
  children.sort(function(a, b) { return b.modelCount - a.modelCount; });

  return {
    name: 'Biological Processes',
    children: children,
    totalModels: totalBp
  };
}

// ================================================================
// Post-pack radius correction for Biological Processes mode
// Prevents sparse-looking large circles by:
//   1. Capping parent radius based on child count
//   2. Scaling children to fill a target fraction of parent area
//   3. Re-packing children within the adjusted parent
// ================================================================
function correctBPRadii(root) {
  if (_tp.mode !== 'processes') return;
  if (!root || !root.children) return;
  // TEMPORARY: set to false to skip correction & see raw pack layout
  var APPLY = true;
  if (!APPLY) {
    console.log('correctBPRadii SKIPPED — showing raw d3.pack layout');
    return;
  }

  // Helper: shift children so the enclosing circle is centered at (0,0).
  // d3.packSiblings spreads children around an arbitrary origin — we MUST
  // re-centre manually, otherwise every child drifts to a corner of the parent.
  function centerChildren(kids) {
    if (kids.length <= 1) { kids[0].x = 0; kids[0].y = 0; return; }
    // Compute the bounding-box centre of all children (reliable, no ext API)
    var x0 =  Infinity, y0 =  Infinity;
    var x1 = -Infinity, y1 = -Infinity;
    for (var i = 0; i < kids.length; i++) {
      var ch = kids[i];
      if (ch.x - ch.r < x0) x0 = ch.x - ch.r;
      if (ch.y - ch.r < y0) y0 = ch.y - ch.r;
      if (ch.x + ch.r > x1) x1 = ch.x + ch.r;
      if (ch.y + ch.r > y1) y1 = ch.y + ch.r;
    }
    var cx = (x0 + x1) / 2;
    var cy = (y0 + y1) / 2;
    for (var i = 0; i < kids.length; i++) {
      kids[i].x -= cx;
      kids[i].y -= cy;
    }
  }

  root.children.forEach(function(cat) {
    var children = cat.children;
    if (!children || !children.length) return;
    var n = children.length;

    // ============================================================
    // 1. Enforce min/max parent radius
    //    - min: 55px so single-model categories aren't invisible
    //    - max: prevents one category from dominating the canvas
    // ============================================================
    var minR = 55;
    var maxR;
    if (n <= 1)       maxR = 90;
    else if (n <= 2)  maxR = 120;
    else if (n <= 3)  maxR = 145;
    else if (n <= 5)  maxR = 170;
    else if (n <= 8)  maxR = 200;
    else if (n <= 13) maxR = 230;
    else              maxR = 280;

    cat.r = Math.max(minR, Math.min(cat.r, maxR));

    // ============================================================
    // 2. Enforce absolute minimum child radius
    // ============================================================
    var absMinChildR = 28;
    children.forEach(function(ch) {
      if (ch.r < absMinChildR) ch.r = absMinChildR;
    });

    // ============================================================
    // 3. Re-pack children, keep them centred inside the parent
    // ============================================================
    if (children.length >= 2) {
      console.log('BEFORE packSiblings — cat:', cat.data.name, 'cat.r:', cat.r.toFixed(1),
        'cat.x:', cat.x.toFixed(1), 'cat.y:', cat.y.toFixed(1),
        'ch0.x:', children[0].x.toFixed(1), 'ch0.y:', children[0].y.toFixed(1));

      d3.packSiblings(children);

      console.log('AFTER packSiblings — ch0.x:', children[0].x.toFixed(1), 'ch0.y:', children[0].y.toFixed(1));

      centerChildren(children);               // ← CRITICAL: centre at origin

      console.log('AFTER centerChildren — ch0.x:', children[0].x.toFixed(1), 'ch0.y:', children[0].y.toFixed(1));

      var maxExtent = 0;
      children.forEach(function(ch) {
        var dist = Math.sqrt(ch.x * ch.x + ch.y * ch.y) + ch.r;
        if (dist > maxExtent) maxExtent = dist;
      });

      var margin = 1.05;
      if (maxExtent * margin < cat.r) {
        cat.r = Math.max(minR, maxExtent * margin);
      } else if (maxExtent > cat.r * 0.96) {
        var shrink = (cat.r * 0.96) / maxExtent;
        children.forEach(function(ch) {
          ch.r = Math.max(absMinChildR, ch.r * shrink);
          ch.x *= shrink;
          ch.y *= shrink;
        });
        d3.packSiblings(children);
        centerChildren(children);             // ← re-centre after re-pack
      }
    } else if (children.length === 1) {
      children[0].x = 0;
      children[0].y = 0;
      // Ensure parent has a clickable ring around the single child
      var idealR = Math.max(minR, children[0].r * 1.30);
      cat.r = Math.max(cat.r, idealR);
    }

    // Offset children to parent absolute centre
    children.forEach(function(ch) {
      ch.x += cat.x;
      ch.y += cat.y;
    });
    if (n >= 2) {
      console.log('AFTER offset — ch0.x:', children[0].x.toFixed(1), 'ch0.y:', children[0].y.toFixed(1),
        'cat.x:', cat.x.toFixed(1), 'cat.y:', cat.y.toFixed(1),
        'diff:', (children[0].x - cat.x).toFixed(1), (children[0].y - cat.y).toFixed(1));
    }
  });
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
  document.getElementById('tax-mode-bar').style.display = 'none';
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
      document.getElementById('tax-mode-bar').style.display = '';
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
      var header = 'ID,DisplayName,OriginalName,Regime,Species,Domain';
      var rows = selected.map(function(m) {
        var dName = getShortName(m) || getDisplayName(m) || (m.name||'');
        return [m.id, '"' + dName + '"', '"' + (m.name||'') + '"', m.regime||'', m.species||0, domainName].join(',');
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

    var rowDisplay = getShortName(m) || getDisplayName(m) || (m.name || '');
    var rowOriginal = (rowDisplay !== (m.name || '') && (m.name || '')) ? ' <span style="color:#4a5f73;font-size:11px;">(' + (m.name || '') + ')</span>' : '';
    html += '<div class="tax-row' + selClass + '" data-id="' + m.id + '">' +
      '<div class="tax-row-main">' +
        '<span class="tax-row-title">' + rowDisplay + rowOriginal + '</span>' +
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

  // Build hierarchy from BIO_MODELS_DATA + BIO_DOMAIN_DATA (or BP data)
  var hierarchyData = _tp.mode === 'processes' ? buildBPHierarchy() : buildHierarchy();
  _tp.root = d3.hierarchy(hierarchyData)
    .sum(function(d) { return d.value || 0; })
    .sort(function(a, b) { return (b.value||0) - (a.value||0); });

  // Pack layout — spacing depends on mode (BP needs more room)
  _tp.pack = d3.pack()
    .size([_tp.W - 30, _tp.H - 50])
    .padding(function(d) {
      return _tp.mode === 'processes' ? (d.depth === 1 ? 14 : 5) : (d.depth === 1 ? 8 : 3);
    });

  _tp.pack(_tp.root);
  correctBPRadii(_tp.root);

  // Color scale — for flat BP mode, use direct color; otherwise parent domain color
  _tp.color = function(d) {
    if (d.data && d.data.isFlat) return d.data.color;
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
  _tp.footerText = _tp.svg.append('text').attr('class','tp-footer')
    .attr('x',_tp.W/2).attr('y',_tp.H-10)
    .attr('text-anchor','middle').style('font-size','11px')
    .style('fill','#556678').style('font-family','sans-serif')
    .style('pointer-events','none')
    .text((_tp.mode === 'processes' ? 'Biological Processes · ' : 'SysBio-Traj · ') +
      (_tp.root.data.totalModels || '1,050') + ' Models · ' + _tp.root.children.length + ' Groups · Click bubble to explore');

  // Container group
  _tp.g = _tp.svg.append('g');
  // Label overlay — renders ABOVE all circles so L1 domain names are never buried.
  // pointer-events:none → clicks pass through to circles underneath
  _tp.labelOverlay = _tp.svg.append('g').attr('class', 'tp-label-overlay')
    .attr('pointer-events', 'none');
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

  // ---- Mode toggle buttons ----
  initModeToggle();
}

// (display name helpers now in js/core/utils.js)

// ================================================================
// Global Search — multi-keyword across name, ID, domain, regime,
// biological process, and species count
// ================================================================
function initGlobalSearch() {
  var input = document.getElementById('tax-global-search');
  var results = document.getElementById('tax-global-results');
  if (!input || !results) return;

  // Update placeholder to reflect new capabilities
  input.placeholder = 'Search by keyword — name, domain, regime, process…';

  // ---- Regime display helpers ----
  var regimeLabels = {
    oscillation:'Oscillation', inc_stable:'Inc-Stable', dec_stable:'Dec-Stable',
    directly_stable:'Direct-Stable', increasing:'Growth', decreasing:'Decay'
  };
  var regimeColors = {
    oscillation:'#9B59B6', inc_stable:'#3498DB', dec_stable:'#E8913A',
    directly_stable:'#50B86C', increasing:'#2ECC71', decreasing:'#E74C3C'
  };

  // ---- Biological keyword alias table ----
  // Maps common biology concepts to model IDs so users can search
  // "calcium", "apoptosis", "MAPK" etc. and find the right models.
  var bioAliases = {
    // Calcium signaling / oscillation (Goldbeter & Tyson models)
    'calcium': ['BIOMD0000000003','BIOMD0000000004','BIOMD0000000005','BIOMD0000000006','BIOMD0000000016','BIOMD0000000036','BIOMD0000000079','BIOMD0000000098','BIOMD0000000057'],
    'ca2+':    ['BIOMD0000000003','BIOMD0000000004'],
    // Circadian rhythm
    'circadian': ['BIOMD0000000079','BIOMD0000000073','BIOMD0000000074','BIOMD0000000078','BIOMD0000000083','BIOMD0000000022'],
    // Apoptosis / cell death
    'apoptosis': ['BIOMD0000000048','BIOMD0000000049','BIOMD0000000054','BIOMD0000000106','BIOMD0000000137','BIOMD0000000762','BIOMD0000000785'],
    // MAPK / EGFR signaling
    'mapk':  ['BIOMD0000000048','BIOMD0000000049','BIOMD0000000054','BIOMD0000000106','BIOMD0000000137','BIOMD0000000647','BIOMD0000000666'],
    'egfr':  ['BIOMD0000000048','BIOMD0000000137','BIOMD0000000762'],
    // NF-κB
    'nfkb':  ['BIOMD0000000048','BIOMD0000000106','BIOMD0000000762'],
    'nf-kb': ['BIOMD0000000048','BIOMD0000000106','BIOMD0000000762'],
    // p53
    'p53':   ['BIOMD0000000049','BIOMD0000000054','BIOMD0000000137'],
    // Cell cycle
    'cell cycle':  ['BIOMD0000000005','BIOMD0000000006','BIOMD0000000036','BIOMD0000000016','BIOMD0000000022'],
    'mitosis':     ['BIOMD0000000005','BIOMD0000000006','BIOMD0000000036'],
    // Immune
    't-cell':   ['BIOMD0000000762','BIOMD0000000763','BIOMD0000000785','BIOMD0000000787'],
    't cell':   ['BIOMD0000000762','BIOMD0000000763','BIOMD0000000785','BIOMD0000000787'],
    'immunotherapy': ['BIOMD0000000762','BIOMD0000000763','BIOMD0000000790'],
    // Metabolism
    'glycolysis': ['BIOMD0000000051','BIOMD0000000061','BIOMD0000000062','BIOMD0000000063'],
    'metabolic':  ['BIOMD0000000051','BIOMD0000000061','BIOMD0000000857','BIOMD0000000858','BIOMD0000000859'],
    // E. coli / bacteria
    'e.coli':  ['BIOMD0000000051','BIOMD0000000062','BIOMD0000000065','BIOMD0000000244'],
    'ecoli':   ['BIOMD0000000051','BIOMD0000000062','BIOMD0000000065','BIOMD0000000244'],
    // Common organisms
    'yeast':   [],  // filled from domain descriptions (Saccharomyces)
    'human':   [],  // filled from domain descriptions (Homo sapiens)
    'mouse':   [],  // Mus musculus
    'rat':     [],  // Rattus
    // Cancer
    'cancer':  ['BIOMD0000000048','BIOMD0000000137','BIOMD0000000666','BIOMD0000000762','BIOMD0000000790','BIOMD0000000810'],
    'tumor':   ['BIOMD0000000650','BIOMD0000000670','BIOMD0000000762','BIOMD0000000763','BIOMD0000000785','BIOMD0000000787','BIOMD0000000790','BIOMD0000000792','BIOMD0000000797'],
  };

  // Expand aliases: also match partials (e.g. "mapk" also finds "map kinase")
  // and add reverse lookup: model ID → extra keywords
  var modelExtraKeywords = {}; // modelId → 'keyword1 keyword2 ...'
  for (var ak in bioAliases) {
    var ids = bioAliases[ak];
    ids.forEach(function(mid) {
      if (!modelExtraKeywords[mid]) modelExtraKeywords[mid] = '';
      modelExtraKeywords[mid] += ' ' + ak;
    });
  }

  // ---- Build rich search index ----
  // Helper: extract clean biological keywords from domain description strings
  function extractBioKeywords(raw) {
    if (!raw) return '';
    // Remove ontology prefixes
    var cleaned = raw
      .replace(/Experimental Factor Ontology-/gi, '')
      .replace(/KEGG Pathway-/gi, '')
      .replace(/KEGG Drug-/gi, '')
      .replace(/Brenda Tissue Ontology-/gi, '')
      .replace(/Gene Ontology-/gi, '')
      .replace(/Reactome-/gi, '')
      .replace(/Human Disease Ontology-/gi, '')
      .replace(/NCIt-/gi, '');
    // Split on common delimiters
    var parts = cleaned.split(/[|;,\n\r]+/);
    // Collect meaningful words (skip short/common words)
    var stopWords = { 'the':1,'and':1,'of':1,'in':1,'to':1,'a':1,'an':1,'or':1,'for':1,'by':1,'is':1,'on':1,'as':1,'at':1,'be':1,'from':1,'with':1,'its':1,'not':1,'are':1,'was':1,'has':1,'have':1,'been':1,'can':1,'may':1,'will':1,'also':1,'but':1,'that':1,'this':1,'into':1,'other':1,'been':1,'no':1,'all':1,'which':1,'via':1,'part':1,'each':1,'than':1 };
    var keywords = [];
    parts.forEach(function(part) {
      var trimmed = part.trim().toLowerCase();
      if (trimmed.length < 3 || stopWords[trimmed]) return;
      // Also add the full phrase (useful for multi-word concepts)
      if (trimmed.split(/\s+/).length >= 2) keywords.push(trimmed);
      // Add individual significant words
      trimmed.split(/\s+/).forEach(function(word) {
        word = word.replace(/[^a-z0-9-]/g, '');
        if (word.length >= 3 && !stopWords[word]) keywords.push(word);
      });
    });
    // De-duplicate
    var seen = {};
    return keywords.filter(function(k) { if (seen[k]) return false; seen[k] = true; return true; }).join(' ');
  }

  var searchIndex = [];
  // Pre-build domain description lookup for BIO_DOMAIN_DATA
  var domainDescMap = {};
  if (typeof BIO_DOMAIN_DATA !== 'undefined') {
    for (var d in BIO_DOMAIN_DATA) {
      BIO_DOMAIN_DATA[d].forEach(function(dm) { domainDescMap[dm.id] = dm.domain || ''; });
    }
  }
  if (typeof BIO_PROCESSES_DATA !== 'undefined') {
    for (var bp in BIO_PROCESSES_DATA) {
      BIO_PROCESSES_DATA[bp].forEach(function(dm) { domainDescMap[dm.id] = dm.domain || ''; });
    }
  }

  if (typeof BIO_MODELS_DATA !== 'undefined') {
    BIO_MODELS_DATA.forEach(function(m) {
      var domain = 'Unclassified', process = '';
      if (typeof BIO_DOMAIN_DATA !== 'undefined') {
        for (var d2 in BIO_DOMAIN_DATA) {
          if (BIO_DOMAIN_DATA[d2].some(function(dm) { return dm.id === m.id; })) { domain = d2; break; }
        }
      }
      if (typeof BIO_PROCESSES_DATA !== 'undefined') {
        for (var bp2 in BIO_PROCESSES_DATA) {
          if (BIO_PROCESSES_DATA[bp2].some(function(dm) { return dm.id === m.id; })) { process = bp2; break; }
        }
      }
      var regimeLabel = regimeLabels[m.regime] || m.regime || '';
      // Extract biological keywords from the domain description
      var descKeywords = extractBioKeywords(domainDescMap[m.id] || '');
      // Add manual alias keywords for common biological concepts
      var aliasKeywords = modelExtraKeywords[m.id] || '';

      // Concatenated lowercased search text for multi-keyword matching
      var searchText = [
        m.name || '', m.id || '', domain, process,
        regimeLabel.toLowerCase(), m.regime || '', String(m.species || ''),
        descKeywords, aliasKeywords,
        getDisplayName(m).toLowerCase(),
        getShortName(m).toLowerCase()
      ].join(' ').toLowerCase();

      searchIndex.push({
        id: m.id, name: m.name, domain: domain, process: process,
        regime: m.regime || '', regimeLabel: regimeLabel,
        species: m.species || 0, model: m,
        searchText: searchText, descKeywords: descKeywords
      });
    });
  }

  var activeIdx = -1;

  // ---- Search function ----
  var _doSearch = debounce(function() {
    var rawQ = input.value.trim();
    if (!rawQ) { results.style.display = 'none'; return; }
    var q = rawQ.toLowerCase();
    // Split into keywords (AND logic: all keywords must match)
    var keywords = q.split(/\s+/).filter(Boolean);

    var matches = searchIndex.filter(function(item) {
      return keywords.every(function(kw) { return item.searchText.indexOf(kw) >= 0; });
    }).slice(0, 20);

    if (!matches.length) {
      results.innerHTML = '<div class="tax-search-empty">No results for "' + rawQ + '"</div>';
      results.style.display = '';
      return;
    }

    activeIdx = -1;
    results.innerHTML = '';
    matches.forEach(function(item, i) {
      // Determine which fields matched which keywords
      var matchTags = [];
      keywords.forEach(function(kw) {
        if (item.name.toLowerCase().indexOf(kw) >= 0 && matchTags.indexOf('name') < 0) matchTags.push('name');
        if (item.id.toLowerCase().indexOf(kw) >= 0 && matchTags.indexOf('id') < 0) matchTags.push('id');
        if (item.domain.toLowerCase().indexOf(kw) >= 0 && matchTags.indexOf('domain') < 0) matchTags.push('domain');
        if (item.process.toLowerCase().indexOf(kw) >= 0 && matchTags.indexOf('process') < 0) matchTags.push('process');
        if (item.regimeLabel.toLowerCase().indexOf(kw) >= 0 && matchTags.indexOf('regime') < 0) matchTags.push('regime');
        if (item.descKeywords && item.descKeywords.indexOf(kw) >= 0 && matchTags.indexOf('bio') < 0) matchTags.push('bio');
      });

      // Build match badge HTML
      var badgesHtml = '';
      if (matchTags.indexOf('regime') >= 0) {
        badgesHtml += '<span class="tax-search-badge regime" style="background:' + (regimeColors[item.regime] || '#666') + '20;color:' + (regimeColors[item.regime] || '#aaa') + ';border:1px solid ' + (regimeColors[item.regime] || '#666') + '40;">' + item.regimeLabel + '</span>';
      }
      if (matchTags.indexOf('domain') >= 0) {
        badgesHtml += '<span class="tax-search-badge domain">' + item.domain + '</span>';
      }
      if (matchTags.indexOf('process') >= 0) {
        badgesHtml += '<span class="tax-search-badge process">📂 ' + item.process + '</span>';
      }
      if (matchTags.indexOf('bio') >= 0 && matchTags.indexOf('process') < 0) {
        badgesHtml += '<span class="tax-search-badge bio">🔬 keyword match</span>';
      }
      if (matchTags.indexOf('id') >= 0 && matchTags.indexOf('name') < 0) {
        badgesHtml += '<span class="tax-search-badge id">ID match</span>';
      }

      var displayName = getDisplayName(item.model);
      var shortName = getShortName(item.model) || displayName;
      var div = document.createElement('div');
      div.className = 'tax-search-item';
      div.innerHTML =
        '<div class="tax-search-row">' +
          '<span class="tax-search-name" title="' + (displayName||item.name) + '">' + (shortName||displayName||item.name) + '</span>' +
          '<span class="tax-search-species">🧬 ' + item.species + '</span>' +
        '</div>' +
        '<div class="tax-search-row2">' +
          badgesHtml +
          '<span class="tax-search-id">' + item.id + '</span>' +
        '</div>';
      div.addEventListener('mousedown', function(e) {
        e.preventDefault();
        selectSearchResult(item);
      });
      results.appendChild(div);
    });
    results.style.display = '';
  }, 180);

  input.addEventListener('input', _doSearch);

  // ---- Keyboard navigation ----
  input.addEventListener('keydown', function(e) {
    var items = results.querySelectorAll('.tax-search-item');
    if (!items.length) return;
    if (e.key === 'ArrowDown') { e.preventDefault(); activeIdx = Math.min(activeIdx + 1, items.length - 1); updateActive(items); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); activeIdx = Math.max(activeIdx - 1, 0); updateActive(items); }
    else if (e.key === 'Enter') {
      e.preventDefault();
      if (activeIdx >= 0 && activeIdx < items.length) {
        var allMatches = getCurrentMatches();
        if (activeIdx < allMatches.length) selectSearchResult(allMatches[activeIdx]);
      }
    } else if (e.key === 'Escape') { results.style.display = 'none'; input.blur(); }
  });

  function updateActive(items) {
    items.forEach(function(item, i) { item.classList.toggle('active', i === activeIdx); });
  }

  function getCurrentMatches() {
    var q = (input.value || '').toLowerCase().trim();
    if (!q) return [];
    var keywords = q.split(/\s+/).filter(Boolean);
    return searchIndex.filter(function(item) {
      return keywords.every(function(kw) { return item.searchText.indexOf(kw) >= 0; });
    }).slice(0, 20);
  }

  function selectSearchResult(item) {
    results.style.display = 'none';
    input.value = item.name + ' (' + item.domain + ')';

    // Try to navigate bubble chart to the matching domain/process first
    if (_tp.init && _tp.root && _tp.focus === _tp.root) {
      var targetDomain = item.process || item.domain;
      var domainNode = null;
      if (_tp.root.children) {
        _tp.root.children.forEach(function(child) {
          if (child.data.name === targetDomain) domainNode = child;
        });
      }
      if (domainNode) {
        _tp.focus = domainNode;
        drawTP();
        setTimeout(function() { openDetailPanel(item.model, 'bio', null); }, 620);
        return;
      }
    }
    // Fallback: just open detail panel
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
  _tp.labelOverlay.transition(t).attr('transform', 'translate(' + tx + ',' + ty + ') scale(' + k + ')');

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
    .attr('visibility', function(d) {
      if (d.r < 2.5) return 'hidden';
      return 'visible';
    })
    // Interactions
    .on('mouseenter', function(event, d) {
      d3.select(this).attr('stroke','#F39C12').attr('stroke-width', 3);
      var html = '';
      var isFlat = _tp.root && _tp.root.data && _tp.root.data.isFlat;
      if (d.depth === 1 && !isFlat) {
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
        var catLabel = d.data.category ? '<div style="color:#F39C12;font-size:11px;margin-top:2px;">📂 ' + d.data.category + '</div>' : '';
        var displayName = getDisplayName(m);
        html = '<div style="font-size:14px;font-weight:700;color:#fff;">' + (displayName||m.name||'') + '</div>' +
          (displayName !== m.name ? '<div style="color:#8899aa;font-size:11px;">' + (m.name||'') + '</div>' : '') +
          '<div style="color:#6a8299;font-size:10px;">' + (m.id||'') + '</div>' +
          (parentName && !isFlat ? '<div style="color:#6a8299;font-size:11px;margin-top:2px;">' + parentName + '</div>' : '') +
          catLabel +
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
      var isFlat = _tp.root && _tp.root.data && _tp.root.data.isFlat;
      if (isFlat && d.data.model) {
        // Flat mode: click any model bubble → open detail directly
        openDetailPanel(d.data.model, 'bio', null);
      } else if (d.depth === 1 && !isFlat) {
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
  var merged = grp.merge(enter);
  merged.transition(t)
    .attr('transform', function(d) { return 'translate(' + d.x + ',' + d.y + ')'; })
    .style('opacity', 1);
  // MUST update circle radii — correctBPRadii may have changed d.r,
  // and drawTP can be called multiple times (zoom, resize, mode-switch).
  merged.select('circle').transition(t)
    .attr('r', function(d) { return Math.max(1.8, d.r); })
    .attr('visibility', function(d) { return d.r < 2.5 ? 'hidden' : 'visible'; });

  // Post-transition: update text
  setTimeout(function() {
    var isFlat = _tp.root && _tp.root.data && _tp.root.data.isFlat;
    G.selectAll('text.tp-name').each(function(d) {
      var elText = d3.select(this);
      elText.selectAll('tspan').remove();
      if (d.r < 12) { elText.text(''); return; }

      var isDomain = (d.depth === 1 && !isFlat);
      var fontSize;
      if (isFlat) {
        fontSize = Math.max(9, Math.min(14, d.r * 0.18));
      } else if (isDomain) {
        fontSize = Math.max(13, Math.min(22, d.r * 0.13));
      } else {
        fontSize = Math.max(8, Math.min(12, d.r * 0.22));
      }
      elText.style('font-size', fontSize + 'px');

      // L1: domain name. L2 model: short descriptive name (no AuthorYear prefix)
      var rawName = d.data.name || (d.data.model ? d.data.model.name : '');
      var name;
      if (isDomain || isFlat) {
        name = rawName;
      } else {
        var short = d.data.model ? getShortName(d.data.model) : '';
        name = short || rawName;
      }
      // For leaf bubbles, truncate to fit
      if (!isDomain && !isFlat && name.length > 18) name = name.substring(0, 16) + '…';

      var maxChars = isDomain ? Math.max(6, Math.floor(d.r * 1.6 / (fontSize * 0.55))) : Math.max(4, Math.floor(d.r * 1.4 / (fontSize * 0.55)));
      var lines = wrapText(name, maxChars, isDomain ? 2 : 1);

      elText.style('fill', isDomain ? '#1a1a2e' : '#e0e6ed');
      elText.style('stroke', isDomain ? 'rgba(255,255,255,0.25)' : 'rgba(0,0,0,0.40)');
      elText.style('stroke-width', isDomain ? '2px' : '1.2px');
      // At overview level hide in-group L1 labels — overlay layer handles them
      if (isDomain && _tp.focus === _tp.root) { elText.style('opacity', 0); }
      else if (isDomain) { elText.style('opacity', null); }

      var lh = fontSize * 1.25;
      var totalH = (lines.length - 1) * lh;
      var startY = -totalH / 2;
      lines.forEach(function(line, i) {
        elText.append('tspan').attr('x', 0)
          .attr('dy', i === 0 ? startY + 'px' : lh + 'px').text(line);
      });
    });

    // "+N more" labels on large domains (only for taxonomy mode)
    if (!isFlat) {
      G.selectAll('g.cn').each(function(d) {
        // BP mode overview: show model count on each category circle
        if (_tp.mode === 'processes' && _tp.focus === _tp.root && d.depth === 1 && d.data.modelCount) {
          var gNode2 = d3.select(this);
          gNode2.selectAll('text.tp-count').remove();
          gNode2.append('text').attr('class','tp-count')
            .attr('text-anchor','middle')
            .style('font-size', Math.max(11, Math.min(16, d.r * 0.10)) + 'px')
            .style('fill','#F39C12').style('font-weight','700')
            .style('font-family','sans-serif').style('pointer-events','none')
            .style('paint-order','stroke').style('stroke','rgba(0,0,0,0.5)').style('stroke-width','3px')
            .attr('y', d.r * 0.48 + 'px')
            .text(d.data.modelCount + ' models');

          // Sparse visual patterns: dotted concentric rings for categories with < 4 children
          var nChild = d.data.modelCount || 0;
          gNode2.selectAll('circle.tp-ring').remove();
          if (nChild <= 3) {
            // Outer ring at 75% radius
            gNode2.append('circle').attr('class','tp-ring')
              .attr('r', d.r * 0.72)
              .attr('fill','none')
              .attr('stroke', d.data.color || '#8899aa')
              .attr('stroke-width', 1.2)
              .attr('stroke-dasharray', '4 6')
              .attr('stroke-opacity', 0.35)
              .attr('pointer-events','none');
            // Inner ring at 50% radius
            if (nChild <= 2) {
              gNode2.append('circle').attr('class','tp-ring')
                .attr('r', d.r * 0.42)
                .attr('fill','none')
                .attr('stroke', d.data.color || '#8899aa')
                .attr('stroke-width', 1.0)
                .attr('stroke-dasharray', '3 5')
                .attr('stroke-opacity', 0.25)
                .attr('pointer-events','none');
            }
          } else if (nChild <= 6) {
            // Single subtle ring for medium-sparse
            gNode2.append('circle').attr('class','tp-ring')
              .attr('r', d.r * 0.68)
              .attr('fill','none')
              .attr('stroke', d.data.color || '#8899aa')
              .attr('stroke-width', 0.8)
              .attr('stroke-dasharray', '3 7')
              .attr('stroke-opacity', 0.25)
              .attr('pointer-events','none');
          }
        }
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
    }

    // ============================================================
    // Overlay domain labels — separate layer ABOVE all child bubbles.
    // Uses text-halo (strong stroke) instead of a badge → no click blocking.
    // Only active at overview level.
    // ============================================================
    _tp.labelOverlay.selectAll('*').remove();
    if (F === _tp.root) {
      var overlayNodes = nodes.filter(function(d) { return d.depth === 1 && d.data.name; });
      overlayNodes.forEach(function(d) {
        var name = d.data.name || '';
        var fontSize = Math.max(13, Math.min(22, d.r * 0.13));
        var maxChars = Math.max(6, Math.floor(d.r * 1.6 / (fontSize * 0.55)));
        var lines = wrapText(name, maxChars, 2);
        var lh = fontSize * 1.25;

        var textG = _tp.labelOverlay.append('g')
          .attr('transform', 'translate(' + d.x + ',' + d.y + ')')
          .attr('pointer-events', 'none');

        // Text halo: heavy dark stroke makes text readable over any background
        var textEl = textG.append('text')
          .attr('text-anchor', 'middle')
          .attr('fill', '#f0f2f5')
          .style('font-size', fontSize + 'px')
          .style('font-weight', '700')
          .style('font-family', '-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif')
          .style('paint-order', 'stroke')
          .style('stroke', 'rgba(8, 16, 28, 0.80)')
          .style('stroke-width', '4.5px')
          .style('stroke-linecap', 'round')
          .style('stroke-linejoin', 'round');

        var startY = -(lines.length - 1) * lh / 2;
        lines.forEach(function(line, i) {
          textEl.append('tspan')
            .attr('x', 0)
            .attr('dy', i === 0 ? startY + 'px' : lh + 'px')
            .text(line);
        });
      });
    }

    // ---- Dynamic footer: show "go back" hint when zoomed in ----
    if (_tp.footerText) {
      if (F === _tp.root) {
        _tp.footerText.text((_tp.mode === 'processes' ? 'Biological Processes · ' : 'SysBio-Traj · ') +
          (_tp.root.data.totalModels || '1,050') + ' Models · ' + _tp.root.children.length + ' Groups · Click bubble to explore');
      } else {
        var fName = F.data.name || '';
        var fCount = F.data.modelCount || (F.children ? F.children.length : 0);
        _tp.footerText.text(fName + ' · ' + fCount + ' models · 🖱 Click empty area to zoom out');
      }
    }
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

  // Rebuild hierarchy to recalculate pack based on current mode
  var hierarchyData = _tp.mode === 'processes' ? buildBPHierarchy() : buildHierarchy();
  _tp.root = d3.hierarchy(hierarchyData)
    .sum(function(d) { return d.value || 0; })
    .sort(function(a, b) { return (b.value||0) - (a.value||0); });
  _tp.pack.size([_tp.W - 30, _tp.H - 50]);
  _tp.pack(_tp.root);
  correctBPRadii(_tp.root);
  _tp.focus = _tp.root;
  drawTP();
}

// ================================================================
// Mode toggle — switch between Taxonomy and Biological Processes
// ================================================================
function initModeToggle() {
  var btns = document.querySelectorAll('.tax-mode-btn');
  btns.forEach(function(btn) {
    btn.classList.remove('active');
    if (btn.dataset.mode === _tp.mode) btn.classList.add('active');
    btn.addEventListener('click', function() {
      var mode = this.dataset.mode;
      if (mode === _tp.mode) return;
      switchTaxonomyMode(mode);
    });
  });
}

function switchTaxonomyMode(mode) {
  _tp.mode = mode;
  _tp.focus = null;

  // Rebuild hierarchy
  var hierarchyData = mode === 'processes' ? buildBPHierarchy() : buildHierarchy();
  _tp.root = d3.hierarchy(hierarchyData)
    .sum(function(d) { return d.value || 0; })
    .sort(function(a, b) { return (b.value||0) - (a.value||0); });

  _tp.pack
    .size([_tp.W - 30, _tp.H - 50])
    .padding(function(d) { return mode === 'processes' ? (d.depth === 1 ? 14 : 5) : (d.depth === 1 ? 8 : 3); });
  _tp.pack(_tp.root);
  correctBPRadii(_tp.root);
  _tp.focus = _tp.root;

  // Update toggle button states
  document.querySelectorAll('.tax-mode-btn').forEach(function(btn) {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });

  // Update BP availability hint
  var hint = document.getElementById('bp-avail-hint');
  if (hint) {
    hint.style.display = (mode === 'taxonomy' && typeof BIO_PROCESSES_DATA !== 'undefined') ? '' : 'none';
  }

  // Update footer text
  if (_tp.footerText) {
    _tp.footerText.text((mode === 'processes' ? 'Biological Processes · ' : 'SysBio-Traj · ') +
      (_tp.root.data.totalModels || '1,050') + ' Models · ' + _tp.root.children.length + ' Groups · Click bubble to explore');
  }

  drawTP();
}

// Debounced resize for taxonomy pack
window.addEventListener('resize', debounce(function() {
  if (_tp.init) resizeTaxonomyPack();
}, 250));
