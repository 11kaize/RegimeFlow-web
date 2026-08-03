/* ================================================================
   RegimeFlow — Detail Panel Component
   Sidebar overlay showing model metadata and HuggingFace file links.
   Used by: taxonomy-pack.js, bubble-chart.js
   ================================================================ */

// ── Public API ──────────────────────────────────────────────────

/**
 * Open the detail side panel for a given model.
 *
 * @param {Object} model  — Model record (from BIO_MODELS_DATA or ML models)
 * @param {string} mode   — 'bio' | 'ml'   (defaults to 'ml')
 */
function openDetailPanel(model, mode) {
  var panel   = document.getElementById('detail-panel');
  var overlay = document.getElementById('detail-overlay');
  var content = document.getElementById('detail-content');
  if (!panel || !content) return;

  mode = mode || 'ml';
  var html = '';

  if (mode === 'bio') {
    var l1key   = REGIME_TO_L1[model.regime] || 'stable';
    var l1info  = REGIME_L1_DEFS[l1key] || {};
    var hfBase  = 'https://huggingface.co/datasets/HengRao/SysBio-Traj/resolve/main/Data/' + model.id + '/' + model.name;

    html +=
      '<div class="dp-header">' +
        '<div class="dp-name">🧬 ' + escapeHtml(model.name || '') + '</div>' +
        '<div class="dp-id">' + escapeHtml(model.id || '') + '</div>' +
      '</div>' +
      '<div class="dp-section">' +
        '<div class="dp-section-title">Basic Info</div>' +
        '<div class="dp-row"><span class="dp-label">Dynamic Type</span><span class="dp-value">' + (l1info.icon || '') + ' ' + (l1info.label || model.regime || '—') + '</span></div>' +
        '<div class="dp-row"><span class="dp-label">Original Regime</span><span class="dp-value">' + escapeHtml(model.regime || '—') + '</span></div>' +
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
        '<button class="btn-predict" data-bio-id="' + escapeHtml(model.id) + '" data-bio-name="' + escapeHtml(model.name) + '" data-bio-species="' + model.species + '">📈 Load Trajectory to Prediction</button>' +
      '</div>';
  } else {
    // ── ML mode ──
    var typeLabel  = model.type === 'probabilistic' ? 'Probabilistic' : model.type === 'point' ? 'Point' : 'Zero-shot';
    var familyName = (typeof t === 'function') ? t('family.' + model.family) : model.family;

    html +=
      '<div class="dp-header">' +
        '<div class="dp-name">📦 ' + escapeHtml(model.name || '') + '</div>' +
        '<div class="dp-id">' + escapeHtml(model.id || '') + '</div>' +
      '</div>' +
      '<div class="dp-section">' +
        '<div class="dp-section-title">Key Parameters</div>' +
        '<div class="dp-row"><span class="dp-label">Family</span><span class="dp-value">' + escapeHtml(familyName || model.family || '—') + '</span></div>' +
        '<div class="dp-row"><span class="dp-label">Type</span><span class="dp-value">' + escapeHtml(typeLabel) + '</span></div>';

    if (model.hidden_dim > 0) {
      html +=
        '<div class="dp-row"><span class="dp-label">Hidden Dim</span><span class="dp-value">' + model.hidden_dim + '</span></div>' +
        '<div class="dp-row"><span class="dp-label">Layers</span><span class="dp-value">' + model.layers + '</span></div>' +
        '<div class="dp-row"><span class="dp-label">Learning Rate</span><span class="dp-value">' + model.lr + '</span></div>';
    } else {
      html += '<div class="dp-row"><span class="dp-label">Pretrained</span><span class="dp-value">' + escapeHtml(model.pretrained || '—') + '</span></div>';
    }

    html +=
        '<div class="dp-row"><span class="dp-label">Context Length</span><span class="dp-value">' + model.context_len + '</span></div>' +
        '<div class="dp-row"><span class="dp-label">Prediction Length</span><span class="dp-value">' + model.pred_len + '</span></div>' +
      '</div>';

    if (model.paper_metrics && Object.keys(model.paper_metrics).length) {
      html += '<div class="dp-section"><div class="dp-section-title">Metrics</div><div class="dp-tags">';
      Object.entries(model.paper_metrics).forEach(function(e) {
        html += '<span class="dp-tag regime">' + escapeHtml(e[0]) + ': ' + escapeHtml(String(e[1])) + '</span>';
      });
      html += '</div></div>';
    }

    var features = model.features || [];
    if (features.length) {
      html += '<div class="dp-section"><div class="dp-section-title">Features</div><div class="dp-tags">';
      features.forEach(function(f) {
        html += '<span class="dp-tag feature">' + escapeHtml(f) + '</span>';
      });
      html += '</div></div>';
    }

    if (model.description) {
      html += '<div class="dp-section"><div class="dp-section-title">Description</div><div class="dp-desc">' + escapeHtml(model.description) + '</div></div>';
    }
  }

  content.innerHTML = html;

  // Bind predict button via event delegation (no inline onclick)
  if (mode === 'bio') {
    var predictBtn = content.querySelector('.btn-predict');
    if (predictBtn) {
      predictBtn.addEventListener('click', function() {
        loadBioTrajectory(
          predictBtn.getAttribute('data-bio-id'),
          predictBtn.getAttribute('data-bio-name'),
          parseInt(predictBtn.getAttribute('data-bio-species'), 10)
        );
      });
    }
  }

  panel.classList.add('active');
  overlay.classList.add('active');
}

/**
 * Close the detail side panel and clear bubble selection highlight.
 */
function closeDetailPanel() {
  var panel   = document.getElementById('detail-panel');
  var overlay = document.getElementById('detail-overlay');
  if (panel)   panel.classList.remove('active');
  if (overlay) overlay.classList.remove('active');

  // Clear bubble highlight (set by bubble-chart.js)
  if (window._bubbleSelected) {
    window._bubbleSelected
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 2.5)
      .attr('filter', null);
    window._bubbleSelected = null;
  }
}

// ── Keyboard shortcut ───────────────────────────────────────────
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeDetailPanel();
});
