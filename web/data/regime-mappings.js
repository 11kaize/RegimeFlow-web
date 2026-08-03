/* ================================================================
   RegimeFlow — Shared Regime Classification Constants
   Source of truth for regime labels, colours, icons.
   Used by: bubble-chart.js, taxonomy-pack.js, detail-panel.js
   ================================================================ */

// Original regime labels → L1 dynamic category
var REGIME_TO_L1 = {
  directly_stable: 'stable',
  oscillation:     'oscillation',
  inc_stable:      'growth',
  increasing:      'growth',
  dec_stable:      'decay',
  decreasing:      'decay'
};

// L1 category definitions (colour, icon, label, description)
var REGIME_L1_DEFS = {
  stable:      { key: 'stable',      color: '#50B86C', icon: '⚖', label: 'Stable',      desc: 'Direct-to-steady-state' },
  oscillation: { key: 'oscillation', color: '#9B59B6', icon: '🔄', label: 'Oscillation', desc: 'Sustained periodic' },
  growth:      { key: 'growth',      color: '#5B9BD5', icon: '📈', label: 'Growth',     desc: 'Increase-then-stable' },
  decay:       { key: 'decay',       color: '#E8913A', icon: '📉', label: 'Decay',      desc: 'Decay-then-stable' }
};

// Colour variants for L2 (species-size tiers within each L1)
var REGIME_L2_COLOR_VARIANTS = {
  stable:      ['#6DC88E','#82D49E','#97E0AE','#ACECBE'],
  oscillation: ['#AD6BC4','#BF7DD4','#D18FE4','#E3A1F4'],
  growth:      ['#6DABE5','#82BBF0','#97CBFB','#ACDBFF'],
  decay:       ['#F0A34A','#F5B560','#FAC776','#FFD98C']
};
