/* ================================================================
   RegimeFlow — Pathway-Classified Bio Models
   Organises real BIOMD models by biological pathway for the
   Prediction page. Each entry maps to an actual CSV trajectory
   on HuggingFace: HengRao/SysBio-Traj/Data/{id}/{name}.csv

   ⚠️ Pathway assignments are BEST-EFFORT. Marked with ⚑ for
      professor review. Edit freely — this file is the mapping.
   ================================================================ */

var PATHWAY_MODELS = [
  {
    pathway: "Signal Transduction",
    icon: "⚡",
    desc: "Intracellular signalling cascades (MAPK, EGFR, Ca²⁺)",
    models: [
      { id: "BIOMD0000000600", name: "Oire2011",          regime: "directly_stable", species: 3,   note: "EGFR-ERK crosstalk" },
      { id: "BIOMD0000000010", name: "Kholodenko2000",    regime: "oscillation",     species: 8,   note: "MAPK cascade (ultrasensitivity)" },
      { id: "BIOMD0000000019", name: "Schoeberl2002",     regime: "directly_stable", species: 93,  note: "EGFR signalling network" },
      { id: "BIOMD0000000048", name: "Kholodenko1999",    regime: "dec_stable",      species: 23,  note: "EGF receptor pathway" },
      { id: "BIOMD0000000084", name: "Hornberg2005",      regime: "dec_stable",      species: 8,   note: "MAPK cascade (ERK)" },
      { id: "BIOMD0000000667", name: "Hornberg2005",      regime: "dec_stable",      species: 103, note: "MAPK cascade (large variant)" },
      { id: "BIOMD0000000650", name: "Owen1998",          regime: "directly_stable", species: 3,   note: "⚑ Ca²⁺ oscillation / signalling" },
      { id: "BIOMD0000000670", name: "Owen1998",          regime: "dec_stable",      species: 3,   note: "⚑ Ca²⁺ signalling variant" },
      { id: "BIOMD0000000647", name: "Kwang2003",         regime: "dec_stable",      species: 11,  note: "⚑ review needed" },
      { id: "BIOMD0000000011", name: "Levchenko2000",     regime: "inc_stable",      species: 22,  note: "Scaffold protein signalling" }
    ]
  },
  {
    pathway: "Immune Response",
    icon: "🛡",
    desc: "Tumour-immune interactions, vaccine models, infection dynamics",
    models: [
      { id: "BIOMD0000000762", name: "Kuznetsov1994",     regime: "oscillation",     species: 2,   note: "Tumour-immune oscillation (classic)" },
      { id: "BIOMD0000000666", name: "Pappalardo2016",    regime: "dec_stable",      species: 35,  note: "Vaccine / immune response" },
      { id: "BIOMD0000000787", name: "Frascoli2014",      regime: "oscillation",     species: 2,   note: "Tumour-immune dynamics" },
      { id: "BIOMD0000000809", name: "Malinzi2018",       regime: "dec_stable",      species: 5,   note: "Oncolytic virotherapy" },
      { id: "BIOMD0000000695", name: "FelixGarza2017",    regime: "dec_stable",      species: 12,  note: "⚑ Immune / infection model" },
      { id: "BIOMD0000000009", name: "Huang1996",         regime: "inc_stable",      species: 26,  note: "⚑ IL-1 / NF-κB signalling" }
    ]
  },
  {
    pathway: "Metabolic Pathways",
    icon: "🧪",
    desc: "Plant & microbial metabolism, glycolysis, TCA cycle",
    models: [
      { id: "BIOMD0000000857", name: "Larbat2016",        regime: "dec_stable",      species: 9,   note: "Plant phenylpropanoid metabolism" },
      { id: "BIOMD0000000858", name: "Larbat2016",        regime: "oscillation",     species: 8,   note: "Plant metabolism (oscillatory)" },
      { id: "BIOMD0000000859", name: "Larbat2016",        regime: "oscillation",     species: 12,  note: "Plant metabolism (large)" }
    ]
  },
  {
    pathway: "Cell Cycle",
    icon: "🔄",
    desc: "Cell division, mitosis, checkpoints",
    models: [
      { id: "BIOMD0000000676", name: "Chen2006",          regime: "inc_stable",      species: 13,  note: "⚑ Cell cycle regulation" },
      { id: "BIOMD0000000828", name: "Jung2019",          regime: "inc_stable",      species: 5,   note: "⚑ Cell cycle / proliferation" },
      { id: "BIOMD0000000829", name: "Jung2019",          regime: "oscillation",     species: 11,  note: "⚑ Cell cycle (oscillatory variant)" }
    ]
  },
  {
    pathway: "Apoptosis & Stress",
    icon: "💀",
    desc: "Programmed cell death, oxidative stress, DNA damage",
    models: [
      { id: "BIOMD0000000785", name: "Costa2003",         regime: "dec_stable",      species: 2,   note: "⚑ Apoptosis / cell death" },
      { id: "BIOMD0000000790", name: "Alvarez2019",       regime: "dec_stable",      species: 4,   note: "⚑ Stress response" },
      { id: "BIOMD0000000797", name: "Hu2018",            regime: "dec_stable",      species: 4,   note: "⚑ Apoptosis pathway" },
      { id: "BIOMD0000000049", name: "Sasagawa2005",      regime: "dec_stable",      species: 94,  note: "⚑ MAPK / stress (large model)" },
      { id: "BIOMD0000000056", name: "Chen2004",          regime: "oscillation",     species: 50,  note: "⚑ p53-Mdm2 oscillation / DNA damage" }
    ]
  },
  {
    pathway: "Other Processes",
    icon: "📦",
    desc: "Additional biological process models — review & reassign",
    models: [
      { id: "BIOMD0000000763", name: "Dritschel2018",     regime: "oscillation",     species: 3,   note: "⚑ review needed" },
      { id: "BIOMD0000000792", name: "Hu2019",            regime: "inc_stable",      species: 6,   note: "⚑ review needed" },
      { id: "BIOMD0000000848", name: "FatehiChenar2018",  regime: "inc_stable",      species: 9,   note: "⚑ review needed" }
    ]
  }
];

// Derived: total model count across all pathways
var PATHWAY_TOTAL_MODELS = PATHWAY_MODELS.reduce(function(sum, p) {
  return sum + p.models.length;
}, 0);
