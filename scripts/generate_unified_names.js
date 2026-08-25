// Regenerates web/data/model-display-names.js with unified names.
// Format: "Description (AuthorYear)" — descriptive name first, AuthorYear at the end.
// Full original titles are preserved in MODEL_FULL_NAMES for cross-reference.
//
// Run: node scripts/generate_unified_names.js
// Reads: web/data/bio-models.js, web/data/bio-processes.js, web/data/model-display-names.js
// Writes: web/data/model-display-names.js

var fs = require("fs");

function load(path) {
  var sandbox = {};
  var code = fs.readFileSync(path, "utf8");
  // strip "var X =" -> just evaluate into sandbox via indirect eval
  eval(code.replace(/var\s+([A-Za-z_$][\w$]*)\s*=/g, "sandbox.$1 ="));
  return sandbox;
}

var models = load("web/data/bio-models.js").BIO_MODELS_DATA;
var procs  = load("web/data/bio-processes.js").BIO_PROCESSES_DATA;
var names  = load("web/data/model-display-names.js");

var FULL = names.MODEL_FULL_NAMES || names.MODEL_DISPLAY_NAMES || {};

// GO term map (id -> go), skip obsolete
var goMap = {};
for (var cat in procs) {
  procs[cat].forEach(function (m) {
    var go = m.go || "";
    if (go && !/^obsolete/i.test(go)) goMap[m.id] = go;
  });
}

var REGIME_LABEL = {
  oscillation: "Oscillation",
  inc_stable: "Growth → Stable",
  dec_stable: "Decay → Stable",
  directly_stable: "Homeostatic",
  increasing: "Monotonic Growth",
  decreasing: "Monotonic Decay"
};

function escRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

function splitCamel(s) {
  return s
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2");
}

function normalize(s) {
  var out = String(s)
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  out = splitCamel(out).replace(/\s+/g, " ").trim();
  out = out.replace(/Ru\s+Bis\s+CO/gi, "RuBisCO"); // restore over-split acronym
  if (out.length) out = out.charAt(0).toUpperCase() + out.slice(1);
  return out;
}

function truncWord(s, n) {
  if (s.length <= n) return s;
  var cut = s.slice(0, n);
  var lastSpace = cut.lastIndexOf(" ");
  if (lastSpace > n * 0.5) cut = cut.slice(0, lastSpace);
  return cut.trim() + "…";
}

function description(m) {
  var name = m.name || "";
  var full = FULL[m.id] || "";

  if (!full) return "";

  // Fallback: "Regime Model (N spp.) — AuthorYear"
  if (/Model \(\d+ spp\.\) — /.test(full)) {
    var go = goMap[m.id];
    if (go && go.length < 48) return normalize(go);
    var reg = REGIME_LABEL[m.regime] || m.regime;
    return reg + " · " + m.species + " spp.";
  }

  // 0. exact "name Description" (space-separated, no dash) — e.g. "Wodarz1999 CTL memory response HIV"
  var m0 = full.match(new RegExp("^" + escRe(name) + "\\s+(?![-–—])(.+)$"));
  if (m0) return normalize(m0[1]);

  // 1. exact "name - Desc" / "name_Desc" / "nameCamelDesc"
  var m1 = full.match(new RegExp("^" + escRe(name) + "\\s*[-–—]\\s*(.+)$"));
  if (m1) return normalize(m1[1]);
  var m2 = full.match(new RegExp("^" + escRe(name) + "[_]\\s*(.+)$"));
  if (m2) return normalize(m2[1]);
  var m3 = full.match(new RegExp("^" + escRe(name) + "([A-Z].+)$"));
  if (m3) return normalize(m3[1]);

  // 2. generic "Author-year - Desc" / "Author-year_Desc" (name may differ from title prefix)
  var g1 = full.match(/^[A-Za-z][A-Za-z0-9_ .'-]*?\d{4}\s*[-–—]\s*(.+)$/);
  if (g1) return normalize(g1[1]);
  var g2 = full.match(/^[A-Za-z][A-Za-z0-9_ .'-]*?\d{4}_\s*(.+)$/);
  if (g2) return normalize(g2[1]);

  // 3. generic "Author_Desc" without a year (e.g. "PetelenzKuehn_osmoadaptation_pfk2627D")
  var g3 = full.match(/^([A-Za-z][A-Za-z0-9]{1,24})_\s*([A-Za-z][A-Za-z0-9 _.-]*)$/);
  if (g3) return normalize(g3[2]);

  // 4. generic "Author Year Description" (space-separated, no dash)
  var g4 = full.match(/^[A-Za-z][A-Za-z0-9_ .'-]*?\d{4}\s+(?=[A-Z])/);
  if (g4) return normalize(full.slice(g4[0].length));

  // no AuthorYear prefix — normalize as-is
  return normalize(full);
}

function stripTrailingYear(desc, year) {
  if (year && desc) {
    var re = new RegExp("\\s*" + year + "$");
    if (re.test(desc)) return desc.replace(re, "").trim();
  }
  return desc;
}

var out = {};
var fullOut = {};
var stats = { go: 0, fallback: 0, title: 0, other: 0 };

models.forEach(function (m) {
  fullOut[m.id] = FULL[m.id] || "";
  var year = String(m.name.match(/\d{4}$/) || "");
  var desc = description(m);

  if (/Model \(\d+ spp\.\) — /.test(FULL[m.id])) {
    stats[goMap[m.id] ? "go" : "fallback"]++;
  } else if (/^[A-Za-z]/.test(FULL[m.id])) {
    stats.title++;
  } else {
    stats.other++;
  }

  desc = stripTrailingYear(desc, year);
  desc = truncWord(desc, 48);
  out[m.id] = desc ? desc + " (" + m.name + ")" : m.name;
});

function q(s) {
  return "'" + String(s).replace(/\\/g, "\\\\").replace(/'/g, "\\'") + "'";
}

var lines = [];
lines.push("// Unified display names — \"Description (AuthorYear)\" format.");
lines.push("// MODEL_DISPLAY_NAMES: short descriptive name, AuthorYear at the end.");
lines.push("// MODEL_FULL_NAMES: original full EBI BioModels title (cross-reference).");
lines.push("// Total: " + models.length + " models (title: " + stats.title +
  ", GO: " + stats.go + ", regime fallback: " + stats.fallback +
  ", other: " + stats.other + ")");
lines.push("var MODEL_DISPLAY_NAMES = {");
models.forEach(function (m) { lines.push("  " + q(m.id) + ": " + q(out[m.id]) + ","); });
lines.push("};");
lines.push("");
lines.push("var MODEL_FULL_NAMES = {");
models.forEach(function (m) { lines.push("  " + q(m.id) + ": " + q(fullOut[m.id]) + ","); });
lines.push("};");
lines.push("");

fs.writeFileSync("web/data/model-display-names.js", lines.join("\n"), "utf8");
console.log("Wrote web/data/model-display-names.js");
console.log("Stats:", JSON.stringify(stats));

// Print samples for eyeballing
var samples = ["BIOMD0000000010","BIOMD0000000016","BIOMD0000000003","BIOMD0000000515",
  "BIOMD0000000011","BIOMD0000000020","BIOMD0000000008","BIOMD0000000048",
  "BIOMD0000000009","BIOMD0000000069","MODEL0975191032","MODEL0910896131",
  "MODEL1002160000","MODEL1102210001","MODEL1004070000","MODEL1005050000","MODEL1101170000"];
console.log("\n=== samples ===");
var byId = {};
models.forEach(function (m) { byId[m.id] = m; });
samples.forEach(function (id) {
  console.log("  " + id);
  console.log("    now:  " + FULL[id]);
  console.log("    new:  " + out[id]);
});
