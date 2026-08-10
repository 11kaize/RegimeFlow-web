#!/usr/bin/env python3
"""
Fetch official BioModels display names from the EBI Search API
and regenerate model-display-names.js with verified data.

BIOMD models → fetched from EBI API (authoritative)
MODEL-prefix models → formulaic from domain/regime data
"""
import json
import re
import time
import urllib.request
import urllib.error
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Step 1: Load all BIOMD IDs
# ---------------------------------------------------------------------------
print("Loading model data...")
# Use Node to parse JS data files into JSON
import subprocess
result = subprocess.run(
    ["node", "-e", """
        var fs = require('fs');
        eval(fs.readFileSync('../web/data/bio-models.js', 'utf8'));
        eval(fs.readFileSync('../web/data/bio-domains.js', 'utf8'));
        eval(fs.readFileSync('../web/data/bio-processes.js', 'utf8'));
        var info = {};
        for (var d in BIO_DOMAIN_DATA) {
            BIO_DOMAIN_DATA[d].forEach(function(m) { info[m.id] = {domain:d, desc:m.domain||''}; });
        }
        for (var p in BIO_PROCESSES_DATA) {
            BIO_PROCESSES_DATA[p].forEach(function(m) {
                if (info[m.id]) { info[m.id].process = p; info[m.id].desc = m.domain||''; }
                else { info[m.id] = {process:p, desc:m.domain||''}; }
            });
        }
        console.log(JSON.stringify({models: BIO_MODELS_DATA, info: info}));
    """],
    capture_output=True, text=True, cwd=os.getcwd()
)
data = json.loads(result.stdout)
bio_models = data["models"]
model_info = data["info"]

# Separate BIOMD and MODEL IDs
biomd_ids = [m["id"] for m in bio_models if m["id"].startswith("BIOMD")]
model_ids = [m["id"] for m in bio_models if not m["id"].startswith("BIOMD")]
print(f"  BIOMD models: {len(biomd_ids)}")
print(f"  MODEL-prefix: {len(model_ids)}")

# ---------------------------------------------------------------------------
# Step 2: Fetch official names from EBI Search API
# ---------------------------------------------------------------------------
REGIME_LABELS = {
    "oscillation": "Oscillation",
    "inc_stable": "Growth → Stable",
    "dec_stable": "Decay → Stable",
    "directly_stable": "Stable / Homeostatic",
    "increasing": "Monotonic Growth",
    "decreasing": "Monotonic Decay",
}

def fetch_batch(ids_batch):
    """Query EBI Search for a batch of BIOMD IDs, return {id: name}."""
    # EBI Search uses Lucene query syntax. We need exact ID matches.
    # Build query like: id:BIOMD0000000003 OR id:BIOMD0000000004
    query_parts = [f"id:{mid}" for mid in ids_batch]
    query = " OR ".join(query_parts)
    url = (
        "https://www.ebi.ac.uk/ebisearch/ws/rest/biomodels"
        "?query=" + query.replace(" ", "%20")
        + f"&format=json&size={len(ids_batch) + 5}&fields=name"
    )
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
        names = {}
        requested = set(ids_batch)
        for entry in result.get("entries", []):
            eid = entry["id"]
            if eid not in requested:
                continue  # Skip fuzzy matches that aren't our targets
            fields = entry.get("fields", {})
            name_list = fields.get("name", [])
            if name_list:
                names[eid] = name_list[0]
        return names
    except Exception as e:
        print(f"    API error: {e}")
        return {}

# Fetch in batches to avoid timeouts
BATCH_SIZE = 12  # EBI Search caps URL length / query complexity
all_names = {}
total_batches = (len(biomd_ids) + BATCH_SIZE - 1) // BATCH_SIZE

print(f"\nFetching {len(biomd_ids)} models from EBI Search API...")
print(f"  Batches: {total_batches} (batch size: {BATCH_SIZE})")

for i in range(0, len(biomd_ids), BATCH_SIZE):
    batch = biomd_ids[i:i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    print(f"  Batch {batch_num}/{total_batches}: {batch[0]} ... {batch[-1]} ({len(batch)} ids)", end=" ")
    names = fetch_batch(batch)
    all_names.update(names)
    print(f"→ {len(names)} retrieved")
    if batch_num < total_batches:
        time.sleep(0.3)  # Be polite to the API

print(f"\n  Total retrieved: {len(all_names)} / {len(biomd_ids)}")
missing = [mid for mid in biomd_ids if mid not in all_names]
if missing:
    print(f"  Missing ({len(missing)}): {missing[:10]}...")

# ---------------------------------------------------------------------------
# Step 3: Clean up BioModels names for display
# ---------------------------------------------------------------------------
def clean_biomodels_name(raw_name):
    """Clean a BioModels display name into a user-friendly label."""
    # Remove trailing whitespace and newlines
    name = raw_name.strip().replace("\n", " ").replace("  ", " ")
    # Truncate at first newline or excessive length
    if len(name) > 120:
        name = name[:117] + "..."
    return name

# ---------------------------------------------------------------------------
# Step 4: Generate formualic names for MODEL-prefix models
# ---------------------------------------------------------------------------
def extract_organism(desc):
    organisms = {
        "homo sapiens": "Homo sapiens", "mus musculus": "Mouse", "rattus": "Rat",
        "saccharomyces cerevisiae": "S. cerevisiae", "drosophila melanogaster": "Drosophila",
        "escherichia coli": "E. coli", "arabidopsis thaliana": "Arabidopsis",
        "xenopus": "Xenopus", "danio rerio": "Zebrafish",
    }
    d = desc.lower()
    for key, label in organisms.items():
        if key in d:
            return label
    words = desc.split()
    return " ".join(words[:3]) if len(words) >= 2 else desc[:30]

# ---------------------------------------------------------------------------
# Step 5: Assemble final display names
# ---------------------------------------------------------------------------
display_names = {}
stats = {"biomodels_api": 0, "formulaic": 0, "fallback": 0}

for m in bio_models:
    mid = m["id"]
    name = m["name"]
    regime = m.get("regime", "")
    species = m.get("species", 0)
    info = model_info.get(mid, {})
    desc = info.get("desc", "")
    domain = info.get("domain", "")
    process = info.get("process", "")
    rlabel = REGIME_LABELS.get(regime, regime)

    if mid in all_names:
        # Use the official BioModels name
        display_names[mid] = clean_biomodels_name(all_names[mid])
        stats["biomodels_api"] += 1
    elif desc and len(desc) > 5:
        # Fallback: use domain description + regime
        org = extract_organism(desc)
        if process:
            display_names[mid] = f"{org} — {rlabel} ({process}, {species} spp.)"
        else:
            display_names[mid] = f"{org} — {rlabel} Model ({species} spp.)"
        stats["formulaic"] += 1
    else:
        display_names[mid] = f"{rlabel} Model ({species} spp.) — {name}"
        stats["fallback"] += 1

# ---------------------------------------------------------------------------
# Step 6: Write output
# ---------------------------------------------------------------------------
out_path = "../web/data/model-display-names.js"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("// Descriptive display names for all 1,050 models.\n")
    f.write("// Fetched from EBI BioModels API + formulaic fallback.\n")
    f.write(f"// BioModels API: {stats['biomodels_api']}, "
            f"Formulaic: {stats['formulaic']}, "
            f"Fallback: {stats['fallback']}\n")
    f.write("var MODEL_DISPLAY_NAMES = {\n")
    for mid in sorted(display_names.keys()):
        escaped = display_names[mid].replace("\\", "\\\\").replace("'", "\\'")
        f.write(f"  '{mid}': '{escaped}',\n")
    f.write("};\n")

print(f"\nGenerated {len(display_names)} display names -> {out_path}")
for k, v in stats.items():
    print(f"  {k}: {v}")

# Show samples
print("\nSample names (from API):")
samples = ["BIOMD0000000003", "BIOMD0000000005", "BIOMD0000000048",
           "BIOMD0000000051", "BIOMD0000000762", "BIOMD0000000063"]
for mid in samples:
    print(f"  {mid}: {display_names.get(mid, 'NOT FOUND')}")
