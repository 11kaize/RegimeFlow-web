#!/usr/bin/env python3
"""Generate descriptive display names for all 1,050 models."""
import json, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

MANUAL_NAMES = {
    "BIOMD0000000003": "Minimal Ca²⁺ Oscillation (Goldbeter 1991, 3 spp.)",
    "BIOMD0000000004": "Ca²⁺ Oscillation with CaM Kinase (Goldbeter 1991, 5 spp.)",
    "BIOMD0000000005": "Xenopus Cell Cycle / MPF Oscillation (Tyson 1991, 6 spp.)",
    "BIOMD0000000006": "Xenopus Cell Cycle — Simplified (Tyson 1991, 2 spp.)",
    "BIOMD0000000016": "Drosophila Circadian Rhythm (Goldbeter 1995, 6 spp.)",
    "BIOMD0000000036": "Fission Yeast Cell Cycle (Tyson 1999, 2 spp.)",
    "BIOMD0000000079": "Mammalian Circadian Clock (Goldbeter 2006, 3 spp.)",
    "BIOMD0000000098": "Minimal Ca²⁺ Oscillator (Goldbeter 1990, 2 spp.)",
    "BIOMD0000000057": "Smooth Muscle Ca²⁺ Dynamics (Sneyd 2002, 6 spp.)",
    "BIOMD0000000022": "Drosophila Circadian Feedback Loop (Ueda 2001)",
    "BIOMD0000000073": "Mammalian Circadian Clock — Simplified (Leloup 2003)",
    "BIOMD0000000074": "Mammalian Circadian Clock — Full (Leloup 2003)",
    "BIOMD0000000078": "Mammalian Circadian Rhythm (Leloup 2003)",
    "BIOMD0000000083": "Mammalian Circadian PER/TIM Loop (Leloup 2003)",
    "BIOMD0000000048": "EGFR Signaling Cascade (Kholodenko 1999, 23 spp.)",
    "BIOMD0000000049": "MAPK Cascade — PC12 Cell Differentiation (Sasagawa 2005, 94 spp.)",
    "BIOMD0000000054": "Blood Coagulation Cascade (Ataullahkhanov 1996, 3 spp.)",
    "BIOMD0000000106": "NF-κB Signaling Pathway (Yang 2007, 23 spp.)",
    "BIOMD0000000137": "Insulin / EGFR Signaling Network (Sedaghat 2002)",
    "BIOMD0000000051": "E. coli Central Carbon Metabolism (Chassagnole 2002, 18 spp.)",
    "BIOMD0000000062": "E. coli PTS Sugar Transport (Bhartiya 2003, 3 spp.)",
    "BIOMD0000000065": "E. coli lac Operon Regulation (Yildirim 2003, 8 spp.)",
    "BIOMD0000000244": "E. coli Stress Response / RelA System (Kotte 2010, 47 spp.)",
    "BIOMD0000000762": "T Cell Mediated Tumor Immunity (Kuznetsov 1994, 2 spp.)",
    "BIOMD0000000763": "T Cell — Tumor Cell Dynamics (Dritschel 2018)",
    "BIOMD0000000785": "T Cell Cytotoxicity Against Tumor (Costa 2003)",
    "BIOMD0000000790": "Immune Response to Tumor Cell (Alvarez 2019, 4 spp.)",
    "BIOMD0000000061": "Yeast Glycolytic Oscillation (Wolf 2001)",
    "BIOMD0000000063": "Yeast Glycolysis — Detailed (Teusink 2000)",
    "BIOMD0000000056": "S. cerevisiae Pentose Phosphate Pathway (Bruggeman 2002)",
    "BIOMD0000000857": "Plant Sucrose / Starch Metabolism (Larbat 2016)",
    "BIOMD0000000858": "Plant Phenolic / Sucrose Metabolism (Larbat 2016)",
    "BIOMD0000000859": "Plant Starch / Sucrose Metabolism (Larbat 2016)",
    "BIOMD0000000059": "Pancreatic β-Cell Excitability (Fridlyand 2003)",
    "BIOMD0000000060": "β-Cell Mitochondrial Metabolism (Magnus 1997)",
    "BIOMD0000000647": "MAPK Signaling Pathway (Kwang 2003)",
    "BIOMD0000000666": "PI3K/AKT/mTOR Signaling in Cancer (Pappalardo 2016)",
    "BIOMD0000000695": "Keratinocyte Blue Light Response / Psoriasis (Felix-Garza 2017)",
    "BIOMD0000000028": "MAPK Cascade with Phosphatase Regulation (Markevich 2004)",
    "BIOMD0000000030": "MAPK Signaling Bistability (Markevich 2004)",
    "BIOMD0000000032": "S. cerevisiae HOG Pathway (Kofahl 2004)",
    "BIOMD0000000033": "Eukaryotic MAPK Pathway (Brown 2004)",
    "BIOMD0000000179": "Wnt / β-Catenin Signaling (Kim 2007)",
    "BIOMD0000000208": "EGFR-ERK Signaling Crosstalk (Deineko 2003)",
}

REGIME_LABELS = {
    "oscillation": "Oscillation",
    "inc_stable": "Growth → Stable",
    "dec_stable": "Decay → Stable",
    "directly_stable": "Stable / Homeostatic",
    "increasing": "Monotonic Growth",
    "decreasing": "Monotonic Decay",
}

with open("../tmp_model_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

bio_models = data["models"]
model_info = data["info"]


def clean_desc(raw, max_len=80):
    if not raw:
        return ""
    for prefix in [
        "Experimental Factor Ontology-", "KEGG Pathway-", "KEGG Drug-",
        "Brenda Tissue Ontology-", "Gene Ontology-", "Reactome-",
        "Human Disease Ontology-", "NCIt-",
    ]:
        raw = raw.replace(prefix, "")
    parts = [p.strip() for p in raw.replace("|", ";").replace(" \x0c", ";").split(";")]
    best = parts[0] if parts else raw
    for p in parts:
        p = p.strip()
        if len(p) > len(best) and len(p.split()) >= 2:
            best = p
    if len(best) > max_len:
        best = best[:max_len].rsplit(" ", 1)[0] + "..."
    return best.strip()


def extract_organism(desc):
    organisms = {
        "homo sapiens": "Homo sapiens",
        "mus musculus": "Mouse",
        "rattus": "Rat",
        "saccharomyces cerevisiae": "S. cerevisiae",
        "drosophila melanogaster": "Drosophila",
        "escherichia coli": "E. coli",
        "arabidopsis thaliana": "Arabidopsis",
        "xenopus": "Xenopus",
        "danio rerio": "Zebrafish",
        "cavia porcellus": "Guinea Pig",
        "dictyostelium discoideum": "Dictyostelium",
        "trypanosoma": "Trypanosoma",
    }
    d = desc.lower()
    for key, label in organisms.items():
        if key in d:
            return label
    words = desc.split()
    return " ".join(words[:3]) if len(words) >= 2 else desc[:30]


display_names = {}
stats = {"manual": 0, "process": 0, "domain": 0, "unclassified": 0, "fallback": 0}

for m in bio_models:
    mid = m["id"]
    name = m["name"]
    regime = m.get("regime", "")
    species = m.get("species", 0)
    info = model_info.get(mid, {})
    domain = info.get("domain", "")
    process = info.get("process", "")
    desc = info.get("desc", "")
    regime_label = REGIME_LABELS.get(regime, regime)

    if mid in MANUAL_NAMES:
        display_names[mid] = MANUAL_NAMES[mid]
        stats["manual"] += 1
    elif process and desc:
        cleaned = clean_desc(desc, 60)
        if cleaned:
            display_names[mid] = f"{cleaned} ({regime_label}, {species} spp.)"
        else:
            display_names[mid] = f"{process} Model ({regime_label}, {species} spp.)"
        stats["process"] += 1
    elif domain and domain != "Unclassified" and desc:
        org = extract_organism(desc)
        display_names[mid] = f"{org} — {regime_label} Model ({species} spp.)"
        stats["domain"] += 1
    elif desc and len(desc) > 5:
        cleaned = clean_desc(desc, 60)
        if cleaned:
            display_names[mid] = f"{cleaned} ({regime_label})"
        else:
            display_names[mid] = f"{regime_label} Model ({species} spp.) — {name}"
        stats["unclassified"] += 1
    else:
        display_names[mid] = f"{regime_label} Model ({species} spp.) — {name}"
        stats["fallback"] += 1

# Write output
out_path = "../web/data/model-display-names.js"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("// Auto-generated descriptive display names for all 1,050 models.\n")
    f.write("// Generated by scripts/generate_display_names.py\n")
    f.write(f"// Manual: {stats['manual']}, Process: {stats['process']}, "
            f"Domain: {stats['domain']}, Unclassified: {stats['unclassified']}, "
            f"Fallback: {stats['fallback']}\n")
    f.write("var MODEL_DISPLAY_NAMES = {\n")
    for mid in sorted(display_names.keys()):
        escaped = display_names[mid].replace("\\", "\\\\").replace("'", "\\'")
        f.write(f"  '{mid}': '{escaped}',\n")
    f.write("};\n")

print(f"Generated {len(display_names)} display names -> {out_path}")
for k, v in stats.items():
    print(f"  {k}: {v}")

print()
print("Sample names:")
samples = [
    "BIOMD0000000003", "BIOMD0000000005", "BIOMD0000000048", "BIOMD0000000051",
    "BIOMD0000000762", "BIOMD0000000008", "BIOMD0000000062", "BIOMD0000000179",
    "BIOMD0000000857", "MODEL0910896131",
]
for mid in samples:
    print(f"  {mid}: {display_names.get(mid, 'NOT FOUND')}")
