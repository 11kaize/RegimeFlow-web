#!/usr/bin/env python3
"""
Fetch Gene Ontology (GO) biological-process annotations for all 1,050
SysBio-Traj models from the BioModels database, and build a full
biological-process classification (replacing the 30-model heuristic
currently in web/data/bio-processes.js).

Pipeline (resumable — caches every phase to disk):
  1. Fetch each model's modelLevelAnnotations from
     https://www.ebi.ac.uk/biomodels/{id}?format=json
     → cache scripts/cache/bio_process_raw.json
  2. Resolve every unique GO accession via QuickGO (name + aspect)
     → cache scripts/cache/go_terms.json
  3. Keep only GO biological_process terms, map to high-level categories,
     write web/data/bio-processes-full.js

Usage:
    python fetch_go_process.py [--limit N] [--no-fetch]
"""
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from collections import defaultdict, Counter

os.chdir(os.path.dirname(os.path.abspath(__file__)))

CACHE_DIR = os.path.join(os.getcwd(), "cache")
RAW_CACHE = os.path.join(CACHE_DIR, "bio_process_raw.json")
GO_CACHE = os.path.join(CACHE_DIR, "go_terms.json")
OUT_PATH = "../web/data/bio-processes-full.js"

UA = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
GO_ASPECT = "biological_process"


def http_json(url, timeout=40):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def load_ids():
    bm = open("../web/data/bio-models.js", encoding="utf-8").read()
    return re.findall(r'id:"([^"]+)"', bm)


def load_cache(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=0)


# ---------------------------------------------------------------------------
# Phase 1: fetch model-level annotations
# ---------------------------------------------------------------------------
def fetch_model(mid):
    """Return list of {qualifier, resource, accession} for one model."""
    d = http_json(f"https://www.ebi.ac.uk/biomodels/{mid}?format=json")
    out = []
    for a in d.get("modelLevelAnnotations", []):
        out.append({
            "qualifier": a.get("qualifier", ""),
            "resource": a.get("resource", ""),
            "accession": a.get("accession", ""),
        })
    return out


def phase1(ids, limit, do_fetch):
    raw = load_cache(RAW_CACHE)
    todo = [i for i in ids if i not in raw]
    if limit:
        todo = todo[:limit]
    print(f"[Phase 1] {len(ids)} models, {len(raw)} cached, fetching {len(todo)} ...")
    for n, mid in enumerate(todo, 1):
        try:
            raw[mid] = fetch_model(mid)
        except Exception as e:
            raw[mid] = {"__error__": f"{type(e).__name__}: {e}"}
        if n % 50 == 0 or n == len(todo):
            save_cache(RAW_CACHE, raw)
            print(f"  {n}/{len(todo)} ...")
        time.sleep(0.12)
    save_cache(RAW_CACHE, raw)
    print(f"[Phase 1] done. cache has {len(raw)} entries.")
    return raw


# ---------------------------------------------------------------------------
# Phase 2: resolve GO terms via QuickGO
# ---------------------------------------------------------------------------
def phase2(raw):
    # collect unique GO accessions
    go_ids = set()
    for mid, anns in raw.items():
        if isinstance(anns, dict) and "__error__" in anns:
            continue
        for a in anns:
            if a.get("resource") == "Gene Ontology" and a.get("accession"):
                go_ids.add(a["accession"])
    print(f"[Phase 2] {len(go_ids)} unique GO accessions to resolve.")

    cache = load_cache(GO_CACHE)
    todo = [g for g in sorted(go_ids) if g not in cache]
    print(f"  resolving {len(todo)} ...")
    for n, g in enumerate(todo, 1):
        try:
            d = http_json(
                f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{g}/complete")
            r = d.get("results", [{}])[0] if d.get("results") else {}
            cache[g] = {"name": r.get("name", ""), "aspect": r.get("aspect", "")}
        except Exception as e:
            cache[g] = {"name": "", "aspect": f"ERROR:{type(e).__name__}"}
        if n % 100 == 0 or n == len(todo):
            save_cache(GO_CACHE, cache)
            print(f"  {n}/{len(todo)} ...")
        time.sleep(0.08)
    save_cache(GO_CACHE, cache)
    print(f"[Phase 2] done. resolved {len(cache)} GO terms.")
    return cache


# ---------------------------------------------------------------------------
# Phase 3: map GO biological_process terms to high-level categories
# ---------------------------------------------------------------------------
# Ordered keyword→category rules applied to the GO term NAME (lowercased).
# First match wins — keep more specific patterns earlier.
CATEGORY_RULES = [
    ("Signal Transduction", [
        "signal transduction", "signaling", "signal", "mapk", "erk", "egf", "jak",
        "stat", "wnt", "notch", "hedgehog", "nf-kappab", "nfkb", "g protein",
        "gpcr", "cAMP", "receptor", "phosphorylation cascade", "kinase cascade",
        "phosphatase", "ras", "rho", "akt", "mtor", "pi3k", "transduction",
        "second messenger", "inositol", "diacylglycerol", "calcium signaling",
    ]),
    ("Cell Cycle & Division", [
        "cell cycle", "mitotic", "meiotic", "mitosis", "meiosis", "cytokinesis",
        "cyclin", "checkpoint", "dna replication", "chromosome segregation",
        "cell division", "centrosome", "spindle", "g1/s", "g2/m", "anaphase",
        "telophase", "prophase", "metaphase",
    ]),
    ("Metabolism", [
        "metabolic", "metabolism", "glycolysis", "gluconeogenesis", "tca cycle",
        "tricarboxylic", "citrate cycle", "pentose", "biosynthetic", "biosynthesis",
        "catabolic", "catabolism", "anabolic", "amino acid", "lipid", "fatty acid",
        "cholesterol", "nucleotide", "glucose", "sucrose", "starch", "glycogen",
        "atp", "oxidation", "respiration", "photosynthesis", "glycosylation",
        "beta-oxidation", "urea cycle", "purine", "pyrimidine", "carbohydrate",
        "redox", "glutathione",
    ]),
    ("Immune Response", [
        "immune", "t cell", "b cell", "antibody", "antigen", "cytokine", "chemokine",
        "inflammat", "macrophage", "lymphocyte", "nk cell", "toll-like", "tumor immunity",
        "immunity", "complement", "phagocyt",
    ]),
    ("Apoptosis & Cell Death", [
        "apoptosis", "cell death", "necrosis", "senescence", "caspase", "ferroptosis",
        "pyroptosis", "autophagy",
    ]),
    ("Gene Expression & Regulation", [
        "transcription", "translation", "gene expression", "rna", "mrna", "trna",
        "splicing", "epigenetic", "chromatin", "promoter", "methylation", "operon",
        "transcriptional regulation", "ribosome", "dna binding",
    ]),
    ("Development & Differentiation", [
        "development", "differentiation", "morphogenesis", "embryonic", "embryo",
        "organogenesis", "regeneration", "homeostasis", "stem cell", "patterning",
        "growth factor", "angiogenesis", "wound",
    ]),
    ("Circadian Rhythm", ["circadian", "rhythmic"]),
    ("Cell Motility & Cytoskeleton", [
        "motility", "cytoskeleton", "actin", "myosin", "microtubule", "contraction",
        "migration", "chemotaxis", "flagellar", "ciliary",
    ]),
    ("Transport & Trafficking", [
        "transport", "trafficking", "secretion", "endocytosis", "exocytosis",
        "ion channel", "ion transport", "membrane transport", "transmembrane",
        "vesicle", "import", "export", "uptake", "efflux", "osmotic", "phagosome",
    ]),
    ("Stress & Damage Response", [
        "stress", "oxidative", "dna damage", "dna repair", "heat shock", "hypoxia",
        "unfolded protein", "er stress", "damage response",
    ]),
    ("Cell Adhesion & Extracellular", [
        "adhesion", "junction", "extracellular matrix", "integrin", "tight junction",
        "gap junction", "cell-cell",
    ]),
    ("Calcium Homeostasis", ["calcium", "ca2+", "calcineurin", "calmodulin"]),
]


def categorize(go_name):
    n = go_name.lower()
    for cat, kws in CATEGORY_RULES:
        for kw in kws:
            if kw in n:
                return cat
    return "Other"


def phase3(raw, go_cache):
    # For each model, gather its biological_process GO terms (names).
    per_model = {}  # id -> [go term names]
    for mid, anns in raw.items():
        if isinstance(anns, dict) and "__error__" in anns:
            continue
        names = []
        for a in anns:
            if a.get("resource") == "Gene Ontology" and a.get("accession"):
                g = go_cache.get(a["accession"], {})
                if g.get("aspect") == GO_ASPECT and g.get("name"):
                    names.append(g["name"])
        per_model[mid] = names

    # Build classification
    cats = defaultdict(list)
    # need id->name mapping for the output
    bm = open("../web/data/bio-models.js", encoding="utf-8").read()
    name_of = dict(zip(re.findall(r'id:"([^"]+)"', bm),
                       re.findall(r'name:"([^"]+)"', bm)))

    unclassified = []
    for mid, names in per_model.items():
        if not names:
            unclassified.append(mid)
            continue
        cat = categorize(names[0])  # primary GO BP term decides category
        cats[cat].append({
            "id": mid,
            "name": name_of.get(mid, mid),
            "domain": " | ".join(names[:3]),   # keep up to 3 GO terms
            "go": names[0],
        })

    # sort categories by size desc
    for c in cats:
        cats[c].sort(key=lambda m: m["id"])

    stats = {c: len(v) for c, v in sorted(cats.items(), key=lambda kv: -len(kv[1]))}
    return cats, stats, unclassified, per_model


def write_output(cats, stats, unclassified, total):
    header = (
        "// Auto-generated from BioModels GO biological-process annotations\n"
        f"// {total} models total, {sum(stats.values())} classified, "
        f"{len(unclassified)} without a GO biological_process term\n"
        f"// Categories: {', '.join(stats)}\n"
        "// L1 category -> array of {id, name, domain, go}\n"
        "var BIO_PROCESSES_DATA = {\n"
    )
    body = ""
    for cat, models in sorted(cats.items(), key=lambda kv: -len(kv[1])):
        body += f'  "{cat}": [\n'
        for m in models:
            body += (f'    {{id:"{m["id"]}",name:"{m["name"]}",'
                     f'domain:"{m["domain"]}",go:"{m["go"]}"}},\n')
        body += "  ],\n"
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(header + body + "};\n")
    print(f"\nWrote {OUT_PATH}")


def main():
    limit = None
    do_fetch = True
    args = sys.argv[1:]
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])
    if "--no-fetch" in args:
        do_fetch = False

    ids = load_ids()
    print(f"Loaded {len(ids)} model ids.")

    raw = load_cache(RAW_CACHE) if not do_fetch else phase1(ids, limit, do_fetch)
    if limit:
        raw = dict(list(raw.items())[:limit])

    go_cache = phase2(raw)
    cats, stats, unclassified, per_model = phase3(raw, go_cache)

    print("\n=== Biological-process classification ===")
    total = len(raw)
    print(f"Total models fetched: {total}")
    print(f"Classified (has GO biological_process): {sum(stats.values())}")
    print(f"Without GO biological_process: {len(unclassified)}")
    print("\nCategory distribution:")
    for c, n in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {c}: {n}")

    write_output(cats, stats, unclassified, total)


if __name__ == "__main__":
    main()
