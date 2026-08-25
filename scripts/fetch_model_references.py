#!/usr/bin/env python3
"""
Fetch paper reference (title / authors / journal / year / PubMed) + model
description for every model from the EBI BioModels REST API, and generate
web/data/model-references.js.

Source:  https://www.ebi.ac.uk/biomodels/{id}  (Accept: application/json)
         .publication  -> paper metadata (PubMed ID, title, authors, journal…)
         .description  -> HTML notes containing dc:description (model summary)

Resumable: caches each fetched record to scripts/.cache/model_refs.json so an
interrupted run can be resumed without re-hitting the API.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Load model IDs
# ---------------------------------------------------------------------------
print("Loading model data…")
result = subprocess.run(
    ["node", "-e",
     "var fs=require('fs');eval(fs.readFileSync('../web/data/bio-models.js','utf8'));"
     "console.log(JSON.stringify(BIO_MODELS_DATA.map(function(m){return m.id;})))"],
    capture_output=True, text=True, cwd=os.getcwd(),
)
model_ids = json.loads(result.stdout)
print(f"  {len(model_ids)} models")

CACHE_PATH = ".cache/model_refs.json"
os.makedirs(".cache", exist_ok=True)
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)
else:
    cache = {}

API = "https://www.ebi.ac.uk/biomodels/{mid}"


def fetch_one(mid):
    if mid in cache:
        return mid, cache[mid], "cache"
    url = API.format(mid=mid)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return mid, None, "404"
        # transient — retry once
        time.sleep(1.0)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            return mid, None, "error"
    except Exception:
        return mid, None, "error"

    pub = data.get("publication") or {}
    desc_html = data.get("description") or ""

    # model summary from <div class="dc:description"><p>…</p></div>
    model_desc = ""
    m = re.search(r'class="dc:description"[^>]*>\s*<p>(.*?)</p>', desc_html, re.S)
    if m:
        model_desc = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", m.group(1))).strip()

    authors = [a.get("name", "") for a in pub.get("authors", []) if a.get("name")]

    record = {
        "title": (pub.get("title") or "").strip(),
        "authors": authors,
        "journal": (pub.get("journal") or "").strip(),
        "year": pub.get("year") or "",
        "volume": pub.get("volume") or "",
        "issue": pub.get("issue") or "",
        "pages": pub.get("pages") or "",
        "pubmedId": pub.get("accession") or "",
        "link": (pub.get("link") or "").strip(),
        "description": model_desc,
    }
    # Drop empty strings to keep the file lean
    record = {k: v for k, v in record.items() if v not in ("", [], None)}
    return mid, record, "ok"


def main():
    todo = [mid for mid in model_ids if mid not in cache]
    print(f"  cached: {len(model_ids) - len(todo)}, to fetch: {len(todo)}")

    stats = {"ok": 0, "404": 0, "error": 0, "cache": 0}
    done = 0
    if todo:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = {ex.submit(fetch_one, mid): mid for mid in todo}
            for fut in as_completed(futures):
                mid, record, status = fut.result()
                stats[status] = stats.get(status, 0) + 1
                if record is not None:
                    cache[mid] = record
                done += 1
                if done % 100 == 0:
                    print(f"    {done}/{len(todo)} fetched…")
                    _save_cache(cache)
        _save_cache(cache)
    else:
        stats["cache"] = len(model_ids)

    print(f"  stats: {stats}")

    _write_js(cache)


def _save_cache(cache):
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    os.replace(tmp, CACHE_PATH)


def _js_string(s):
    return "'" + str(s).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _write_js(cache):
    out = "../web/data/model-references.js"
    lines = [
        "// Paper reference + model description for each model.",
        "// Fetched from EBI BioModels REST API (https://www.ebi.ac.uk/biomodels/{id}).",
        "// Keys: title (paper), authors, journal, year, volume, issue, pages,",
        "//       pubmedId, link, description (model summary).",
        "var MODEL_REFERENCES = {",
    ]
    for mid in model_ids:
        rec = cache.get(mid)
        if not rec:
            continue
        parts = []
        for key in ("title", "authors", "journal", "year", "volume",
                    "issue", "pages", "pubmedId", "link", "description"):
            if key in rec:
                val = rec[key]
                if isinstance(val, list):
                    # Output a real JS array literal (not a JSON string), so the
                    # front-end can call .join() on it directly.
                    parts.append(f"{key}: [{', '.join(_js_string(v) for v in val)}]")
                else:
                    parts.append(f"{key}: {_js_string(val)}")
        lines.append("  " + _js_string(mid) + ": { " + ", ".join(parts) + " },")
    lines.append("};")
    lines.append("")

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out} ({len(lines) - 2} references)")


if __name__ == "__main__":
    main()
