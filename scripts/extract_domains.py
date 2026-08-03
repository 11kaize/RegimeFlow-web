"""Extract model_id -> domain mapping from paper PDF text."""
import re, json, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

with open(os.path.join(PROJECT_ROOT, 'rf_paper.txt'), 'r', encoding='utf-8') as f:
    text = f.read()

# Find Table 11 section
start = text.find('Table 11. Full list of 1,050')
end = text.find('D.2. Systems Biology Model Specifications')
if end < 0:
    end = len(text)
table_text = text[start:end]

lines = table_text.split('\n')

# Extract all model ID + domain pairs
all_model_ids = []
all_domains = []

for line in lines:
    line = line.strip()
    if not line:
        continue

    # Lines with model IDs (BIOMD... or MODEL...)
    ids = re.findall(r'(BIOMD\d{10}|MODEL\d{3,15})', line)
    if ids and 'Domain' not in line and 'Table' not in line:
        all_model_ids.extend(ids)

    # Domain lines - these have taxonomy info
    if 'Taxonomy-' in line or re.match(r'^(cell |blood |immune )', line):
        all_domains.append(line)

print(f"Model IDs: {len(all_model_ids)}")
print(f"Domain lines: {len(all_domains)}")

# Count unique taxonomy prefixes
tax_counts = {}
for d in all_domains:
    # Simplify: get first meaningful part
    parts = d.replace('Taxonomy-', '').split('|')[0].strip()
    key = parts.split(' ')[0][:20]
    tax_counts[key] = tax_counts.get(key, 0) + 1

print("\nDomain categories:")
for k, v in sorted(tax_counts.items(), key=lambda x: -x[1])[:20]:
    print(f"  {k}: {v}")
