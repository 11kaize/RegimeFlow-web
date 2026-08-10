#!/usr/bin/env python3
"""Parse Table 11 domain classification from rf_table.txt and generate bio-domains.js."""
import re, json, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open('../rf_table.txt', 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()

# Find Table 11
t11_start = text.find('Table 11.')
if t11_start < 0:
    print("ERROR: Table 11 not found")
    sys.exit(1)

# Find end — look for "References" or "Appendix"
for marker in ['\nReferences\n', '\nAppendix\n', '\nA Regime-Aware Trajectory']:
    pos = text.find(marker, t11_start + 100)
    if pos > 0:
        t11_end = pos
        break
else:
    t11_end = len(text)

# Re-scan: find the last BIOMD entry
last_biomd = t11_start
for m in re.finditer(r'^BIOMD\d+', text[t11_start:t11_end], re.MULTILINE):
    last_biomd = t11_start + m.start()

# Find where the next section starts after the last BIOMD
after_last = text.find('\n\n', last_biomd + 50)
if after_last > 0:
    t11_end = after_last

table_text = text[t11_start:t11_end]
lines = table_text.split('\n')

# Collect all BIOMD entries with their domain info
model_pattern = re.compile(r'^(BIOMD\d+)\s+(\S+)\s+(.*)')
entries = []
current = None
domain_lines = []
collecting_domain = False

for line in lines:
    stripped = line.rstrip()
    if not stripped:
        continue

    m = model_pattern.match(stripped)
    if m:
        # Save previous entry
        if current:
            domain_str = ' '.join(domain_lines)
            # Extract taxonomy
            tax_match = re.search(r'Taxonomy-([^|]+)', domain_str)
            tax_value = tax_match.group(1).strip() if tax_match else '-'
            entries.append((current[0], current[1], tax_value, domain_str[:200]))

        current = (m.group(1), m.group(2))
        domain_lines = []

        # Check if domain is on the same line
        rest = m.group(3)
        # Try to match domain + sys_dim + par_dim + time_span
        # Domain format: text followed by multiple spaces then digits
        # Also handle multi-word domain before the numeric columns
        dm = re.match(r'^(.+?)\s{2,}(\d+)\s{2,}(\d+)\s{2,}(.+)', rest)
        if dm:
            domain_lines.append(dm.group(1).strip())
        elif rest.strip():
            # Might be continuation of domain without numeric cols yet
            # Check if it looks like a domain (contains Taxonomy-, Gene Ontology-, KEGG-)
            if re.search(r'Taxonomy-|Gene Ontology|KEGG|Biological Process|NCIt-|Reactome-|Experimental Factor', rest):
                domain_lines.append(rest.strip())
            elif not re.match(r'^\d+', rest.strip()):
                domain_lines.append(rest.strip())

        collecting_domain = True
    elif current and collecting_domain:
        s = stripped
        # Stop collecting if we hit a page number or "Continued" marker
        if re.match(r'^\d+$', s) or 'Continued on next page' in s or 'Model ID' in s:
            continue
        # Stop if line looks like it starts with numbers (next entry's data)
        # Actually, BIOMD lines are already caught above, so this is continuation
        if re.search(r'Taxonomy-|Gene Ontology|KEGG|Biological Process|NCIt-|Reactome-|Experimental Factor|Ontology', s):
            domain_lines.append(s)
        elif not re.match(r'^\d+\s', s) and not re.match(r'^\s{20,}\d', s):
            # Check if this is continuation of a wrapped domain name
            if domain_lines and not re.match(r'^\d', s):
                domain_lines.append(s)

# Don't forget last entry
if current:
    domain_str = ' '.join(domain_lines)
    tax_match = re.search(r'Taxonomy-([^|]+)', domain_str)
    tax_value = tax_match.group(1).strip() if tax_match else '-'
    entries.append((current[0], current[1], tax_value, domain_str[:200]))

print(f"Parsed {len(entries)} model entries from Table 11")

# ========================================
# L1 Category Mapping (expanded)
# ========================================
l1_mapping = {
    # === Mammals ===
    'Homo sapiens': 'Mammals',
    'Mus musculus': 'Mammals',
    'Mammalia': 'Mammals',
    'Rattus norvegicus': 'Mammals',
    'Rattus rattus': 'Mammals',
    'Rattus': 'Mammals',
    'Murinae': 'Mammals',
    'Bos taurus': 'Mammals',
    'Oryctolagus': 'Mammals',
    'Oryctolagus cuniculus': 'Mammals',
    'Sus scrofa': 'Mammals',
    'Canis lupus': 'Mammals',
    'Pan troglodytes': 'Mammals',
    'Macaca': 'Mammals',
    'Macaca mulatta': 'Mammals',
    'Cricetulus': 'Mammals',
    'Cricetulus griseus': 'Mammals',
    'Cricetinae': 'Mammals',
    'Mesocricetus': 'Mammals',
    'Mesocricetus auratus': 'Mammals',
    'Cavia porcellus': 'Mammals',
    'Felis catus': 'Mammals',
    'Equus caballus': 'Mammals',
    'Ovis aries': 'Mammals',
    'Capra hircus': 'Mammals',
    'Neovison vison': 'Mammals',
    'Octodon degus': 'Mammals',
    'Amniota': 'Mammals',

    # === Vertebrates (non-mammal) ===
    'Xenopus laevis': 'Vertebrates',
    'Xenopus': 'Vertebrates',
    'Amphibia': 'Vertebrates',
    'Chordata': 'Vertebrates',
    'Aves': 'Vertebrates',
    'Gallus gallus': 'Vertebrates',
    'Danio rerio': 'Vertebrates',
    'Loligo': 'Vertebrates',
    'Loligo forbesii': 'Vertebrates',
    'Vertebrata': 'Vertebrates',

    # === Plants & Algae ===
    'Arabidopsis thaliana': 'Plants & Algae',
    'Arabidopsis': 'Plants & Algae',
    'Viridiplantae': 'Plants & Algae',
    'Embryophyta': 'Plants & Algae',
    'Nicotiana tabacum': 'Plants & Algae',
    'Saccharum officinarum': 'Plants & Algae',
    'Armoracia rusticana': 'Plants & Algae',
    'Chlamydomonas': 'Plants & Algae',
    'Chlamydomonas reinhardtii': 'Plants & Algae',
    'Solanum': 'Plants & Algae',
    'Solanum lycopersicum': 'Plants & Algae',
    'Solanum tuberosum': 'Plants & Algae',
    'Zea mays': 'Plants & Algae',
    'Oryza sativa': 'Plants & Algae',
    'Glycine max': 'Plants & Algae',
    'Marchantia': 'Plants & Algae',
    'Marchantia polymorpha': 'Plants & Algae',
    'Physcomitrium': 'Plants & Algae',
    'Physcomitrium patens': 'Plants & Algae',
    'Selaginella': 'Plants & Algae',
    'Picea': 'Plants & Algae',
    'Brassica': 'Plants & Algae',
    'Spinacia oleracea': 'Plants & Algae',
    'Hordeum vulgare': 'Plants & Algae',
    'Triticum aestivum': 'Plants & Algae',
    'Chlorella': 'Plants & Algae',
    'Volvox': 'Plants & Algae',
    'Ostreococcus tauri': 'Plants & Algae',
    'Ostreococcus': 'Plants & Algae',

    # === Fungi ===
    'Saccharomyces cerevisiae': 'Fungi',
    'Schizosaccharomyces pombe': 'Fungi',
    'Schizosaccharomycetaceae': 'Fungi',
    'Neurospora crassa': 'Fungi',
    'Aspergillus': 'Fungi',
    'Aspergillus niger': 'Fungi',
    'Aspergillus nidulans': 'Fungi',
    'Candida albicans': 'Fungi',
    'Fungi': 'Fungi',

    # === Bacteria ===
    'Escherichia coli': 'Bacteria',
    'Escherichia': 'Bacteria',
    'Bacillus subtilis': 'Bacteria',
    'Mycobacterium tuberculosis': 'Bacteria',
    'Lactococcus lactis': 'Bacteria',
    'Bacteria': 'Bacteria',
    'Pseudomonas': 'Bacteria',
    'Pseudomonas aeruginosa': 'Bacteria',
    'Streptomyces': 'Bacteria',
    'Synechocystis': 'Bacteria',
    'Synechococcus': 'Bacteria',
    'Salmonella': 'Bacteria',
    'Salmonella typhimurium': 'Bacteria',
    'Vibrio': 'Bacteria',
    'Helicobacter pylori': 'Bacteria',
    'Caulobacter': 'Bacteria',
    'Caulobacter crescentus': 'Bacteria',
    'Corynebacterium': 'Bacteria',
    'Staphylococcus aureus': 'Bacteria',
    'Streptococcus': 'Bacteria',
    'Clostridium': 'Bacteria',
    'Thermus': 'Bacteria',
    'Rhodobacter': 'Bacteria',
    'Sinorhizobium': 'Bacteria',
    'Agrobacterium': 'Bacteria',
    'Buchnera': 'Bacteria',
    'Mycoplasma': 'Bacteria',
    'Chlamydia': 'Bacteria',
    'Bordetella pertussis': 'Bacteria',
    'Bordetella': 'Bacteria',

    # === Invertebrates ===
    'Drosophila melanogaster': 'Invertebrates',
    'Drosophila': 'Invertebrates',
    'Caenorhabditis elegans': 'Invertebrates',
    'Physarum polycephalum': 'Invertebrates',
    'Anopheles': 'Invertebrates',
    'Aedes': 'Invertebrates',
    'Bombyx mori': 'Invertebrates',
    'Apis mellifera': 'Invertebrates',
    'Schistosoma': 'Invertebrates',
    'Strongylocentrotus': 'Invertebrates',
    'Aplysia': 'Invertebrates',

    # === Protists ===
    'Trypanosoma brucei': 'Protists',
    'Leishmania': 'Protists',
    'Dictyostelium discoideum': 'Protists',
    'Plasmodium': 'Protists',
    'Trypanosoma': 'Protists',
    'Toxoplasma gondii': 'Protists',
    'Giardia': 'Protists',

    # === Viruses ===
    'HIV': 'Viruses',
    'HIV-1': 'Viruses',
    'Influenza A': 'Viruses',
    'SARS-CoV-2': 'Viruses',
    'Hepatitis C': 'Viruses',
    'Hepatitis B': 'Viruses',
    'Bacteriophage': 'Viruses',
    'Virus': 'Viruses',

    # === General Eukaryotes ===
    'cellular organisms': 'General Eukaryotes',
    'Eukaryota': 'General Eukaryotes',
    'Opisthokonta': 'General Eukaryotes',
    'Metazoa': 'General Eukaryotes',
}

# Classify
from collections import defaultdict

domain_stats = defaultdict(list)
unclassified = []

for model_id, ref, tax_value, domain_str in entries:
    l1 = None

    if tax_value == '-' or tax_value == 'Unknown' or tax_value == '':
        # Check if domain string contains useful info
        # Look for known organism names
        for key, value in sorted(l1_mapping.items(), key=lambda x: -len(x[0])):
            if key.lower() in domain_str.lower():
                l1 = value
                tax_value = key  # update taxonomy
                break

        if l1 is None:
            # Check if it's a biological process
            bp_regex = r'immune|signal|metabolic|apoptosis|cell cycle|pathway|process|response|regulation|biosynthesis|muscle|protein|drug|transcription|translation|degradation|transport|phosphorylation|oxidation|binding|secretion|homeostasis|development|differentiation|proliferation|growth|death|repair|replication|contraction|folding|receptor|expression|activation|inhibition|gene|DNA|RNA|synthesis|catabolic|anabolic|cycle|kinase|phosphatase|calcium|ion|channel|tumor|cancer|infection|inflammation'
            if re.search(bp_regex, domain_str.lower()):
                l1 = 'Biological Processes'
            else:
                l1 = 'Unclassified'
    else:
        # Try exact match first
        if tax_value in l1_mapping:
            l1 = l1_mapping[tax_value]
        else:
            # Try prefix/substring match (longest first)
            for key, value in sorted(l1_mapping.items(), key=lambda x: -len(x[0])):
                if tax_value.lower().startswith(key.lower()) or key.lower().startswith(tax_value.lower()):
                    l1 = value
                    break

            if l1 is None:
                # Search in full domain string
                for key, value in sorted(l1_mapping.items(), key=lambda x: -len(x[0])):
                    if key.lower() in domain_str.lower():
                        l1 = value
                        break

            if l1 is None:
                # Check for bio processes
                bp_regex = r'immune|signal|metabolic|apoptosis|cell cycle|pathway|process|response|regulation|biosynthesis|muscle|protein|drug|transcription|translation|degradation|transport|phosphorylation|oxidation|binding|secretion|homeostasis|development|differentiation|proliferation|growth|death|repair|replication|contraction|folding|receptor|expression|activation|inhibition|gene|DNA|RNA|synthesis|catabolic|anabolic|cycle|kinase|phosphatase|calcium|ion|channel|tumor|cancer|infection|inflammation'
                if re.search(bp_regex, domain_str.lower()):
                    l1 = 'Biological Processes'
                else:
                    l1 = 'Unclassified'

    # Determine domain field value:
    # - For taxonomy models: use the organism name (tax_value)
    # - For Biological Processes: use the pathway description (domain_str, cleaned)
    # - For Unclassified: use "-" or the raw domain_str if available
    if l1 == 'Biological Processes':
        # Clean up domain_str: remove extra whitespace, truncate
        bp_desc = ' '.join(domain_str.split())[:200]
        if not bp_desc or bp_desc == '-':
            bp_desc = 'Unknown process'
        stored_domain = bp_desc
    elif l1 == 'Unclassified':
        stored_domain = tax_value if tax_value != '-' else (' '.join(domain_str.split())[:200] or '-')
    else:
        stored_domain = tax_value

    if l1 == 'Unclassified':
        unclassified.append((model_id, ref, tax_value, domain_str[:120]))

    domain_stats[l1].append({
        'id': model_id,
        'name': ref,
        'domain': stored_domain
    })

print(f"\n=== Domain Classification ({len(entries)} models) ===")
for domain, models in sorted(domain_stats.items(), key=lambda x: -len(x[1])):
    print(f"  {domain}: {len(models)} models")
print(f"  TOTAL: {sum(len(v) for v in domain_stats.values())}")

if unclassified:
    print(f"\n=== Unclassified ({len(unclassified)}) ===")
    for uid, uref, utax, udom in unclassified[:25]:
        try:
            print(f"  {uid} {uref} | Tax: {utax} | {udom[:80]}")
        except Exception:
            pass  # skip unicode errors in console

# ========================================
# Generate bio-domains.js (taxonomic categories only)
# ========================================
taxonomic_domains = {k: v for k, v in domain_stats.items()
                     if k not in ('Biological Processes', 'Unclassified')}
# Include Unclassified in taxonomy
if 'Unclassified' in domain_stats:
    taxonomic_domains['Unclassified'] = domain_stats['Unclassified']

output_path = '../web/data/bio-domains.js'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('// Auto-generated from Table 11 of RegimeFlow paper\n')
    f.write(f'// {len(entries)} models classified into {len(taxonomic_domains)} taxonomic domains\n')
    f.write('// NOTE: Biological Processes models are in bio-processes.js\n')
    f.write('// L1 category -> array of {id, name, domain}\n')
    f.write('var BIO_DOMAIN_DATA = {\n')
    for domain, models in sorted(taxonomic_domains.items(), key=lambda x: -len(x[1])):
        f.write(f'  "{domain}": [\n')
        for m in models:
            f.write(f'    {{id:"{m["id"]}",name:"{m["name"]}",domain:"{m["domain"]}"}},\n')
        f.write('  ],\n')
    f.write('};\n')

print(f"\nDone: {output_path}")
print(f"Taxonomic categories: {len(taxonomic_domains)}, Models: {sum(len(v) for v in taxonomic_domains.values())}")

# ========================================
# Generate bio-processes.js (Biological Processes only)
# ========================================
bp_models = domain_stats.get('Biological Processes', [])
if bp_models:
    # Sub-categorize by keyword matching on domain description
    bp_subcats = {
        'Immune Response':      r'immune|tumor|cancer|infection|inflammation|T cell|B cell|antibody|antigen|NK cell|macrophage|cytokine|chemokine',
        'Signal Transduction':  r'signal|receptor|kinase|phosphatase|MAPK|JAK|STAT|Wnt|Notch|Hedgehog|NF.kB|TGF|GPCR|cAMP|IP3|DAG|calcium|channel|transduction',
        'Metabolic Pathways':   r'metabolic|glycolysis|TCA|citrate|pentose|phosphate|lipid|fatty acid|cholesterol|amino acid|nucleotide|ATP|NADH|FADH|biosynthesis|catabolic|anabolic|glutathione|methionine|serine|threonine',
        'Cell Cycle & Division':r'cell cycle|mitosis|meiosis|cyclin|CDK|checkpoint|replication|division|proliferation|growth factor',
        'Gene Regulation':      r'transcription|translation|gene|DNA|RNA|mRNA|tRNA|promoter|enhancer|silencer|expression|activation|inhibition|repressor|inducer|operator|operon',
        'Protein & Folding':    r'protein|folding|unfolded|chaperone|degradation|proteasome|ubiquitin|phosphorylation|binding|secretion|endoplasmic|Golgi|ER |ribosome',
        'Apoptosis & Stress':   r'apoptosis|death|stress|oxidative|ROS|DNA damage|repair|p53|caspase|Bcl|Bax|survival',
        'Development':          r'development|differentiation|morphogen|pattern|embryo|organogenesis|homeostasis|regeneration',
        'Muscle & Contraction': r'muscle|contraction|actin|myosin|sarcomere|calcium|troponin',
        'Drug & Transport':     r'drug|transport|membrane|channel|pump|carrier|transporter|excretion|absorption|Gemcitabine|pharmacokinetic',
    }

    # Classify each BP model
    bp_categorized = {cat: [] for cat in bp_subcats}
    bp_categorized['Other Processes'] = []
    bp_categorized['models'] = []  # flat list for reference

    for m in bp_models:
        desc = m['domain'].lower()
        matched = False
        for cat, regex in bp_subcats.items():
            if re.search(regex, desc):
                bp_categorized[cat].append(m)
                matched = True
                break
        if not matched:
            bp_categorized['Other Processes'].append(m)
        bp_categorized['models'].append(m)

    with open('../web/data/bio-processes.js', 'w', encoding='utf-8') as f:
        f.write('// Auto-generated from Table 11 of RegimeFlow paper\n')
        f.write('// Biological Processes — models without taxonomic annotation\n')
        f.write(f'// {len(bp_models)} models, sub-categorized by pathway keywords\n')
        f.write('var BIO_PROCESSES_DATA = {\n')
        for cat in list(bp_subcats.keys()) + ['Other Processes']:
            models = bp_categorized[cat]
            if models:
                f.write(f'  "{cat}": [\n')
                for m in models:
                    f.write(f'    {{id:"{m["id"]}",name:"{m["name"]}",domain:"{m["domain"]}"}},\n')
                f.write('  ],\n')
        f.write('};\n')

    print(f"Biological Processes: {len(bp_models)} models → {sum(1 for k in bp_subcats if bp_categorized[k]) + (1 if bp_categorized['Other Processes'] else 0)} sub-categories")
    for cat in list(bp_subcats.keys()) + ['Other Processes']:
        n = len(bp_categorized.get(cat, []))
        if n > 0:
            print(f"  {cat}: {n}")

print(f"\nAll done!")
