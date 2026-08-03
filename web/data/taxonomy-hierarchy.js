// ==========================================================================
// Domain Taxonomy Hierarchy for Circle Packing
// Source: RegimeFlow Paper Table 11 — 1,050 Systems Biology Models
// Each node: { name, count } where count = number of models in that category
// ==========================================================================
var TAXONOMY_HIERARCHY = {
  name: "SysBio-Traj (1,050)",
  children: [
    // ── MAMMALS ──
    {
      name: "Mammals",
      children: [
        { name: "Homo sapiens", count: 320 },
        { name: "Mus musculus", count: 80 },
        { name: "Mammalia (unspecified)", count: 62 },
        { name: "Rattus norvegicus", count: 20 },
        { name: "Murinae", count: 6 },
        { name: "Bos taurus", count: 4 },
        { name: "Other Mammals", count: 8 }
      ]
    },
    // ── VERTEBRATES (non-mammal) ──
    {
      name: "Vertebrates",
      children: [
        { name: "Xenopus laevis (Amphibian)", count: 27 },
        { name: "Chordata (unspecified)", count: 8 },
        { name: "Amphibia", count: 4 },
        { name: "Aves (Birds)", count: 2 },
        { name: "Other Vertebrates", count: 5 }
      ]
    },
    // ── PLANTS & ALGAE ──
    {
      name: "Plants & Algae",
      children: [
        { name: "Arabidopsis thaliana", count: 27 },
        { name: "Viridiplantae", count: 14 },
        { name: "Embryophyta", count: 2 },
        { name: "Other Plants", count: 6 }
      ]
    },
    // ── FUNGI ──
    {
      name: "Fungi",
      children: [
        { name: "Saccharomyces cerevisiae", count: 70 },
        { name: "Schizosaccharomyces pombe", count: 3 },
        { name: "Neurospora crassa", count: 3 },
        { name: "Other Fungi", count: 2 }
      ]
    },
    // ── BACTERIA ──
    {
      name: "Bacteria",
      children: [
        { name: "Escherichia coli", count: 24 },
        { name: "Bacillus subtilis", count: 4 },
        { name: "Mycobacterium tuberculosis", count: 3 },
        { name: "Lactococcus lactis", count: 3 },
        { name: "Other Bacteria", count: 6 }
      ]
    },
    // ── INVERTEBRATES ──
    {
      name: "Invertebrates",
      children: [
        { name: "Drosophila melanogaster", count: 11 },
        { name: "Drosophila spp.", count: 4 },
        { name: "Other Invertebrates", count: 8 }
      ]
    },
    // ── PROTISTS ──
    {
      name: "Protists",
      children: [
        { name: "Trypanosoma brucei", count: 10 },
        { name: "Leishmania spp.", count: 4 },
        { name: "Dictyostelium discoideum", count: 3 },
        { name: "Other Protists", count: 5 }
      ]
    },
    // ── VIRUSES ──
    {
      name: "Viruses",
      children: [
        { name: "HIV", count: 3 },
        { name: "Influenza A", count: 4 },
        { name: "SARS-CoV-2", count: 2 },
        { name: "Other Viruses", count: 3 }
      ]
    },
    // ── GENERAL EUKARYOTES ──
    {
      name: "General Eukaryotes",
      children: [
        { name: "cellular organisms", count: 57 },
        { name: "Eukaryota", count: 23 },
        { name: "Opisthokonta", count: 4 },
        { name: "Metazoa", count: 2 }
      ]
    },
    // ── BIOLOGICAL PROCESSES ──
    {
      name: "Biological Processes",
      children: [
        { name: "Immune Response", count: 12 },
        { name: "Signal Transduction", count: 8 },
        { name: "Metabolic Pathways", count: 5 },
        { name: "Cell Cycle", count: 3 },
        { name: "Apoptosis & Stress", count: 3 },
        { name: "Other Processes", count: 5 }
      ]
    },
    // ── UNCLASSIFIED ──
    {
      name: "Unclassified",
      children: [
        { name: "Unknown / \"-\"", count: 115 },
        { name: "Other / Mixed", count: 15 }
      ]
    }
  ]
};
