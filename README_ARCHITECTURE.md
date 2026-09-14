# Peptide Substitution Scan & Optimization Pipeline

## Overview

Full-stack peptide optimization pipeline with **rigorous confidence scoring** and **evidence-based predictions**. Designed for systematic single-position substitution scanning to identify gain-of-function improvements.

### Core Philosophy

- **Never fabricate scores**: If a quantitative prediction cannot be computed from real data, state the limitation plainly.
- **Evidence tiers over black boxes**: All predictions backed by explicit reasoning tied to a Bayesian-weighted evidence tier.
- **Audit trail**: Every prediction logged for later calibration against known variants on test panel (IGF-1, insulin, GLP-1, BPC-157).

---

## Architecture

### Module Structure

```
peptide_suite/
├── core/                 # Shared infrastructure
│   ├── __init__.py      # Data structures (Effect, SubstitutionRecommendation, PeptideContext)
│   ├── confidence_scoring.py    # Evidence-tier framework, Brier calibration
│   ├── charge_calculator.py     # Henderson-Hasselbalch (Eq #11)
│   ├── conservation.py          # Shannon entropy per position (Eq #43)
│   ├── substitution_predictor.py  # Tier 1 effect prediction (sequence-only)
│   ├── peptide_manager.py       # Sequence loading, validation, basic properties
│   ├── evidence_retrieval.py    # NCBI/PubMed/UniProt APIs (optional)
│   └── effect_cache.py          # (Future) Memoization of known substitutions
│
├── workflows/
│   ├── optimize.py      # Workflow 1: "Optimize This Peptide" (v1 complete)
│   └── find_peptides.py # Workflow 2: "Find Peptides" (v2 roadmap)
│
├── data/
│   ├── test_panel.json  # Known peptides (IGF-1, insulin, GLP-1, BPC-157)
│   ├── ncaa_reference.json     # Non-canonical AA properties (v1.1+)
│   └── protease_motifs.json    # Known cleavage sites
│
├── tests/
│   ├── test_core_modules.py
│   └── test_optimize_workflow.py
│
├── logs/
│   └── prediction_calibration.jsonl  # Audit trail for Brier-score tracking
│
└── main.py              # CLI entry points
```

---

## Key Modules

### 1. **confidence_scoring.py** — Evidence-Tier Weighting

Implements Bayesian-style confidence assignment:

```
Evidence Tier Weights:
  Direct Experimental:    1.0  (e.g., published mutagenesis data)
  Homolog Experimental:   0.8  (e.g., ortholog with known variant)
  Biochemical Principle:  0.6  (e.g., "Pro removal increases flexibility")
  Inference Only:         0.3  (e.g., pure computational inference)
```

Every effect includes:
- `score`: 0–1 confidence
- `evidence_tier`: Source category
- `reasoning`: Explicit justification
- `equation_refs`: Which physics/biochemistry informed it

**Calibration Logging**: Every prediction is logged to `prediction_calibration.jsonl` for later Brier-score validation.

### 2. **charge_calculator.py** — Henderson-Hasselbalch (Equation #11)

Computes effective charge at any pH:

```
α (ionization fraction) = 1 / (1 + 10^(pKa - pH))
```

- Uses literature pKa values (Stryer, Nelson & Cox)
- Adjusts for burial (rough heuristic)
- Computes per-position charges and net peptide charge
- Default: pH 7.4 (physiological), configurable per analysis

### 3. **conservation.py** — Shannon Entropy (Equation #43)

Per-position conservation from homolog MSA:

```
H(position) = -Σ f_i * log₂(f_i)
```

Ranges:
- 0.0–0.5:   Highly conserved (red flag for substitution)
- 0.5–2.0:   Moderately conserved (orange flag)
- 2.0+:      Variable (green, tolerant to substitution)

Converts to tolerance score: `tolerance = 1 / (1 + exp(-1.5 * (entropy - 1.5)))`

### 4. **substitution_predictor.py** — Tier 1 Effect Prediction

**Scope**: Sequence-only, no structure prediction.

For each substitution, predicts:

#### Primary Effect (goal-dependent)
- **Protease Resistance**: Checks for disruption of known DPP4/neprilysin/trypsin motifs
- **Binding Affinity**: Electrostatic (Coulomb heuristic) + hydrophobicity changes
- **Generic**: Reports likely properties without assuming a goal

#### Off-Target Effects (always assessed)
1. **Residue Property Loss**: Loss of Pro's flexibility, Cys's disulfide, aromatic π-π stacking
2. **Conservation Penalty**: Red flag if mutating a highly conserved position
3. **Backbone Impact**: Rigidity changes (e.g., Pro introduction/removal)
4. **Charge Redistribution**: Local electrostatic environment shifts

### 5. **peptide_manager.py** — Sequence Handling

- Validates amino acid sequences (canonical AA check)
- Loads FASTA or raw sequences
- Computes basic properties (composition, charge, aromaticity, Pro/Cys count)
- Finds motifs (simple wildcard matching, e.g., "RXR")
- Six-frame DNA translation (scaffolding for gene→protein lookup)

### 6. **evidence_retrieval.py** — Literature & Homolog Lookups

Graceful fallback architecture:

- **NCBI Entrez**: Retrieves homologs (caches results)
- **UniProt**: Annotations and tissue specificity hints
- **PubMed**: Literature context (mocked in v1)
- **HPA/GTEx**: Expression profiles (scaffolding for Workflow 2)
- **Gene Ontology**: Function→cell-type mapping

All queries gracefully handle network failures and return empty results + status message rather than crashing.

---

## Workflow 1: "Optimize This Peptide" (v1 Complete)

### Input
```
peptide_sequence = "MGFPGLQPRRVSCGQAK..."
confirmed_goal = "binding_affinity"  # or None; will infer from name
```

### Execution Flow

**Step 1: Parse & Infer Function**
- Load sequence, validate
- Infer likely function from gene name (e.g., "IGF1" → IGF-1 receptor binding)
- Display to user, wait for confirmation (or auto-confirm for testing)

**Step 2: Retrieve Homologs & Conservation**
- Query NCBI for RefSeq homologs (or use test panel)
- Build MSA (simple padding in v1; upgrade to Clustal/MAFFT in production)
- Compute Shannon entropy per position

**Step 3: Run Substitution Scan**
- For each position 0...len(seq):
  - For each candidate AA (19 other canonical):
    - Predict primary effect
    - Predict off-target effects (always 4 types)
    - Combine scores into net benefit
    - Log prediction for calibration

**Step 4: Rank & Filter**
- Sort by net score (descending)
- Return top 3–5 recommendations
- Each includes primary effect, off-targets, evidence tiers, reasoning

### Output Example

```
=============================================================
Position 12: L → I
=============================================================
Category: binding_affinity
Net Recommendation: RECOMMEND
Net Score: +0.35 | Confidence: Medium

📌 PRIMARY EFFECT:
   Charge change: -0.00e; hydrophobicity shift: -0.3
   Evidence: BIOCHEMICAL_PRINCIPLE, Confidence: Medium
   Score: 0.60
   Reasoning: Charge-based prediction...

⚠️  OFF-TARGET EFFECTS:
   (1) Loss of L biochemical properties
       Evidence: BIOCHEMICAL_PRINCIPLE, Score: 0.40
       Reasoning: L→I is a conservative substitution...
   ...
```

---

## Equations Reference

From attached "Peptide Interaction Equation Cheat Sheet". Used selectively in v1:

### Tier 1 (v1)
- **#43**: Shannon Entropy (conservation)
- **#11**: Henderson-Hasselbalch (charge state)
- **#1**: Coulomb-style qualitative reasoning (heuristic, not computed energy)
- **#22, #23**: Michaelis-Menten framing (conceptual for protease resistance, not actual kcat)

### Tier 2 (v1.1+, gated behind structure prediction)
- **#3**: Lennard-Jones clash checking
- **#10**: SASA / hydrophobic burial
- **#38**: Generalized Born rescoring
- **#32**: MM/PBSA triage

### Not Implemented (Out of Scope)
- **#30, #31**: FEP/TI (free energy perturbation / thermodynamic integration)
- **#36**: QM/MM (quantum/classical hybrid)
- Explicit solvent MD (too expensive)

---

## Calibration & Validation

### Prediction Logging

Every substitution prediction is logged:

```json
{
  "timestamp": "2025-09-14T...",
  "peptide": "IGF1",
  "position": 5,
  "wt_aa": "G",
  "mutant_aa": "A",
  "predicted_effect": "...",
  "predicted_score": 0.65,
  "predicted_confidence": "Medium",
  "actual_outcome": null,  // Filled in later by user/experiment
  "actual_score": null
}
```

### Brier Score Computation

Once actual experimental data arrives (e.g., user uploads known variants):

```
Brier = mean((predicted_confidence - actual_outcome)²)
```

Tracks calibration over time; informs confidence-threshold tuning.

### Test Panel

Known peptides with characterized variants:

- **IGF-1**: Growth signaling, known binding-site mutations
- **Insulin**: Glucose homeostasis, engineered variants (e.g., faster-acting)
- **GLP-1 agonists**: Incretin effect, protease-resistant variants (Semaglutide, etc.)
- **BPC-157**: Wound healing, tissue repair (empirically robust to substitutions)

---

## Roadmap

### v1.0 (Complete)
- ✅ Confidence scoring framework
- ✅ Henderson-Hasselbalch charge calculations
- ✅ Shannon entropy conservation analysis
- ✅ Tier 1 substitution prediction (sequence-only)
- ✅ Workflow 1: Optimize This Peptide (canonical AAs)
- ✅ Audit logging for calibration

### v1.1
- 📋 Non-canonical AA reference table + weighting
- 📋 Protease resistance tuning (DPP4, neprilysin, etc.)
- 📋 Brier-score calibration reporting

### v2.0
- 📋 Structure prediction integration (ESMFold/ColabFold)
- 📋 Lennard-Jones clash checking
- 📋 Generalized Born rescoring
- 📋 Workflow 2: Find Peptides (expression-based discovery)

---

## Running the Pipeline

### Installation

```bash
pip install -r requirements.txt
```

### CLI Usage

#### Workflow 1: Optimize a Peptide

```bash
# Auto-infer function from known peptide name
python -m peptide_suite.main optimize IGF1 --auto-confirm

# Raw sequence with explicit goal
python -m peptide_suite.main optimize "MGFPGLQPRRVSCGQAK" \
    --goal "binding_affinity"

# Custom pH
python -m peptide_suite.main optimize IGF1 --auto-confirm --ph 6.5
```

#### Calibration Tests

```bash
# Run test panel (IGF-1, insulin, GLP-1, BPC-157)
python -m peptide_suite.main test
```

#### Workflow 2 (Placeholder)

```bash
python -m peptide_suite.main find "myelinating_peptides"
# Not yet implemented
```

### Unit Tests

```bash
python -m pytest peptide_suite/tests/test_core_modules.py -v
```

---

## Hard Rules

(From spec, enforced in code)

1. **No fabricated scores**: If a quantitative value can't be computed, state the limitation instead.
2. **Evidence tiers mandatory**: Every claim includes evidence_tier and reasoning.
3. **Off-targets always assessed**: Never present a primary benefit without evaluating secondary effects.
4. **Confidence logged**: Predictions stored in `prediction_calibration.jsonl` for later calibration.
5. **No fake kinetics**: Protease resistance framed as "predicted cleavage liability" based on known motifs, never as fabricated kcat/KM.

---

## Contact & Future Work

This pipeline is designed as an **auditable, interview-ready system** for peptide engineering. Every prediction can be traced back to:
- Its evidence tier
- The literature or principle behind it
- The confidence score and calibration data

The goal is honest science: predict what we're confident about, flag limitations, and improve calibration over time.
