# Deliverables

The completion summary Addendum 3 asks for. Written to be read by someone
deciding whether to look at the code, and to be checkable against it.

## What this is

A peptide engineering analysis platform built around one rule: **a score
appears only if it was computed from real input data.** Where a factor could
not be computed — no structure, no assay, no parameters, no homologs — the
output says so rather than estimating. Most of the engineering is machinery for
keeping that true when a chart, an average, or an API response would rather
have a number.

Three analysis workflows (substitution scanning, transformations against an
eight-objective vector, functional peptide search), five gates that refuse
rather than guess, and an ML layer kept in its own provenance lane.

## Repository architecture

![Architecture](architecture.svg)

| Path | What lives there |
|---|---|
| `policy/` | The scoring artifact. Every weight, threshold and cutoff. The engine holds none. |
| `peptide_suite/core/` | Analysis: electrostatics, conservation, substitution prediction, the landscape grid, the five gates, the explanatory-depth contracts. |
| `peptide_suite/workflows/` | Optimize, transform, find-peptides. |
| `peptide_suite/api.py` | FastAPI transport. Computes nothing itself. |
| `peptide_suite/static/` | The browser interface. Renders; never scores. |
| `peptide_suite/tests/` | 621 tests, and none of them reaches a network or a third-party package. |
| `ml/` | PyTorch layer: encoders, splitting, metrics, training, model cards, tracking, the representation explorer, the multimodal comparison. |
| `ml/tests/` | 111 tests. |
| `tools/` | Policy pack generation, the engine/policy boundary check. |

About 23,000 lines of Python across the two packages.

## Datasets and licences

Eight tasks are declared in `ml/datasets/registry.py`. Seven are **unavailable
in this environment** and say so, with the source and licence each would need.
None is backed by invented labels.

| Task | Status | Source | Licence |
|---|---|---|---|
| `amp_activity` | unavailable | APD3 / DBAASP / DRAMP | academic use; redistribution differs per database |
| `toxicity` | unavailable | ToxinPred / DBAASP haemolysis annotations | academic use |
| `solubility` | unavailable | eSOL / PROSO II style *E. coli* sets | varies by source; check before redistribution |
| `cell_penetration` | unavailable | CPPsite 2.0 | academic use |
| `aggregation` | unavailable | AmyLoad / WALTZ-DB | academic use |
| `stability` | unavailable | Rocklin et al. mini-protein stability | published; check terms |
| `protease_stability` | unavailable | no standard set; would need assembling from primary literature | n/a |
| `split_methodology_demo` | **synthetic, method only** | generated in-process from a known rule | n/a |

The one available task is synthetic by design and marked
`SYNTHETIC_METHOD_ONLY`. It settles a question about evaluation protocol, which
is the kind of question a constructed dataset *can* settle, and supports no
biological claim.

## Models implemented

| Model | Where | Note |
|---|---|---|
| Composition baseline | `ml/experiments/split_gap.py` | The arm that most often embarrasses a deep model. |
| One-hot MLP | `ml/experiments/split_gap.py` | The memorisation exhibit. |
| Deterministic encoders (composition, positional one-hot) | `ml/embeddings/encoder.py` | Real encodings, no learned content, never reported as embeddings. |
| ESM-2 encoder | `ml/embeddings/encoder.py` | Raises. Weights unreachable here; no silent fallback. |
| Single-modality MLP, naive concatenation, gated fusion | `ml/multimodal/fusion.py` | Same width and budget across arms. |
| PCA projection | `ml/embeddings/explorer.py` | SVD with a fixed sign convention and reported explained variance. |

## Evaluation results

**Sequence-similarity-aware splitting** (`ml/experiments/split_gap.py`, run
live at `/api/ml/split-comparison`):

| Arm | Random split | Sequence-clustered split | Gap |
|---|---|---|---|
| Composition baseline | 0.929 AUC | 0.814 AUC | +0.114 |
| One-hot MLP | **1.000 AUC** | **0.205 AUC** | **+0.795** |

47 of 48 random-split test sequences have a ≥80%-identity neighbour in train.
Zero do under the clustered split. The MLP's 1.000 is entirely recall, and
reported the usual way this experiment would have concluded that the neural
model wins.

**Multimodal fusion** (`ml/experiments/fusion_benefit.py`, run live at
`/api/ml/fusion-benefit`): all four arms land inside each other's bootstrap
intervals and the fusion gate collapses onto the sequence modality. Verdict:
no claim that fusion helps. A permutation null over shuffled labels confirms
the test had the resolution to detect an effect of that size, so the tie is a
result rather than an absence of one.

**Representation explorer**: two components of a positional one-hot encoding of
a 31-mer's substitution scan carry about 4% of the variation. Reported at size,
above the plot, because that is what makes the plot readable.

## Screenshots

In `docs/screenshots/`, all taken from a live run against the demonstration
policy pack. The two worth looking at first are `landscape.png` and
`landscape-not-computed.png`: the same grid, filled and hatched, which is the
whole thesis in two images.

## Known limitations

- **The demonstration policy pack is not fitted.** No weight in it came from
  data, and 15 of 26 thresholds are placeholders. Every scored response says
  so. Rankings show the shape of a result; the magnitudes are not a claim.
- **No real labelled dataset is reachable here**, so no model in this
  repository is trained on real labels. Both ML experiments are synthetic and
  marked.
- **ESM-2 weights are unreachable**, so the four-arm transfer-learning
  comparison the addendum asks for cannot be run. The arms that exist are the
  ones that can be run honestly.
- **QM is installed; MD and NBO are not.** GFN2-xTB (pip) and Psi4 1.11 with
  CREST and the `resp` package (conda-forge) run here — an earlier version of
  this document said the tooling was unreachable, which was wrong. Conformer
  ensembles and SAPT decomposition remain contracts because nothing is wired
  into them yet, not because the engines are missing. NBO is genuinely blocked:
  it is commercially licensed.
- **The parameterized-residue registry holds no usable residue**, so every
  non-canonical proposal is still a research request. Aib now has three of six
  stages recorded with real HF/6-31G* artifacts behind them and is reported as
  `IN_PROGRESS`, which licenses nothing. A record and a result are deliberately
  different things here: writing one is an explicit act, never a side effect of
  a calculation returning without an exception.
- **Homolog retrieval is not wired up**, so conservation is not computed unless
  sequences are supplied. The landscape hatches that entire metric rather than
  filling it with zeros.
- **Engine/policy boundary debt: 59 coefficients** still live in engine code,
  tracked in `policy/BOUNDARY_DEBT.txt` and ratcheted — the check fails if any
  count rises.
- **The suite is hermetic by construction, not by accident.** It disables live
  UniProt lookups for itself. Before that, three tests took the reference-set
  path in a sandbox with no route to uniprot.org and the live path on a CI
  runner that has one, so the same commit passed in one place and failed in the
  other with nothing to indicate why.

## Remaining roadmap

Blocked on data, weights or tooling rather than on design:

1. Real labelled datasets for the seven declared tasks.
2. ESM-2 weights, then the four-arm comparison per task.
3. Aib's remaining three stages: torsion scans (tractable — a few hours of
   HF/6-31G* on this machine), the force-field fit, and validation against
   experimental conformational data.
4. Wire the installed CREST into `conformer_ensemble.generate` and Psi4's SAPT
   into `contact_decomposition.decompose`. NBO stays blocked on a licence.
5. Attribution, once a model trained on real labels exists.
6. The receptor-prediction validation: take the learned properties together
   with the physics pipeline's and predict the binding site, ignoring the
   literature, then check against known biology.

## Reproduction

```bash
export PEPTIDE_SUITE_POLICY=policy/demo.v1.json

# The analysis suite needs no third-party package at all — not FastAPI, not
# PyTorch, not requests. CI runs it with nothing installed, which is what keeps
# that true.
python -m unittest discover -t . -s peptide_suite/tests     # 621 tests

# The ML layer is optional; PyTorch and numpy are its only dependencies.
pip install --index-url https://download.pytorch.org/whl/cpu torch numpy
python -m unittest discover -t . -s ml/tests                # 111 tests

# The web interface needs the runtime dependencies.
pip install -r requirements.txt

# the engine/policy boundary, which only ratchets down
python tools/check_policy_boundary.py --scope all

# the two experiments, from the command line
python -c "from ml.experiments.split_gap import run; print(run().report())"
python -c "from ml.experiments.fusion_benefit import run; print(run().report())"

# the interface
python -m uvicorn peptide_suite.api:app --reload    # then http://127.0.0.1:8000
```

`-t .` on the test commands matters: it lets the test package load the
demonstration pack before any test imports. Without it the suite fails at the
first coefficient read, which is the boundary working rather than a broken
suite.

Environment this was built and verified in: Python 3.11, PyTorch 2.14 on CPU.
No GPU, and no network access to model or dataset hosts.

## What this repository demonstrates

**To an ML engineer.** That the evaluation is the product. Sequence-clustered
splitting with a tested no-cross-cluster guarantee, bootstrap intervals on
every metric, a verdict rule that refuses to call overlapping intervals a win,
a permutation null that decides whether a comparison had the resolution to
detect anything, model cards that refuse to omit their limitations, and run
records that report a dirty tree as unreproducible rather than as its nearest
commit. The headline exhibit is a model at 1.000 AUC that is at 0.205 once
near-duplicates are removed — and the repository leads with it rather than
burying it. Where a capability could not be built honestly, it raises: the
pretrained encoder does not fall back to amino-acid counts, and attribution is
absent rather than run against a synthetic model.

**To a computational biologist.** That the refusals are real. A structure
template hierarchy whose last tier is refusal, with free-state and precursor
structures rejected on identity rather than on score. A parameterized-residue
registry that is empty, so semaglutide-style chemistry becomes a research
request naming the QM work required. A native-contact classifier that freezes
an essential footprint. A class B1 rule that will not let an affinity gain
stand alone as an improvement. A synthetic-feasibility gate that blocks GLP-1's
Lys26/Lys34 regioselectivity conflict rather than costing it. Conservation
entropy that is not computed at all when there are too few homologs, because
entropy over one sequence is zero everywhere by construction and would report
the whole peptide as perfectly conserved. And an engine that cannot score
without a policy artifact, so a number can always be traced to what weighted
it.
