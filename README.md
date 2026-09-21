# Peptide-Suite

Peptide engineering analysis with the provenance of every number carried
alongside it.

The organising rule: **a score appears only if it was computed from real input
data.** Where a factor could not be computed — no structure, no assay, no
parameters — the output says so instead of estimating. Most of what follows is
machinery for keeping that true under pressure.

Three workflows:

- **Optimize** — systematic single-position substitution scan against a
  confirmed goal, with each candidate's primary effect, off-target effects and
  confidence reported separately.
- **Transform** — discrete, reviewable modifications scored against an
  eight-objective vector, never collapsed into one number.
- **Find peptides** — functional keywords to capable cell types, checking known
  answers before ranking anything.

And four gates that refuse rather than guess: a structure-template hierarchy
whose last tier is refusal, a parameterized-residue registry that is empty (so
non-canonical chemistry becomes a research request, not a recommendation), a
native-contact classifier that freezes essential footprints, and a class B1
placement rule that will not let an affinity gain stand alone as an improvement.

## Running

The engine holds no coefficients. Every weight, threshold and cutoff comes from
a policy artifact loaded at startup, and the system refuses to start without
one rather than falling back to built-in values. A demonstration pack is
committed so this repository runs out of the box.

```bash
pip install -r requirements.txt

export PEPTIDE_SUITE_POLICY=policy/demo.v1.json
python -m uvicorn peptide_suite.api:app --reload
```

Then open http://127.0.0.1:8000.

`policy/demo.v1.json` is a demonstration pack: no weight in it was fitted to
data. Every scored response says so, in the API payload and in the UI, so a
number produced under it is never mistaken for a measured one. See
`policy/README.md`.

## Why there is a QM arm

The honest justification, and the one that generalises: for non-canonical
chemistry, quantum mechanics is not an enhancement that sharpens an existing
number. It is the entry ticket to having a number at all.

Aib, Nle, Cha, N-methylated backbones, beta-amino acids, hydrocarbon staples,
fatty-acid linkers, gamma-Glu and OEG spacers — none of these exist in ff19SB,
CHARMM36m, or any standard library. A force field asked to simulate one either
refuses or silently substitutes something else, and the second failure mode is
the dangerous one: it produces a number, the number looks like every other
number, and nothing downstream indicates that the molecule simulated was not
the molecule proposed.

So the system maintains a parameterized-residue registry, and the proposal
engine draws only from it. Getting a residue into that registry means running
the whole pipeline:

    geometry optimization -> ESP -> RESP charges -> torsion scans
      -> fit to GAFF2/OpenFF -> validate against experimental conformational data

Every stage is required. The last one especially: parameters that reproduce the
QM are not the same as parameters that reproduce the molecule.

**The registry is currently empty**, and that is its correct state rather than an
omission. This project has parameterized nothing, so every proposal involving
non-canonical chemistry is emitted as a research request carrying the cost of
parameterizing it — never as a recommendation. On GLP-1(7-36), even with a bound
experimental structure supplied, three of five generated proposals land there:
the Aib substitution, the i,i+4 staple, and the C18 diacid with its gamma-Glu
and OEG linkers.

That cost is stated rather than invented. It says which pipeline stages remain
and what the expensive stage scales with — Aib has no side-chain dihedral to
scan but needs its backbone phi/psi surface mapped, a staple's cost scales with
its span, an OEG spacer's with the number of glycol units. It does not say hours
or dollars, because those depend on hardware, level of theory and how much of
the work is already automated, and a figure here would travel downstream as
though it had been estimated.

One residue class in the catalogue needs no new parameters: a D-enantiomer of a
canonical amino acid. Standard force-field functional forms are achiral, so the
bonded and non-bonded terms are the L values and the inversion is carried by the
sign of the C-alpha improper dihedral. What still has to be checked is that the
improper is actually set for the D configuration — a silently-L improper gives
the wrong enantiomer and reports no error.

This gate is independent of the structure-template gate. A proposal can have a
perfect bound experimental structure and still be unrankable because its
chemistry has no parameters, and the two refusals have different remedies: one
needs a structure, the other needs weeks of QM. Both are reported, because
suppressing one because the other fired would understate what the proposal
actually needs.

## Why sequence-aware evaluation matters

Peptide datasets are full of near-duplicates: alanine scans, single-point
variants, truncation series, the same peptide from two papers. A random split
puts a peptide in train and its point mutant in test, and the model gets credit
for recalling a sequence it has already seen. The score is real; what it
measures is memorisation.

`ml/experiments/split_gap.py` settles this rather than asserting it. Forty
families of related sequences, a label that is a genuine function of the
sequence (a motif present or absent), and two splits over the same data:

| Arm | Random split | Sequence-clustered split | Gap |
|---|---|---|---|
| Composition baseline | 0.929 AUC | 0.814 AUC | +0.114 |
| One-hot MLP | **1.000 AUC** | **0.205 AUC** | **+0.795** |

47 of 48 test sequences have a ≥80%-identity neighbour in training under the
random split. Zero do under the clustered one.

Two things to read off that table.

The MLP's 1.000 is entirely recall. Held-out families drop it to 0.205 — *below
chance*, meaning it learned family-specific patterns that actively mislead on
sequences it has not seen. The composition baseline cannot represent a specific
sequence at all, so it has less to memorise and loses far less.

And the deep model looks better than the baseline on the random split (1.000 vs
0.929) and is much worse on the honest one (0.205 vs 0.814). Reported the usual
way, this experiment would have concluded that the neural model wins.

The clustering is union-find single-linkage, and the guarantee is exhaustively
tested: no two sequences in different clusters are within the identity
threshold. Greedy first-match assignment does not provide that — a sequence
similar to members of two clusters joins only the first, and the pair it leaves
behind straddles the split. That leaked three sequences past a clustered split
before it was fixed.

This experiment is marked `SYNTHETIC_METHOD_ONLY` in the dataset registry. It
demonstrates a fact about evaluation protocol and supports no biological claim.

## Why named-entity holdouts overstate the result

The obvious way to test whether the system can rediscover a known drug is to
remove every document mentioning it and see whether the modifications come back.
For semaglutide that protocol does not hold, and the reason is checkable from
the data rather than rhetorical.

Semaglutide carries four engineering motifs: Aib8, Arg34, a C18 diacid, and a
gamma-Glu linker. After excluding every document naming semaglutide:

| Motif | Still present in |
|---|---|
| Aib | taspoglutide, tirzepatide |
| lipidation | liraglutide, tirzepatide, insulin degludec |
| gamma-Glu linker | liraglutide, tirzepatide, insulin degludec |
| regioselectivity substitution | liraglutide |

Every motif survives. Liraglutide carries Arg34 and gamma-Glu-linked acylation
at Lys26; taspoglutide is [Aib8, Aib35]-GLP-1. The union of two other published
drugs is the complete answer, so what such an experiment demonstrates is correct
retrieval and recombination of established engineering motifs.

That is a real and useful capability and the system claims exactly that. It is
not evidence of de novo discovery, and `GET /api/holdout` will produce the table
above for any drug in the golden set.

Three protocols replace it: motif-level ablation (exclude by chemistry, not by
name), temporal holdout (freeze the corpus at a cutoff year), and decoy controls
(run on peptides where the motif is known not to help — a system that proposes
it anyway has a prior, not a prediction).

**Results are never reported as a binary hit.** The contract is the rank of the
true modification within the full proposal list, plus the count ranked above it,
plus the decoy false-positive rate:

> 'Aib8' ranked 2nd of 47 proposals, with 1 ranked above it under motif_ablation
> (ablated: aib). 'aib' was proposed in 2 of 18 decoy trials (false-positive rate
> 0.11).

"Predicted two of three modifications" is not supportable. The sentence above
is. `RankedOutcome` has no boolean hit field anywhere on it, because a system
whose output can be reduced to yes-or-no will be — and 2nd of 47 would get
written up alongside 2nd of 3.

## API

Typed request and response schemas; interactive docs at `/docs` when running.

| Endpoint | Purpose |
|---|---|
| `POST /api/infer-function` | Identify a sequence or name, and propose a goal for confirmation |
| `POST /api/optimize` | Run the substitution scan against a confirmed goal |
| `POST /api/transform` | Ranked transformations, objective vectors, gates and research requests |
| `POST /api/find-peptides` | Functional keyword search |
| `GET /api/policy` | Which policy produced the numbers, and how much of it is placeholder |
| `GET /api/parameterization` | The residue registry and the pipeline that would fill it |
| `GET /api/holdout` | Holdout protocols, and the named-entity leakage table |
| `GET /api/goals` | Goals with a real scoring path |
| `GET /api/calibration` | Prediction-vs-outcome log summary |
| `GET /api/health` | Liveness |

Every scored response carries its policy provenance, so a number cannot be read
without knowing what weighted it.

## Tests

```bash
python -m unittest discover -t . -s peptide_suite/tests
```

`-t .` matters. It lets the test package load the demonstration pack before any
test imports — without it the suite fails at the first coefficient read, which
is the boundary working as intended rather than a broken suite.

```bash
python tools/check_policy_boundary.py --scope all
```
