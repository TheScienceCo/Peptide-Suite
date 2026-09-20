# Peptide-Suite
Peptide optimizer based on quantum biochemical principles and robust relevant research retrieval

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
