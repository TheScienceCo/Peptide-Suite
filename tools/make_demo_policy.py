#!/usr/bin/env python3
"""
Generate the demonstration policy pack.  [Addendum 1 step 3]

A demonstration pack exists so the system can be run end to end by someone who
does not hold the production policy. It is not a starting point to tune and it
is not a worse version of the real thing.

That forces a constraint on how its numbers are chosen. Inventing plausible
values would produce an artifact indistinguishable at a glance from a fitted
one, and the first person to copy it into production would be shipping numbers
nobody measured. So the pack contains exactly two kinds of value:

  derived              a value with a stated, checkable source, which the
                       production pack would very likely carry unchanged
  placeholder_midpoint the arithmetic midpoint of the key's declared valid
                       range, chosen by nothing at all

Weights are uniform, for the same reason: no weight here was fitted, and equal
weights say so more clearly than any spread of numbers could. A test asserts
both properties, so the demo pack cannot be quietly tuned into looking real.

Regenerate with:
    python tools/make_demo_policy.py
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from peptide_suite.policy.loader import compute_digest, validate_document, SCHEMA_VERSION
from peptide_suite.policy.opaque import mint_term_id
from peptide_suite.policy.registry import FEATURE_FAMILIES, THRESHOLDS

OUTPUT = REPO / "policy" / "demo.v1.json"
POLICY_VERSION = "1.0.0"

# The salt used for the demonstration pack's aggregate-term ids is published
# here on purpose. Its terms are placeholders with no chemical meaning, so there
# is nothing for the salt to protect; publishing it keeps the pack reproducible.
# A production pack never uses this salt.
DEMO_SALT = b"peptide-suite-demonstration-pack-salt-not-private"

DEMO_TERM_LABELS = [
    "demonstration aggregate term one",
    "demonstration aggregate term two",
]

# Thresholds with a source. Each entry is (value, where it comes from). Nothing
# goes in this table without an answer to "who says so"; everything else falls
# through to the midpoint rule below.
DERIVED = {
    # Physiological reference points. Conventions, not fitted quantities.
    "chemistry.reference_ph":      (7.4,   "physiological blood pH, standard reference condition"),
    "chemistry.interstitial_ph":   (6.5,   "Addendum 2: declared multi-pH protonation set"),
    "chemistry.endosomal_ph":      (5.5,   "Addendum 2: declared multi-pH protonation set"),

    # AlphaFold's own published confidence banding, not a tuned cutoff.
    "structure_gate.min_plddt":    (70.0,  "AlphaFold pLDDT banding: below 70 is not a confident region"),

    # Structural facts about the computation rather than choices about it.
    "conservation.min_distinct_sequences": (
        3.0, "below three distinct sequences positional entropy is zero by construction, "
             "so every position would read as conserved"),
    "identification.min_containment_length": (
        6.0, "shortest substring for which a database containment match is not routinely "
             "satisfied by chance across the reference set"),
    "identification.peptide_length_ceiling": (
        100.0, "boundary this system draws between peptide and protein handling"),

    # Specified directly by the addenda.
    "parameter_budget.measured_ratio_divisor": (
        10.0, "Addendum 2: free parameters may not exceed N_measured / 10"),
}


def midpoint(lo: float, hi: float) -> float:
    return round((lo + hi) / 2.0, 6)


def build() -> dict:
    uniform = round(1.0 / len(FEATURE_FAMILIES), 6)

    thresholds = {}
    for tid, spec in THRESHOLDS.items():
        if tid in DERIVED:
            value, _source = DERIVED[tid]
            basis = "derived"
        else:
            value = midpoint(*spec.valid_range)
            basis = "placeholder_midpoint"
        thresholds[tid] = {
            "value": value,
            "basis": basis,
            "semantic_type": spec.semantic_type,
            "units": spec.units,
            "valid_range": list(spec.valid_range),
        }

    document = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "pack_kind": "demonstration",
        "provenance": {
            "created_utc": "2026-01-01T00:00:00Z",
            "created_by": "tools/make_demo_policy.py",
            "derivation": (
                "DEMONSTRATION PACK. No value here was fitted to data. Weights are "
                "uniform. Thresholds are either derived from a stated source or the "
                "arithmetic midpoint of their declared range, marked per key by the "
                "'basis' field. Output produced under this pack shows the shape of a "
                "result and carries no claim about its magnitude."
            ),
            "n_measured": 0,
            "target_class": "none",
        },
        "integrity": {"algorithm": "sha256", "digest": "0" * 64},
        "feature_families": {
            fid: {
                "semantic_type": spec.semantic_type,
                "units": spec.units,
                "valid_range": list(spec.valid_range),
            }
            for fid, spec in FEATURE_FAMILIES.items()
        },
        "weights": {fid: uniform for fid in FEATURE_FAMILIES},
        "thresholds": thresholds,
        "aggregate_terms": {
            mint_term_id(label, DEMO_SALT): {
                "source_tiers": ["CLASSICAL_SIM"],
                "reduction": "mean",
                "max_inputs": 8,
            }
            for label in DEMO_TERM_LABELS
        },
    }
    document["integrity"]["digest"] = compute_digest(document)
    return document


def main() -> int:
    document = build()

    problems = validate_document(document)
    if problems:
        print("Generated pack does not validate:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    OUTPUT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")

    derived = sum(1 for t in document["thresholds"].values() if t["basis"] == "derived")
    placeholder = len(document["thresholds"]) - derived
    print(f"Wrote {OUTPUT.relative_to(REPO)}")
    print(f"  {len(document['weights'])} weights, uniform at {document['weights'][next(iter(document['weights']))]}")
    print(f"  {len(document['thresholds'])} thresholds: {derived} derived, {placeholder} placeholder")
    print(f"  digest {document['integrity']['digest'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
