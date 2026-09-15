"""
Policy artifact loading and validation.  [Addendum 1 sections 1-2]

The engine holds no coefficients. Every weight, threshold and cutoff is supplied
by a versioned policy artifact loaded at runtime.

If the artifact is absent, malformed, or fails validation, the system fails at
startup. There is no fallback to built-in values, because a fallback would let a
misconfigured deployment run silently on defaults that are not the policy — the
outputs would look normal and be wrong, and nothing would say so.

Validation is strict in all three directions the specification requires:
unknown keys are rejected, out-of-range values are rejected, and missing
required fields are rejected.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .registry import FEATURE_FAMILIES, THRESHOLDS, KeySpec, free_parameter_count

SCHEMA_VERSION = "1.0.0"
KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")

VALID_SEMANTIC_TYPES = {
    "bounded_unit", "signed_unit", "probability", "energy_kcal_per_mol",
    "ph", "angstrom", "count", "fraction", "dimensionless",
}
VALID_PACK_KINDS = {"demonstration", "production"}
VALID_REDUCTIONS = {"sum", "mean", "max_abs", "min", "max", "count_above_threshold"}
VALID_AGGREGATE_TIERS = {"QM", "SEMIEMPIRICAL", "CLASSICAL_SIM", "MEASURED_STRUCTURE"}

TOP_LEVEL_REQUIRED = {
    "schema_version", "policy_version", "pack_kind", "provenance",
    "integrity", "feature_families", "weights", "thresholds", "aggregate_terms",
}


class PolicyError(Exception):
    """Base for every policy failure. Always fatal; never caught to fall back."""


class PolicyNotFound(PolicyError):
    pass


class PolicyValidationError(PolicyError):
    def __init__(self, problems: List[str]):
        self.problems = problems
        super().__init__(
            f"Policy artifact failed validation with {len(problems)} problem(s):\n  - "
            + "\n  - ".join(problems)
        )


class PolicyIntegrityError(PolicyError):
    pass


def compute_digest(document: Dict[str, Any]) -> str:
    """
    SHA-256 over the canonical document with the integrity block removed.

    Excluding the block is what makes the digest checkable: it cannot contain
    its own hash.
    """
    payload = {k: v for k, v in document.items() if k != "integrity"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Policy:
    """A validated policy artifact. The only source of coefficients in the system."""
    policy_version: str
    pack_kind: str
    provenance: Dict[str, Any]
    digest: str
    _weights: Dict[str, float]
    _thresholds: Dict[str, float]
    _families: Dict[str, Dict[str, Any]]
    _aggregate_terms: Dict[str, Dict[str, Any]]
    source_path: Optional[Path] = None

    @property
    def is_demonstration(self) -> bool:
        return self.pack_kind == "demonstration"

    def weight(self, family_id: str) -> float:
        if family_id not in self._weights:
            raise PolicyError(
                f"No weight for feature family '{family_id}' in policy "
                f"{self.policy_version}. The engine requires it; the pack does not supply it."
            )
        return self._weights[family_id]

    def threshold(self, threshold_id: str) -> float:
        if threshold_id not in self._thresholds:
            raise PolicyError(
                f"No threshold '{threshold_id}' in policy {self.policy_version}. "
                f"The engine requires it; the pack does not supply it."
            )
        return self._thresholds[threshold_id]

    def aggregate_term(self, term_id: str) -> Dict[str, Any]:
        if term_id not in self._aggregate_terms:
            raise PolicyError(
                f"'{term_id}' is not a declared aggregate term in policy {self.policy_version}."
            )
        return dict(self._aggregate_terms[term_id])

    @property
    def declared_aggregate_terms(self) -> List[str]:
        return sorted(self._aggregate_terms)

    @property
    def free_parameters(self) -> int:
        return len(self._weights) + len(self._thresholds)

    def banner(self) -> str:
        kind = "DEMONSTRATION PACK — not the production policy" if self.is_demonstration else "production policy"
        return (f"policy {self.policy_version} ({kind}), "
                f"{self.free_parameters} free parameters, digest {self.digest[:12]}")


def _check_key_block(block: Dict[str, Any], required: Dict[str, KeySpec],
                     block_name: str, problems: List[str], value_of) -> None:
    """Unknown keys, missing keys, and out-of-range values, in that order."""
    supplied = set(block)
    expected = set(required)

    for unknown in sorted(supplied - expected):
        problems.append(f"{block_name}: unknown key '{unknown}' (not declared by the engine)")

    for missing in sorted(expected - supplied):
        problems.append(f"{block_name}: missing required key '{missing}'")

    for key in sorted(supplied & expected):
        spec = required[key]
        try:
            value = value_of(block[key])
        except (TypeError, ValueError, KeyError):
            problems.append(f"{block_name}: '{key}' has no readable numeric value")
            continue
        lo, hi = spec.valid_range
        if not (lo <= value <= hi):
            problems.append(
                f"{block_name}: '{key}' = {value} is outside its declared range [{lo}, {hi}]"
            )


def validate_document(document: Dict[str, Any], *, verify_integrity: bool = True) -> List[str]:
    """Return every problem found. An empty list means the artifact is valid."""
    problems: List[str] = []

    if not isinstance(document, dict):
        return ["policy artifact is not a JSON object"]

    for unknown in sorted(set(document) - TOP_LEVEL_REQUIRED):
        problems.append(f"unknown top-level key '{unknown}'")
    for missing in sorted(TOP_LEVEL_REQUIRED - set(document)):
        problems.append(f"missing required top-level key '{missing}'")
    if problems:
        return problems

    if document["schema_version"] != SCHEMA_VERSION:
        problems.append(
            f"schema_version is '{document['schema_version']}', this engine requires "
            f"'{SCHEMA_VERSION}'"
        )
    if not SEMVER.match(str(document["policy_version"])):
        problems.append(f"policy_version '{document['policy_version']}' is not semantic version")
    if document["pack_kind"] not in VALID_PACK_KINDS:
        problems.append(f"pack_kind '{document['pack_kind']}' is not one of {sorted(VALID_PACK_KINDS)}")

    provenance = document["provenance"]
    for field in ("created_utc", "created_by", "derivation"):
        if not provenance.get(field):
            problems.append(f"provenance: missing '{field}'")

    integrity = document["integrity"]
    if integrity.get("algorithm") != "sha256":
        problems.append("integrity: algorithm must be sha256")
    digest = str(integrity.get("digest", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        problems.append("integrity: digest is not a 64-character hex sha256")
    elif verify_integrity:
        actual = compute_digest(document)
        if actual != digest:
            problems.append(
                f"integrity: digest mismatch — recorded {digest[:12]}..., computed {actual[:12]}.... "
                f"The artifact has been modified since it was signed."
            )

    families = document["feature_families"]
    for family_id, spec in sorted(families.items()):
        if not KEY_PATTERN.match(family_id):
            problems.append(f"feature_families: '{family_id}' is not a valid namespaced key")
        if spec.get("semantic_type") not in VALID_SEMANTIC_TYPES:
            problems.append(f"feature_families: '{family_id}' has invalid semantic_type")
        if not isinstance(spec.get("valid_range"), list) or len(spec.get("valid_range", [])) != 2:
            problems.append(f"feature_families: '{family_id}' has no two-element valid_range")

    _check_key_block(document["weights"], FEATURE_FAMILIES, "weights", problems, lambda v: float(v))
    _check_key_block(document["thresholds"], THRESHOLDS, "thresholds", problems,
                     lambda v: float(v["value"]))

    for weight_id in sorted(document["weights"]):
        if weight_id not in families:
            problems.append(
                f"weights: '{weight_id}' has no matching entry in feature_families"
            )

    for term_id, term in sorted(document["aggregate_terms"].items()):
        if not KEY_PATTERN.match(term_id):
            problems.append(f"aggregate_terms: '{term_id}' is not a valid namespaced key")
        bad_tiers = set(term.get("source_tiers", [])) - VALID_AGGREGATE_TIERS
        if bad_tiers:
            problems.append(f"aggregate_terms: '{term_id}' declares invalid source tiers {sorted(bad_tiers)}")
        if term.get("reduction") not in VALID_REDUCTIONS:
            problems.append(f"aggregate_terms: '{term_id}' has invalid reduction")
        max_inputs = term.get("max_inputs")
        if not isinstance(max_inputs, int) or not (1 <= max_inputs <= 64):
            problems.append(f"aggregate_terms: '{term_id}' max_inputs must be an integer in [1, 64]")

    return problems


def load_policy(path: Optional[Path] = None, *, verify_integrity: bool = True) -> Policy:
    """
    Load and validate a policy artifact, or fail.

    Never returns a partially valid policy and never falls back to defaults.
    """
    if path is None:
        import os
        env = os.environ.get("PEPTIDE_SUITE_POLICY")
        if not env:
            raise PolicyNotFound(
                "No policy artifact specified. Set PEPTIDE_SUITE_POLICY to a policy file, "
                "or pass an explicit path. The engine contains no coefficients and cannot "
                "score without a policy; it will not fall back to built-in values."
            )
        path = Path(env)

    path = Path(path)
    if not path.exists():
        raise PolicyNotFound(
            f"Policy artifact not found at {path}. The engine contains no coefficients "
            f"and will not fall back to built-in values."
        )

    try:
        document = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise PolicyValidationError([f"{path} is not valid JSON: {e}"])

    problems = validate_document(document, verify_integrity=verify_integrity)
    if problems:
        raise PolicyValidationError(problems)

    return Policy(
        policy_version=document["policy_version"],
        pack_kind=document["pack_kind"],
        provenance=dict(document["provenance"]),
        digest=document["integrity"]["digest"],
        _weights={k: float(v) for k, v in document["weights"].items()},
        _thresholds={k: float(v["value"]) for k, v in document["thresholds"].items()},
        _families=dict(document["feature_families"]),
        _aggregate_terms=dict(document["aggregate_terms"]),
        source_path=path,
    )
