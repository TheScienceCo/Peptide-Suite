"""
Structure template hierarchy, confidence gates, and the refusal path.
[Addendum 2 sections 4 and 6c]

Every conformation-dependent claim rests on some structure. This module decides
which structure that is allowed to be, and refuses when the answer is "none good
enough".

The hierarchy is strictly ordered:

  1. Experimental structure of THIS peptide bound to THIS receptor
  2. Experimental structure of a close homolog bound to the SAME receptor
  3. Predicted complex that passes the confidence gate
  4. Refuse -- emit UNRESOLVED and block all conformation-dependent proposals

Tier 4 is the point of the module. A system without a usable structure does not
degrade gracefully into guessing; it stops. Gracefully degrading is precisely
the failure being designed against, because a conformational proposal made
without a conformation looks exactly like one made with it.

Two templates are excluded by name rather than by score, because they are
plausible, available, and wrong:

  Free-state conformation. Linear peptide hormones are disordered in solution
  and fold on binding. GLP-1 is a random coil free and helical bound. A design
  built on the free state is built on the wrong molecule.

  Precursor conformation. A fragment's conformation inside its parent is a fact
  about the parent. Addendum 2 names this the most likely way for the
  parent-molecule module to actively mislead.

Both are rejected at any confidence. A well-resolved crystal structure of the
free peptide is still the wrong template, and its resolution is not the issue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TemplateTier(Enum):
    """
    Where a conformational claim's structure came from. Lower is better.

    The numeric value is the rank, so tiers compare directly and a caller cannot
    accidentally treat a predicted complex as equivalent to a bound experimental
    structure.
    """
    EXPERIMENTAL_THIS_COMPLEX = 1
    EXPERIMENTAL_HOMOLOG_COMPLEX = 2
    PREDICTED_COMPLEX_GATED = 3
    REFUSED = 4

    @property
    def is_usable(self) -> bool:
        return self is not TemplateTier.REFUSED

    @property
    def label(self) -> str:
        return {
            TemplateTier.EXPERIMENTAL_THIS_COMPLEX:
                "experimental structure of this peptide bound to this receptor",
            TemplateTier.EXPERIMENTAL_HOMOLOG_COMPLEX:
                "experimental structure of a close homolog bound to the same receptor",
            TemplateTier.PREDICTED_COMPLEX_GATED:
                "predicted complex, confidence gate passed",
            TemplateTier.REFUSED:
                "no admissible template",
        }[self]


class RejectedTemplateKind(Enum):
    """
    Templates rejected on what they are, not on how good they are.

    Kept as an enum rather than a string so a caller offering one gets a named
    refusal instead of silently falling through to the next candidate.
    """
    FREE_STATE = "free_state"
    PRECURSOR = "precursor"
    UNBOUND_RECEPTOR = "unbound_receptor"
    DIFFERENT_RECEPTOR = "different_receptor"


REJECTION_REASON = {
    RejectedTemplateKind.FREE_STATE: (
        "Free-state conformation. Linear peptide hormones are disordered in solution and "
        "fold on binding, so the free state is a different molecule from the one that "
        "binds. Resolution is not the issue: a perfect structure of the wrong state is "
        "still the wrong template."
    ),
    RejectedTemplateKind.PRECURSOR: (
        "Precursor conformation. A fragment's conformation inside its parent is a fact "
        "about the parent, not about the fragment in isolation or bound to a receptor."
    ),
    RejectedTemplateKind.UNBOUND_RECEPTOR: (
        "Apo receptor structure. The binding-site geometry of an unoccupied receptor is "
        "not the geometry the peptide sees, and class B1 receptors in particular "
        "rearrange substantially on peptide engagement."
    ),
    RejectedTemplateKind.DIFFERENT_RECEPTOR: (
        "Complex with a different receptor. The bound conformation is a property of the "
        "pair, so a structure against another receptor constrains this one only weakly."
    ),
}


@dataclass(frozen=True)
class TemplateCandidate:
    """
    One structure offered as a possible template.

    `kind` is set when the candidate is one of the named-wrong categories; it is
    checked before any confidence metric, so a high-confidence free-state
    structure is rejected on identity rather than accepted on score.
    """
    identifier: str
    is_experimental: bool
    is_complex: bool                       # peptide bound to a receptor
    same_peptide: bool = False
    same_receptor: bool = False
    homolog_identity: Optional[float] = None   # 0-1, to the modelled peptide
    kind: Optional[RejectedTemplateKind] = None
    metadata: Dict[str, Any] = field(default_factory=dict)  # plddt / pae / iptm / seed_count
    source: str = ""


@dataclass
class TemplateDecision:
    """
    Which template was selected, or why none was.

    Carries the full audit trail rather than just the winner: a refusal that
    cannot say what it looked at and why each candidate failed is
    indistinguishable from a system that never looked.
    """
    tier: TemplateTier
    template: Optional[TemplateCandidate] = None
    considered: List[TemplateCandidate] = field(default_factory=list)
    rejections: List[str] = field(default_factory=list)
    gate_failures: List[str] = field(default_factory=list)

    @property
    def is_refusal(self) -> bool:
        return self.tier is TemplateTier.REFUSED

    @property
    def permits_conformational_claims(self) -> bool:
        return self.tier.is_usable

    def summary(self) -> str:
        if not self.is_refusal:
            return (f"Conformational claims admitted on tier {self.tier.value}: "
                    f"{self.tier.label} ({self.template.identifier}).")
        if not self.considered:
            return (
                "No structure was supplied, so no conformational claim can be made. "
                "Conformation-dependent proposals are blocked rather than estimated: a "
                "proposal made without a conformation is indistinguishable in the output "
                "from one made with it."
            )
        return (
            f"No admissible template among {len(self.considered)} candidate(s), so "
            f"conformation-dependent proposals are blocked. Refusing is the designed "
            f"behaviour here, not a gap."
        )

    def detail_lines(self) -> List[str]:
        return list(self.rejections) + list(self.gate_failures)


# Conformation-dependent moves. Each of these makes a claim about the shape the
# peptide adopts, so each is blocked by a refusal. Side-chain substitution,
# terminal capping and charge engineering are not here: their primary rationale
# is sequence-level and survives without a structure.
CONFORMATION_DEPENDENT_MOVES = frozenset({
    "backbone_constraint",
    "cyclization_stapling",
    "disulfide_surrogate",
})


def _gate():
    """
    The confidence gate, with its thresholds from the active policy.

    Previously hardcoded in provenance.StructureGate. The direction of the
    comparisons is engine (higher pLDDT is better, lower PAE is better); where
    the line falls is policy.
    """
    from ..runtime import threshold, int_threshold
    return {
        "min_plddt": threshold("structure_gate.min_plddt"),
        "max_pae": threshold("structure_gate.max_interface_pae"),
        "min_iptm": threshold("structure_gate.min_iptm"),
        "min_seed_count": int_threshold("structure_gate.min_seed_count"),
    }


def evaluate_confidence_gate(metadata: Dict[str, Any]) -> List[str]:
    """
    Reasons a predicted complex fails the gate. Empty means it passes.

    A missing metric is a failure, not a pass. An unreported pLDDT is not
    evidence of a good structure, and defaulting it to acceptable is how an
    ungated structure enters the pipeline looking gated.
    """
    limits = _gate()
    failures: List[str] = []

    plddt = metadata.get("plddt")
    if plddt is None:
        failures.append("pLDDT not reported")
    elif plddt < limits["min_plddt"]:
        failures.append(f"pLDDT {plddt} below {limits['min_plddt']}")

    pae = metadata.get("pae")
    if pae is None:
        failures.append("interface PAE not reported")
    elif pae > limits["max_pae"]:
        failures.append(f"interface PAE {pae} above {limits['max_pae']}")

    iptm = metadata.get("iptm")
    if iptm is None:
        failures.append("ipTM not reported")
    elif iptm < limits["min_iptm"]:
        failures.append(f"ipTM {iptm} below {limits['min_iptm']}")

    seeds = metadata.get("seed_count")
    if seeds is None:
        failures.append("seed count not reported")
    elif seeds < limits["min_seed_count"]:
        failures.append(f"seed count {seeds} below {limits['min_seed_count']}")

    return failures


def _min_homolog_identity() -> float:
    from ..runtime import threshold
    return threshold("identification.similarity_floor")


def select_template(candidates: Optional[List[TemplateCandidate]] = None) -> TemplateDecision:
    """
    Walk the hierarchy and return the best admissible template, or a refusal.

    Strictly ordered: a tier-1 candidate wins outright, and a tier-3 candidate
    is never consulted while a tier-2 one exists. Scanning for "the best
    scoring" candidate across tiers would let a confident prediction outrank a
    real structure, which inverts the whole hierarchy.
    """
    candidates = list(candidates or [])
    decision = TemplateDecision(tier=TemplateTier.REFUSED, considered=candidates)

    admissible: List[TemplateCandidate] = []
    for candidate in candidates:
        # Identity first. A named-wrong template is rejected at any confidence.
        if candidate.kind is not None:
            decision.rejections.append(
                f"{candidate.identifier}: {REJECTION_REASON[candidate.kind]}")
            continue
        if not candidate.is_complex:
            decision.rejections.append(
                f"{candidate.identifier}: not a peptide-receptor complex, so it carries no "
                f"bound conformation.")
            continue
        if not candidate.same_receptor:
            decision.rejections.append(
                f"{candidate.identifier}: {REJECTION_REASON[RejectedTemplateKind.DIFFERENT_RECEPTOR]}")
            continue
        admissible.append(candidate)

    # Tier 1
    for candidate in admissible:
        if candidate.is_experimental and candidate.same_peptide:
            decision.tier = TemplateTier.EXPERIMENTAL_THIS_COMPLEX
            decision.template = candidate
            return decision

    # Tier 2
    floor = _min_homolog_identity()
    for candidate in admissible:
        if not (candidate.is_experimental and not candidate.same_peptide):
            continue
        identity = candidate.homolog_identity
        if identity is None:
            decision.rejections.append(
                f"{candidate.identifier}: homolog template with no stated identity to the "
                f"modelled peptide, so closeness cannot be established.")
            continue
        if identity < floor:
            decision.rejections.append(
                f"{candidate.identifier}: homolog identity {identity:.2f} below the "
                f"admissible floor {floor:.2f}.")
            continue
        decision.tier = TemplateTier.EXPERIMENTAL_HOMOLOG_COMPLEX
        decision.template = candidate
        return decision

    # Tier 3
    for candidate in admissible:
        if candidate.is_experimental:
            continue
        failures = evaluate_confidence_gate(candidate.metadata)
        if failures:
            decision.gate_failures.append(
                f"{candidate.identifier}: predicted complex failed the confidence gate "
                f"({'; '.join(failures)}).")
            continue
        decision.tier = TemplateTier.PREDICTED_COMPLEX_GATED
        decision.template = candidate
        return decision

    # Tier 4
    return decision


def blocked_moves(decision: TemplateDecision) -> frozenset:
    """Move types a refusal blocks. Empty when a template was admitted."""
    return CONFORMATION_DEPENDENT_MOVES if decision.is_refusal else frozenset()
