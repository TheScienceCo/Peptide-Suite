"""
Class B1 GPCR restricted zone, and the affinity/efficacy/bias contract.
[Addendum 2 section 9, build step 6g]

Class B1 peptide receptors engage their ligands through two domains, and the two
halves of the peptide do different jobs:

  C-terminal half    engages the extracellular domain and dominates AFFINITY
  N-terminal residues insert into the transmembrane core and dominate EFFICACY,
                     G-protein coupling and arrestin BIAS

The consequence is a placement rule, not a preference. Half-life chemistry and
conjugation go C-terminal or mid-chain; the N-terminal residues are a restricted
zone, and a modification inside it must report its predicted efficacy and bias
consequences rather than its affinity alone. This is why semaglutide's lipid
sits at Lys26 and not near the N-terminus.

The harder rule is the reporting one, and it applies everywhere rather than only
inside the zone: the system may never report an affinity gain as an improvement
on its own. Every proposal carries three separate fields, and any of them may be
UNKNOWN. UNKNOWN is an expected and acceptable value. A fabricated number is
not.

That asymmetry is the whole design. A peptide agonist that binds better and
signals worse is a worse drug, and a single "predicted improvement" number
cannot express that — it can only average it away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# The receptors the rule applies to. Listed explicitly rather than inferred from
# a family string: getting this wrong in the permissive direction would apply a
# two-domain model to a receptor that does not have one.
CLASS_B1_RECEPTORS = frozenset({
    "GLP1R", "GIPR", "GCGR", "PTH1R", "CTR", "CRF1R", "SCTR", "VIPR", "PAC1",
})

# Common spellings, so a caller writing "GLP-1R" is not silently treated as a
# non-B1 receptor and exempted from the rule.
_ALIASES = {
    "GLP-1R": "GLP1R", "GLP1-R": "GLP1R", "GLP-1 RECEPTOR": "GLP1R",
    "GIP-R": "GIPR", "GIP RECEPTOR": "GIPR",
    "GCG-R": "GCGR", "GLUCAGON RECEPTOR": "GCGR",
    "PTH-1R": "PTH1R", "PTH1-R": "PTH1R", "PTHR1": "PTH1R",
    "CALCR": "CTR", "CALCITONIN RECEPTOR": "CTR",
    "CRFR1": "CRF1R", "CRF-R1": "CRF1R", "CRHR1": "CRF1R",
    "SECRETIN RECEPTOR": "SCTR",
    "VPAC1": "VIPR", "VPAC2": "VIPR", "VIP RECEPTOR": "VIPR",
    "ADCYAP1R1": "PAC1", "PAC1R": "PAC1",
}


def normalise_receptor(name: str) -> str:
    key = (name or "").strip().upper()
    return _ALIASES.get(key, key)


def is_class_b1(receptor: str) -> bool:
    return normalise_receptor(receptor) in CLASS_B1_RECEPTORS


def restricted_zone_length() -> int:
    """
    How many N-terminal residues are restricted.

    The two-domain model is engine; where the line falls is a policy question
    like any other cutoff.
    """
    from ..runtime import int_threshold
    return int_threshold("class_b1.restricted_zone_residues")


class Effect(Enum):
    """
    A predicted directional effect on one axis.

    UNKNOWN is a first-class value, not a failure. The section is explicit that
    it is expected and acceptable, and that a fabricated number is not — so the
    enum has no "neutral" member that an unexamined axis could quietly take.
    """
    IMPROVES = "improves"
    DEGRADES = "degrades"
    NEUTRAL = "neutral"
    UNKNOWN = "UNKNOWN"

    @property
    def is_known(self) -> bool:
        return self is not Effect.UNKNOWN


@dataclass
class ThreeFieldPrediction:
    """
    Affinity, efficacy and bias as separate fields.

    Never collapsed into one number. A peptide agonist that binds better and
    signals worse is a worse drug, and a single "predicted improvement" cannot
    express that; it can only average it away.
    """
    affinity: Effect = Effect.UNKNOWN
    efficacy: Effect = Effect.UNKNOWN
    bias: Effect = Effect.UNKNOWN
    affinity_basis: str = ""
    efficacy_basis: str = ""
    bias_basis: str = ""

    @property
    def known_axes(self) -> List[str]:
        return [name for name in ("affinity", "efficacy", "bias")
                if getattr(self, name).is_known]

    @property
    def is_affinity_only(self) -> bool:
        """An affinity call with nothing said about signalling."""
        return (self.affinity.is_known
                and not self.efficacy.is_known and not self.bias.is_known)

    def improvement_claim_permitted(self) -> bool:
        """
        Whether this may be described as an improvement.

        Requires that the affinity gain is not standing alone. An affinity-only
        improvement is the specific claim the section forbids.
        """
        if self.affinity is not Effect.IMPROVES:
            return True
        return not self.is_affinity_only

    def summary(self) -> str:
        return (f"affinity {self.affinity.value}, efficacy {self.efficacy.value}, "
                f"bias {self.bias.value}")


class RestrictedZoneViolation(Exception):
    """Raised when a proposal in the restricted zone reports affinity alone."""


@dataclass
class ZoneRuling:
    """Whether a proposal satisfies the restricted-zone reporting requirement."""
    position: Optional[int]
    in_restricted_zone: bool
    receptor: str
    prediction: ThreeFieldPrediction
    violations: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    # Set when affinity improves with nothing reported on efficacy or bias. The
    # proposal still emits; its affinity gain just does not count toward rank.
    affinity_claim_discounted: bool = False

    @property
    def permits_emission(self) -> bool:
        return not self.violations


def evaluate(proposal: str, position: Optional[int], receptor: str,
             prediction: ThreeFieldPrediction) -> ZoneRuling:
    """
    Apply the section 9 rules to one proposal.

    Two rules, with different scopes. The placement rule applies inside the
    restricted zone: a modification there must say what it does to efficacy and
    bias. The reporting rule applies everywhere: an affinity gain may never
    stand alone as an improvement.
    """
    zone = restricted_zone_length() if is_class_b1(receptor) else 0
    inside = bool(is_class_b1(receptor) and position is not None and 1 <= position <= zone)

    ruling = ZoneRuling(position=position, in_restricted_zone=inside,
                        receptor=normalise_receptor(receptor), prediction=prediction)

    if not is_class_b1(receptor):
        ruling.notes.append(
            f"{receptor or 'The receptor'} is not in the class B1 set, so the two-domain "
            f"placement rule does not apply. The reporting rule still does.")

    if inside:
        if not prediction.efficacy.is_known and not prediction.bias.is_known:
            ruling.violations.append(
                f"Position {position} is inside the N-terminal restricted zone (first "
                f"{zone} residues). Those residues insert into the transmembrane core and "
                f"dominate efficacy and bias, so a modification there must report its "
                f"predicted efficacy and bias consequences. Reporting affinity alone "
                f"describes the half of the pharmacology this position does not control.")
        else:
            ruling.notes.append(
                f"Inside the restricted zone (first {zone} residues) and reporting "
                f"{', '.join(prediction.known_axes)}.")
    elif is_class_b1(receptor) and position is not None:
        ruling.notes.append(
            f"Position {position} is outside the N-terminal restricted zone. Half-life "
            f"chemistry and conjugation belong here — semaglutide's lipid sits at Lys26 "
            f"for this reason.")

    # The reporting rule is not a block. UNKNOWN is an expected and acceptable
    # value, so a proposal with unknown efficacy and bias is still a legitimate
    # proposal -- what it may not do is have its affinity gain counted as an
    # improvement. In a ranked system, counting it IS reporting it, so the
    # enforcement is that the affinity contribution does not drive the ranking
    # and the output says why.
    if not prediction.improvement_claim_permitted():
        ruling.affinity_claim_discounted = True
        ruling.notes.append(
            "Predicted affinity improves and neither efficacy nor bias was assessed. The "
            "affinity gain is therefore not counted toward this proposal's rank: a peptide "
            "agonist that binds better and signals worse is a worse drug, and ranking on "
            "affinity alone is what reporting it as an improvement would mean. The "
            "prediction is still shown, on all three axes.")

    return ruling


def conjugation_site_guidance(sequence: str, receptor: str) -> Dict[str, object]:
    """
    Where half-life chemistry may go on a class B1 ligand.

    Returns the restricted span and the permitted span as positions, so a caller
    proposing an acylation site can check it rather than reading prose.
    """
    if not is_class_b1(receptor):
        return {"applies": False,
                "reason": f"{receptor or 'The receptor'} is not class B1."}
    zone = restricted_zone_length()
    return {
        "applies": True,
        "receptor": normalise_receptor(receptor),
        "restricted_positions": list(range(1, min(zone, len(sequence)) + 1)),
        "permitted_positions": list(range(zone + 1, len(sequence) + 1)),
        "rule": ("Half-life chemistry and conjugation go C-terminal or mid-chain. The "
                 f"first {zone} residues insert into the transmembrane core and dominate "
                 "efficacy and bias."),
        "worked_example": ("Semaglutide's lipid sits at Lys26, not near the N-terminus."),
    }
