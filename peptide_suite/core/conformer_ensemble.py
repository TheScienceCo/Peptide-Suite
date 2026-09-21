"""
Conformer ensembles, and the length at which they stop being honest.
[Addendum 2 section 4, post-6i]

CREST / GFN2-xTB ensembles are generated for cyclic, stapled,
disulfide-constrained, or short peptides. Above that length the ensemble is
reported as unavailable rather than generated, and the distinction is the whole
point: an undersampled ensemble is not a worse ensemble, it is a different
object. Boltzmann-weighting populations over conformers that were never found
produces numbers with the shape of a distribution and none of its content, and
nothing downstream can tell the difference.

So this module answers one question — is an ensemble admissible for this
peptide — and refuses to answer the second, because no conformational search
engine is available in this deployment. The eligibility logic is real and runs;
the generation is stubbed, and the stub raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ConstraintKind(Enum):
    """
    Why a peptide's conformational space is small enough to sample.

    Constraint matters more than length. A stapled or disulfide-closed peptide
    has a drastically reduced accessible space, which is exactly what makes an
    ensemble tractable; a linear peptide of the same length does not.
    """
    CYCLIC = "cyclic"
    STAPLED = "stapled"
    DISULFIDE = "disulfide_constrained"
    SHORT_LINEAR = "short_linear"
    UNCONSTRAINED = "unconstrained"


class EnsembleStatus(Enum):
    ELIGIBLE = "eligible"                  # a search would be meaningful
    UNAVAILABLE_TOO_LONG = "unavailable_too_long"
    UNAVAILABLE_NO_ENGINE = "unavailable_no_engine"

    @property
    def permits_conformational_claims(self) -> bool:
        return False  # never, in this build: eligible still needs an engine


class EnsembleNotAvailable(NotImplementedError):
    """Raised when an ensemble is requested and cannot honestly be produced."""


def ensemble_length_ceiling() -> int:
    """
    The length above which a search is reported unavailable.

    The rule is engine -- above some length the accessible space outruns any
    affordable search -- and where the line falls is policy.
    """
    from ..runtime import int_threshold
    return int_threshold("conformer.max_ensemble_length")


@dataclass
class EnsembleAssessment:
    """Whether an ensemble is admissible, and what it would take."""
    sequence_length: int
    constraint: ConstraintKind
    status: EnsembleStatus
    ceiling: int
    reason: str
    required_tooling: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def is_eligible(self) -> bool:
        return self.status is EnsembleStatus.ELIGIBLE

    def summary(self) -> str:
        return f"Conformer ensemble: {self.status.value}. {self.reason}"


def detect_constraint(sequence: str, description: str = "") -> ConstraintKind:
    """
    What constrains this peptide's conformational space, from what is known.

    Read from the sequence and from any proposal text describing it. Two
    cysteines is evidence of a possible disulfide rather than proof of one, and
    the assessment says so rather than assuming the bond is formed.
    """
    text = (description or "").lower()
    if any(k in text for k in ("cyclis", "cycliz", "head-to-tail", "macrocycl", "lactam")):
        return ConstraintKind.CYCLIC
    if any(k in text for k in ("staple", "stapling", "ring-closing metathesis")):
        return ConstraintKind.STAPLED
    if (sequence or "").upper().count("C") >= 2:
        return ConstraintKind.DISULFIDE
    return ConstraintKind.UNCONSTRAINED


def assess(sequence: str, description: str = "") -> EnsembleAssessment:
    """
    Whether a conformer ensemble is admissible for this peptide.

    Two independent reasons it may not be: the peptide is too long for a search
    to be complete, or no search engine exists here. They are reported
    separately because they have different remedies -- one needs a different
    method, the other needs an install.
    """
    sequence = (sequence or "").upper()
    length = len(sequence)
    ceiling = ensemble_length_ceiling()
    constraint = detect_constraint(sequence, description)

    if constraint is ConstraintKind.UNCONSTRAINED and length <= ceiling:
        constraint = ConstraintKind.SHORT_LINEAR

    notes = []
    if constraint is ConstraintKind.DISULFIDE:
        notes.append(
            "Two or more cysteines are present, which makes a disulfide possible rather "
            "than certain. If the bonds are not actually formed the accessible space is "
            "that of the linear peptide, and the eligibility below does not hold.")

    if length > ceiling and constraint in (ConstraintKind.SHORT_LINEAR,
                                           ConstraintKind.UNCONSTRAINED):
        return EnsembleAssessment(
            sequence_length=length, constraint=constraint,
            status=EnsembleStatus.UNAVAILABLE_TOO_LONG, ceiling=ceiling,
            reason=(f"{length} residues, unconstrained, above the {ceiling}-residue "
                    f"ceiling. A search here would be undersampled, and an undersampled "
                    f"ensemble is not a worse ensemble -- it is a different object. "
                    f"Boltzmann-weighting over conformers that were never found gives "
                    f"numbers shaped like a distribution with none of its content."),
            notes=notes)

    return EnsembleAssessment(
        sequence_length=length, constraint=constraint,
        status=EnsembleStatus.UNAVAILABLE_NO_ENGINE, ceiling=ceiling,
        reason=(f"{length} residues, {constraint.value}: the conformational space is "
                f"small enough for a search to be meaningful, but no conformational "
                f"search engine is available in this deployment, so no ensemble was "
                f"generated."),
        required_tooling=["CREST with GFN2-xTB, or an equivalent conformational search"],
        notes=notes)


def generate(sequence: str, description: str = ""):
    """
    Generate the ensemble. Not implemented, and raises rather than returning one.

    A stub returning a single conformer would be the worst available outcome: it
    has the type of an ensemble, so everything downstream would Boltzmann-weight
    a population of one and report the result as a conformational average.
    """
    assessment = assess(sequence, description)
    raise EnsembleNotAvailable(
        f"No conformer ensemble was generated. {assessment.reason} "
        f"Conformational claims that depend on an ensemble are unsupported here; they "
        f"are not approximated."
    )
