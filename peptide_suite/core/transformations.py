"""
Transformation output frame.

The unit of output is a discrete transformation drawn from a controlled
vocabulary, never a free-form optimised sequence. A sequence hides which moves
were made and why; a transformation list is reviewable move by move.

Each transformation is scored against the full objective vector rather than a
single number, because peptide optimisation is inherently multi-objective and a
scalar hides the trades. The scalarised score is reported alongside the vector,
never instead of it, and always with its weights stated.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .epistemics import Assumption, Claim, ClaimType, LeakageFlag


class MoveType(Enum):
    """
    Controlled vocabulary of transformation types.

    Extending this vocabulary requires explicit justification: an uncontrolled
    move space makes outputs incomparable across runs and lets vague suggestions
    masquerade as specific ones.
    """
    BACKBONE_CONSTRAINT = "backbone_constraint"
    SIDE_CHAIN_SUBSTITUTION = "side_chain_substitution"
    LIPIDATION = "lipidation"
    CYCLIZATION_STAPLING = "cyclization_stapling"
    TERMINAL_CAPPING = "terminal_capping"
    GLYCOSYLATION = "glycosylation"
    DISULFIDE_SURROGATE = "disulfide_surrogate"
    CHARGE_ENGINEERING = "charge_engineering"
    LIABILITY_REMOVAL = "liability_removal"


# Moves that act by restricting conformational freedom. Each of these must ship
# a pre-organization proxy: interaction-energy features alone cannot see
# pre-organization, so an entropic benefit would otherwise go unreported.
CONFORMATIONAL_CONSTRAINT_MOVES = {
    MoveType.BACKBONE_CONSTRAINT,
    MoveType.CYCLIZATION_STAPLING,
    MoveType.DISULFIDE_SURROGATE,
}


class Objective(Enum):
    """The eight axes every transformation is scored against."""
    POTENCY = "potency"
    FUNCTIONAL_SELECTIVITY = "functional_selectivity_bias"
    PROTEOLYTIC_HALF_LIFE = "proteolytic_half_life"
    ALBUMIN_FCRN = "albumin_fcrn_engagement"
    AGGREGATION = "aggregation_propensity"
    SOLUBILITY = "solubility_at_formulation_ph"
    IMMUNOGENICITY = "immunogenicity_risk"
    SYNTHESIZABILITY = "synthesizability"


class Direction(Enum):
    IMPROVES = "improves"
    DEGRADES = "degrades"
    NEUTRAL = "neutral"
    NOT_ASSESSED = "not_assessed"


# Default scalarisation weights. These are a stated editorial choice, not a
# derived optimum: they are reported with every scalarised score so a reader can
# see what the single number is actually optimising for and substitute their own.
DEFAULT_WEIGHTS: Dict[Objective, float] = {
    Objective.POTENCY: 0.25,
    Objective.FUNCTIONAL_SELECTIVITY: 0.10,
    Objective.PROTEOLYTIC_HALF_LIFE: 0.20,
    Objective.ALBUMIN_FCRN: 0.10,
    Objective.AGGREGATION: 0.10,
    Objective.SOLUBILITY: 0.10,
    Objective.IMMUNOGENICITY: 0.10,
    Objective.SYNTHESIZABILITY: 0.05,
}


@dataclass
class ObjectiveDelta:
    """
    The predicted effect of a transformation on one objective.

    `magnitude` is None when the objective was not assessed. It is never 0 as a
    placeholder: "no effect" and "not looked at" are different statements and
    collapsing them silently turns an unexamined axis into a clean bill of health.
    """
    objective: Objective
    direction: Direction
    magnitude: Optional[float]      # 0-1, signed by `direction`; None if unassessed
    claim: Claim
    basis: str = ""

    @property
    def assessed(self) -> bool:
        return self.direction is not Direction.NOT_ASSESSED and self.magnitude is not None

    @property
    def signed(self) -> Optional[float]:
        if not self.assessed:
            return None
        if self.direction is Direction.IMPROVES:
            return self.magnitude
        if self.direction is Direction.DEGRADES:
            return -self.magnitude
        return 0.0

    @classmethod
    def not_assessed(cls, objective: Objective, reason: str) -> "ObjectiveDelta":
        return cls(
            objective=objective,
            direction=Direction.NOT_ASSESSED,
            magnitude=None,
            claim=Claim.inferred(
                f"{objective.value}: not assessed",
                inference_step=reason,
            ),
            basis=reason,
        )


@dataclass
class PreOrganizationProxy:
    """
    Evidence that a constraint move acts by restricting the accessible ensemble.

    Required for every conformational-constraint move. Without it, an entropic
    benefit is invisible to interaction-energy features and the move looks inert.
    """
    helicity_delta: Optional[float] = None       # fractional helicity change
    basin_restriction: str = ""                  # Ramachandran basins removed
    rmsf_change: Optional[float] = None          # requires an ensemble (Tier 2)
    mechanism: str = ""                          # "entropic" | "enthalpic" | "mixed"
    claim: Optional[Claim] = None

    def render(self) -> str:
        parts = []
        if self.helicity_delta is not None:
            parts.append(f"predicted helicity {self.helicity_delta:+.2f}")
        if self.basin_restriction:
            parts.append(self.basin_restriction)
        if self.rmsf_change is not None:
            parts.append(f"ensemble RMSF {self.rmsf_change:+.2f} A")
        else:
            parts.append("ensemble RMSF not available (requires Tier 2 sampling)")
        if self.mechanism:
            parts.append(f"predicted benefit is primarily {self.mechanism}")
        return "; ".join(parts)


@dataclass
class Transformation:
    """One discrete, reviewable change to the peptide."""
    move: MoveType
    position: Optional[int]         # 0-indexed; None for whole-molecule moves
    description: str                # e.g. "Ala2 -> Aib"
    rationale: str
    objective_deltas: List[ObjectiveDelta] = field(default_factory=list)
    evidence_tier: str = ""
    assumptions: List[Assumption] = field(default_factory=list)
    preorganization: Optional[PreOrganizationProxy] = None
    tradeoff_label: str = ""        # Set when the move trades one axis for another
    leakage_flag: Optional[LeakageFlag] = None
    physics_tier_reached: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def display_position(self) -> Optional[int]:
        return None if self.position is None else self.position + 1

    def delta(self, objective: Objective) -> Optional[ObjectiveDelta]:
        for d in self.objective_deltas:
            if d.objective is objective:
                return d
        return None

    def unassessed_objectives(self) -> List[Objective]:
        assessed = {d.objective for d in self.objective_deltas if d.assessed}
        return [o for o in Objective if o not in assessed]

    def scalarize(self, weights: Optional[Dict[Objective, float]] = None) -> Dict:
        """
        Collapse the objective vector to one number, and say how.

        Only assessed objectives contribute, and the weight actually applied is
        renormalised over them. `coverage` reports what fraction of the intended
        weight was backed by an assessment, so a score resting on two of eight
        axes cannot be mistaken for a score resting on all eight.
        """
        weights = weights or DEFAULT_WEIGHTS

        total_weight = sum(weights.get(o, 0.0) for o in Objective)
        applied, accumulated = 0.0, 0.0

        for delta in self.objective_deltas:
            if not delta.assessed:
                continue
            w = weights.get(delta.objective, 0.0)
            applied += w
            accumulated += w * delta.signed

        score = accumulated / applied if applied else 0.0
        coverage = applied / total_weight if total_weight else 0.0

        return {
            "score": round(score, 4),
            "coverage": round(coverage, 4),
            "weights": {o.value: weights.get(o, 0.0) for o in Objective},
            "assessed_objectives": [d.objective.value for d in self.objective_deltas if d.assessed],
            "unassessed_objectives": [o.value for o in self.unassessed_objectives()],
            "caveat": (
                f"Scalarised over {coverage:.0%} of the intended objective weight. "
                f"The remaining objectives were not assessed and are excluded rather "
                f"than assumed neutral."
            ),
        }

    def validate(self) -> List[str]:
        """
        Check the transformation against the output contract.

        Returns a list of violations; an empty list means the transformation is
        safe to emit. Callers should refuse to emit a violating transformation
        rather than emitting it with a warning.
        """
        problems = []

        if self.move in CONFORMATIONAL_CONSTRAINT_MOVES and self.preorganization is None:
            problems.append(
                f"{self.move.value} is a conformational-constraint move and must report a "
                f"pre-organization proxy (helicity delta, basin restriction, or RMSF change). "
                f"Interaction-energy features alone cannot see pre-organization."
            )

        for delta in self.objective_deltas:
            if delta.claim.claim_type is ClaimType.COMPUTED and delta.claim.tier is None:
                problems.append(
                    f"Objective '{delta.objective.value}' is COMPUTED but names no physics tier."
                )

        potency = self.delta(Objective.POTENCY)
        exposure = self.delta(Objective.ALBUMIN_FCRN)
        trades_potency_for_exposure = (
            potency is not None and potency.direction is Direction.DEGRADES
            and exposure is not None and exposure.direction is Direction.IMPROVES
        )
        if trades_potency_for_exposure and not self.tradeoff_label:
            problems.append(
                "This move degrades potency while improving exposure and must carry an "
                "explicit tradeoff label. Presenting it as an unqualified improvement "
                "misrepresents what it does."
            )

        return problems
