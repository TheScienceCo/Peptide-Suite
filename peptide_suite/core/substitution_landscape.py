"""
Substitution landscape: the whole scan as a grid.  [Addendum 3 section 9]

The optimize workflow ranks every position-by-residue substitution and then
reports the best five. That is the right answer to "what should I change" and
the wrong shape for "what does this peptide tolerate". The landscape keeps the
same computation and drops the ranking cut, so a reader sees the cold columns
as well as the hot cells -- where nothing helps is a finding too.

Three things this module exists to get right:

1. **A cell with no computed value is not a cell with the value zero.** The
   conservation term is the live case: with too few homologs there is no
   entropy to compute, and the pipeline says so rather than emitting 0. A grid
   would happily paint that as "no conservation cost", which is the opposite of
   what the pipeline said. So a cell carries a state, and NOT_COMPUTED is one
   of them, with the reason attached.

2. **The wild-type residue is its own state.** The diagonal is not a
   substitution with no effect; it is not a substitution.

3. **A one-signed quantity does not get a diverging scale.** Off-target cost
   runs 0..1 and has no meaningful midpoint, so it declares a sequential
   encoding and the renderer gives it one hue. Only genuinely signed
   quantities -- net score, charge change -- get two poles and a neutral
   middle, where the middle means "no change" because that is what zero means
   for them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from . import PeptideContext, SubstitutionRecommendation
from .charge_calculator import ChargeCalculator
from .substitution_predictor import (
    HydrophobicityScale,
    TERM_CONSERVATION,
)

# Amino acids down the y-axis, grouped by side-chain chemistry rather than
# alphabetically. Alphabetical order scatters the chemistry, and the eye reads
# bands: if every basic residue is bad at a position, that should be one block,
# not five rows spread over the axis.
AA_ROWS: Tuple[str, ...] = (
    "A", "V", "L", "I", "M",        # aliphatic
    "F", "W", "Y",                  # aromatic
    "S", "T", "N", "Q",             # polar uncharged
    "C", "G", "P",                  # special: thiol, no side chain, ring
    "D", "E",                       # acidic
    "K", "R", "H",                  # basic
)

# How long a peptide the grid will render. This is a display limit -- a chart
# 200 columns wide is unreadable at any width -- and deliberately NOT a policy
# threshold: the policy artifact holds scoring coefficients, and a number that
# describes a screen is not one of those.
MAX_LANDSCAPE_LENGTH = 80


class CellState(str, Enum):
    """
    What a cell of the grid is.

    COMPUTED carries a number. WILD_TYPE is the residue already there.
    NOT_COMPUTED is a term the pipeline declined to evaluate, and it must never
    be rendered as the midpoint of a scale: the midpoint reads as "no change",
    and "no data" is a different claim.
    """
    COMPUTED = "COMPUTED"
    WILD_TYPE = "WILD_TYPE"
    NOT_COMPUTED = "NOT_COMPUTED"


@dataclass(frozen=True)
class LandscapeCell:
    position: int              # 0-indexed
    mutant_aa: str
    state: CellState
    value: Optional[float]     # None unless state is COMPUTED
    detail: str                # what the number means, or why there isn't one
    confidence: str = ""       # the recommendation's overall confidence label


@dataclass(frozen=True)
class LandscapeMetric:
    """
    One selectable quantity, and how it is allowed to be drawn.

    `encoding` is part of the metric rather than a choice the renderer makes,
    because it follows from what the number is. A quantity with no sign has no
    midpoint to diverge about.
    """
    key: str
    label: str
    units: str
    encoding: str              # "diverging" | "sequential"
    midpoint_meaning: str      # what the centre of the scale means; "" if sequential
    description: str
    source: str                # which part of the pipeline produced it


METRICS: Tuple[LandscapeMetric, ...] = (
    LandscapeMetric(
        key="net_score",
        label="Net score",
        units="dimensionless, -1..+1",
        encoding="diverging",
        midpoint_meaning="predicted benefit and predicted off-target cost cancel",
        description=(
            "Expected benefit against the confirmed goal, minus the combined off-target "
            "cost. Benefit is the primary effect's magnitude discounted by confidence; "
            "cost combines the off-target terms as independent risks."
        ),
        source="ConfidenceScorer.explain_net_score",
    ),
    LandscapeMetric(
        key="off_target_cost",
        label="Off-target cost",
        units="dimensionless, 0..1",
        encoding="sequential",
        midpoint_meaning="",
        description=(
            "The combined off-target term alone, with the goal-directed benefit removed. "
            "Reads as how disruptive a substitution is regardless of whether it helps."
        ),
        source="ConfidenceScorer.explain_net_score",
    ),
    LandscapeMetric(
        key="conservation_cost",
        label="Conservation cost",
        units="dimensionless, 0..1",
        encoding="sequential",
        midpoint_meaning="",
        description=(
            "The conservation penalty on its own, from per-position Shannon entropy over "
            "the aligned homologs. Computed only when enough distinct homologous sequences "
            "were available; otherwise every cell is marked not computed, because entropy "
            "over one sequence is zero everywhere by construction and would paint the whole "
            "peptide as perfectly conserved."
        ),
        source="SubstitutionPredictor conservation term (Shannon entropy)",
    ),
    LandscapeMetric(
        key="charge_delta",
        label="Charge change",
        units="elementary charge, at the selected pH",
        encoding="diverging",
        midpoint_meaning="the substitution does not change charge at this pH",
        description=(
            "Effective charge of the mutant residue minus the wild-type residue, from "
            "Henderson-Hasselbalch at the requested pH. A property of the substitution, "
            "not a prediction about binding: direction of benefit is unknown without a "
            "receptor structure."
        ),
        source="ChargeCalculator.charge_at_ph",
    ),
    LandscapeMetric(
        key="hydrophobicity_delta",
        label="Hydrophobicity change",
        units="Kyte-Doolittle units",
        encoding="diverging",
        midpoint_meaning="the substitution does not change side-chain hydrophobicity",
        description=(
            "Kyte-Doolittle hydrophobicity of the mutant minus the wild type. Like the "
            "charge change, it describes the perturbation, not its consequence."
        ),
        source="Kyte-Doolittle scale",
    ),
)

METRICS_BY_KEY: Dict[str, LandscapeMetric] = {m.key: m for m in METRICS}
DEFAULT_METRIC = "net_score"


class LandscapeError(ValueError):
    pass


@dataclass(frozen=True)
class PositionSummary:
    """
    One column of the grid, reduced.

    The grid answers "what happens if I change this residue to that one". The
    column marginal answers the question people actually arrive with: which
    positions tolerate change at all. It is a reduction of the same computed
    cells, never a separate calculation -- so it cannot disagree with the grid
    above it.
    """
    position: int
    wild_type_aa: str
    n_computed: int
    n_not_computed: int
    best: Optional[float]      # None when nothing in this column was computed
    worst: Optional[float]
    mean: Optional[float]
    best_substitution: str     # "" when nothing was computed
    detail: str


@dataclass
class SubstitutionLandscape:
    """The grid, plus everything needed to read it without guessing."""
    sequence: str
    name: str
    goal: str
    ph: float
    metric: LandscapeMetric
    cells: List[LandscapeCell]
    rows: Tuple[str, ...] = AA_ROWS
    scale_bound: Optional[float] = None    # diverging: +/- this; sequential: 0..this
    scale_basis: str = ""
    n_computed: int = 0
    n_not_computed: int = 0
    not_computed_reason: str = ""
    notes: List[str] = field(default_factory=list)
    profile: List[PositionSummary] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.sequence)


def _conservation_term(rec: SubstitutionRecommendation):
    """The conservation effect, found by key rather than by its wording."""
    for effect in rec.off_target_effects or []:
        if effect.term_key == TERM_CONSERVATION:
            return effect
    return None


def _net_score(rec, ctx, charge_calc, ph):
    return rec.net_score, CellState.COMPUTED, (
        f"net {rec.net_score:+.2f} = benefit {rec.score_breakdown['expected_benefit']:.2f} "
        f"- off-target cost {rec.score_breakdown['combined_cost']:.2f}"
    )


def _off_target_cost(rec, ctx, charge_calc, ph):
    cost = rec.score_breakdown["combined_cost"]
    return cost, CellState.COMPUTED, f"combined off-target cost {cost:.2f}"


def _conservation_cost(rec, ctx, charge_calc, ph):
    effect = _conservation_term(rec)
    if effect is None:
        # The term is absent rather than zero. Nothing in the pipeline produces
        # this today, but a cell must not invent a value if it ever does.
        return None, CellState.NOT_COMPUTED, "No conservation term was produced for this substitution."
    if not effect.computed:
        return None, CellState.NOT_COMPUTED, effect.reasoning
    contribution = effect.magnitude * effect.score
    return contribution, CellState.COMPUTED, (
        f"conservation cost {contribution:.2f} (magnitude {effect.magnitude:.2f} "
        f"x confidence {effect.score:.2f}) — {effect.reasoning}"
    )


def _charge_delta(rec, ctx, charge_calc, ph):
    wt = charge_calc.charge_at_ph(rec.wild_type_aa, ph=ph, position=rec.position)
    mut = charge_calc.charge_at_ph(rec.mutant_aa, ph=ph, position=rec.position)
    delta = mut.effective_charge - wt.effective_charge
    return delta, CellState.COMPUTED, (
        f"Δq = {delta:+.2f}e at pH {ph} ({rec.wild_type_aa} {wt.effective_charge:+.2f}e → "
        f"{rec.mutant_aa} {mut.effective_charge:+.2f}e)"
    )


def _hydrophobicity_delta(rec, ctx, charge_calc, ph):
    delta = (
        HydrophobicityScale.hydrophobicity(rec.mutant_aa)
        - HydrophobicityScale.hydrophobicity(rec.wild_type_aa)
    )
    return delta, CellState.COMPUTED, (
        f"ΔKyte-Doolittle = {delta:+.1f} "
        f"({rec.wild_type_aa} {HydrophobicityScale.hydrophobicity(rec.wild_type_aa):+.1f} → "
        f"{rec.mutant_aa} {HydrophobicityScale.hydrophobicity(rec.mutant_aa):+.1f})"
    )


_EXTRACTORS: Dict[str, Callable] = {
    "net_score": _net_score,
    "off_target_cost": _off_target_cost,
    "conservation_cost": _conservation_cost,
    "charge_delta": _charge_delta,
    "hydrophobicity_delta": _hydrophobicity_delta,
}


def build_landscape(
    context: PeptideContext,
    recommendations: List[SubstitutionRecommendation],
    metric_key: str = DEFAULT_METRIC,
    ph: float = 7.4,
) -> SubstitutionLandscape:
    """
    Assemble the grid from a completed full scan.

    Takes the scan's output rather than running it, so the same computation can
    be re-projected onto a different metric without being redone -- and so this
    function cannot quietly become a second, divergent scoring path.
    """
    metric = METRICS_BY_KEY.get(metric_key)
    if metric is None:
        raise LandscapeError(
            f"Unknown metric '{metric_key}'. Available: {', '.join(sorted(METRICS_BY_KEY))}"
        )
    if not context.sequence:
        raise LandscapeError("No sequence to build a landscape over.")
    if len(context.sequence) > MAX_LANDSCAPE_LENGTH:
        raise LandscapeError(
            f"Sequence is {len(context.sequence)} residues; the landscape renders up to "
            f"{MAX_LANDSCAPE_LENGTH}. This is a readability limit on the chart, not a limit "
            f"on the scan — run the optimize workflow for the full ranking."
        )

    extract = _EXTRACTORS[metric.key]
    charge_calc = ChargeCalculator()

    by_cell: Dict[Tuple[int, str], SubstitutionRecommendation] = {
        (rec.position, rec.mutant_aa): rec for rec in recommendations
    }

    cells: List[LandscapeCell] = []
    computed_values: List[float] = []
    not_computed_reasons: List[str] = []

    for position, wt_aa in enumerate(context.sequence):
        for mutant_aa in AA_ROWS:
            if mutant_aa == wt_aa:
                cells.append(LandscapeCell(
                    position=position,
                    mutant_aa=mutant_aa,
                    state=CellState.WILD_TYPE,
                    value=None,
                    detail=f"{wt_aa} is the wild-type residue at position {position + 1}.",
                ))
                continue

            rec = by_cell.get((position, mutant_aa))
            if rec is None:
                cells.append(LandscapeCell(
                    position=position,
                    mutant_aa=mutant_aa,
                    state=CellState.NOT_COMPUTED,
                    value=None,
                    detail="The scan produced no result for this substitution.",
                ))
                not_computed_reasons.append("scan produced no result")
                continue

            value, state, detail = extract(rec, context, charge_calc, ph)
            if state is CellState.COMPUTED:
                computed_values.append(value)
            else:
                not_computed_reasons.append(detail)
            cells.append(LandscapeCell(
                position=position,
                mutant_aa=mutant_aa,
                state=state,
                value=value,
                detail=detail,
                confidence=rec.overall_confidence,
            ))

    scale_bound, scale_basis = _scale(metric, computed_values)
    profile = _profile(context.sequence, cells, metric)

    n_not_computed = sum(1 for c in cells if c.state is CellState.NOT_COMPUTED)
    landscape = SubstitutionLandscape(
        sequence=context.sequence,
        name=context.name,
        goal=context.confirmed_goal or context.inferred_function or "",
        ph=ph,
        metric=metric,
        cells=cells,
        scale_bound=scale_bound,
        scale_basis=scale_basis,
        n_computed=len(computed_values),
        n_not_computed=n_not_computed,
        # One reason serves for the whole grid when every uncomputed cell shares
        # it, which is the usual case: the conservation term either has homologs
        # or it does not.
        not_computed_reason=(
            not_computed_reasons[0] if len(set(not_computed_reasons)) == 1 and not_computed_reasons
            else ""
        ),
        notes=list(context.data_notes),
        profile=profile,
    )
    return landscape


def _profile(sequence: str, cells: List[LandscapeCell], metric: LandscapeMetric
             ) -> List[PositionSummary]:
    """
    Reduce each column of the grid.

    `best` means the most favourable computed value in that column, which for a
    diverging metric is the largest and for a one-signed cost metric is the
    smallest -- the direction is a property of what the number means, not a
    convention chosen here. A column with nothing computed reports None rather
    than an aggregate over an empty set, which numpy and Python both make
    tempting to report as zero or nan.
    """
    by_position: Dict[int, List[LandscapeCell]] = {}
    for cell in cells:
        if cell.state is CellState.WILD_TYPE:
            continue
        by_position.setdefault(cell.position, []).append(cell)

    favourable = max if metric.encoding == "diverging" else min
    summaries: List[PositionSummary] = []
    for position, wt_aa in enumerate(sequence):
        column = by_position.get(position, [])
        computed = [c for c in column if c.state is CellState.COMPUTED]
        n_not = len(column) - len(computed)
        if not computed:
            summaries.append(PositionSummary(
                position=position, wild_type_aa=wt_aa, n_computed=0, n_not_computed=n_not,
                best=None, worst=None, mean=None, best_substitution="",
                detail=(f"Nothing was computed at position {position + 1}, so this column "
                        f"has no summary. It is not a position where nothing helps."),
            ))
            continue
        values = [c.value for c in computed]
        pick = favourable(computed, key=lambda c: c.value)
        other = (min if favourable is max else max)(values)
        summaries.append(PositionSummary(
            position=position, wild_type_aa=wt_aa,
            n_computed=len(computed), n_not_computed=n_not,
            best=pick.value, worst=other, mean=sum(values) / len(values),
            best_substitution=f"{wt_aa}{position + 1}{pick.mutant_aa}",
            detail=(f"{len(computed)} substitutions computed at position {position + 1}; "
                    f"most favourable is {wt_aa}{position + 1}{pick.mutant_aa} at "
                    f"{pick.value:+.2f}" if metric.encoding == "diverging" else
                    f"{len(computed)} substitutions computed at position {position + 1}; "
                    f"least costly is {wt_aa}{position + 1}{pick.mutant_aa} at "
                    f"{pick.value:.2f}"),
        ))
    return summaries


def _scale(metric: LandscapeMetric, values: List[float]) -> Tuple[Optional[float], str]:
    """
    The outer bound of the colour scale, taken from the data actually computed.

    Derived from this grid rather than fixed, because a fixed theoretical range
    flattens a real result: a scan whose scores all fall within +/-0.3 rendered
    against +/-1.0 is a uniformly pale chart that says nothing. The basis
    travels with the number so two charts are never compared as if they shared
    a scale.
    """
    if not values:
        return None, "No cell in this grid has a computed value, so there is no scale."
    bound = max(abs(v) for v in values) if metric.encoding == "diverging" else max(values)
    if bound <= 0:
        return None, (
            "Every computed cell in this grid holds the same value, so there is no range "
            "to spread a scale across."
        )
    if metric.encoding == "diverging":
        return bound, (
            f"Scale runs ±{bound:.2f} {metric.units.split(',')[0]}, the largest magnitude "
            f"computed in this grid. It is not a fixed range — two grids do not share it."
        )
    return bound, (
        f"Scale runs 0 to {bound:.2f}, the largest value computed in this grid. "
        f"It is not a fixed range — two grids do not share it."
    )
