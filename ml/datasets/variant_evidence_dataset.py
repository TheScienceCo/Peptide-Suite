"""
Turning measured variant outcomes into training data, and refusing to.  [Phase 3]

The variant-evidence store holds published measurements. That makes it look
like a labelled dataset, and it very nearly is one. What stops it being one
straight off is that most of what a store like this contains cannot be a label,
for reasons that have nothing to do with how many rows there are.

FOUR ROWS THAT ARE NOT LABELS.

  A record with no measured outcome. It is a design -- someone made this
  construct -- and a design has no target value. Training on it means inventing
  one.

  A confounded record. A construct with a backbone substitution, a sequence
  substitution and a lipid chain has one measured half-life and three changes.
  Attaching that number to any one of them teaches the model an attribution
  nobody made.

  A qualitative outcome. "Increased" is a real result and is not a regression
  target. It can be a classification target, which is a different task, and the
  two are kept apart rather than one being coerced into the other.

  An outcome with no citation. It caps at BIOCHEMICAL_PRINCIPLE under
  `Provenance.max_tier`, and training a model on unverifiable numbers produces
  a model whose outputs inherit the unverifiability while looking like
  predictions.

LEAKAGE IS THE SPECIFIC RISK HERE, more than in an ordinary peptide dataset.
This store is built out of variant series: an alanine scan is thirty records
sharing one parent, and a random split puts most of them in train and the rest
in test. The model then predicts a peptide it has effectively seen. Splitting
by parent molecule is the floor -- `check_leakage` refuses a split where any
parent, or any exact variant, appears on both sides -- and sequence clustering
on top of it is what `ml/datasets/splitting.py` already does.

The store is currently empty, so every builder here returns nothing and says
why. That is the correct output, not a placeholder: a dataset builder that
manufactured rows to have something to return would be the failure this whole
layer is written against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from peptide_suite.core import EvidenceTier
from peptide_suite.core.variant_evidence import (
    Direction, MeasuredOutcome, OutcomeMeasure, VariantEvidenceStore, VariantRecord,
)


class ExclusionReason(Enum):
    """Why a stored record did not become a training row. Counted, never silent."""

    NO_MEASURED_OUTCOME = "NO_MEASURED_OUTCOME"
    CONFOUNDED_COMBINATION = "CONFOUNDED_COMBINATION"
    NOT_QUANTITATIVE = "NOT_QUANTITATIVE"
    BELOW_TIER_FLOOR = "BELOW_TIER_FLOOR"
    WRONG_MEASURE = "WRONG_MEASURE"

    @property
    def explanation(self) -> str:
        return {
            ExclusionReason.NO_MEASURED_OUTCOME:
                "The record states a modification and no measurement. A design has no "
                "target value, and training on it means inventing one.",
            ExclusionReason.CONFOUNDED_COMBINATION:
                "More than one thing changed at once, so the measured value belongs to "
                "the whole construct. Attaching it to one change teaches the model an "
                "attribution nobody made.",
            ExclusionReason.NOT_QUANTITATIVE:
                "The outcome is a direction without a magnitude. That is a real result "
                "and a classification target, not a regression target.",
            ExclusionReason.BELOW_TIER_FLOOR:
                "The outcome carries no citation, so it caps below the tier floor. A "
                "model trained on unverifiable numbers produces outputs that inherit the "
                "unverifiability and look like predictions.",
            ExclusionReason.WRONG_MEASURE:
                "The outcome measures something other than the requested target.",
        }[self]


@dataclass(frozen=True)
class TrainingRow:
    """One usable (variant, measurement) pair."""

    parent: str
    variant_name: str
    modification_label: str
    position: Optional[int]
    wild_type: str
    mutant: str
    measure: OutcomeMeasure
    target_value: float
    target_is_fold_change: bool
    comparator: str
    assay: str
    tier: EvidenceTier

    @property
    def group(self) -> str:
        """
        The key a split must not straddle.

        The parent molecule, not the variant: an alanine scan is thirty records
        sharing one parent, and splitting on the variant puts twenty-nine of
        them in train and one in test.
        """
        return self.parent.strip().upper()


@dataclass
class DatasetBuild:
    """Rows, and a full account of what was left out."""

    measure: OutcomeMeasure
    tier_floor: EvidenceTier
    rows: List[TrainingRow] = field(default_factory=list)
    excluded: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def is_trainable(self) -> bool:
        return bool(self.rows)

    def exclude(self, reason: ExclusionReason, what: str) -> None:
        self.excluded.setdefault(reason.value, []).append(what)

    def report(self) -> str:
        if not self.rows:
            head = (f"No training rows for {self.measure.value}: nothing in the "
                    f"variant-evidence store qualifies.")
        else:
            head = (f"{len(self.rows)} training row(s) for {self.measure.value} across "
                    f"{len({r.group for r in self.rows})} parent molecule(s).")
        for reason, items in sorted(self.excluded.items()):
            head += (f" {len(items)} excluded as {reason}: "
                     f"{ExclusionReason[reason].explanation}")
        return head

    def to_dict(self) -> Dict[str, Any]:
        return {"measure": self.measure.value, "tier_floor": self.tier_floor.name,
                "rows": len(self.rows), "groups": sorted({r.group for r in self.rows}),
                "excluded": {k: len(v) for k, v in self.excluded.items()},
                "is_trainable": self.is_trainable, "report": self.report()}


#: The weakest evidence a training label may come from. HOMOLOG_EXPERIMENTAL is
#: the floor because it is the weakest tier that still means somebody measured
#: something; below it the "label" is a curated assertion.
DEFAULT_TIER_FLOOR = EvidenceTier.HOMOLOG_EXPERIMENTAL


def _tier_ok(tier: EvidenceTier, floor: EvidenceTier) -> bool:
    order = list(EvidenceTier)
    return order.index(tier) <= order.index(floor)


def build_dataset(measure: OutcomeMeasure,
                  store: Optional[VariantEvidenceStore] = None,
                  tier_floor: EvidenceTier = DEFAULT_TIER_FLOOR) -> DatasetBuild:
    """
    Rows for one measure, with every exclusion counted and explained.

    Exclusions are returned rather than logged. A builder that quietly drops
    four fifths of a store reports a clean dataset, and the number that matters
    -- how much of the evidence was usable -- is the one that disappears.
    """
    store = store or VariantEvidenceStore()
    build = DatasetBuild(measure=measure, tier_floor=tier_floor)

    for record in store.records:
        if not record.has_evidence:
            build.exclude(ExclusionReason.NO_MEASURED_OUTCOME, record.name)
            continue
        if not record.is_attributable:
            build.exclude(ExclusionReason.CONFOUNDED_COMBINATION, record.name)
            continue
        modification = record.modifications[0]
        for outcome in record.outcomes:
            label = f"{record.name}:{outcome.measure.value}"
            if outcome.measure is not measure:
                build.exclude(ExclusionReason.WRONG_MEASURE, label)
                continue
            if not _tier_ok(outcome.tier, tier_floor):
                build.exclude(ExclusionReason.BELOW_TIER_FLOOR, label)
                continue
            if outcome.fold_change is None and outcome.value is None:
                build.exclude(ExclusionReason.NOT_QUANTITATIVE, label)
                continue
            build.rows.append(TrainingRow(
                parent=record.parent or record.name, variant_name=record.name,
                modification_label=modification.label, position=modification.position,
                wild_type=modification.wild_type, mutant=modification.mutant,
                measure=outcome.measure,
                target_value=float(outcome.fold_change if outcome.fold_change is not None
                                   else outcome.value),
                target_is_fold_change=outcome.fold_change is not None,
                comparator=outcome.comparator, assay=outcome.assay, tier=outcome.tier))
    return build


# ---- leakage ----------------------------------------------------------------

@dataclass(frozen=True)
class LeakageReport:
    """Whether a split lets the model see its test set in training."""

    is_clean: bool
    shared_parents: Tuple[str, ...] = ()
    shared_variants: Tuple[str, ...] = ()

    def statement(self) -> str:
        if self.is_clean:
            return ("No parent molecule and no variant appears on both sides of the "
                    "split.")
        bits = []
        if self.shared_variants:
            bits.append(f"{len(self.shared_variants)} variant(s) appear in both train "
                        f"and test: {', '.join(sorted(self.shared_variants)[:5])}.")
        if self.shared_parents:
            bits.append(
                f"{len(self.shared_parents)} parent molecule(s) are split across train "
                f"and test: {', '.join(sorted(self.shared_parents)[:5])}. A variant "
                f"series shares a parent, so the model is scored on peptides it has "
                f"effectively already seen, and the number measures recall.")
        return " ".join(bits)


class LeakageError(ValueError):
    """Raised when a split would train and test on the same molecule."""


def check_leakage(train: Sequence[TrainingRow],
                  test: Sequence[TrainingRow]) -> LeakageReport:
    """
    Whether a proposed split keeps parent molecules whole.

    Splitting on the variant is not enough and is the trap this store sets: an
    alanine scan is one molecule wearing thirty names, and a variant-level split
    keeps twenty-nine of them in training.
    """
    train_parents: Set[str] = {r.group for r in train}
    test_parents: Set[str] = {r.group for r in test}
    train_variants = {r.variant_name for r in train}
    test_variants = {r.variant_name for r in test}
    shared_parents = tuple(sorted(train_parents & test_parents))
    shared_variants = tuple(sorted(train_variants & test_variants))
    return LeakageReport(is_clean=not shared_parents and not shared_variants,
                         shared_parents=shared_parents, shared_variants=shared_variants)


def require_no_leakage(train: Sequence[TrainingRow],
                       test: Sequence[TrainingRow]) -> LeakageReport:
    """`check_leakage`, but refuses rather than reporting. For use before training."""
    report = check_leakage(train, test)
    if not report.is_clean:
        raise LeakageError(report.statement())
    return report


def split_by_parent(rows: Sequence[TrainingRow], test_fraction: float = 0.2,
                    seed: int = 0) -> Tuple[List[TrainingRow], List[TrainingRow]]:
    """
    A grouped split that cannot leak by construction.

    Whole parent molecules go to one side or the other. The fraction is over
    groups rather than rows, so a parent with thirty variants does not drag the
    split with it -- and the resulting row counts will not match the requested
    fraction, which is the correct behaviour and not an approximation error.
    """
    import random as _random

    groups = sorted({r.group for r in rows})
    rng = _random.Random(seed)
    rng.shuffle(groups)
    n_test = max(1, round(len(groups) * test_fraction)) if groups else 0
    test_groups = set(groups[:n_test])
    train = [r for r in rows if r.group not in test_groups]
    test = [r for r in rows if r.group in test_groups]
    return train, test


def status_report(store: Optional[VariantEvidenceStore] = None) -> Dict[str, Any]:
    """Whether any measure in the store is trainable at all, and why not."""
    store = store or VariantEvidenceStore()
    builds = {m.value: build_dataset(m, store=store).to_dict() for m in OutcomeMeasure}
    trainable = [name for name, b in builds.items() if b["is_trainable"]]
    return {
        "store": store.status(),
        "trainable_measures": trainable,
        "statement": (
            "No measure in the variant-evidence store has usable training rows, because "
            "the store holds no records. A builder that manufactured rows to have "
            "something to return would be the failure this layer is written against."
            if not trainable else
            f"{len(trainable)} measure(s) have usable rows: {', '.join(trainable)}."),
        "builds": builds,
    }
