"""
Does fusion actually help?  [Addendum 3, section 5]

The comparison the addendum asks for: single modality, naive concatenation,
learned fusion, on held-out data, with an honest answer. The honest answer is
frequently "this experiment does not distinguish them", and the machinery here
is built to be able to say that.

The verdict rule is the deliverable. Two arms whose bootstrap intervals overlap
have not been distinguished by this test, however far apart their point
estimates are; reporting the larger number as the winner is the single most
common way a fusion result is oversold. So the rule is applied mechanically and
the intervals are printed beside the verdict.

Two modalities, both computed and neither learned:

- **sequence** — position-aware one-hot. Sees order, so it can see a motif.
- **physicochemistry** — net charge at pH 7.4, mean hydrophobicity, and the
  charged/aromatic/proline fractions. Sees bulk properties, so it cannot see a
  motif at all.

The label needs both to be satisfied, which is what makes the comparison worth
running rather than decorative. Note what that does NOT guarantee: the sequence
modality contains the information the physicochemical one summarises, so a
model with enough data could recover charge from one-hot alone. Finding that
fusion adds nothing here would therefore be a real result about these two
modalities rather than a failure of the machinery, and it is reported that way.

Synthetic by construction, like the split-gap experiment, and marked the same
way: it demonstrates that the comparison can be run and read, and supports no
biological claim whatever the numbers say.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..datasets.splitting import compare as split_compare
from ..evaluation.metrics import Metric, evaluate_classification
from ..multimodal.fusion import (
    Modality, build_fusion_model, concatenation_model, single_modality_model,
)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
MOTIF = "WKF"

# Kyte-Doolittle. Duplicated from the analysis engine rather than imported: the
# ML layer must not require a loaded policy artifact to run, and pulling the
# engine in would drag that requirement across the boundary.
HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
POSITIVE = set("KR")
NEGATIVE = set("DE")
AROMATIC = set("FWY")


def net_charge(sequence: str) -> float:
    """Charge at neutral pH, counted rather than titrated -- a bulk descriptor."""
    return sum(1 for a in sequence if a in POSITIVE) - sum(1 for a in sequence if a in NEGATIVE)


def physicochemistry(sequence: str) -> List[float]:
    length = max(len(sequence), 1)
    return [
        net_charge(sequence) / length,
        sum(HYDROPHOBICITY.get(a, 0.0) for a in sequence) / length,
        sum(1 for a in sequence if a in POSITIVE or a in NEGATIVE) / length,
        sum(1 for a in sequence if a in AROMATIC) / length,
        sequence.count("P") / length,
    ]


def one_hot(sequence: str, length: int) -> List[float]:
    vector = [0.0] * (length * len(AMINO_ACIDS))
    for i, residue in enumerate(sequence[:length]):
        index = AMINO_ACIDS.find(residue)
        if index >= 0:
            vector[i * len(AMINO_ACIDS) + index] = 1.0
    return vector


def build_dataset(n_families: int = 40, per_family: int = 8, length: int = 24,
                  seed: int = 0) -> Tuple[List[str], List[int]]:
    """
    Families of related sequences, with a label that needs both modalities.

    Positive iff the motif is present AND the net charge is positive. Neither
    modality alone is sufficient by construction: the physicochemical features
    cannot represent a motif, and the charge condition is a second, independent
    requirement.
    """
    rng = random.Random(seed)
    sequences: List[str] = []
    for family in range(n_families):
        founder = "".join(rng.choice(AMINO_ACIDS) for _ in range(length))
        if family % 2 == 0:
            at = rng.randrange(length - len(MOTIF))
            founder = founder[:at] + MOTIF + founder[at + len(MOTIF):]
        # Push roughly half the families to each side of the charge condition,
        # independently of whether they carry the motif, so the two conditions
        # are not the same condition wearing two names.
        if family % 4 < 2:
            founder = founder.replace("D", "K", 2).replace("E", "R", 2)
        else:
            founder = founder.replace("K", "D", 2).replace("R", "E", 2)
        for _ in range(per_family):
            member = list(founder)
            for _ in range(rng.randint(1, 3)):
                position = rng.randrange(length)
                motif_at = founder.find(MOTIF)
                if motif_at >= 0 and motif_at <= position < motif_at + len(MOTIF):
                    continue
                member[position] = rng.choice(AMINO_ACIDS)
            sequences.append("".join(member))
    labels = [1 if (MOTIF in s and net_charge(s) > 0) else 0 for s in sequences]
    return sequences, labels


@dataclass
class Arm:
    name: str
    kind: str                     # "single" | "concatenation" | "fusion"
    metrics: Dict[str, Metric]
    gate_share: Optional[List[float]] = None   # fusion only: mean weight per modality

    @property
    def auc(self) -> Optional[float]:
        return self.metrics["roc_auc"].value


@dataclass
class FusionComparison:
    arms: List[Arm] = field(default_factory=list)
    modalities: List[str] = field(default_factory=list)
    split: str = ""
    n_train: int = 0
    n_test: int = 0
    skipped: List[str] = field(default_factory=list)
    permutation_null: List[float] = field(default_factory=list)
    verdict: str = ""
    resolution: str = ""

    def best(self) -> Optional[Arm]:
        scored = [a for a in self.arms if a.auc is not None]
        return max(scored, key=lambda a: a.auc) if scored else None

    def report(self) -> str:
        lines = [
            f"Modalities: {', '.join(self.modalities)}",
            f"Split: {self.split} ({self.n_train} train, {self.n_test} test)",
            "",
        ]
        for arm in self.arms:
            lines.append(f"  {arm.name:>26s}  {arm.metrics['roc_auc']}")
            if arm.gate_share:
                shares = ", ".join(f"{m} {w:.0%}" for m, w in zip(self.modalities, arm.gate_share))
                lines.append(f"  {'':>26s}  mean gate weight: {shares}")
        if self.permutation_null:
            lines.append(
                f"  {'shuffled-label null':>26s}  {len(self.permutation_null)} runs, ROC-AUC "
                f"{min(self.permutation_null):.3f}–{max(self.permutation_null):.3f}")
        for note in self.skipped:
            lines.append(f"  SKIPPED  {note}")
        lines += ["", self.resolution, "", self.verdict, "",
                  "SYNTHETIC -- this demonstrates that the comparison can be run and read, "
                  "and supports no biological claim whatever the numbers say."]
        return "\n".join(lines)


def intervals_overlap(a: Metric, b: Metric) -> Optional[bool]:
    """
    Whether two metrics' bootstrap intervals overlap.

    None when either interval is missing, because "no interval" is not "no
    overlap" -- treating it as a clean separation is exactly the error this
    function exists to prevent.
    """
    if None in (a.ci_low, a.ci_high, b.ci_low, b.ci_high):
        return None
    return not (a.ci_high < b.ci_low or b.ci_high < a.ci_low)


def resolution_for(arms: Sequence[Arm], null: Sequence[float]) -> str:
    """
    Whether this test could have detected anything at all.

    "The intervals overlap" is only informative if the test has the resolution
    to separate arms that really differ, so the comparison runs against a null
    built by retraining on shuffled labels.

    A PERMUTATION null, not a single control run, and that distinction was
    learned the hard way: one shuffled-label run on this construction returned
    0.783 ROC-AUC, which looks like leakage and is not -- over ten shuffles the
    same arm spans roughly 0.28 to 0.56 around a mean of 0.46. A single draw
    from a distribution that wide cannot certify anything, and using one as a
    resolution check would have reported an under-powered test as a leaking
    one, or the reverse, depending on the seed.
    """
    real = [a for a in arms if a.auc is not None]
    if not null or not real:
        return "No shuffled-label null completed, so the resolution of this test is unknown."

    best = max(real, key=lambda a: a.auc)
    ceiling = max(null)
    if best.auc > ceiling:
        return (
            f"Resolution check: the best arm ({best.name}, {best.auc:.3f}) beats all "
            f"{len(null)} shuffled-label runs (highest {ceiling:.3f}), so this test can "
            f"detect a real effect — p < {1 / (len(null) + 1):.2f} by permutation, which is "
            f"as fine as {len(null)} shuffles can resolve."
        )
    beaten = sum(1 for value in null if best.auc > value)
    return (
        f"Under-powered: the best arm ({best.name}, {best.auc:.3f}) beats only {beaten} of "
        f"{len(null)} shuffled-label runs (highest {ceiling:.3f}). Nothing below distinguishes "
        f"any two arms, and that is a fact about this test set's size rather than about the "
        f"architectures."
    )


def verdict_for(arms: Sequence[Arm]) -> str:
    """
    Whether this experiment distinguished the fusion arm from its baselines.

    The rule is mechanical and applied to every comparison, not only the ones
    where it is convenient.
    """
    fusion = next((a for a in arms if a.kind == "fusion" and a.auc is not None), None)
    if fusion is None:
        return "No fusion arm completed, so this experiment says nothing about fusion."

    baselines = [a for a in arms if a.kind != "fusion" and a.auc is not None]
    if not baselines:
        return "No baseline completed, so the fusion arm has nothing to be compared against."

    best_baseline = max(baselines, key=lambda a: a.auc)
    overlap = intervals_overlap(fusion.metrics["roc_auc"], best_baseline.metrics["roc_auc"])
    delta = fusion.auc - best_baseline.auc

    if overlap is None:
        return (
            f"Fusion {fusion.auc:.3f} against {best_baseline.name} {best_baseline.auc:.3f}, but "
            f"at least one interval could not be computed. No comparison is claimed: a missing "
            f"interval is not a narrow one."
        )
    if overlap:
        note = ""
        if fusion.gate_share and max(fusion.gate_share) > 0.95:
            note = (f" Its gate also put {max(fusion.gate_share):.0%} of the weight on one "
                    f"modality, so the fusion mechanism was not being exercised.")
        return (
            f"This experiment does not distinguish learned fusion ({fusion.auc:.3f}) from "
            f"{best_baseline.name} ({best_baseline.auc:.3f}): the bootstrap intervals overlap, "
            f"so the {delta:+.3f} difference is within what resampling this test set produces. "
            f"No claim that fusion helps.{note}"
        )
    if fusion.gate_share and max(fusion.gate_share) > 0.95:
        dominant = fusion.gate_share.index(max(fusion.gate_share))
        return (
            f"Learned fusion ({fusion.auc:.3f}) separates from {best_baseline.name} "
            f"({best_baseline.auc:.3f}) by {delta:+.3f}, but its gate put "
            f"{max(fusion.gate_share):.0%} of the weight on a single modality, so what is "
            f"being compared is that modality with extra layers rather than a mixture."
        )
    direction = "above" if delta > 0 else "below"
    return (
        f"Learned fusion ({fusion.auc:.3f}) sits {direction} {best_baseline.name} "
        f"({best_baseline.auc:.3f}) with non-overlapping bootstrap intervals, a difference of "
        f"{delta:+.3f} on this test set. That is one held-out set and one construction, so it "
        f"is evidence about this comparison, not about fusion in general."
    )


def _train(model, parts_train, y_train, parts_test, seed: int, epochs: int = 120,
           fused: bool = False):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    xs = [torch.tensor(p, dtype=torch.float32) for p in parts_train]
    yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    inputs = xs if fused else torch.cat(xs, dim=1)

    for _ in range(epochs):
        optimiser.zero_grad()
        loss = loss_fn(model(inputs), yt)
        loss.backward()
        optimiser.step()

    model.eval()
    with torch.no_grad():
        xt = [torch.tensor(p, dtype=torch.float32) for p in parts_test]
        out = model(xt if fused else torch.cat(xt, dim=1))
        scores = torch.sigmoid(out).squeeze(1).tolist()
    gate = None
    if fused and getattr(model, "last_gate", None) is not None:
        gate = model.last_gate.mean(dim=0).tolist()
    return scores, gate


def run(n_families: int = 40, per_family: int = 8, length: int = 24,
        threshold: float = 0.8, seed: int = 0,
        n_permutations: int = 10) -> FusionComparison:
    """
    Run all four arms on the sequence-clustered split.

    Clustered rather than random, deliberately: a fusion comparison on a random
    split over redundant families measures which architecture memorises fastest.
    """
    sequences, labels = build_dataset(n_families, per_family, length, seed)
    comparison = split_compare(sequences, seed=seed, threshold=threshold, labels=labels)
    split = comparison.clustered

    modalities = [
        Modality("sequence", lambda s: one_hot(s, length),
                 "position-aware one-hot; sees order, so it can see a motif"),
        Modality("physicochemistry", physicochemistry,
                 "charge, hydrophobicity and composition fractions; cannot see a motif"),
    ]
    result = FusionComparison(
        modalities=[m.name for m in modalities],
        split=split.strategy.value,
        n_train=len(split.train), n_test=len(split.test),
    )

    y_train = [labels[i] for i in split.train]
    y_test = [labels[i] for i in split.test]
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        result.skipped.append(
            f"every arm: the clustered partition is single-class "
            f"(train {sorted(set(y_train))}, test {sorted(set(y_test))}). Too few clusters "
            f"at this scale for a usable held-out set."
        )
        result.verdict = verdict_for(result.arms)
        result.resolution = resolution_for(result.arms, result.permutation_null)
        return result

    parts_train = [[m.featurise(sequences[i]) for i in split.train] for m in modalities]
    parts_test = [[m.featurise(sequences[i]) for i in split.test] for m in modalities]
    dims = [len(parts_train[i][0]) for i in range(len(modalities))]

    def record(name, kind, scores, gate=None):
        result.arms.append(Arm(
            name=name, kind=kind, gate_share=gate,
            metrics=evaluate_classification(y_test, scores, n_resamples=300, seed=seed)))

    for i, modality in enumerate(modalities):
        scores, _ = _train(single_modality_model(dims[i]),
                           [parts_train[i]], y_train, [parts_test[i]], seed=seed)
        record(f"{modality.name} only", "single", scores)

    scores, _ = _train(concatenation_model(dims), parts_train, y_train, parts_test, seed=seed)
    record("naive concatenation", "concatenation", scores)

    scores, gate = _train(build_fusion_model(dims), parts_train, y_train, parts_test,
                          seed=seed, fused=True)
    record("learned fusion", "fusion", scores, gate)

    # The null: the same architecture and budget, retrained on shuffled labels,
    # several times. Several because one run of it spans a quarter of the AUC
    # range on a test set this size -- see resolution_for.
    from ..evaluation.metrics import roc_auc
    for k in range(n_permutations):
        shuffled = list(y_train)
        random.Random(seed * 1000 + k).shuffle(shuffled)
        scores, _ = _train(concatenation_model(dims), parts_train, shuffled, parts_test,
                           seed=seed + k)
        value = roc_auc(y_test, scores)
        if value is not None:
            result.permutation_null.append(value)

    result.verdict = verdict_for(result.arms)
    result.resolution = resolution_for(result.arms, result.permutation_null)
    return result
