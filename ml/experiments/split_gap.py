"""
Why sequence-aware evaluation matters, demonstrated rather than asserted.
[Addendum 3, section 4 and section 12]

The claim: a random split over a redundant peptide dataset reports a score that
is partly recall, and a clustered split does not. The claim is about the
EVALUATION PROTOCOL, not about biology, so it can be settled on constructed
sequences -- and it has to be, because settling it on a real dataset would
confound the protocol question with whatever the model does or does not learn.

The construction: families of related sequences, and a label that is a genuine
function of the sequence (a motif is present or absent). The label is learnable,
so a model that generalises will score well on both splits. If the random score
exceeds the clustered score, the excess is recall of near-duplicates, because
by construction there is nothing else it can be.

This is marked SYNTHETIC_METHOD_ONLY in the dataset registry and supports no
biological claim whatever the numbers say.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from ..datasets.splitting import SplitStrategy, compare
from ..evaluation.metrics import Metric, evaluate_classification

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
MOTIF = "WKF"


def build_dataset(n_families: int = 40, per_family: int = 8, length: int = 24,
                  seed: int = 0) -> Tuple[List[str], List[int]]:
    """
    Redundant families with a learnable label.

    Half the families carry a motif and half do not; members of a family are
    point mutants of its founder. The redundancy is the realistic part -- an
    alanine scan or a truncation series produces exactly this structure -- and
    the motif makes the label a real function of the sequence rather than noise.
    """
    rng = random.Random(seed)
    sequences, labels = [], []
    for family in range(n_families):
        founder = "".join(rng.choice(AMINO_ACIDS) for _ in range(length))
        positive = family % 2 == 0
        if positive:
            at = rng.randrange(length - len(MOTIF))
            founder = founder[:at] + MOTIF + founder[at + len(MOTIF):]
        else:
            founder = founder.replace(MOTIF, "AAA")
        for _ in range(per_family):
            member = list(founder)
            # Mutate away from the motif so a family stays one label.
            for _ in range(rng.randint(1, 3)):
                position = rng.randrange(length)
                if positive and founder.find(MOTIF) <= position < founder.find(MOTIF) + len(MOTIF):
                    continue
                member[position] = rng.choice(AMINO_ACIDS)
            candidate = "".join(member)
            if not positive and MOTIF in candidate:
                candidate = candidate.replace(MOTIF, "AAA")
            sequences.append(candidate)
            labels.append(1 if positive else 0)
    return sequences, labels


def one_hot(sequence: str, length: int) -> List[float]:
    vector = [0.0] * (length * len(AMINO_ACIDS))
    for i, residue in enumerate(sequence[:length]):
        index = AMINO_ACIDS.find(residue)
        if index >= 0:
            vector[i * len(AMINO_ACIDS) + index] = 1.0
    return vector


def composition_baseline(sequence: str) -> List[float]:
    """
    Amino-acid composition: the physicochemical-feature baseline.

    Included because it is the arm that most often embarrasses a deep model.
    Composition cannot see order, so it cannot represent a motif at all -- which
    makes it the right control for a motif task and the honest floor.
    """
    length = max(len(sequence), 1)
    return [sequence.count(a) / length for a in AMINO_ACIDS]


@dataclass
class ArmResult:
    name: str
    split: str
    metrics: Dict[str, Metric]

    @property
    def auc(self):
        return self.metrics["roc_auc"].value


@dataclass
class SplitGapResult:
    """The two splits, the two arms, and the gap between them."""
    arms: List[ArmResult] = field(default_factory=list)
    leakage: Dict[str, int] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)
    n_sequences: int = 0
    n_clusters: int = 0
    threshold: float = 0.0

    def gap_for(self, arm: str):
        random_auc = next((a.auc for a in self.arms
                           if a.name == arm and a.split == "random"), None)
        clustered_auc = next((a.auc for a in self.arms
                              if a.name == arm and a.split == "sequence_clustered"), None)
        if random_auc is None or clustered_auc is None:
            return None
        return random_auc - clustered_auc

    def report(self) -> str:
        lines = [
            f"{self.n_sequences} sequences in {self.n_clusters} clusters at "
            f"{self.threshold:.0%} identity.",
            f"Test sequences with a near-duplicate in train: "
            f"{self.leakage.get('random')} under a random split, "
            f"{self.leakage.get('clustered')} under a clustered one.",
            "",
        ]
        for arm in self.arms:
            lines.append(f"  {arm.name:>22s}  {arm.split:<20s} {arm.metrics['roc_auc']}")
        lines.append("")
        for name in sorted({a.name for a in self.arms}):
            gap = self.gap_for(name)
            if gap is not None:
                lines.append(f"  {name}: random exceeds clustered by {gap:+.3f} AUC")
        for note in self.skipped:
            lines.append(f"  SKIPPED  {note}")
        if self.skipped:
            lines.append("")
        lines.append(
            "Any positive gap is recall of near-duplicates: the label is a function of "
            "the sequence and both splits see the same label rule, so there is nothing "
            "else the difference can be. SYNTHETIC -- this demonstrates the evaluation "
            "protocol and supports no biological claim.")
        return "\n".join(lines)


def _train_mlp(x_train, y_train, x_test, seed: int, epochs: int = 60):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(len(x_train[0]), 32), nn.ReLU(), nn.Linear(32, 1))
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    for _ in range(epochs):
        optimiser.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        optimiser.step()

    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.tensor(x_test, dtype=torch.float32)))
    return scores.squeeze(1).tolist()


def run(n_families: int = 40, per_family: int = 8, length: int = 24,
        threshold: float = 0.8, seed: int = 0) -> SplitGapResult:
    sequences, labels = build_dataset(n_families, per_family, length, seed)
    comparison = compare(sequences, seed=seed, threshold=threshold, labels=labels)
    leakage = comparison.leakage()

    result = SplitGapResult(
        leakage=leakage, n_sequences=len(sequences),
        n_clusters=comparison.clustered.n_clusters or 0, threshold=threshold)

    featurisers = {
        "composition baseline": lambda s: composition_baseline(s),
        "one-hot MLP": lambda s: one_hot(s, length),
    }

    for split in (comparison.random, comparison.clustered):
        for arm_name, featurise in featurisers.items():
            x_train = [featurise(sequences[i]) for i in split.train]
            y_train = [labels[i] for i in split.train]
            x_test = [featurise(sequences[i]) for i in split.test]
            y_test = [labels[i] for i in split.test]
            if len(set(y_train)) < 2 or len(set(y_test)) < 2:
                # Recorded, not skipped silently. An arm that vanishes looks
                # like the experiment not running; saying why is the
                # difference between a partial result and a missing one.
                result.skipped.append(
                    f"{arm_name} / {split.strategy.value}: partition is single-class "
                    f"(train {sorted(set(y_train))}, test {sorted(set(y_test))}). Too few "
                    f"clusters at this scale for a usable held-out set.")
                continue
            scores = _train_mlp(x_train, y_train, x_test, seed=seed)
            result.arms.append(ArmResult(
                name=arm_name, split=split.strategy.value,
                metrics=evaluate_classification(y_test, scores, n_resamples=300, seed=seed)))
    return result
