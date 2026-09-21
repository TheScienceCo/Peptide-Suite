"""
Random versus sequence-clustered splitting.  [Addendum 3, section 4]

Random peptide splits overestimate generalization, and the mechanism is
specific rather than general: peptide datasets are full of near-duplicates --
alanine scans, single-point variants, truncation series, the same peptide from
two papers. A random split puts a peptide in train and its single-point mutant
in test, and the model gets credit for recalling a sequence it has already
seen. The reported number is real; what it measures is memorisation.

Clustering by sequence identity first, then splitting whole clusters, removes
that. The gap between the two numbers is the interesting quantity, and it is
usually large enough that a paper reporting only the random number is reporting
a different result than it thinks.

Both splits are produced so the gap can be shown rather than asserted.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class SplitStrategy(Enum):
    RANDOM = "random"
    CLUSTERED = "sequence_clustered"

    @property
    def measures(self) -> str:
        return {
            SplitStrategy.RANDOM:
                "performance on sequences that may have near-duplicates in training. "
                "Optimistic by an amount that depends on how redundant the dataset is.",
            SplitStrategy.CLUSTERED:
                "performance on sequence families held out entirely. This is the number "
                "that estimates generalization to a new peptide.",
        }[self]


def identity(a: str, b: str) -> float:
    """
    Fractional identity between two sequences, ungapped, over the shorter one.

    Deliberately simple and deliberately generous: a real pipeline would align.
    Being generous is the safe direction here, because over-estimating identity
    puts more pairs in the same cluster, which makes the held-out split harder
    rather than easier. A clustering that errs toward merging cannot inflate
    the clustered score.
    """
    a, b = (a or "").upper(), (b or "").upper()
    if not a or not b:
        return 0.0
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    best = 0
    for offset in range(len(long_) - len(short) + 1):
        matches = sum(1 for i, ch in enumerate(short) if long_[offset + i] == ch)
        best = max(best, matches)
    return best / len(short)


def cluster_sequences(sequences: Sequence[str], threshold: float) -> List[List[int]]:
    """
    Greedy single-linkage clustering by identity, in the spirit of CD-HIT.

    Sequences are processed longest first so the longest member seeds its
    cluster, which is the convention CD-HIT uses and makes the representative
    the most informative member rather than an arbitrary one.

    Single-linkage and transitive: if A is similar to B and B to C, all three
    go together even when A and C are not. That over-merges, and over-merging
    is the safe direction -- it makes the held-out evaluation harder, never
    easier.

    Transitivity has to be real rather than approximate. The guarantee this
    function exists to provide is that no two sequences in different clusters
    are within `threshold` of each other, and a greedy first-match assignment
    does not provide it.
    """
    n = len(sequences)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    # Every similar pair is united, not just the first match found. A greedy
    # first-match assignment does NOT give the property this function claims:
    # a sequence similar to members of two different clusters joins only the
    # one checked first, and the pair it left behind then straddles the split.
    # That leaked three sequences past a clustered split before this was
    # union-find.
    for i in range(n):
        for j in range(i + 1, n):
            if identity(sequences[i], sequences[j]) >= threshold:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    # Ordered by longest member first, which keeps the representative the most
    # informative one and makes the output deterministic.
    return sorted(groups.values(),
                  key=lambda members: (-max(len(sequences[i] or "") for i in members),
                                       members[0]))


@dataclass
class Split:
    """One train/validation/test partition, with what produced it."""
    strategy: SplitStrategy
    train: List[int]
    validation: List[int]
    test: List[int]
    seed: int
    identity_threshold: Optional[float] = None
    n_clusters: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    @property
    def sizes(self) -> Dict[str, int]:
        return {"train": len(self.train), "validation": len(self.validation),
                "test": len(self.test)}

    def leakage_against(self, sequences: Sequence[str], threshold: float) -> int:
        """
        Test sequences with a near-neighbour in train, above `threshold`.

        The measurement that makes the argument concrete. For a random split
        this is usually not zero, and the count is how much of the reported
        score could be recall.
        """
        leaked = 0
        for test_index in self.test:
            if any(identity(sequences[test_index], sequences[train_index]) >= threshold
                   for train_index in self.train):
                leaked += 1
        return leaked


def _partition(items: List, fractions: Tuple[float, float, float]) -> Tuple[List, List, List]:
    train_end = int(len(items) * fractions[0])
    val_end = train_end + int(len(items) * fractions[1])
    return items[:train_end], items[train_end:val_end], items[val_end:]


def random_split(sequences: Sequence[str], seed: int,
                 fractions: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> Split:
    """A shuffled split. Deterministic given the seed."""
    indices = list(range(len(sequences)))
    random.Random(seed).shuffle(indices)
    train, validation, test = _partition(indices, fractions)
    return Split(strategy=SplitStrategy.RANDOM, train=train, validation=validation,
                 test=test, seed=seed,
                 notes=["Near-duplicates may straddle train and test. Any score from "
                        "this split includes whatever the model recalls rather than "
                        "generalises."])


def clustered_split(sequences: Sequence[str], seed: int, threshold: float,
                    fractions: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                    labels: Optional[Sequence] = None) -> Split:
    """
    A split where whole clusters go to one side.

    Clusters are shuffled, not sorted by size: assigning the largest clusters
    to train first would make the split deterministic in a way that correlates
    cluster size with partition, and the test set would end up systematically
    composed of small, unusual families.
    """
    clusters = cluster_sequences(sequences, threshold)
    rng = random.Random(seed)

    if labels is None:
        order = list(range(len(clusters)))
        rng.shuffle(order)
        train_c, val_c, test_c = _partition(order, fractions)
    else:
        # Stratify by label. Without this a small dataset routinely produces a
        # single-class test partition, and every metric on it is undefined --
        # which looks like the experiment not running rather than the split
        # being unusable.
        by_label: Dict[object, List[int]] = {}
        for cluster_id, members in enumerate(clusters):
            key = labels[members[0]]
            by_label.setdefault(key, []).append(cluster_id)
        train_c, val_c, test_c = [], [], []
        for key in sorted(by_label, key=str):
            group = by_label[key]
            rng.shuffle(group)
            a, b, c = _partition(group, fractions)
            train_c += a
            val_c += b
            test_c += c

    def flatten(cluster_ids):
        return sorted(i for cid in cluster_ids for i in clusters[cid])

    return Split(strategy=SplitStrategy.CLUSTERED, train=flatten(train_c),
                 validation=flatten(val_c), test=flatten(test_c), seed=seed,
                 identity_threshold=threshold, n_clusters=len(clusters),
                 notes=[f"{len(clusters)} clusters at {threshold:.0%} identity. Whole "
                        f"clusters are held out, so no test sequence has a near-duplicate "
                        f"in training."])


@dataclass
class SplitComparison:
    """
    The two splits side by side, and the leakage that separates them.

    Held as one object because reporting either number alone is the failure
    this module exists to prevent.
    """
    random: Split
    clustered: Split
    sequences: Sequence[str]
    threshold: float

    def leakage(self) -> Dict[str, int]:
        return {"random": self.random.leakage_against(self.sequences, self.threshold),
                "clustered": self.clustered.leakage_against(self.sequences, self.threshold)}

    def report(self) -> str:
        leak = self.leakage()
        random_test = max(len(self.random.test), 1)
        return (
            f"Random split: {leak['random']} of {len(self.random.test)} test sequences "
            f"({leak['random'] / random_test:.0%}) have a >={self.threshold:.0%}-identity "
            f"neighbour in training. Clustered split: {leak['clustered']}. "
            f"{self.clustered.n_clusters} clusters from {len(self.sequences)} sequences. "
            f"Any gap between the two scores is the part of the random number that was "
            f"recall."
        )


def compare(sequences: Sequence[str], seed: int, threshold: float,
            fractions: Tuple[float, float, float] = (0.7, 0.15, 0.15),
            labels: Optional[Sequence] = None) -> SplitComparison:
    """
    Both splits over the same data.

    Pass `labels` to stratify the clustered split. On a small dataset an
    unstratified clustered split routinely yields a single-class test set,
    which makes every metric undefined -- and an undefined metric reads as the
    experiment not running rather than as the split being unusable.
    """
    return SplitComparison(
        random=random_split(sequences, seed, fractions),
        clustered=clustered_split(sequences, seed, threshold, fractions, labels=labels),
        sequences=list(sequences), threshold=threshold)
