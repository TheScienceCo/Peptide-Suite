"""
Nearest examples and out-of-distribution warning.  [Addendum 3, section 6]

Two of the four explainability asks are answerable without a trained model,
and those are the two built here. "Which sequences in the reference set is
this one closest to" and "is this one further out than the reference set's own
members are from each other" are questions about a representation, not about a
prediction, so they work the moment an encoder exists.

The third ask, residue-level sensitivity, is already answered by the
substitution landscape's column marginal: the most favourable computed
substitution at each position, over the real pipeline rather than over a model.
It is not duplicated here.

The fourth, attribution, is deliberately absent. Attribution explains a model's
output, and no model in this deployment is trained on real labels -- the only
one available is the synthetic split-gap demonstration. Attribution over it
would attribute a model's recall of constructed sequences, which is a picture
of nothing. It arrives with the labelled data, not before.

The warning this module exists to give is a specific one. A prediction about a
sequence unlike anything the reference set contains is an extrapolation, and a
model will make it with exactly the same confidence it uses for an
interpolation. The distance is computable; the confidence is not corrected by
it; so the distance is reported beside the prediction rather than folded into
it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..embeddings.encoder import Embedding, EncoderKind

# A percentile over fewer than this many reference distances is not a
# distribution, it is a handful of numbers with a percentage sign on it.
MIN_REFERENCE = 8


class NeighbourError(ValueError):
    pass


@dataclass(frozen=True)
class Neighbour:
    label: str
    sequence: str
    distance: float
    rank: int


@dataclass
class OutOfDistributionReport:
    """
    How far outside the reference set a query sits, in the reference set's own
    terms.

    A raw distance answers nothing -- 0.4 is close in one space and remote in
    another. What carries information is where that distance falls in the
    distribution of distances the reference set already contains, which is a
    statement the reference set itself supplies.
    """
    query_label: str
    nearest: List[Neighbour]
    distance_to_nearest: float
    reference_median_nn_distance: float
    percentile: float            # share of reference members closer to their own nearest
    is_outside: bool
    n_coincident: int            # reference members whose nearest neighbour is at distance 0
    encoder_model: str
    encoder_has_learned_content: bool
    verdict: str
    caveat: str


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _require_one_space(embeddings: Sequence[Embedding]) -> Embedding:
    head = embeddings[0]
    for other in embeddings[1:]:
        if not head.comparable_with(other):
            raise NeighbourError(
                f"Embeddings come from different encoders ({head.model} vs {other.model}). "
                f"A distance between two unrelated spaces is a number with no meaning, and "
                f"it will look exactly like a meaningful one."
            )
    return head


def nearest_examples(
    query: Embedding,
    reference: Sequence[Embedding],
    labels: Sequence[str],
    k: int = 5,
) -> List[Neighbour]:
    """The k closest reference sequences, by Euclidean distance in the pooled space."""
    if len(reference) != len(labels):
        raise NeighbourError("Every reference embedding needs a label.")
    if not reference:
        raise NeighbourError("No reference set to search.")
    _require_one_space([query, *reference])

    scored = sorted(
        ((_distance(query.pooled, r.pooled), label, r.sequence)
         for r, label in zip(reference, labels)),
        key=lambda t: (t[0], t[1]),
    )
    return [Neighbour(label=label, sequence=sequence, distance=distance, rank=i + 1)
            for i, (distance, label, sequence) in enumerate(scored[:k])]


def out_of_distribution(
    query: Embedding,
    reference: Sequence[Embedding],
    labels: Sequence[str],
    query_label: str = "query",
    k: int = 5,
    percentile_cutoff: float = 0.95,
) -> OutOfDistributionReport:
    """
    Place the query's distance-to-nearest inside the reference set's own
    nearest-neighbour distance distribution.

    The cutoff is a reporting convention rather than a fitted quantity -- it
    says which side of the reference set's own spread the query falls on, and
    changing it changes a label, not a number. Both the percentile and the
    distance are returned so a reader can ignore the label entirely.
    """
    if len(reference) < MIN_REFERENCE:
        raise NeighbourError(
            f"{len(reference)} reference sequences is too few to say what 'outside' means; "
            f"{MIN_REFERENCE} is the floor. Without a distribution to compare against, a "
            f"distance is just a number, and calling a sequence out-of-distribution from it "
            f"would be a verdict with nothing behind it."
        )
    head = _require_one_space([query, *reference])

    neighbours = nearest_examples(query, reference, labels, k=k)
    distance = neighbours[0].distance

    # Each reference member's distance to its own nearest neighbour, excluding
    # itself. This is the spread the query is being judged against.
    internal: List[float] = []
    for i, member in enumerate(reference):
        others = [_distance(member.pooled, other.pooled)
                  for j, other in enumerate(reference) if j != i]
        internal.append(min(others))
    internal.sort()
    median = internal[len(internal) // 2]
    # An encoder that maps distinct sequences to one vector puts zeros in this
    # distribution, which drags the median to zero and makes every query look
    # far out. That is a fact about the encoder, and it is reported rather than
    # smoothed away.
    coincident = sum(1 for d in internal if d == 0.0)
    closer = sum(1 for d in internal if d < distance)
    percentile = closer / len(internal)
    is_outside = percentile >= percentile_cutoff

    if is_outside:
        verdict = (
            f"Further from this reference set than {percentile:.0%} of its own members are "
            f"from their nearest neighbour. A prediction about this sequence would be an "
            f"extrapolation, and a model will make it with the same confidence it uses for "
            f"an interpolation."
        )
    else:
        verdict = (
            f"Sits inside the reference set's own spread: {percentile:.0%} of members are "
            f"closer to their nearest neighbour than this sequence is to its nearest. That "
            f"is not evidence a prediction would be correct, only that it would not be an "
            f"extrapolation."
        )

    if coincident:
        verdict += (
            f" {coincident} of {len(internal)} reference sequences sit exactly on top of "
            f"another, because this encoder maps them to the same vector; that pulls the "
            f"comparison distribution toward zero and makes any query look further out "
            f"than it would in a space that separates them."
        )

    caveat = (
        "Distance here is measured in a space with no learned content, so it reflects "
        "sequence composition or position and nothing about function. Being 'inside the "
        "distribution' of a descriptor space is a weaker statement than being inside a "
        "learned one."
        if head.kind is not EncoderKind.PRETRAINED_LANGUAGE_MODEL else
        "Distance is measured in the pretrained model's space, so it reflects what that "
        "model learned rather than functional similarity."
    )

    return OutOfDistributionReport(
        query_label=query_label,
        nearest=neighbours,
        distance_to_nearest=distance,
        reference_median_nn_distance=median,
        percentile=percentile,
        is_outside=is_outside,
        n_coincident=coincident,
        encoder_model=head.model,
        encoder_has_learned_content=head.kind.has_learned_content,
        verdict=verdict,
        caveat=caveat,
    )
