"""
Representation explorer: variants projected into two dimensions.
[Addendum 3, section 2 -- "PCA/UMAP visualization of WT vs single vs
multi-substitution candidates in representation space"]

A two-dimensional scatter of high-dimensional vectors is the most
over-interpreted figure in computational biology. Points that land near each
other look related; the plot says nothing about whether they are, and usually
nothing about how much of the original variation survived the projection. So
three things ride with every projection here:

1. **Explained variance, per component and cumulative.** Without it the axes
   are unlabelled. A PC1+PC2 that carries 12% of the variance produces a
   picture in which proximity means almost nothing, and the caller is entitled
   to know that before reading anything off it.

2. **Which encoder made the vectors.** Distance in a deterministic descriptor
   space is a statement about amino-acid composition, not about function. In a
   pretrained language-model space it is a statement about what the model
   learned. Those are different claims and the projection carries which one it
   is making.

3. **A refusal to mix spaces.** Two vectors of equal dimension from different
   models occupy unrelated spaces, and a PCA over the mixture produces axes
   that mean nothing at all. That is checked rather than assumed.

PCA rather than UMAP, deliberately. UMAP's layout depends on hyperparameters
that change cluster structure, has no explained-variance analogue to report,
and its distances are not metric -- everything above would become unstatable.
When the pretrained encoder is available and a UMAP is genuinely wanted, it
belongs beside a PCA, not instead of one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from .encoder import Embedding, EncoderKind

# Two components: this exists to be plotted. More would be a different tool.
N_COMPONENTS = 2

# Below this many points a principal-component direction is not estimated, it
# is interpolated between the few points present.
MIN_POINTS = 4


class VariantClass(str, Enum):
    """
    How far a sequence is from the reference, by edit count rather than by
    anything learned. Three classes because that is the comparison the
    addendum asks for; the count itself rides along so a reader is not left
    guessing where "multi" starts.
    """
    WILD_TYPE = "WILD_TYPE"
    SINGLE = "SINGLE"
    MULTI = "MULTI"
    UNRELATED = "UNRELATED"     # different length: no substitution count exists


class ProjectionError(ValueError):
    pass


@dataclass(frozen=True)
class ProjectedPoint:
    label: str
    sequence: str
    x: float
    y: float
    variant_class: VariantClass
    n_substitutions: Optional[int]   # None when the sequences are not alignable by index


@dataclass
class Projection:
    points: List[ProjectedPoint]
    explained_variance_ratio: List[float]
    encoder_model: str
    encoder_version: str
    encoder_kind: str
    encoder_has_learned_content: bool
    input_dim: int
    interpretation: str
    warnings: List[str] = field(default_factory=list)

    @property
    def cumulative_explained(self) -> float:
        return sum(self.explained_variance_ratio)


def classify_variant(reference: str, sequence: str) -> Tuple[VariantClass, Optional[int]]:
    """
    Edit distance by position, for equal-length sequences only.

    Unequal lengths are reported as UNRELATED with no count rather than being
    forced through an alignment: a count produced by an alignment this module
    did not compute would be a number with no derivation behind it.
    """
    if sequence == reference:
        return VariantClass.WILD_TYPE, 0
    if len(sequence) != len(reference):
        return VariantClass.UNRELATED, None
    n = sum(1 for a, b in zip(reference, sequence) if a != b)
    return (VariantClass.SINGLE if n == 1 else VariantClass.MULTI), n


def project(
    embeddings: Sequence[Embedding],
    labels: Sequence[str],
    reference: str,
) -> Projection:
    """
    Project pooled embeddings onto their first two principal components.

    Implemented on the SVD of the mean-centred matrix rather than on an
    eigendecomposition of a covariance estimate: same answer, better
    conditioning, and the singular values give the explained variance directly.
    """
    import numpy as np

    if len(embeddings) != len(labels):
        raise ProjectionError("Every embedding needs a label.")
    if len(embeddings) < MIN_POINTS:
        raise ProjectionError(
            f"{len(embeddings)} points is too few to estimate a principal-component "
            f"direction; {MIN_POINTS} is the floor. A projection of three points is a "
            f"drawing, not a decomposition."
        )

    head = embeddings[0]
    for other in embeddings[1:]:
        if not head.comparable_with(other):
            raise ProjectionError(
                f"Embeddings come from different encoders ({head.model} "
                f"{head.model_version} vs {other.model} {other.model_version}). Equal "
                f"dimensionality is not enough: the two occupy unrelated spaces and a "
                f"PCA over the mixture produces axes that mean nothing."
            )

    matrix = np.asarray([e.pooled for e in embeddings], dtype=float)
    if matrix.ndim != 2:
        raise ProjectionError("Pooled embeddings must all be vectors of the same length.")

    centred = matrix - matrix.mean(axis=0, keepdims=True)
    total_variance = float((centred ** 2).sum())
    # Relative, not absolute. Centring a set of identical vectors leaves
    # rounding dust rather than exact zeros -- a mean computed as a sum over n
    # is not bitwise equal to the value it averages -- and an exact `<= 0`
    # test lets that dust through as a decomposition of noise, which projects
    # to a plot of pure floating-point error presented as structure.
    scale = float((matrix ** 2).sum())
    if total_variance <= max(1e-24, scale * 1e-12):
        raise ProjectionError(
            "Every embedding in this set is identical, so there is no variation to "
            "decompose. A projection here would place every point at the origin and "
            "look like perfect agreement."
        )

    u, singular, _ = np.linalg.svd(centred, full_matrices=False)
    variances = singular ** 2
    ratios = [float(v / variances.sum()) for v in variances[:N_COMPONENTS]]
    while len(ratios) < N_COMPONENTS:
        ratios.append(0.0)

    # Sign is arbitrary in an SVD, so fix it: without this the plot mirrors
    # itself between runs on the same data and looks like a different result.
    scores = u[:, :N_COMPONENTS] * singular[:N_COMPONENTS]
    for axis in range(scores.shape[1]):
        column = scores[:, axis]
        if column[int(np.argmax(np.abs(column)))] < 0:
            scores[:, axis] = -column

    points: List[ProjectedPoint] = []
    for i, (embedding, label) in enumerate(zip(embeddings, labels)):
        variant, n = classify_variant(reference, embedding.sequence)
        points.append(ProjectedPoint(
            label=label, sequence=embedding.sequence,
            x=float(scores[i, 0]), y=float(scores[i, 1]),
            variant_class=variant, n_substitutions=n))

    warnings: List[str] = []
    cumulative = sum(ratios)
    if cumulative < 0.5:
        warnings.append(
            f"The two components shown carry {cumulative:.0%} of the variation in these "
            f"vectors. Most of what distinguishes these sequences is in the directions "
            f"not drawn, so proximity on this plot is weak evidence of similarity."
        )
    if head.kind is not EncoderKind.PRETRAINED_LANGUAGE_MODEL:
        warnings.append(
            "These vectors have no learned content. Distance here is a statement about "
            "amino-acid composition and nothing else — in particular it is not a "
            "statement about function, activity or binding."
        )

    return Projection(
        points=points,
        explained_variance_ratio=ratios,
        encoder_model=head.model,
        encoder_version=head.model_version,
        encoder_kind=head.kind.value,
        encoder_has_learned_content=head.kind.has_learned_content,
        input_dim=head.dim,
        interpretation=_interpretation(head.kind),
        warnings=warnings,
    )


def _interpretation(kind: EncoderKind) -> str:
    if kind is EncoderKind.PRETRAINED_LANGUAGE_MODEL:
        return (
            "Axes are the first two principal components of a pretrained language "
            "model's pooled representations. Proximity reflects what that model learned "
            "about these sequences, which is not the same as functional similarity."
        )
    return (
        "Axes are the first two principal components of amino-acid composition. "
        "Proximity means similar composition. Two sequences with the same residues in a "
        "different order land in exactly the same place, which is a limitation of the "
        "encoder rather than a finding about the peptides."
    )


def single_substitution_variants(reference: str, alphabet: str = "ACDEFGHIKLMNPQRSTVWY"
                                 ) -> List[Tuple[str, str]]:
    """Every single substitution of a reference sequence, as (label, sequence)."""
    out = []
    for position, wild in enumerate(reference.upper()):
        for mutant in alphabet:
            if mutant == wild:
                continue
            out.append((f"{wild}{position + 1}{mutant}",
                        reference[:position] + mutant + reference[position + 1:]))
    return out
