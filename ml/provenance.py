"""
Where an ML number came from, and what that permits.  [Addendum 3, section 1]

The existing system already separates experimentally observed data, external
annotations and deterministic descriptors. Machine learning adds three more
categories that must never be silently merged with those or with each other:

  PRETRAINED_EMBEDDING   a vector from a protein language model. Not a
                         measurement and not a prediction -- a learned
                         representation whose meaning depends entirely on the
                         model and version that produced it.
  ML_PREDICTION          a model's output. Carries the model's training
                         distribution with it, which is why an
                         out-of-distribution flag travels alongside.
  ML_UNCERTAINTY         an estimate of how wrong the prediction might be.
                         Separate from the prediction, and separate again from
                         confidence in the evidence tier sense.

The last distinction is the one that gets collapsed. A model that outputs 0.9
with a wide predictive interval and a model that outputs 0.9 having seen a
thousand near-identical training examples are saying different things, and a
single "0.9" cannot tell them apart.

This module exists before any model does, deliberately. Retrofitting provenance
onto a trained pipeline means deciding what a number meant after someone has
already acted on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class MLProvenance(Enum):
    """The three ML categories, kept distinct from the existing tiers."""
    PRETRAINED_EMBEDDING = "PRETRAINED_EMBEDDING"
    ML_PREDICTION = "ML_PREDICTION"
    ML_UNCERTAINTY = "ML_UNCERTAINTY"

    @property
    def may_be_a_regression_target(self) -> bool:
        """
        None of them. A model's output is not evidence about the world, and
        fitting one model to another's predictions launders the second model's
        errors into the first as though they were measurements.
        """
        return False


class MLProvenanceError(Exception):
    """Raised when an ML quantity is used in a way its category forbids."""


# What every ML quantity must be able to say about itself. A prediction whose
# model version is unknown cannot be reproduced, compared against a later
# version, or withdrawn when that version is found to be wrong.
REQUIRED_METADATA: Dict[MLProvenance, frozenset] = {
    MLProvenance.PRETRAINED_EMBEDDING: frozenset({
        "model", "model_version", "embedding_dim", "pooling", "preprocessing",
    }),
    MLProvenance.ML_PREDICTION: frozenset({
        "model", "model_version", "task", "training_dataset", "dataset_version",
        "split_strategy", "git_commit", "seed",
    }),
    MLProvenance.ML_UNCERTAINTY: frozenset({
        "model", "model_version", "method", "calibrated",
    }),
}


@dataclass
class MLQuantity:
    """
    One ML-derived value with its provenance.

    `value` and `uncertainty` are separate fields rather than one interval,
    because they are separate categories: the prediction is what the model says
    and the uncertainty is a claim about that claim. Collapsing them is how a
    confident-looking number reaches a decision with its error bar left behind.
    """
    name: str
    value: Any
    provenance: MLProvenance
    metadata: Dict[str, Any] = field(default_factory=dict)
    uncertainty: Optional["MLQuantity"] = None
    out_of_distribution: Optional[bool] = None
    ood_basis: str = ""

    def __post_init__(self):
        missing = REQUIRED_METADATA[self.provenance] - set(self.metadata)
        if missing:
            raise MLProvenanceError(
                f"{self.provenance.value} quantity '{self.name}' is missing required "
                f"metadata: {', '.join(sorted(missing))}. A prediction whose model "
                f"version is unknown cannot be reproduced, compared against a later "
                f"version, or withdrawn when that version is found to be wrong."
            )
        if (self.provenance is MLProvenance.ML_PREDICTION
                and self.uncertainty is not None
                and self.uncertainty.provenance is not MLProvenance.ML_UNCERTAINTY):
            raise MLProvenanceError(
                f"'{self.name}' attaches an uncertainty that is not itself tagged "
                f"ML_UNCERTAINTY. The estimate of error is a different kind of claim "
                f"from the prediction and carries its own provenance."
            )

    @property
    def is_reportable(self) -> bool:
        """
        Whether this may be shown as a prediction.

        A prediction outside its training distribution is not a prediction; it
        is the model extrapolating, and the honest report is the flag rather
        than the number.
        """
        if self.provenance is not MLProvenance.ML_PREDICTION:
            return True
        return self.out_of_distribution is not True

    def require_in_distribution(self) -> None:
        if not self.is_reportable:
            raise MLProvenanceError(
                f"'{self.name}' is flagged out of distribution ({self.ood_basis}). "
                f"The model has not seen anything like this input, so its output is an "
                f"extrapolation rather than a prediction and is not reported as one."
            )

    def describe(self) -> str:
        model = self.metadata.get("model", "?")
        version = self.metadata.get("model_version", "?")
        line = f"{self.name} = {self.value} [{self.provenance.value} via {model} {version}]"
        if self.uncertainty is not None:
            line += f" ± {self.uncertainty.value} ({self.uncertainty.metadata.get('method')})"
        if self.out_of_distribution:
            line += "  OUT OF DISTRIBUTION"
        return line


def forbid_as_target(quantity: MLQuantity) -> None:
    """
    Refuse to use an ML output as a regression target.

    Fitting a model to another model's predictions launders the second model's
    errors into the first as though they were measurements, and the resulting
    system reports high confidence about a distribution it invented.
    """
    raise MLProvenanceError(
        f"'{quantity.name}' is {quantity.provenance.value} and may not be used as a "
        f"regression target. Only MEASURED quantities may be fitted against. Training on "
        f"predictions launders their error into the next model as though it were data."
    )


@dataclass
class AttributionClaim:
    """
    A feature-attribution result, with the causal reading refused.

    Attribution says which inputs the model's output was sensitive to. It does
    not say which inputs matter to the biology, and the two come apart exactly
    where a model has learned a confound. The disclaimer is part of the object
    rather than a note beside it, so it cannot be dropped in transit.
    """
    positions: Sequence[int]
    scores: Sequence[float]
    method: str
    model: str
    model_version: str

    @property
    def disclaimer(self) -> str:
        return (
            "Attribution shows what the model's output was sensitive to, not what the "
            "biology depends on. Where the model has learned a confound the two come "
            "apart, and this method cannot distinguish the cases."
        )

    def as_mechanism(self):
        raise MLProvenanceError(
            f"{self.method} attribution may not be reported as a mechanism. It ranks the "
            f"model's sensitivity to its inputs; a mechanistic claim needs the physics, "
            f"which lives in the tiered layer rather than here."
        )
