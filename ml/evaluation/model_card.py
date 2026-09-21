"""
Model cards, generated rather than written.  [Addendum 3, section 7]

A hand-written card drifts from the model it describes, and the drift is
invisible: the card still reads correctly. So the card is built from the run
record, the dataset spec and the evaluation results, and anything that cannot be
filled from those is printed as a gap rather than left out.

The two fields that get omitted from most real model cards are here by
construction: the sequence-similarity control used for the split, and the
limitations. Both are required, and the generator refuses to produce a card
without them, because a card whose limitations section is blank reads as a model
without limitations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

from ..datasets.registry import DatasetSpec
from ..evaluation.metrics import Metric
from ..reproducibility import RunRecord


class IncompleteModelCard(Exception):
    """Raised when a card is missing a section that cannot be left blank."""


@dataclass
class ModelCard:
    """Everything a reader needs to decide whether to trust a prediction."""
    task: str
    architecture: str
    dataset: DatasetSpec
    run: RunRecord
    split_strategy: str
    identity_threshold: Optional[float]
    metrics: Dict[str, Metric] = field(default_factory=dict)
    comparison_metrics: Dict[str, Dict[str, Metric]] = field(default_factory=dict)
    calibration: str = ""
    limitations: List[str] = field(default_factory=list)
    domain_restrictions: List[str] = field(default_factory=list)
    date_trained: str = field(default_factory=lambda: date.today().isoformat())

    def __post_init__(self):
        if not self.limitations:
            raise IncompleteModelCard(
                f"The card for '{self.task}' has no limitations. A blank limitations "
                f"section reads as a model without limitations, which is never the case. "
                f"If they are unknown, say that -- it is itself a limitation.")
        if not self.split_strategy:
            raise IncompleteModelCard(
                f"The card for '{self.task}' does not say how the data was split. Without "
                f"it the metrics cannot be interpreted: a random split over a redundant "
                f"peptide set reports partly recall.")

    @property
    def supports_biological_claim(self) -> bool:
        return self.dataset.status.supports_biological_claim

    def to_markdown(self) -> str:
        lines = [
            f"# Model card: {self.task}",
            "",
            f"**Trained** {self.date_trained} · **Architecture** {self.architecture}",
            f"**Commit** `{self.run.git_commit}` · **Seed** {self.run.seed} · "
            f"**Device** {self.run.device}",
            "",
            "## Prediction target",
            "",
            f"{self.task} ({self.dataset.kind.value})",
            "",
            "## Dataset",
            "",
            f"- Source: {self.dataset.source}",
            f"- Licence: {self.dataset.licence}",
            f"- Size: {self.dataset.approximate_size}",
            f"- Version: {self.run.dataset_version or 'not recorded'}",
            f"- Status: **{self.dataset.status.value}**",
            "",
            self.dataset.claim_guidance(),
            "",
            "## Sequence similarity control",
            "",
            f"Split strategy: **{self.split_strategy}**",
        ]
        if self.identity_threshold is not None:
            lines.append(f"Clustered at {self.identity_threshold:.0%} identity; whole "
                         f"clusters held out, so no test sequence has a near-duplicate "
                         f"in training.")
        else:
            lines.append("No identity threshold recorded. If the split was random, the "
                         "metrics below include whatever the model recalls rather than "
                         "generalises.")

        lines += ["", "## Performance", ""]
        if self.metrics:
            lines.append("| Metric | Value | 95% interval | n |")
            lines.append("|---|---|---|---|")
            for name, metric in self.metrics.items():
                if metric.is_determined:
                    interval = (f"{metric.ci_low:.3f}–{metric.ci_high:.3f}"
                                if metric.ci_low is not None else "not computed")
                    lines.append(f"| {metric.name} | {metric.value:.3f} | {interval} "
                                 f"| {metric.n} |")
                else:
                    lines.append(f"| {metric.name} | not computable | — | {metric.n} |")
        else:
            lines.append("No metrics recorded.")

        if self.comparison_metrics:
            lines += ["", "## Split comparison", "",
                      "| Split | " + " | ".join(
                          sorted(next(iter(self.comparison_metrics.values())))) + " |",
                      "|---|" + "---|" * len(next(iter(self.comparison_metrics.values())))]
            for split_name, metrics in self.comparison_metrics.items():
                cells = [f"{metrics[k].value:.3f}" if metrics[k].is_determined else "—"
                         for k in sorted(metrics)]
                lines.append(f"| {split_name} | " + " | ".join(cells) + " |")

        lines += ["", "## Calibration", "", self.calibration or "Not assessed.",
                  "", "## Limitations", ""]
        lines += [f"- {item}" for item in self.limitations]

        if self.domain_restrictions:
            lines += ["", "## Known domain restrictions", ""]
            lines += [f"- {item}" for item in self.domain_restrictions]

        lines += ["", "## Reproducibility", "", self.run.reproducibility_note()]
        if not self.supports_biological_claim:
            lines += ["", "> This model was not trained on real biological labels. Its "
                      "metrics describe a method, not a property of peptides."]
        return "\n".join(lines)
