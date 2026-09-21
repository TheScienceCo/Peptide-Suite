"""
What data each task needs, and why it is not here.  [Addendum 3, sections 1 and 3]

The governing rule: do not build an ML pipeline around synthetic labels. A model
trained on invented targets produces metrics with the exact shape of real ones
and no content, and every downstream consumer -- the evaluation dashboard, the
model card, the substitution landscape -- renders them identically.

So no task here ships with data it does not have. Each declares its real public
source, its licence, and its status. A task whose data cannot be obtained is
UNAVAILABLE and its model cannot be trained; it is not quietly backed by random
numbers.

The distinction that keeps this usable: a SYNTHETIC dataset is admissible for
demonstrating a METHOD -- that a clustered split scores lower than a random one
is a fact about splitting, provable on constructed sequences -- and never for
demonstrating BIOLOGY. The two are different claims and the status field says
which one a dataset can support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class DatasetStatus(Enum):
    AVAILABLE = "available"              # real labels, present on disk
    UNAVAILABLE = "unavailable"          # real source named, not obtainable here
    SYNTHETIC_METHOD_ONLY = "synthetic_method_only"

    @property
    def supports_biological_claim(self) -> bool:
        return self is DatasetStatus.AVAILABLE

    @property
    def supports_method_claim(self) -> bool:
        return self is not DatasetStatus.UNAVAILABLE


class TaskKind(Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    REGRESSION = "regression"


@dataclass
class DatasetSpec:
    """
    A prediction task and the data it would need.

    `licence` is recorded because a model card has to state it, and finding it
    after training is harder than recording it before.
    """
    task: str
    kind: TaskKind
    status: DatasetStatus
    source: str
    licence: str
    approximate_size: str
    why_unavailable: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def is_trainable(self) -> bool:
        return self.status is not DatasetStatus.UNAVAILABLE

    def claim_guidance(self) -> str:
        if self.status is DatasetStatus.AVAILABLE:
            return f"Real labels from {self.source}. Biological claims are supportable."
        if self.status is DatasetStatus.SYNTHETIC_METHOD_ONLY:
            return ("Synthetic labels. This dataset demonstrates a METHOD and supports no "
                    "biological claim whatever the metrics say.")
        return (f"No data. {self.why_unavailable} The real source is {self.source} "
                f"({self.licence}); nothing is trained until it is present.")


# The candidate tasks from Addendum 3 section 3. Four would be implemented first
# on real data; the rest are declared so the shape of the work is visible, which
# is what the addendum asks for -- placeholders rather than a wider feature list.
TASKS: Dict[str, DatasetSpec] = {
    "solubility": DatasetSpec(
        task="solubility", kind=TaskKind.BINARY_CLASSIFICATION,
        status=DatasetStatus.UNAVAILABLE,
        source="eSOL / PROSO II style E. coli solubility sets",
        licence="varies by source; must be checked per set before redistribution",
        approximate_size="tens of thousands of proteins",
        why_unavailable="No network access to the dataset hosts from this environment.",
        notes=["Mostly full-length proteins rather than peptides, so transfer to short "
               "sequences is itself a question rather than an assumption."]),
    "amp_activity": DatasetSpec(
        task="amp_activity", kind=TaskKind.BINARY_CLASSIFICATION,
        status=DatasetStatus.UNAVAILABLE,
        source="APD3, DBAASP, or DRAMP antimicrobial peptide databases",
        licence="academic use; redistribution terms differ per database",
        approximate_size="thousands of peptides with activity annotations",
        why_unavailable="No network access to the database hosts from this environment.",
        notes=["The best-matched task on this list: genuinely peptide-scale, and the "
               "negatives are the hard part -- most sets use random sequences as "
               "negatives, which makes the task easier than the real one."]),
    "cell_penetration": DatasetSpec(
        task="cell_penetration", kind=TaskKind.BINARY_CLASSIFICATION,
        status=DatasetStatus.UNAVAILABLE,
        source="CPPsite 2.0",
        licence="academic use",
        approximate_size="around two thousand peptides",
        why_unavailable="No network access from this environment.",
        notes=["Heavily enriched in arginine-rich sequences, so a model can score well "
               "by counting arginine. Any result needs that baseline reported beside it."]),
    "toxicity": DatasetSpec(
        task="toxicity", kind=TaskKind.BINARY_CLASSIFICATION,
        status=DatasetStatus.UNAVAILABLE,
        source="ToxinPred / DBAASP haemolysis annotations",
        licence="academic use",
        approximate_size="thousands of peptides",
        why_unavailable="No network access from this environment."),
    "aggregation": DatasetSpec(
        task="aggregation", kind=TaskKind.BINARY_CLASSIFICATION,
        status=DatasetStatus.UNAVAILABLE,
        source="AmyLoad / WALTZ-DB amyloid hexapeptide sets",
        licence="academic use",
        approximate_size="thousands of hexapeptides",
        notes=["Hexapeptides only in most sets, so the length distribution is far from "
               "the peptides this suite analyses."],
        why_unavailable="No network access from this environment."),
    "stability": DatasetSpec(
        task="stability", kind=TaskKind.REGRESSION,
        status=DatasetStatus.UNAVAILABLE,
        source="Rocklin et al. designed mini-protein stability measurements",
        licence="published dataset; check terms",
        approximate_size="tens of thousands of designed sequences",
        why_unavailable="No network access from this environment.",
        notes=["Designed sequences rather than natural ones, which is a distribution "
               "shift that has to be reported rather than assumed away."]),
    "protease_stability": DatasetSpec(
        task="protease_stability", kind=TaskKind.REGRESSION,
        status=DatasetStatus.UNAVAILABLE,
        source="no single standard set; would need assembling from primary literature",
        licence="n/a",
        approximate_size="unknown",
        why_unavailable=("There is no established benchmark. Assembling one from primary "
                         "sources is a project in itself and is not pretended otherwise."),
        notes=["The suite already scores protease liability from documented P1 "
               "specificities, which is a different and better-evidenced route than a "
               "model fitted to a set nobody has assembled."]),
    "split_methodology_demo": DatasetSpec(
        task="split_methodology_demo", kind=TaskKind.BINARY_CLASSIFICATION,
        status=DatasetStatus.SYNTHETIC_METHOD_ONLY,
        source="generated in-process from a known rule over constructed families",
        licence="n/a",
        approximate_size="configurable",
        notes=["Exists to demonstrate that a clustered split scores lower than a random "
               "one, which is a fact about splitting and is provable on constructed "
               "sequences. It supports no biological claim, and the status field is "
               "what stops it being read as one."]),
}


def trainable_tasks() -> List[str]:
    return sorted(name for name, spec in TASKS.items() if spec.is_trainable)


def unavailable_tasks() -> List[str]:
    return sorted(name for name, spec in TASKS.items() if not spec.is_trainable)


def status_report() -> str:
    lines = [f"{len(TASKS)} declared tasks; {len(trainable_tasks())} trainable here."]
    for name in sorted(TASKS):
        spec = TASKS[name]
        lines.append(f"  {name:26s} {spec.status.value:22s} {spec.source[:44]}")
    return "\n".join(lines)
