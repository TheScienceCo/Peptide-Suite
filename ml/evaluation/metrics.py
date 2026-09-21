"""
Evaluation metrics, with bootstrap intervals.  [Addendum 3, section 4]

Implemented directly rather than imported, for one reason that matters and one
that is convenience. The convenience: no scikit-learn dependency for CI. The
reason that matters: every metric here returns an interval alongside the point
estimate, because a peptide test set is usually small enough that the interval
is wider than the differences people report.

An AUC of 0.81 on 40 test sequences, with a 95% interval of 0.66-0.93, is not
meaningfully better than 0.76. Reporting the point estimate alone turns noise
into a finding, and on datasets this size that is the normal outcome rather
than an unlucky one.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple


@dataclass
class Metric:
    """A point estimate with its uncertainty and the sample it rests on."""
    name: str
    value: Optional[float]
    n: int
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    note: str = ""

    @property
    def is_determined(self) -> bool:
        return self.value is not None

    def __str__(self) -> str:
        if not self.is_determined:
            return f"{self.name}: not computable ({self.note})"
        base = f"{self.name}: {self.value:.3f}"
        if self.ci_low is not None:
            base += f" [{self.ci_low:.3f}, {self.ci_high:.3f}]"
        return f"{base} (n={self.n})"

    @property
    def interval_width(self) -> Optional[float]:
        if self.ci_low is None or self.ci_high is None:
            return None
        return self.ci_high - self.ci_low


def roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> Optional[float]:
    """
    ROC-AUC by rank, with ties handled by averaging ranks.

    Ties matter here: a model that outputs the same score for everything has an
    AUC of exactly 0.5, and a tie-unaware implementation reports anywhere from
    0 to 1 depending on input order.
    """
    pairs = sorted(zip(y_score, y_true))
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = average
        i = j + 1

    rank_sum = sum(r for r, (_s, y) in zip(ranks, pairs) if y == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def pr_auc(y_true: Sequence[int], y_score: Sequence[float]) -> Optional[float]:
    """
    Average precision. The right metric when positives are rare, because
    ROC-AUC stays flattering under class imbalance.
    """
    n_pos = sum(1 for y in y_true if y == 1)
    if n_pos == 0:
        return None
    order = sorted(range(len(y_score)), key=lambda i: -y_score[i])
    tp = 0
    total = 0.0
    for rank, index in enumerate(order, start=1):
        if y_true[index] == 1:
            tp += 1
            total += tp / rank
    return total / n_pos


def _confusion(y_true, y_pred) -> Tuple[int, int, int, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return tp, tn, fp, fn


def precision(y_true, y_pred) -> Optional[float]:
    tp, _tn, fp, _fn = _confusion(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) else None


def recall(y_true, y_pred) -> Optional[float]:
    tp, _tn, _fp, fn = _confusion(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) else None


def f1(y_true, y_pred) -> Optional[float]:
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    if p is None or r is None or (p + r) == 0:
        return None
    return 2 * p * r / (p + r)


def mcc(y_true, y_pred) -> Optional[float]:
    """
    Matthews correlation. Preferred over F1 on imbalanced data because it uses
    all four cells: F1 ignores true negatives, so a model that predicts the
    majority class everywhere can score well on one and near zero on this.
    """
    tp, tn, fp, fn = _confusion(y_true, y_pred)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return None
    return (tp * tn - fp * fn) / denominator


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> Optional[float]:
    if not y_true:
        return None
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> Optional[float]:
    if not y_true:
        return None
    return math.sqrt(sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true))


def r2(y_true: Sequence[float], y_pred: Sequence[float]) -> Optional[float]:
    """
    Coefficient of determination. None when the targets have no variance: R^2
    is undefined there, and returning 0 would read as "explains nothing" when
    the truth is "the question does not apply".
    """
    if len(y_true) < 2:
        return None
    mean = sum(y_true) / len(y_true)
    ss_tot = sum((t - mean) ** 2 for t in y_true)
    if ss_tot == 0:
        return None
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    return 1 - ss_res / ss_tot


def bootstrap(metric_fn: Callable, y_true: Sequence, y_pred: Sequence,
              n_resamples: int = 1000, seed: int = 0,
              alpha: float = 0.05) -> Tuple[Optional[float], Optional[float]]:
    """
    Percentile bootstrap interval.

    Resamples that cannot produce the metric -- a classification resample with
    one class, say -- are dropped rather than scored as zero. Scoring them
    would drag the interval toward a value the metric never took.
    """
    rng = random.Random(seed)
    n = len(y_true)
    if n < 2:
        return None, None
    values = []
    for _ in range(n_resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        value = metric_fn([y_true[i] for i in indices], [y_pred[i] for i in indices])
        if value is not None:
            values.append(value)
    if len(values) < n_resamples * 0.5:
        return None, None
    values.sort()
    low = values[int(len(values) * (alpha / 2))]
    high = values[min(len(values) - 1, int(len(values) * (1 - alpha / 2)))]
    return low, high


def evaluate_classification(y_true: Sequence[int], y_score: Sequence[float],
                            threshold: float = 0.5, seed: int = 0,
                            n_resamples: int = 1000) -> Dict[str, Metric]:
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    n = len(y_true)

    def build(name, fn, pred):
        value = fn(y_true, pred)
        low, high = bootstrap(fn, y_true, pred, n_resamples=n_resamples, seed=seed)
        note = "" if value is not None else "one class only, or an empty cell"
        return Metric(name=name, value=value, n=n, ci_low=low, ci_high=high, note=note)

    return {
        "roc_auc": build("ROC-AUC", roc_auc, y_score),
        "pr_auc": build("PR-AUC", pr_auc, y_score),
        "f1": build("F1", f1, y_pred),
        "precision": build("precision", precision, y_pred),
        "recall": build("recall", recall, y_pred),
        "mcc": build("MCC", mcc, y_pred),
    }


def evaluate_regression(y_true: Sequence[float], y_pred: Sequence[float],
                        seed: int = 0, n_resamples: int = 1000) -> Dict[str, Metric]:
    n = len(y_true)

    def build(name, fn):
        value = fn(y_true, y_pred)
        low, high = bootstrap(fn, y_true, y_pred, n_resamples=n_resamples, seed=seed)
        note = "" if value is not None else "targets have no variance, or n < 2"
        return Metric(name=name, value=value, n=n, ci_low=low, ci_high=high, note=note)

    return {"mae": build("MAE", mae), "rmse": build("RMSE", rmse), "r2": build("R2", r2)}


@dataclass
class CalibrationCurve:
    """
    Predicted probability against observed frequency, binned.

    A model can rank perfectly and still be badly calibrated -- AUC is invariant
    to any monotone transform of the scores, so it cannot see this at all. When
    a prediction is going to be read as a probability rather than as a ranking,
    calibration is the property that matters and AUC is silent about it.
    """
    bin_edges: List[float]
    predicted: List[float]
    observed: List[float]
    counts: List[int]

    @property
    def expected_calibration_error(self) -> Optional[float]:
        total = sum(self.counts)
        if not total:
            return None
        return sum(c / total * abs(p - o)
                   for p, o, c in zip(self.predicted, self.observed, self.counts) if c)

    def summary(self) -> str:
        ece = self.expected_calibration_error
        if ece is None:
            return "Calibration not computable: no predictions."
        return (f"Expected calibration error {ece:.3f} over {sum(self.counts)} "
                f"predictions in {sum(1 for c in self.counts if c)} populated bins.")


def calibration_curve(y_true: Sequence[int], y_score: Sequence[float],
                      n_bins: int = 10) -> CalibrationCurve:
    edges = [i / n_bins for i in range(n_bins + 1)]
    predicted, observed, counts = [], [], []
    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        members = [(s, t) for s, t in zip(y_score, y_true)
                   if (low <= s < high) or (i == n_bins - 1 and s == 1.0)]
        counts.append(len(members))
        predicted.append(sum(s for s, _ in members) / len(members) if members else 0.0)
        observed.append(sum(t for _, t in members) / len(members) if members else 0.0)
    return CalibrationCurve(bin_edges=edges, predicted=predicted,
                            observed=observed, counts=counts)


def confusion_matrix(y_true, y_pred) -> Dict[str, int]:
    tp, tn, fp, fn = _confusion(y_true, y_pred)
    return {"true_positive": tp, "true_negative": tn,
            "false_positive": fp, "false_negative": fn}
