"""
Aim as a view onto the run record, never as the run record.  [Phase 2]

Aim is good at the thing this repository does not have: metrics over time,
across runs, in a browser. It is not good at being a system of record, and it
must not become one here. `ml/tracking.py` writes an append-only JSONL line per
run and `RunRecord` decides whether that run is reproducible; both keep working
with Aim absent, uninstalled, or broken. What this module does is mirror an
already-recorded run into Aim so it can be looked at.

The direction matters. Mirroring *from* the record means a machine with no Aim
loses a dashboard and nothing else, and it means the two can never disagree
about what a run was -- there is one writer. Logging to Aim first and
reconstructing the record from it afterwards would invert that, and the
reconstruction would quietly become the truth.

THREE THINGS THIS REFUSES.

`None` is not a metric. `Experiment.metrics` holds `Optional[float]` because a
metric that was not computed is recorded as not computed. Aim would take
`float('nan')` without complaint and draw it as a gap, which is indistinguishable
in a chart from a step that was skipped; it would take `0.0` and draw it as a
result. Unmeasured metrics are dropped from the mirror and counted, and the
count is reported, so "this run logged 4 of 7 metrics" is visible rather than
being a chart with three flat lines at zero.

A silent no-op is not a success. When Aim is not installed -- the ordinary case,
including in CI -- every call still works and every call still reports that
nothing was transmitted. A tracker that swallows its own failure is worse than
one that raises, because the run appears to have been logged.

Two runs are not comparable because they are both in Aim. Dataset version,
encoder identity and split strategy decide that, and `comparability()` computes
it in this module rather than leaving it to whoever is looking at two lines on
the same axes. Aim will happily overlay a run trained on v1 with a run trained
on v2.

EMBEDDINGS ARE PROJECTED BEFORE THEY GET HERE. `ml/embeddings/explorer.py`
computes the PCA, on the SVD of the mean-centred matrix, with its own floors --
four points minimum, a refusal when every vector is identical, a warning when
the first two components explain little. Aim receives finished coordinates and
the explained-variance ratio that produced them. Letting a visualiser perform
the reduction would put the axes' derivation somewhere nobody records, and the
axes are the part that gets interpreted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .reproducibility import RunRecord
from .tracking import Experiment

#: Set to "0"/"false"/"no" to keep the mirror off on a machine that has Aim
#: installed. There is deliberately no value that turns it *on* when Aim is
#: absent: the switch can only ever reduce what is attempted.
AIM_ENV_VAR = "PEPTIDE_SUITE_AIM"

_OFF = {"0", "false", "no", "off"}


class AimStatus(Enum):
    """Why the mirror is or is not transmitting. Never collapsed to a bool."""

    LIVE = "LIVE"                       # a run is open and calls reach Aim
    READY = "READY"                     # installed and enabled, nothing opened yet
    NOT_INSTALLED = "NOT_INSTALLED"     # the ordinary case; not an error
    DISABLED = "DISABLED"               # installed, switched off by the operator
    FAILED = "FAILED"                   # installed, and opening the repo raised
    CLOSED = "CLOSED"                   # was live, has been finalised

    @property
    def is_transmitting(self) -> bool:
        return self is AimStatus.LIVE

    @property
    def explanation(self) -> str:
        return {
            AimStatus.LIVE: "Aim is open and receiving.",
            AimStatus.READY: (
                "Aim is installed and enabled, but no run has been opened, so nothing "
                "has been transmitted yet."),
            AimStatus.NOT_INSTALLED: (
                "Aim is not installed, so nothing was transmitted. The JSONL run record "
                "is unaffected: it is the system of record and Aim is a view onto it."),
            AimStatus.DISABLED: (
                f"Aim is installed but switched off via {AIM_ENV_VAR}, so nothing was "
                f"transmitted."),
            AimStatus.FAILED: (
                "Aim is installed but its repository could not be opened, so nothing was "
                "transmitted. This is reported rather than raised because losing a "
                "dashboard must not lose a training run."),
            AimStatus.CLOSED: "The Aim run has been finalised; no further calls reach it.",
        }[self]


def aim_available() -> bool:
    """Whether the Aim package can be imported. Does not consider the switch."""
    try:
        import aim  # noqa: F401
    except Exception:
        return False
    return True


def aim_enabled() -> bool:
    """Whether the operator has left the mirror switched on."""
    return os.environ.get(AIM_ENV_VAR, "1").strip().lower() not in _OFF


# ---- comparability ---------------------------------------------------------

#: Fields on which two runs must agree before their metrics mean the same
#: thing. Each maps to why. Aim overlays any two runs on request; this is the
#: part it cannot know.
COMPARABILITY_KEYS: Dict[str, str] = {
    "dataset_version": (
        "the two runs were trained and evaluated on different data, so a difference "
        "between their metrics is a difference between the datasets as much as between "
        "the models"),
    "split_strategy": (
        "one run's test set may be an easier partition than the other's; a random split "
        "and a similarity-aware split do not produce comparable held-out numbers"),
    "encoder_model": (
        "the representations come from different encoders, so the models are not solving "
        "the same problem in the same space"),
    "encoder_version": (
        "the same encoder at a different version is a different representation; equal "
        "dimensionality does not make two spaces the same space"),
    "metric_definition": (
        "the number is computed differently, so the two values are not the same quantity "
        "under one name"),
}


@dataclass(frozen=True)
class Comparability:
    """Whether two runs may be read against each other, and what differs."""

    comparable: bool
    differing: Tuple[str, ...] = ()
    reasons: Tuple[str, ...] = ()
    unknown: Tuple[str, ...] = ()

    def summary(self) -> str:
        if self.comparable and not self.unknown:
            return "Comparable: dataset, split and encoder agree."
        parts = []
        if self.differing:
            parts.append("NOT comparable. " + " ".join(
                f"{key}: {reason}." for key, reason in zip(self.differing, self.reasons)))
        if self.unknown:
            parts.append(
                f"Not established for {', '.join(self.unknown)}: at least one run does "
                f"not record it, so agreement cannot be checked and is not assumed.")
        return " ".join(parts)


def comparability(left: Dict[str, Any], right: Dict[str, Any]) -> Comparability:
    """
    Whether two runs' metrics may be read against each other.

    A field absent from either run is reported as unknown rather than as
    agreement. Two runs that both fail to record their dataset version are not
    thereby known to share one, and treating silence as a match is how an
    incomparable pair ends up on the same axes.
    """
    differing: List[str] = []
    reasons: List[str] = []
    unknown: List[str] = []
    for key, reason in COMPARABILITY_KEYS.items():
        a, b = left.get(key), right.get(key)
        if a in (None, "") or b in (None, ""):
            unknown.append(key)
        elif a != b:
            differing.append(key)
            reasons.append(reason)
    return Comparability(comparable=not differing and not unknown,
                         differing=tuple(differing), reasons=tuple(reasons),
                         unknown=tuple(unknown))


# ---- the mirror ------------------------------------------------------------

@dataclass
class MirrorReport:
    """
    What actually reached Aim. Returned by every mirroring call.

    Holds `skipped_metrics` because the count is the point: a run whose loss was
    never computed should read as a run missing a metric, not as a run whose
    loss was zero.
    """

    status: AimStatus
    transmitted: int = 0
    skipped_metrics: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def reached_aim(self) -> bool:
        return self.status is AimStatus.LIVE and self.transmitted > 0

    def describe(self) -> str:
        head = f"{self.transmitted} value(s) sent to Aim." if self.reached_aim \
            else f"Nothing sent to Aim. {self.status.explanation}"
        if self.skipped_metrics:
            head += (f" {len(self.skipped_metrics)} metric(s) were not computed and were "
                     f"omitted rather than sent as zero or NaN: "
                     f"{', '.join(sorted(self.skipped_metrics))}.")
        return " ".join([head] + self.notes)


class AimMirror:
    """
    Mirrors a recorded run into Aim, and reports honestly when it cannot.

    Every method is safe to call in every state. None of them raise on a
    missing or broken Aim: a training run must not die because a dashboard is
    unavailable. What they do instead is return a `MirrorReport` whose status
    says what happened, so a caller that wants to insist on Aim can check, and
    a caller that does not gets the same behaviour it had before Aim existed.
    """

    def __init__(self, repo: Optional[str] = None, experiment: str = "peptide-suite"):
        self.repo = repo
        self.experiment = experiment
        self._run = None
        self._status = self._initial_status()

    def _initial_status(self) -> AimStatus:
        if not aim_available():
            return AimStatus.NOT_INSTALLED
        if not aim_enabled():
            return AimStatus.DISABLED
        return AimStatus.READY

    @property
    def status(self) -> AimStatus:
        return self._status

    @property
    def is_transmitting(self) -> bool:
        return self._status.is_transmitting

    # -- lifecycle --

    def open(self, record: RunRecord) -> MirrorReport:
        """
        Open an Aim run mirroring a RunRecord, and stamp its provenance on it.

        The reproducibility verdict goes on as a tag, not only as a parameter.
        Aim's run list filters on tags, and a comparison view that cannot filter
        out runs made from a dirty tree will eventually include one.
        """
        if self._status in (AimStatus.NOT_INSTALLED, AimStatus.DISABLED,
                            AimStatus.FAILED):
            return MirrorReport(status=self._status)
        try:
            from aim import Run
            self._run = Run(repo=self.repo, experiment=self.experiment)
        except Exception as exc:
            self._status = AimStatus.FAILED
            return MirrorReport(status=AimStatus.FAILED,
                                notes=[f"Opening the Aim repository raised: {exc}"])

        self._status = AimStatus.LIVE
        payload = record.to_dict()
        self._run["run_record"] = payload
        self._run["reproducible"] = record.is_reproducible
        self._run["reproducibility_note"] = record.reproducibility_note()
        try:
            self._run.add_tag(
                "reproducible" if record.is_reproducible else "NOT-reproducible")
        except Exception:
            pass
        notes = []
        if not record.is_reproducible:
            notes.append(
                "Tagged NOT-reproducible in Aim: " + record.reproducibility_note())
        return MirrorReport(status=AimStatus.LIVE, transmitted=len(payload), notes=notes)

    def close(self) -> MirrorReport:
        """
        Finalise the run, and index it so that it is actually visible.

        Closing is not enough, which is the sort of thing only a real Aim
        install tells you. Aim writes each run to its own chunk and builds the
        index from a separate daemon that `aim up` starts; `Repo.iter_runs`
        reads the index. A run mirrored and closed by a plain script is on disk
        and absent from every view of the repository until something indexes
        it -- which, for a script that exits, is never.

        That is the failure this whole module is against, arriving from the
        other direction: a report reading "sent to Aim" while the run cannot be
        found. So the index is written here, and when it cannot be, the report
        says the run is stored but not yet visible rather than saying nothing.
        """
        if self._status is not AimStatus.LIVE:
            return MirrorReport(status=self._status)
        run_hash = getattr(self._run, "hash", "")
        try:
            self._run.close()
        except Exception:
            pass
        self._status = AimStatus.CLOSED
        self._run = None
        return MirrorReport(status=AimStatus.CLOSED, notes=self._index(run_hash))

    def _index(self, run_hash: str) -> List[str]:
        """Make a finished run visible to Aim's repository views."""
        if not run_hash:
            return ["The run had no hash, so it could not be indexed and will not appear "
                    "in the Aim UI until the indexing daemon reaches it."]
        try:
            from aim import Repo
            from aim.sdk.index_manager import RepoIndexManager
            RepoIndexManager.get_index_manager(Repo(self.repo) if self.repo else Repo.default_repo()).index(run_hash)
        except Exception as exc:
            return [f"The run was written but could not be indexed ({exc}), so it will "
                    f"not appear in the Aim UI until `aim up` indexes it. The JSONL "
                    f"record is unaffected."]
        return [f"Run {run_hash[:12]} indexed and visible in the Aim repository."]

    # -- content --

    def log_metrics(self, metrics: Dict[str, Optional[float]], step: Optional[int] = None,
                    context: Optional[Dict[str, Any]] = None) -> MirrorReport:
        """
        Send computed metrics. Metrics that were not computed are omitted.

        `None` means the metric was not computed, and there is no faithful way
        to draw that. NaN renders as a gap, which reads as a skipped step; zero
        renders as a result. Both assert something about a number that does not
        exist, so the name is recorded as skipped and nothing is sent for it.
        """
        skipped = [name for name, value in metrics.items() if not _is_real_number(value)]
        if self._status is not AimStatus.LIVE:
            return MirrorReport(status=self._status, skipped_metrics=skipped)

        sent = 0
        notes = []
        for name, value in metrics.items():
            if not _is_real_number(value):
                continue
            try:
                self._run.track(float(value), name=name, step=step, context=context or {})
                sent += 1
            except Exception as exc:
                notes.append(f"Aim rejected '{name}': {exc}")
        return MirrorReport(status=AimStatus.LIVE, transmitted=sent,
                            skipped_metrics=skipped, notes=notes)

    def log_experiment(self, experiment: Experiment) -> MirrorReport:
        """
        Mirror a completed `Experiment` -- the object `Tracker` writes.

        Takes the tracker's own type on purpose. A separate shape for Aim would
        be a second definition of what a run is, and the two would drift.
        """
        report = self.open(experiment.run)
        if report.status is not AimStatus.LIVE:
            report.skipped_metrics = [
                name for name, value in experiment.metrics.items()
                if not _is_real_number(value)]
            return report

        self._run["experiment_name"] = experiment.name
        self._run["checkpoint"] = experiment.checkpoint
        self._run["embedding_model"] = experiment.embedding_model
        self._run["notes"] = list(experiment.notes)
        self._run["reproduce_command"] = experiment.reproduce_command()
        for key, value in comparison_keys(experiment).items():
            self._run[key] = value

        metrics = self.log_metrics(experiment.metrics)
        return MirrorReport(
            status=AimStatus.LIVE,
            transmitted=report.transmitted + metrics.transmitted,
            skipped_metrics=metrics.skipped_metrics,
            notes=report.notes + metrics.notes)

    def log_projection(self, projection, name: str = "embedding_projection") -> MirrorReport:
        """
        Send a projection that was computed elsewhere.

        Takes the finished `Projection` from `ml/embeddings/explorer.py`: the
        coordinates, the explained-variance ratio and the interpretation string
        that says what the axes are. This module performs no reduction and asks
        Aim to perform none, because a reduction whose parameters nobody
        recorded produces axes that get interpreted anyway.

        The warnings travel with the points. A projection whose first two
        components explain little of the variance still plots, and the plot
        looks exactly as convincing as one that explains most of it.
        """
        if self._status is not AimStatus.LIVE:
            return MirrorReport(status=self._status)

        payload = {
            "computed_by": "ml.embeddings.explorer.project (PCA on the mean-centred SVD)",
            "encoder_model": projection.encoder_model,
            "encoder_version": projection.encoder_version,
            "encoder_kind": projection.encoder_kind,
            "encoder_has_learned_content": projection.encoder_has_learned_content,
            "input_dim": projection.input_dim,
            "explained_variance_ratio": list(projection.explained_variance_ratio),
            "cumulative_explained": projection.cumulative_explained,
            "interpretation": projection.interpretation,
            "warnings": list(projection.warnings),
            "distinct_positions": projection.distinct_positions,
            "distinct_projected": projection.distinct_projected,
            "points": [{"label": p.label, "sequence": p.sequence, "x": p.x, "y": p.y,
                        "variant_class": p.variant_class.value,
                        "n_substitutions": p.n_substitutions}
                       for p in projection.points],
        }
        try:
            self._run[name] = payload
        except Exception as exc:
            return MirrorReport(status=AimStatus.LIVE,
                                notes=[f"Aim rejected the projection: {exc}"])
        notes = ["Axes were computed before transmission; Aim performed no reduction."]
        if projection.warnings:
            notes.append("Projection warnings travelled with the points: "
                         + " ".join(projection.warnings))
        return MirrorReport(status=AimStatus.LIVE, transmitted=len(payload["points"]),
                            notes=notes)

    def describe(self) -> str:
        return self._status.explanation


def _is_real_number(value: Any) -> bool:
    """
    A value that can honestly be plotted.

    Rejects None (not computed), NaN and the infinities. bool is rejected too:
    `True` is a float in Python's eyes and would plot as 1.0, which is a
    measurement-shaped rendering of a flag.
    """
    if value is None or isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    value = float(value)
    return value == value and value not in (float("inf"), float("-inf"))


def comparison_keys(experiment: Experiment) -> Dict[str, Any]:
    """
    The fields `comparability()` reads, pulled from a recorded experiment.

    Absent fields are omitted rather than defaulted. A run whose config does not
    name a split strategy has not got one recorded, and filling in "random"
    would manufacture the agreement the comparison is supposed to test.
    """
    config = experiment.run.config or {}
    candidates = {
        "dataset_version": experiment.run.dataset_version,
        "split_strategy": config.get("split_strategy") or config.get("split"),
        "encoder_model": experiment.embedding_model or config.get("encoder_model"),
        "encoder_version": config.get("encoder_version"),
        "metric_definition": config.get("metric_definition"),
    }
    return {key: value for key, value in candidates.items() if value not in (None, "")}


def mirror_tracked_runs(tracker, repo: Optional[str] = None,
                        experiment: str = "peptide-suite") -> List[MirrorReport]:
    """
    Mirror every run a `Tracker` already holds.

    Reads from the JSONL rather than intercepting the write, which is what keeps
    the record independent: if this function is deleted, every run is still
    recorded and still replayable.
    """
    reports = []
    for entry in tracker.all():
        record = RunRecord(
            name=entry["run"]["name"], seed=entry["run"]["seed"],
            device=entry["run"]["device"], git_commit=entry["run"]["git_commit"],
            dataset_version=entry["run"].get("dataset_version", ""),
            config=entry["run"].get("config", {}),
            library_versions=entry["run"].get("library_versions", {}),
            seeding_caveats=entry["run"].get("seeding_caveats", []))
        mirror = AimMirror(repo=repo, experiment=experiment)
        reports.append(mirror.log_experiment(Experiment(
            name=entry["name"], run=record, metrics=entry.get("metrics", {}),
            checkpoint=entry.get("checkpoint", ""),
            embedding_model=entry.get("embedding_model", ""),
            notes=entry.get("notes", []))))
        mirror.close()
    return reports


def status_report(repo: Optional[str] = None) -> Dict[str, Any]:
    """
    Whether Aim would do anything here, for a caller that wants to know before
    starting a run rather than after it.
    """
    mirror = AimMirror(repo=repo)
    installed = aim_available()
    return {
        "installed": installed,
        "enabled": aim_enabled(),
        "status": mirror.status.value,
        "explanation": mirror.status.explanation,
        "system_of_record": "ml/tracking.py (JSONL) — unaffected by any of the above",
    }
