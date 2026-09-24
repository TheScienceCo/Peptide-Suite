"""
The Aim mirror, and the four ways a tracker lies.

It reports a no-op as a success. It plots a metric that was never computed. It
becomes the system of record without anyone deciding that it should. It puts
two incomparable runs on one axis because both happen to be in the database.

Aim is not installed in CI, on purpose -- the ml job installs torch and numpy
and nothing else -- so the absent path is the one that runs for real here, and
the live path runs against a fake `aim` module. Which means the fake is doing
load-bearing work, and it records what it was asked to do rather than accepting
and discarding it, so a test can assert on what would have been transmitted.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

from ml.aim_tracking import (
    AIM_ENV_VAR, AimMirror, AimStatus, Comparability, MirrorReport,
    aim_available, aim_enabled, comparability, comparison_keys,
    mirror_tracked_runs, status_report,
)
from ml.reproducibility import RunRecord
from ml.tracking import Experiment, Tracker


def record(commit: str = "a" * 40, **kwargs) -> RunRecord:
    """A RunRecord built without torch, which `capture` would import."""
    fields = dict(name="run", seed=7, device="cpu", git_commit=commit,
                  dataset_version="v1", config={}, library_versions={},
                  seeding_caveats=[])
    fields.update(kwargs)
    return RunRecord(**fields)


# ---- the fake ---------------------------------------------------------------

class FakeRun:
    """Records what was set and tracked instead of writing anything."""

    hash = "0123456789abcdef01234567"

    def __init__(self, repo=None, experiment=None):
        self.repo = repo
        self.experiment = experiment
        self.params = {}
        self.tracked = []
        self.tags = []
        self.closed = False

    def __setitem__(self, key, value):
        self.params[key] = value

    def track(self, value, name, step=None, context=None):
        self.tracked.append({"name": name, "value": value, "step": step,
                             "context": context})

    def add_tag(self, tag):
        self.tags.append(tag)

    def close(self):
        self.closed = True


@contextlib.contextmanager
def fake_aim(run_factory=FakeRun):
    """Install a fake `aim` module for the duration of a test."""
    module = types.ModuleType("aim")
    created = []

    def Run(repo=None, experiment=None):
        made = run_factory(repo=repo, experiment=experiment)
        created.append(made)
        return made

    module.Run = Run
    previous = sys.modules.get("aim")
    sys.modules["aim"] = module
    try:
        yield created
    finally:
        if previous is None:
            sys.modules.pop("aim", None)
        else:
            sys.modules["aim"] = previous


@contextlib.contextmanager
def no_aim():
    """
    Make `import aim` fail, whatever the machine has installed.

    Written after these tests passed for the wrong reason: they asserted the
    absent path while merely running on a machine without Aim, so installing
    it turned five of them red. A test of what happens when a dependency is
    missing has to make it missing.
    """
    previous = sys.modules.get("aim", ...)
    sys.modules["aim"] = None          # `import aim` raises ImportError on None
    try:
        yield
    finally:
        if previous is ...:
            sys.modules.pop("aim", None)
        else:
            sys.modules["aim"] = previous


@contextlib.contextmanager
def env(**values):
    previous = {k: os.environ.get(k) for k in values}
    os.environ.update({k: v for k, v in values.items() if v is not None})
    for k, v in values.items():
        if v is None:
            os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---- absence ----------------------------------------------------------------

class TestAbsenceIsSafeAndVisible(unittest.TestCase):
    """
    The ordinary case. Aim is not installed, and the requirement is that
    everything still works AND that nothing claims to have been logged.
    """

    def test_every_call_is_safe_without_aim(self):
        with no_aim():
            mirror = AimMirror()
            self.assertIs(mirror.status, AimStatus.NOT_INSTALLED)
            for report in (mirror.open(record()),
                           mirror.log_metrics({"loss": 0.4}),
                           mirror.close()):
                self.assertIsInstance(report, MirrorReport)

    def test_a_no_op_does_not_report_success(self):
        with no_aim():
            report = AimMirror().log_metrics({"loss": 0.4})
        self.assertFalse(report.reached_aim)
        self.assertEqual(report.transmitted, 0)
        self.assertIn("Nothing sent to Aim", report.describe())

    def test_absence_is_stated_as_absence_not_as_an_error(self):
        explanation = AimStatus.NOT_INSTALLED.explanation
        self.assertIn("not installed", explanation)
        self.assertIn("system of record", explanation)

    def test_the_switch_can_only_turn_the_mirror_off(self):
        """
        There is no value of the variable that makes an uninstalled Aim
        transmit. A switch that can turn a capability on is a switch that can
        be set by someone who then believes the capability is present.
        """
        with no_aim(), env(**{AIM_ENV_VAR: "1"}):
            self.assertIs(AimMirror().status, AimStatus.NOT_INSTALLED)
        with fake_aim():
            with env(**{AIM_ENV_VAR: "0"}):
                self.assertIs(AimMirror().status, AimStatus.DISABLED)

    def test_status_report_names_the_system_of_record(self):
        with no_aim():
            report = status_report()
        self.assertFalse(report["installed"])
        self.assertIn("tracking.py", report["system_of_record"])

    def test_status_report_says_so_when_aim_is_present(self):
        with fake_aim():
            report = status_report()
        self.assertTrue(report["installed"])
        self.assertEqual(report["status"], AimStatus.READY.value)
        self.assertIn("tracking.py", report["system_of_record"])


class TestFailureIsReportedNotRaised(unittest.TestCase):

    def test_a_broken_repository_does_not_kill_the_run(self):
        class Exploding(FakeRun):
            def __init__(self, repo=None, experiment=None):
                raise RuntimeError("repo is locked")

        with fake_aim(run_factory=Exploding):
            mirror = AimMirror()
            report = mirror.open(record())
        self.assertIs(report.status, AimStatus.FAILED)
        self.assertFalse(report.reached_aim)
        self.assertIn("repo is locked", " ".join(report.notes))

    def test_a_rejected_metric_is_named_rather_than_swallowed(self):
        class Picky(FakeRun):
            def track(self, value, name, step=None, context=None):
                if name == "auc":
                    raise ValueError("no")
                super().track(value, name, step=step, context=context)

        with fake_aim(run_factory=Picky):
            mirror = AimMirror()
            mirror.open(record())
            report = mirror.log_metrics({"loss": 0.2, "auc": 0.9})
        self.assertEqual(report.transmitted, 1)
        self.assertIn("auc", " ".join(report.notes))


# ---- what must never be plotted --------------------------------------------

class TestAnUncomputedMetricIsNotPlotted(unittest.TestCase):
    """
    The failure this class exists for: `metrics` holds Optional[float] because
    a metric that was not computed is recorded as not computed. Sending NaN
    draws a gap, which is what a skipped step looks like; sending 0.0 draws a
    result.
    """

    METRICS = {"loss": 0.25, "auc": None, "spearman": float("nan"),
               "grad_norm": float("inf"), "converged": True}

    def test_only_real_numbers_are_transmitted(self):
        with fake_aim() as runs:
            mirror = AimMirror()
            mirror.open(record())
            report = mirror.log_metrics(self.METRICS)
        self.assertEqual([t["name"] for t in runs[0].tracked], ["loss"])
        self.assertEqual(report.transmitted, 1)

    def test_the_omissions_are_counted_and_named(self):
        with fake_aim():
            mirror = AimMirror()
            mirror.open(record())
            report = mirror.log_metrics(self.METRICS)
        self.assertEqual(sorted(report.skipped_metrics),
                         ["auc", "converged", "grad_norm", "spearman"])
        self.assertIn("not computed", report.describe())

    def test_no_omitted_metric_is_substituted_with_zero(self):
        with fake_aim() as runs:
            mirror = AimMirror()
            mirror.open(record())
            mirror.log_metrics(self.METRICS)
        values = [t["value"] for t in runs[0].tracked]
        self.assertNotIn(0.0, values)
        self.assertFalse(any(isinstance(v, float) and math.isnan(v) for v in values))

    def test_a_flag_is_not_a_measurement(self):
        """
        `True` is a float to Python and would plot as 1.0 -- a flag rendered in
        the shape of a measurement, on the same axes as the measurements.
        """
        with fake_aim() as runs:
            mirror = AimMirror()
            mirror.open(record())
            mirror.log_metrics({"converged": True, "diverged": False})
        self.assertEqual(runs[0].tracked, [])

    def test_omissions_are_reported_even_when_aim_is_absent(self):
        """
        A caller learns which metrics were missing whether or not a dashboard
        exists: the fact is about the run, not about Aim.
        """
        with no_aim():
            report = AimMirror().log_metrics(self.METRICS)
        self.assertIs(report.status, AimStatus.NOT_INSTALLED)
        self.assertIn("auc", report.skipped_metrics)


# ---- provenance -------------------------------------------------------------

class TestProvenanceTravels(unittest.TestCase):

    def test_a_dirty_run_is_tagged_not_reproducible(self):
        with fake_aim() as runs:
            mirror = AimMirror()
            report = mirror.open(record(commit="b" * 40 + "-dirty"))
        self.assertIn("NOT-reproducible", runs[0].tags)
        self.assertFalse(runs[0].params["reproducible"])
        self.assertIn("uncommitted changes", " ".join(report.notes))

    def test_a_clean_run_carries_its_commit_and_seed(self):
        with fake_aim() as runs:
            AimMirror().open(record())
        stored = runs[0].params["run_record"]
        self.assertEqual(stored["git_commit"], "a" * 40)
        self.assertEqual(stored["seed"], 7)
        self.assertIn("reproducible", runs[0].tags)

    def test_the_reproduce_command_is_mirrored_verbatim(self):
        experiment = Experiment(name="fit", run=record(), metrics={"loss": 0.1})
        with fake_aim() as runs:
            AimMirror().log_experiment(experiment)
        self.assertEqual(runs[0].params["reproduce_command"],
                         experiment.reproduce_command())

    def test_an_unreproducible_run_mirrors_the_refusal_not_a_command(self):
        experiment = Experiment(name="fit", run=record(commit="UNKNOWN"),
                                metrics={"loss": 0.1})
        with fake_aim() as runs:
            AimMirror().log_experiment(experiment)
        self.assertIn("NOT reproducible", runs[0].params["reproduce_command"])


# ---- comparability ----------------------------------------------------------

class TestTwoRunsAreNotComparableJustBecauseBothAreLogged(unittest.TestCase):

    BASE = {"dataset_version": "v1", "split_strategy": "similarity",
            "encoder_model": "esm2", "encoder_version": "t33",
            "metric_definition": "spearman"}

    def test_identical_provenance_is_comparable(self):
        self.assertTrue(comparability(self.BASE, dict(self.BASE)).comparable)

    def test_a_different_dataset_is_not_comparable(self):
        other = dict(self.BASE, dataset_version="v2")
        verdict = comparability(self.BASE, other)
        self.assertFalse(verdict.comparable)
        self.assertIn("dataset_version", verdict.differing)
        self.assertIn("difference between the datasets", verdict.summary())

    def test_each_key_carries_its_own_reason(self):
        for key in self.BASE:
            with self.subTest(key=key):
                verdict = comparability(self.BASE, dict(self.BASE, **{key: "other"}))
                self.assertFalse(verdict.comparable)
                self.assertEqual(verdict.differing, (key,))
                self.assertTrue(verdict.reasons[0])

    def test_a_missing_field_is_unknown_rather_than_agreement(self):
        """
        Two runs that both fail to record their dataset version are not thereby
        known to share one. Reading silence as a match is how an incomparable
        pair ends up on one axis.
        """
        blank = dict(self.BASE, dataset_version="")
        verdict = comparability(blank, dict(blank))
        self.assertFalse(verdict.comparable)
        self.assertIn("dataset_version", verdict.unknown)
        self.assertEqual(verdict.differing, ())
        self.assertIn("cannot be checked and is not assumed", verdict.summary())

    def test_comparison_keys_omit_what_was_not_recorded(self):
        experiment = Experiment(name="fit", run=record(config={"split_strategy": "random"}),
                                metrics={}, embedding_model="esm2")
        keys = comparison_keys(experiment)
        self.assertEqual(keys["split_strategy"], "random")
        self.assertEqual(keys["encoder_model"], "esm2")
        self.assertNotIn("encoder_version", keys)
        self.assertNotIn("metric_definition", keys)


# ---- projections ------------------------------------------------------------

def projection(warnings=()):
    from ml.embeddings.explorer import ProjectedPoint, Projection, VariantClass
    points = [ProjectedPoint(label=f"p{i}", sequence="AAAA", x=float(i), y=-float(i),
                             variant_class=VariantClass.SINGLE, n_substitutions=1)
              for i in range(4)]
    return Projection(points=points, explained_variance_ratio=[0.6, 0.2],
                      encoder_model="esm2", encoder_version="t33",
                      encoder_kind="PRETRAINED", encoder_has_learned_content=True,
                      input_dim=1280, interpretation="Axes are the first two PCs.",
                      warnings=list(warnings), distinct_positions=4, distinct_projected=4)


class TestTheReductionHappensBeforeAim(unittest.TestCase):

    def test_the_points_are_sent_already_projected(self):
        with fake_aim() as runs:
            mirror = AimMirror()
            mirror.open(record())
            report = mirror.log_projection(projection())
        payload = runs[0].params["embedding_projection"]
        self.assertEqual(len(payload["points"]), 4)
        self.assertEqual(payload["explained_variance_ratio"], [0.6, 0.2])
        self.assertIn("explorer.project", payload["computed_by"])
        self.assertIn("Aim performed no reduction", " ".join(report.notes))

    def test_the_axes_say_what_they_are(self):
        with fake_aim() as runs:
            mirror = AimMirror()
            mirror.open(record())
            mirror.log_projection(projection())
        payload = runs[0].params["embedding_projection"]
        self.assertIn("first two PCs", payload["interpretation"])
        self.assertEqual(payload["encoder_version"], "t33")

    def test_a_weak_projections_warning_travels_with_its_points(self):
        """
        A projection explaining little of the variance still plots, and the
        plot is exactly as convincing as one that explains most of it.
        """
        warning = "The first two components explain 12% of the variance."
        with fake_aim() as runs:
            mirror = AimMirror()
            mirror.open(record())
            report = mirror.log_projection(projection(warnings=[warning]))
        self.assertIn(warning, runs[0].params["embedding_projection"]["warnings"])
        self.assertIn(warning, " ".join(report.notes))

    def test_projecting_without_aim_transmits_nothing_and_says_so(self):
        with no_aim():
            report = AimMirror().log_projection(projection())
        self.assertFalse(report.reached_aim)
        self.assertIn("not installed", report.describe())


# ---- the record stays the record -------------------------------------------

class TestAimDoesNotBecomeTheSystemOfRecord(unittest.TestCase):

    def test_the_jsonl_is_written_whether_or_not_aim_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = Tracker(directory=Path(directory))
            tracker.log(Experiment(name="fit", run=record(), metrics={"loss": 0.1}))
            self.assertEqual(len(tracker.all()), 1)
            self.assertTrue(tracker.reproducible())

    def test_mirroring_reads_the_record_rather_than_intercepting_the_write(self):
        """
        The direction is the design. One writer means the two can never
        disagree about what a run was, and deleting this module loses a
        dashboard rather than a run.
        """
        with tempfile.TemporaryDirectory() as directory:
            tracker = Tracker(directory=Path(directory))
            tracker.log(Experiment(name="fit", run=record(), metrics={"loss": 0.1}))
            before = tracker.all()
            with fake_aim() as runs:
                reports = mirror_tracked_runs(tracker)
            self.assertEqual(tracker.all(), before)
        self.assertEqual(len(reports), 1)
        self.assertEqual(runs[0].params["experiment_name"], "fit")
        self.assertTrue(runs[0].closed)

    def test_a_failing_mirror_leaves_the_record_intact(self):
        class Exploding(FakeRun):
            def __init__(self, repo=None, experiment=None):
                raise RuntimeError("down")

        with tempfile.TemporaryDirectory() as directory:
            tracker = Tracker(directory=Path(directory))
            tracker.log(Experiment(name="fit", run=record(), metrics={"loss": 0.1}))
            with fake_aim(run_factory=Exploding):
                reports = mirror_tracked_runs(tracker)
            self.assertEqual(len(tracker.all()), 1)
        self.assertIs(reports[0].status, AimStatus.FAILED)

    def test_the_existing_tracking_modules_are_untouched(self):
        """
        Item 11 of the specification: the Aim adapter adds a view, it does not
        replace provenance, reproducibility, tracking, RunRecord or checkpoint
        metadata. Asserted against the modules rather than trusted.
        """
        import ml.provenance, ml.reproducibility, ml.tracking
        self.assertTrue(hasattr(ml.tracking, "Tracker"))
        self.assertTrue(hasattr(ml.reproducibility, "RunRecord"))
        self.assertTrue(hasattr(ml.provenance, "MLProvenance"))
        for module in (ml.provenance, ml.reproducibility, ml.tracking):
            with self.subTest(module=module.__name__):
                self.assertNotIn("aim", module.__dict__)
                self.assertNotIn("aim_tracking", (module.__doc__ or "").lower())


if __name__ == "__main__":
    unittest.main()


class TestTheStatesAreDistinguishable(unittest.TestCase):
    """
    "Not opened yet" and "finalised" are different facts, and a mirror that
    reported the first as the second would tell a caller its run had ended
    before it began.
    """

    def test_ready_is_not_closed(self):
        with fake_aim():
            mirror = AimMirror()
            self.assertIs(mirror.status, AimStatus.READY)
            self.assertIn("no run has been opened", mirror.describe())
            mirror.open(record())
            self.assertIs(mirror.status, AimStatus.LIVE)
            mirror.close()
            self.assertIs(mirror.status, AimStatus.CLOSED)
            self.assertIn("finalised", mirror.describe())

    def test_only_live_transmits(self):
        for status in AimStatus:
            with self.subTest(status=status.value):
                self.assertEqual(status.is_transmitting, status is AimStatus.LIVE)

    def test_every_status_explains_itself(self):
        for status in AimStatus:
            with self.subTest(status=status.value):
                self.assertTrue(status.explanation.strip())

    def test_a_failed_open_is_not_retried_into_a_false_live(self):
        """
        Once opening has failed the mirror stays failed rather than trying
        again and reporting LIVE on a second call: a caller that saw the
        failure and carried on must not later find the run apparently logged.
        """
        class Exploding(FakeRun):
            def __init__(self, repo=None, experiment=None):
                raise RuntimeError("down")

        with fake_aim(run_factory=Exploding):
            mirror = AimMirror()
            mirror.open(record())
            second = mirror.open(record())
        self.assertIs(second.status, AimStatus.FAILED)
        self.assertFalse(second.reached_aim)


class TestAClosedRunIsActuallyVisible(unittest.TestCase):
    """
    Found against a real Aim install, which is the only place it shows.

    Aim writes each run to its own chunk directory and builds the searchable
    index from a separate daemon that `aim up` starts. `Repo.iter_runs` reads
    the index. So a run mirrored and closed by a script that then exits is on
    disk and invisible: the first version of this module reported "10 values
    sent to Aim" for a run that `iter_runs` could not find at all.

    That is the module's own stated failure mode arriving from the other side
    -- a success report for something that did not happen -- so closing indexes
    the run, and says so when it cannot.
    """

    def test_closing_reports_whether_the_run_became_visible(self):
        with fake_aim():
            mirror = AimMirror()
            mirror.open(record())
            report = mirror.close()
        self.assertIs(report.status, AimStatus.CLOSED)
        self.assertTrue(report.notes, "closing said nothing about visibility")

    def test_a_failed_index_says_the_run_is_stored_but_not_visible(self):
        """
        The fake has no `Repo`, so indexing raises -- which is the case that
        matters: the run is safe on disk and will not appear in the UI, and a
        silent close would leave someone looking for it.
        """
        with fake_aim():
            mirror = AimMirror()
            mirror.open(record())
            report = mirror.close()
        note = " ".join(report.notes)
        self.assertIn("not appear in the Aim UI", note)
        self.assertIn("JSONL record is unaffected", note)


@unittest.skipUnless(aim_available(), "Aim is not installed in this environment")
class TestAgainstRealAim(unittest.TestCase):
    """
    The fake asserts what this module asks Aim to do. Only a real install
    asserts that Aim does it -- and the indexing gap above is exactly the
    difference between the two, so this runs wherever Aim is present and skips
    loudly where it is not, rather than being quietly absent.
    """

    def mirror_one(self, directory, metrics):
        experiment = Experiment(
            name="integration", run=record(config={"split_strategy": "similarity"}),
            metrics=metrics, checkpoint="ckpt.pt", embedding_model="esm2")
        mirror = AimMirror(repo=directory, experiment="peptide-suite-tests")
        report = mirror.log_experiment(experiment)
        mirror.log_projection(projection())
        closing = mirror.close()
        return report, closing

    def test_a_mirrored_run_can_be_found_and_read_back(self):
        from aim import Repo
        with tempfile.TemporaryDirectory() as directory:
            report, closing = self.mirror_one(
                directory, {"loss": 0.25, "auc": None, "spearman": float("nan")})
            self.assertTrue(report.reached_aim)
            self.assertIn("indexed", " ".join(closing.notes))

            runs = list(Repo(directory).iter_runs())
            self.assertEqual(len(runs), 1, "the run was written but cannot be found")
            stored = runs[0][...]
            self.assertEqual(stored["run_record"]["git_commit"], "a" * 40)
            self.assertEqual(stored["experiment_name"], "integration")
            self.assertEqual(stored["split_strategy"], "similarity")
            self.assertEqual(len(stored["embedding_projection"]["points"]), 4)
            self.assertIn("reproducible", list(runs[0].props.tags))

    def test_an_uncomputed_metric_has_no_sequence_in_a_real_repository(self):
        """
        The strongest form of the claim: not that the module declined to send
        it, but that no such series exists in the database afterwards. Aim's
        own `__system__` sequences are excluded -- they are the host's CPU and
        memory, tracked by Aim itself, and are not this run's metrics.
        """
        from aim import Repo
        with tempfile.TemporaryDirectory() as directory:
            self.mirror_one(directory, {"loss": 0.25, "auc": None,
                                        "spearman": float("nan"), "converged": True})
            run = next(iter(Repo(directory).iter_runs()))
            names = {m.name for m in run.metrics()
                     if not m.name.startswith("__system__")}
        self.assertEqual(names, {"loss"})
