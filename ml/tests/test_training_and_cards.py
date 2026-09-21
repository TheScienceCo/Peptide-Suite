"""
Training, checkpoints, model cards and tracking.  [Addendum 3, sections 1, 7, 8]

The loop itself is unremarkable. What is tested is what it refuses to lose: a
checkpoint that cannot say what produced it, a card with a blank limitations
section, a tracked run that prints a command which would not reproduce it.
"""

import json
import tempfile
import unittest
from pathlib import Path

from ml.datasets.registry import TASKS
from ml.evaluation.metrics import evaluate_classification
from ml.evaluation.model_card import IncompleteModelCard, ModelCard
from ml.reproducibility import RunRecord
from ml.tracking import Experiment, Tracker
from ml.training.trainer import (
    EarlyStopping, TrainingConfig, load_checkpoint, record_path, save_checkpoint, train,
)


def a_run(**over):
    kwargs = dict(name="r", seed=0, device="cpu", git_commit="a" * 40,
                  dataset_version="v1")
    kwargs.update(over)
    return RunRecord(**kwargs)


def a_model():
    import torch.nn as nn
    return nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 1))


def batches(x, y, start, end, size=24):
    def generator():
        for i in range(start, end, size):
            yield x[i:i + size], y[i:i + size]
    return generator


class TestEarlyStopping(unittest.TestCase):

    def test_min_delta_rejects_noise_as_improvement(self):
        """
        Without it a run continues on changes in the sixth decimal place, which
        is not early stopping but a slower way to overfit.
        """
        stopper = EarlyStopping(patience=3, min_delta=0.01, mode="min")
        stopper.step(1.0, 1)
        self.assertFalse(stopper.is_improvement(0.9999))
        self.assertTrue(stopper.is_improvement(0.95))

    def test_patience_is_counted_and_triggers(self):
        stopper = EarlyStopping(patience=2, min_delta=0.0, mode="min")
        self.assertFalse(stopper.step(1.0, 1))
        self.assertFalse(stopper.step(1.5, 2))
        self.assertTrue(stopper.step(1.6, 3))

    def test_max_mode_inverts_the_comparison(self):
        stopper = EarlyStopping(patience=2, min_delta=0.0, mode="max")
        stopper.step(0.5, 1)
        self.assertTrue(stopper.is_improvement(0.6))
        self.assertFalse(stopper.is_improvement(0.4))


class TestTrainingRestoresTheBestWeights(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import torch
        import torch.nn as nn
        torch.manual_seed(0)
        cls.x = torch.randn(120, 6)
        cls.y = (cls.x[:, :3].sum(1, keepdim=True) > 0).float()
        cls.result = train(
            a_model(), batches(cls.x, cls.y, 0, 96), batches(cls.x, cls.y, 96, 120),
            nn.BCEWithLogitsLoss(),
            TrainingConfig(epochs=200, patience=8, learning_rate=5e-2),
            name="test", dataset_version="v1")

    def test_it_stops_early_on_patience(self):
        self.assertTrue(self.result.stopped_early)
        self.assertLess(self.result.epochs_run, 200)

    def test_the_best_epoch_is_not_the_last(self):
        """
        The run stopped because the metric got worse, so the final weights are
        by construction not the best ones. Keeping the last is the common
        default and it is wrong.
        """
        self.assertIsNotNone(self.result.best_epoch)
        self.assertLess(self.result.best_epoch, self.result.epochs_run)

    def test_the_history_covers_every_epoch_run(self):
        self.assertEqual(len(self.result.history), self.result.epochs_run)
        self.assertTrue(all(e.val_loss is not None for e in self.result.history))

    def test_the_run_record_travels_with_the_result(self):
        self.assertEqual(self.result.run.dataset_version, "v1")
        self.assertIn(self.result.run.device, ("cpu", "cuda", "mps"))


class TestCheckpointsCarryTheirRecord(unittest.TestCase):

    def test_a_record_is_written_beside_the_weights(self):
        import torch.nn as nn
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.pt"
            x = __import__("torch").randn(48, 6)
            y = (x[:, :3].sum(1, keepdim=True) > 0).float()
            result = train(a_model(), batches(x, y, 0, 48), None, nn.BCEWithLogitsLoss(),
                           TrainingConfig(epochs=3), checkpoint=path)
            self.assertTrue(path.exists())
            self.assertTrue(record_path(path).exists())
            record = json.loads(record_path(path).read_text())
            self.assertIn("run", record)
            self.assertIn("config", record)

    def test_loading_without_a_record_is_refused(self):
        """
        A checkpoint that cannot say which commit and dataset produced it
        cannot be reproduced, compared, or withdrawn.
        """
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "orphan.pt"
            torch.save(a_model().state_dict(), path)
            with self.assertRaises(FileNotFoundError) as ctx:
                load_checkpoint(path, a_model())
            self.assertIn("cannot be reproduced", str(ctx.exception))

    def test_two_checkpoints_do_not_share_one_record(self):
        """
        model.pt and model.bin both map to model.run.json under with_suffix,
        so one would silently overwrite the other's record.
        """
        self.assertNotEqual(record_path(Path("m.pt")), record_path(Path("m.bin")))
        self.assertEqual(record_path(Path("m.pt")).name, "m.pt.run.json")

    def test_a_saved_model_round_trips(self):
        import torch
        import torch.nn as nn
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.pt"
            x = torch.randn(48, 6)
            y = (x[:, :3].sum(1, keepdim=True) > 0).float()
            model = a_model()
            train(model, batches(x, y, 0, 48), None, nn.BCEWithLogitsLoss(),
                  TrainingConfig(epochs=3), checkpoint=path)
            reloaded, _record = load_checkpoint(path, a_model())
            with torch.no_grad():
                self.assertTrue(torch.allclose(model(x[:4]), reloaded(x[:4]), atol=1e-6))


class TestModelCardsRefuseToBeBlank(unittest.TestCase):

    def _metrics(self):
        return evaluate_classification([0, 1] * 12, [0.2, 0.8] * 12, n_resamples=100)

    def test_a_card_without_limitations_is_refused(self):
        with self.assertRaises(IncompleteModelCard) as ctx:
            ModelCard(task="t", architecture="MLP", dataset=TASKS["amp_activity"],
                      run=a_run(), split_strategy="clustered", identity_threshold=0.8)
        self.assertIn("model without limitations", str(ctx.exception))

    def test_a_card_without_a_split_strategy_is_refused(self):
        with self.assertRaises(IncompleteModelCard) as ctx:
            ModelCard(task="t", architecture="MLP", dataset=TASKS["amp_activity"],
                      run=a_run(), split_strategy="", identity_threshold=None,
                      limitations=["x"])
        self.assertIn("reports partly recall", str(ctx.exception))

    def test_the_card_states_the_similarity_control(self):
        card = ModelCard(task="t", architecture="MLP", dataset=TASKS["amp_activity"],
                         run=a_run(), split_strategy="sequence_clustered",
                         identity_threshold=0.8, metrics=self._metrics(),
                         limitations=["no real labels here"])
        markdown = card.to_markdown()
        self.assertIn("Sequence similarity control", markdown)
        self.assertIn("80% identity", markdown)

    def test_a_random_split_card_says_the_metrics_include_recall(self):
        card = ModelCard(task="t", architecture="MLP", dataset=TASKS["amp_activity"],
                         run=a_run(), split_strategy="random", identity_threshold=None,
                         limitations=["x"])
        self.assertIn("recalls rather than generalises", card.to_markdown())

    def test_a_synthetic_dataset_card_disclaims_biology(self):
        card = ModelCard(task="demo", architecture="MLP",
                         dataset=TASKS["split_methodology_demo"], run=a_run(),
                         split_strategy="sequence_clustered", identity_threshold=0.8,
                         limitations=["synthetic"])
        self.assertFalse(card.supports_biological_claim)
        self.assertIn("not a property of peptides", card.to_markdown())

    def test_intervals_are_shown_beside_point_estimates(self):
        card = ModelCard(task="t", architecture="MLP", dataset=TASKS["amp_activity"],
                         run=a_run(), split_strategy="clustered", identity_threshold=0.8,
                         metrics=self._metrics(), limitations=["x"])
        self.assertIn("95% interval", card.to_markdown())


class TestTracking(unittest.TestCase):

    def test_a_run_is_logged_and_read_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = Tracker(Path(tmp))
            tracker.log(Experiment(name="e1", run=a_run(), metrics={"roc_auc": 0.8}))
            entries = tracker.all()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["metrics"]["roc_auc"], 0.8)

    def test_a_reproducible_run_prints_a_command(self):
        command = Experiment(name="e", run=a_run()).reproduce_command()
        self.assertIn("git checkout", command)
        self.assertIn("--seed 0", command)

    def test_a_dirty_run_refuses_to_print_one(self):
        """A command that silently does not reproduce is worse than none."""
        command = Experiment(name="e", run=a_run(git_commit="abc-dirty")).reproduce_command()
        self.assertIn("NOT reproducible", command)
        self.assertNotIn("git checkout", command)

    def test_the_summary_counts_unreplayable_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = Tracker(Path(tmp))
            tracker.log(Experiment(name="clean", run=a_run()))
            tracker.log(Experiment(name="dirty", run=a_run(git_commit="abc-dirty")))
            summary = tracker.summary()
            self.assertIn("2 runs recorded, 1 reproducible", summary)
            self.assertIn("cannot be replayed", summary)

    def test_an_empty_tracker_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(Tracker(Path(tmp)).summary(), "No runs recorded.")


if __name__ == "__main__":
    unittest.main()
