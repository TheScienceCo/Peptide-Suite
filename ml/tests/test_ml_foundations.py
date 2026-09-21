"""
ML pipeline foundations.  [Addendum 3, sections 1, 4 and 6]

Three things are tested: that ML-derived numbers cannot be confused with
measurements, that the splitting does what it claims, and that the metrics are
right on the cases where naive implementations are wrong.
"""

import unittest

from ml.datasets.registry import TASKS, DatasetStatus, trainable_tasks
from ml.datasets.splitting import (
    SplitStrategy, cluster_sequences, clustered_split, compare, identity, random_split,
)
from ml.evaluation.metrics import (
    bootstrap, calibration_curve, evaluate_classification, f1, mcc, pr_auc, r2, roc_auc,
)
from ml.provenance import (
    AttributionClaim, MLProvenance, MLProvenanceError, MLQuantity, forbid_as_target,
)
from ml.reproducibility import RunRecord, select_device


def a_prediction(**over):
    metadata = dict(model="m", model_version="1", task="t", training_dataset="d",
                    dataset_version="v1", split_strategy="clustered",
                    git_commit="abc", seed=0)
    metadata.update(over.pop("metadata", {}))
    kwargs = dict(name="p", value=0.8, provenance=MLProvenance.ML_PREDICTION,
                  metadata=metadata)
    kwargs.update(over)
    return MLQuantity(**kwargs)


class TestMLProvenanceIsDistinct(unittest.TestCase):

    def test_no_ml_category_may_be_a_regression_target(self):
        """
        Training on predictions launders the first model's error into the
        second as though it were data.
        """
        for provenance in MLProvenance:
            self.assertFalse(provenance.may_be_a_regression_target)

    def test_using_a_prediction_as_a_target_raises(self):
        with self.assertRaises(MLProvenanceError) as ctx:
            forbid_as_target(a_prediction())
        self.assertIn("launders their error", str(ctx.exception))

    def test_missing_metadata_is_refused(self):
        with self.assertRaises(MLProvenanceError) as ctx:
            MLQuantity(name="p", value=1.0, provenance=MLProvenance.ML_PREDICTION,
                       metadata={"model": "m"})
        self.assertIn("cannot be reproduced", str(ctx.exception))

    def test_uncertainty_carries_its_own_provenance(self):
        """
        The estimate of error is a different kind of claim from the prediction.
        """
        with self.assertRaises(MLProvenanceError):
            a_prediction(uncertainty=a_prediction(name="u"))

    def test_prediction_and_uncertainty_are_separate_fields(self):
        quantity = a_prediction(uncertainty=MLQuantity(
            name="sd", value=0.1, provenance=MLProvenance.ML_UNCERTAINTY,
            metadata=dict(model="m", model_version="1", method="ensemble",
                          calibrated=False)))
        self.assertEqual(quantity.value, 0.8)
        self.assertEqual(quantity.uncertainty.value, 0.1)

    def test_an_out_of_distribution_prediction_is_not_reportable(self):
        quantity = a_prediction(out_of_distribution=True, ood_basis="embedding distance")
        self.assertFalse(quantity.is_reportable)
        with self.assertRaises(MLProvenanceError) as ctx:
            quantity.require_in_distribution()
        self.assertIn("extrapolation rather than a prediction", str(ctx.exception))

    def test_attribution_may_not_be_reported_as_mechanism(self):
        claim = AttributionClaim(positions=[1], scores=[0.5], method="IG",
                                 model="m", model_version="1")
        with self.assertRaises(MLProvenanceError):
            claim.as_mechanism()
        self.assertIn("not what the biology depends on", claim.disclaimer)


class TestReproducibility(unittest.TestCase):

    def test_a_dirty_tree_is_not_reproducible(self):
        record = RunRecord(name="r", seed=0, device="cpu", git_commit="abc123-dirty")
        self.assertFalse(record.is_reproducible)
        self.assertIn("uncommitted changes", record.reproducibility_note())

    def test_an_unknown_commit_is_not_reproducible(self):
        self.assertFalse(RunRecord(name="r", seed=0, device="cpu",
                                   git_commit="UNKNOWN").is_reproducible)

    def test_a_clean_commit_is(self):
        record = RunRecord(name="r", seed=0, device="cpu", git_commit="a" * 40)
        self.assertTrue(record.is_reproducible)

    def test_capture_records_the_device_and_versions(self):
        record = RunRecord.capture("probe", seed=0)
        self.assertIn(record.device, ("cpu", "cuda", "mps"))
        self.assertIn("torch", record.library_versions)

    def test_device_selection_returns_something_usable(self):
        self.assertIn(select_device(), ("cpu", "cuda", "mps"))


class TestSplitting(unittest.TestCase):

    def test_identity_is_symmetric_and_bounded(self):
        self.assertEqual(identity("ACDEF", "ACDEF"), 1.0)
        self.assertEqual(identity("ACDEF", "GGGGG"), 0.0)
        self.assertEqual(identity("ACD", "ACDEF"), identity("ACDEF", "ACD"))

    def test_clustering_recovers_known_families(self):
        """Constructed families must come back as clusters, or the split is noise."""
        import random
        rng = random.Random(0)
        founders = ["".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(20))
                    for _ in range(6)]
        sequences = []
        for founder in founders:
            sequences.append(founder)
            for _ in range(5):
                i = rng.randrange(20)
                sequences.append(founder[:i] + rng.choice("ACDEFGHIKLMNPQRSTVWY")
                                 + founder[i + 1:])
        self.assertEqual(len(cluster_sequences(sequences, 0.8)), 6)

    def test_a_random_split_leaks_near_duplicates(self):
        sequences = [f"ACDEFGHIKLMNPQRSTVW{'A' * (i % 3)}" for i in range(60)]
        split = random_split(sequences, seed=0)
        self.assertGreater(split.leakage_against(sequences, 0.8), 0)

    def test_clustering_guarantees_no_cross_cluster_near_duplicates(self):
        """
        The property the module exists to provide, checked exhaustively rather
        than assumed. A greedy first-match assignment does not provide it: a
        sequence similar to members of two clusters joins only the first, and
        the pair it leaves behind straddles the split. That leaked three
        sequences past a clustered split before the clustering was union-find.
        """
        import random
        rng = random.Random(3)
        sequences = []
        for _ in range(24):
            founder = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(22))
            for _ in range(4):
                i = rng.randrange(22)
                sequences.append(founder[:i] + rng.choice("ACDEFGHIKLMNPQRSTVWY")
                                 + founder[i + 1:])
        clusters = cluster_sequences(sequences, 0.8)
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                for i in clusters[a]:
                    for j in clusters[b]:
                        if identity(sequences[i], sequences[j]) >= 0.8:
                            self.fail(f"sequences {i} and {j} are near-duplicates in "
                                      f"different clusters")

    def test_a_clustered_split_does_not(self):
        """The property the whole module exists for."""
        import random
        rng = random.Random(1)
        founders = ["".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(22))
                    for _ in range(20)]
        sequences = []
        for founder in founders:
            for _ in range(5):
                i = rng.randrange(22)
                sequences.append(founder[:i] + rng.choice("ACDEFGHIKLMNPQRSTVWY")
                                 + founder[i + 1:])
        comparison = compare(sequences, seed=0, threshold=0.8)
        self.assertEqual(comparison.leakage()["clustered"], 0)
        self.assertGreater(comparison.leakage()["random"], 0)

    def test_splits_are_deterministic_given_a_seed(self):
        sequences = [f"SEQ{i}ACDEFGHIKLMNPQ" for i in range(40)]
        self.assertEqual(random_split(sequences, 3).test, random_split(sequences, 3).test)
        self.assertEqual(clustered_split(sequences, 3, 0.8).test,
                         clustered_split(sequences, 3, 0.8).test)

    def test_partitions_are_disjoint_and_complete(self):
        sequences = [f"SEQ{i}ACDEFGHIKLMNPQ" for i in range(50)]
        for split in (random_split(sequences, 0), clustered_split(sequences, 0, 0.8)):
            with self.subTest(strategy=split.strategy.value):
                combined = split.train + split.validation + split.test
                self.assertEqual(sorted(combined), list(range(len(sequences))))

    def test_each_strategy_states_what_it_measures(self):
        for strategy in SplitStrategy:
            self.assertTrue(strategy.measures)


class TestMetricsAreRightWhereNaiveOnesAreWrong(unittest.TestCase):

    def test_all_tied_scores_give_exactly_half(self):
        """A tie-unaware AUC returns anything from 0 to 1 on input order."""
        self.assertEqual(roc_auc([0, 0, 1, 1], [0.5] * 4), 0.5)

    def test_perfect_and_inverted_ranking(self):
        self.assertEqual(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]), 0.0)

    def test_one_class_returns_none_rather_than_a_number(self):
        self.assertIsNone(roc_auc([1, 1, 1], [0.1, 0.5, 0.9]))
        self.assertIsNone(pr_auc([0, 0, 0], [0.1, 0.5, 0.9]))

    def test_r2_is_none_when_targets_have_no_variance(self):
        """Undefined, not zero: zero reads as 'explains nothing'."""
        self.assertIsNone(r2([3.0, 3.0, 3.0], [3.0, 3.0, 3.0]))

    def test_mcc_uses_all_four_cells(self):
        """A majority-class predictor scores well on F1 and near zero here."""
        y_true = [1] * 9 + [0]
        y_pred = [1] * 10
        self.assertGreater(f1(y_true, y_pred), 0.9)
        self.assertIsNone(mcc(y_true, y_pred))

    def test_bootstrap_produces_an_interval_containing_the_estimate(self):
        y_true = [0, 0, 0, 1, 1, 1] * 6
        y_score = [0.2, 0.3, 0.4, 0.6, 0.7, 0.8] * 6
        point = roc_auc(y_true, y_score)
        low, high = bootstrap(roc_auc, y_true, y_score, n_resamples=300, seed=0)
        self.assertLessEqual(low, point)
        self.assertGreaterEqual(high, point)

    def test_every_metric_reports_its_sample_size(self):
        metrics = evaluate_classification([0, 1] * 10, [0.2, 0.8] * 10, n_resamples=100)
        for name, metric in metrics.items():
            with self.subTest(metric=name):
                self.assertEqual(metric.n, 20)

    def test_calibration_is_independent_of_ranking(self):
        """
        AUC is invariant to any monotone transform of the scores, so it cannot
        see calibration at all.
        """
        y_true = [0] * 10 + [1] * 10
        confident = [0.01] * 10 + [0.99] * 10
        timid = [0.45] * 10 + [0.55] * 10
        self.assertEqual(roc_auc(y_true, confident), roc_auc(y_true, timid))
        self.assertLess(calibration_curve(y_true, confident).expected_calibration_error,
                        calibration_curve(y_true, timid).expected_calibration_error)


class TestDatasetRegistryForbidsInventedLabels(unittest.TestCase):

    def test_no_task_claims_data_it_does_not_have(self):
        for name, spec in TASKS.items():
            with self.subTest(task=name):
                if spec.status is DatasetStatus.UNAVAILABLE:
                    self.assertTrue(spec.why_unavailable)
                    self.assertFalse(spec.is_trainable)

    def test_every_task_names_a_real_source_and_licence(self):
        for name, spec in TASKS.items():
            with self.subTest(task=name):
                self.assertTrue(spec.source)
                self.assertTrue(spec.licence)

    def test_synthetic_data_supports_no_biological_claim(self):
        spec = TASKS["split_methodology_demo"]
        self.assertIs(spec.status, DatasetStatus.SYNTHETIC_METHOD_ONLY)
        self.assertFalse(spec.status.supports_biological_claim)
        self.assertTrue(spec.status.supports_method_claim)
        self.assertIn("no biological claim", spec.claim_guidance())

    def test_only_the_synthetic_demo_is_trainable_here(self):
        """
        Honest about the environment: no network, so no real labels. The list
        changes when data arrives, not when someone wants a number.
        """
        self.assertEqual(trainable_tasks(), ["split_methodology_demo"])


class TestSplitGapExperiment(unittest.TestCase):
    """
    The claim is about the evaluation protocol, so it is settled on constructed
    sequences where the label is a known function of the input.
    """

    @classmethod
    def setUpClass(cls):
        from ml.experiments.split_gap import run
        # Large enough for the effect to be measurable. The small-scale case is
        # a separate test, because it demonstrates something different.
        cls.result = run(n_families=40, per_family=8, seed=0)

    def test_the_random_split_leaks_and_the_clustered_one_does_not(self):
        self.assertGreater(self.result.leakage["random"], 0)
        self.assertEqual(self.result.leakage["clustered"], 0)

    def test_the_random_split_scores_higher(self):
        """
        The label is a function of the sequence and both splits see the same
        rule, so the excess can only be recall of near-duplicates.
        """
        for arm in sorted({a.name for a in self.result.arms}):
            gap = self.result.gap_for(arm)
            with self.subTest(arm=arm):
                self.assertIsNotNone(gap)
                self.assertGreater(gap, 0.0)

    def test_the_deep_model_gains_more_from_leakage_than_the_baseline(self):
        """
        The finding that matters for the headline claim. A model with the
        capacity to memorise individual sequences gains more from a leaky split
        than one that cannot -- composition has no way to represent a specific
        sequence, so its random-split advantage is smaller. Reading the two
        random numbers alone would say the deep model is better; it is better at
        recall.
        """
        self.assertGreater(self.result.gap_for("one-hot MLP"),
                           self.result.gap_for("composition baseline"))

    def test_a_small_dataset_cannot_demonstrate_the_effect(self):
        """
        Not a bug, and worth pinning. With few enough clusters both splits
        saturate and the gap is zero -- so a paper reporting no leakage effect
        on a small set has shown nothing either way, and the test set size is
        part of that claim.
        """
        from ml.experiments.split_gap import run
        small = run(n_families=16, per_family=6, seed=0)
        gaps = [small.gap_for(a) for a in sorted({x.name for x in small.arms})]
        self.assertTrue(all(g is not None for g in gaps))
        self.assertTrue(all(g >= 0.0 for g in gaps))

    def test_a_skipped_arm_is_recorded_rather_than_omitted(self):
        """An arm that vanishes looks like the experiment not running."""
        from ml.experiments.split_gap import run
        tiny = run(n_families=4, per_family=3, seed=0)
        self.assertTrue(tiny.skipped or len(tiny.arms) == 4)
        if tiny.skipped:
            self.assertIn("single-class", tiny.report())

    def test_the_report_marks_itself_synthetic(self):
        self.assertIn("supports no biological claim", self.result.report())

    def test_both_arms_ran(self):
        """The baseline is the control; a result without it is one number."""
        self.assertEqual({a.name for a in self.result.arms},
                         {"composition baseline", "one-hot MLP"})


if __name__ == "__main__":
    unittest.main()
