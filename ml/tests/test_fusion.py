"""
Multimodal fusion and whether it helps.  [Addendum 3, section 5]

The architectures are the easy part. What has to be right is the reporting:
a fusion arm that beats its baseline by 0.02 with overlapping intervals has
not beaten it, and a comparison that cannot separate its arms from a
shuffled-label null has not compared anything.
"""

import unittest

from ml.evaluation.metrics import Metric
from ml.experiments.fusion_benefit import (
    MOTIF,
    Arm,
    build_dataset,
    intervals_overlap,
    net_charge,
    physicochemistry,
    resolution_for,
    run,
    verdict_for,
)
from ml.multimodal.fusion import (
    build_fusion_model, concatenation_model, single_modality_model,
)


def metric(value, low, high, n=100):
    return Metric(name="ROC-AUC", value=value, n=n, ci_low=low, ci_high=high)


def arm(name, kind, value, low, high, gate=None):
    return Arm(name=name, kind=kind, gate_share=gate,
               metrics={"roc_auc": metric(value, low, high)})


class TestTheTaskNeedsBothModalities(unittest.TestCase):

    def setUp(self):
        self.sequences, self.labels = build_dataset(40, 8, 24, seed=0)

    def test_the_label_is_not_the_motif_alone(self):
        motif = [1 if MOTIF in s else 0 for s in self.sequences]
        self.assertNotEqual(motif, self.labels,
                            "if the motif alone decided the label, the physicochemical "
                            "modality would be decoration")
        self.assertTrue(any(m == 1 and y == 0 for m, y in zip(motif, self.labels)))

    def test_the_label_is_not_the_charge_alone(self):
        charged = [1 if net_charge(s) > 0 else 0 for s in self.sequences]
        self.assertNotEqual(charged, self.labels)
        self.assertTrue(any(c == 1 and y == 0 for c, y in zip(charged, self.labels)))

    def test_the_physicochemical_modality_cannot_see_a_motif(self):
        # Same residues, different order: identical features, and only one of
        # them carries the motif.
        with_motif = MOTIF + "AAAA"
        without = "".join(sorted(MOTIF)) + "AAAA"
        self.assertNotEqual(with_motif, without)
        self.assertEqual(physicochemistry(with_motif), physicochemistry(without))

    def test_both_classes_are_present(self):
        self.assertGreater(sum(self.labels), 0)
        self.assertLess(sum(self.labels), len(self.labels))


class TestTheVerdictRule(unittest.TestCase):

    def test_overlapping_intervals_are_not_a_win(self):
        arms = [arm("naive concatenation", "concatenation", 0.80, 0.74, 0.86),
                arm("learned fusion", "fusion", 0.84, 0.78, 0.90)]
        said = verdict_for(arms)
        self.assertIn("does not distinguish", said)
        self.assertIn("No claim that fusion helps", said)

    def test_separated_intervals_are_reported_as_one_comparison_not_a_law(self):
        arms = [arm("naive concatenation", "concatenation", 0.60, 0.55, 0.65),
                arm("learned fusion", "fusion", 0.90, 0.86, 0.94)]
        said = verdict_for(arms)
        self.assertIn("non-overlapping", said)
        self.assertIn("not about fusion in general", said)

    def test_a_collapsed_gate_is_disclosed_even_when_fusion_wins(self):
        arms = [arm("naive concatenation", "concatenation", 0.60, 0.55, 0.65),
                arm("learned fusion", "fusion", 0.90, 0.86, 0.94, gate=[1.0, 0.0])]
        said = verdict_for(arms)
        self.assertIn("single modality", said)
        self.assertIn("extra layers", said)

    def test_a_missing_interval_is_not_a_narrow_one(self):
        self.assertIsNone(intervals_overlap(metric(0.9, None, None), metric(0.6, 0.5, 0.7)))
        arms = [arm("naive concatenation", "concatenation", 0.60, 0.55, 0.65),
                Arm(name="learned fusion", kind="fusion",
                    metrics={"roc_auc": metric(0.90, None, None)})]
        self.assertIn("No comparison is claimed", verdict_for(arms))

    def test_no_fusion_arm_means_no_statement_about_fusion(self):
        arms = [arm("naive concatenation", "concatenation", 0.60, 0.55, 0.65)]
        self.assertIn("says nothing about fusion", verdict_for(arms))


class TestTheResolutionCheck(unittest.TestCase):
    """
    A single shuffled-label run spans a quarter of the AUC range on a test set
    this size, so the null is a permutation over several shuffles and the check
    is stated in those terms.
    """

    def test_beating_every_shuffle_gives_a_permutation_bound(self):
        arms = [arm("sequence only", "single", 0.90, 0.85, 0.95)]
        said = resolution_for(arms, [0.44, 0.52, 0.39, 0.61, 0.48])
        self.assertIn("beats all 5", said)
        self.assertIn("p <", said)

    def test_failing_to_beat_the_null_is_called_under_powered(self):
        arms = [arm("sequence only", "single", 0.55, 0.45, 0.65)]
        said = resolution_for(arms, [0.44, 0.52, 0.39, 0.61, 0.78])
        self.assertIn("Under-powered", said)
        self.assertIn("3 of 5", said)  # 0.55 beats 0.44, 0.52 and 0.39

    def test_no_null_means_the_resolution_is_unknown_not_fine(self):
        arms = [arm("sequence only", "single", 0.90, 0.85, 0.95)]
        self.assertIn("unknown", resolution_for(arms, []))


class TestArchitectures(unittest.TestCase):

    def test_the_gate_is_a_distribution_over_modalities(self):
        import torch
        model = build_fusion_model([6, 4], width=8)
        parts = [torch.randn(5, 6), torch.randn(5, 4)]
        model(parts)
        self.assertIsNotNone(model.last_gate)
        self.assertEqual(tuple(model.last_gate.shape), (5, 2))
        for row in model.last_gate.sum(dim=1).tolist():
            self.assertAlmostEqual(row, 1.0, places=5)

    def test_the_baseline_is_not_handicapped(self):
        # Same hidden width as the fusion head, so a difference between them is
        # about the mechanism rather than about capacity.
        concat = concatenation_model([6, 4], width=8)
        single = single_modality_model(6, width=8)
        self.assertEqual(concat[0].in_features, 10)
        self.assertEqual(concat[0].out_features, 8)
        self.assertEqual(single[0].out_features, 8)


class TestTheExperimentRuns(unittest.TestCase):

    def test_four_arms_and_a_null(self):
        result = run(n_families=24, per_family=6, seed=0, n_permutations=3)
        kinds = [a.kind for a in result.arms]
        if result.skipped:
            self.assertIn("single-class", result.skipped[0])
            return
        self.assertEqual(kinds.count("single"), 2)
        self.assertEqual(kinds.count("concatenation"), 1)
        self.assertEqual(kinds.count("fusion"), 1)
        self.assertEqual(len(result.permutation_null), 3)
        self.assertTrue(result.verdict)
        self.assertTrue(result.resolution)
        self.assertIn("SYNTHETIC", result.report())

    def test_a_single_class_partition_is_recorded_not_dropped(self):
        # Too few families to hold out both classes. The experiment must say so
        # rather than returning an empty arm list that looks like a crash.
        result = run(n_families=4, per_family=4, seed=0, n_permutations=2)
        if not result.arms:
            self.assertTrue(result.skipped)
            self.assertIn("single-class", result.skipped[0])


if __name__ == "__main__":
    unittest.main()
