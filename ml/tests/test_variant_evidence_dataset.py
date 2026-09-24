"""
Turning measured outcomes into labels, and the four rows that are not labels.

The store looks like a labelled dataset and very nearly is one. What stops it
is not row count: most of what a store like this holds cannot be a target
value, for reasons that survive the store being large.
"""

from __future__ import annotations

import unittest

from ml.datasets.variant_evidence_dataset import (
    DEFAULT_TIER_FLOOR, ExclusionReason, LeakageError, TrainingRow,
    build_dataset, check_leakage, require_no_leakage, split_by_parent, status_report,
)
from peptide_suite.core import EvidenceTier
from peptide_suite.core.biological_context import Provenance, SourceKind
from peptide_suite.core.variant_evidence import (
    Direction, Extraction, MeasuredOutcome, Modification, ModificationKind,
    OutcomeMeasure, VariantEvidenceStore, VariantRecord,
)

CITED = Provenance(kind=SourceKind.PRIMARY_LITERATURE, pmid="99999999", year=2020)
UNCITED = Provenance(kind=SourceKind.CURATED_UNVERIFIED, needs_verification=True)
MEASURE = OutcomeMeasure.PROTEASE_STABILITY


def outcome(**kwargs) -> MeasuredOutcome:
    # The extraction is stated because it has to be: an outcome that does not
    # say who read it out caps at BIOCHEMICAL_PRINCIPLE, below the tier floor,
    # so a fixture omitting it tests the floor rather than the rule under test.
    fields = dict(measure=MEASURE, direction=Direction.INCREASED,
                  comparator="the parent peptide", assay="synthetic, this test only",
                  provenance=CITED, fold_change=3.0,
                  extraction=Extraction.CURATOR_READ_FULL_TEXT)
    fields.update(kwargs)
    return MeasuredOutcome(**fields)


def record(name, parent="PARENT-A", modifications=None, outcomes=()) -> VariantRecord:
    return VariantRecord(
        name=name, parent=parent, provenance=UNCITED, outcomes=tuple(outcomes),
        modifications=tuple(modifications or [
            Modification(kind=ModificationKind.SUBSTITUTION, description="A8G",
                         position=8, wild_type="A", mutant="G")]))


def fake_store(records) -> VariantEvidenceStore:
    store = VariantEvidenceStore()
    store._records = list(records)
    return store


class TestFourRowsThatAreNotLabels(unittest.TestCase):

    def build(self, records):
        return build_dataset(MEASURE, store=fake_store(records))

    def test_a_design_has_no_target_value(self):
        build = self.build([record("design")])
        self.assertFalse(build.is_trainable)
        self.assertIn("design", build.excluded[ExclusionReason.NO_MEASURED_OUTCOME.value])

    def test_a_confounded_record_is_excluded(self):
        combination = record("combo", modifications=[
            Modification(kind=ModificationKind.SUBSTITUTION, description="A8G",
                         position=8, wild_type="A", mutant="G"),
            Modification(kind=ModificationKind.LIPIDATION, description="C18 chain",
                         position=26)],
            outcomes=[outcome()])
        build = self.build([combination])
        self.assertFalse(build.is_trainable)
        self.assertIn("combo",
                      build.excluded[ExclusionReason.CONFOUNDED_COMBINATION.value])
        self.assertIn("attribution nobody made", build.report())

    def test_a_direction_without_a_magnitude_is_not_a_regression_target(self):
        build = self.build([record("qualitative", outcomes=[
            outcome(direction=Direction.NOT_DETERMINED, fold_change=None)])])
        self.assertFalse(build.is_trainable)
        self.assertTrue(build.excluded[ExclusionReason.NOT_QUANTITATIVE.value])

    def test_an_uncited_outcome_is_below_the_tier_floor(self):
        build = self.build([record("uncited",
                                   outcomes=[outcome(provenance=UNCITED)])])
        self.assertFalse(build.is_trainable)
        self.assertTrue(build.excluded[ExclusionReason.BELOW_TIER_FLOOR.value])
        self.assertIn("inherit the unverifiability", build.report())

    def test_a_cited_attributable_quantitative_outcome_becomes_a_row(self):
        build = self.build([record("good", outcomes=[outcome()])])
        self.assertTrue(build.is_trainable)
        self.assertEqual(len(build.rows), 1)
        row = build.rows[0]
        self.assertEqual(row.target_value, 3.0)
        self.assertTrue(row.target_is_fold_change)
        self.assertIs(row.tier, EvidenceTier.DIRECT_EXPERIMENTAL)

    def test_every_exclusion_reason_explains_itself(self):
        for reason in ExclusionReason:
            with self.subTest(reason=reason.value):
                self.assertTrue(reason.explanation.strip())

    def test_exclusions_are_reported_rather_than_dropped(self):
        """
        A builder that quietly drops four fifths of a store reports a clean
        dataset, and the number that matters -- how much of the evidence was
        usable -- is the one that disappears.
        """
        build = self.build([record("design"),
                            record("good", outcomes=[outcome()]),
                            record("uncited", outcomes=[outcome(provenance=UNCITED)])])
        self.assertEqual(len(build.rows), 1)
        self.assertEqual(sum(len(v) for v in build.excluded.values()), 2)
        self.assertIn("excluded", build.report())

    def test_the_tier_floor_is_the_weakest_tier_that_still_means_measured(self):
        self.assertIs(DEFAULT_TIER_FLOOR, EvidenceTier.HOMOLOG_EXPERIMENTAL)


class TestLeakageIsTheSpecificRiskHere(unittest.TestCase):
    """
    This store is built out of variant series. An alanine scan is thirty
    records sharing one parent, so a variant-level split keeps twenty-nine of
    them in training and scores the model on a peptide it has already seen.
    """

    def rows(self, parent, names):
        return [TrainingRow(parent=parent, variant_name=n, modification_label="A8G",
                            position=8, wild_type="A", mutant="G", measure=MEASURE,
                            target_value=2.0, target_is_fold_change=True,
                            comparator="parent", assay="synthetic",
                            tier=EvidenceTier.DIRECT_EXPERIMENTAL)
                for n in names]

    def test_a_shared_parent_is_leakage_even_with_distinct_variants(self):
        report = check_leakage(self.rows("GLP-1", ["v1", "v2"]),
                               self.rows("GLP-1", ["v3"]))
        self.assertFalse(report.is_clean)
        self.assertEqual(report.shared_parents, ("GLP-1",))
        self.assertEqual(report.shared_variants, ())
        self.assertIn("effectively already seen", report.statement())

    def test_a_shared_variant_is_leakage(self):
        report = check_leakage(self.rows("A", ["v1"]), self.rows("B", ["v1"]))
        self.assertFalse(report.is_clean)
        self.assertEqual(report.shared_variants, ("v1",))

    def test_distinct_parents_are_clean(self):
        report = check_leakage(self.rows("A", ["v1"]), self.rows("B", ["v2"]))
        self.assertTrue(report.is_clean)
        self.assertIn("No parent molecule", report.statement())

    def test_require_no_leakage_refuses_rather_than_reporting(self):
        with self.assertRaises(LeakageError):
            require_no_leakage(self.rows("A", ["v1"]), self.rows("A", ["v2"]))

    def test_the_grouped_split_cannot_leak_by_construction(self):
        rows = (self.rows("A", [f"a{i}" for i in range(30)])
                + self.rows("B", ["b1", "b2"])
                + self.rows("C", ["c1"]))
        train, test = split_by_parent(rows, test_fraction=0.34, seed=1)
        self.assertTrue(check_leakage(train, test).is_clean)
        self.assertTrue(train and test)

    def test_the_split_fraction_is_over_groups_not_rows(self):
        """
        A parent with thirty variants does not drag the split with it, so the
        row counts will not match the requested fraction. That is correct
        behaviour rather than an approximation error.
        """
        rows = self.rows("A", [f"a{i}" for i in range(30)]) + self.rows("B", ["b1"])
        train, test = split_by_parent(rows, test_fraction=0.5, seed=0)
        self.assertEqual({r.group for r in train} & {r.group for r in test}, set())
        self.assertNotEqual(len(test) / len(rows), 0.5)

    def test_the_group_is_the_parent_not_the_variant(self):
        row = self.rows("GLP-1 (7-37)", ["v1"])[0]
        self.assertEqual(row.group, "GLP-1 (7-37)".upper())


class TestTheEmptyStoreProducesNothingAndSaysWhy(unittest.TestCase):

    def test_no_measure_is_trainable(self):
        report = status_report()
        self.assertEqual(report["trainable_measures"], [])
        self.assertIn("would be the failure this layer is written against",
                      report["statement"])

    def test_every_measure_is_accounted_for(self):
        report = status_report()
        self.assertEqual(len(report["builds"]), len(list(OutcomeMeasure)))
        for name, build in report["builds"].items():
            with self.subTest(measure=name):
                self.assertFalse(build["is_trainable"])
                self.assertEqual(build["rows"], 0)

    def test_the_store_status_travels_with_the_report(self):
        self.assertEqual(status_report()["store"]["records"], 0)


if __name__ == "__main__":
    unittest.main()
