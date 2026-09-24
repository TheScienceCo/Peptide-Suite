"""
Empirical variant evidence: the schema, and the four things it refuses.

The store holds what somebody measured about a modified peptide. Everything
here is about keeping that separate from the three things it is constantly
mistaken for: a list of modifications somebody made, a number with no baseline,
and an effect attributed to one change out of several.
"""

import json
import pathlib
import unittest

from peptide_suite.core import EvidenceTier
from peptide_suite.core.biological_context import Provenance, SourceKind
from peptide_suite.core.variant_evidence import (
    DATA_PATH, Direction, EMPTY_STORE_STATEMENT, EvidenceMatch, EvidenceSchemaError,
    Extraction,
    MatchKind, MeasuredOutcome, Modification, ModificationKind, OutcomeMeasure,
    PrecedentResult, VariantEvidenceStore, VariantRecord, precedent_for, store,
)

CITED = Provenance(kind=SourceKind.PRIMARY_LITERATURE, pmid="99999999", year=2020)
UNCITED = Provenance(kind=SourceKind.CURATED_UNVERIFIED, needs_verification=True)


def outcome(**kwargs) -> MeasuredOutcome:
    fields = dict(measure=OutcomeMeasure.PROTEASE_STABILITY,
                  direction=Direction.INCREASED, comparator="the parent peptide",
                  assay="synthetic assay, this test only", provenance=CITED,
                  fold_change=3.0)
    fields.update(kwargs)
    return MeasuredOutcome(**fields)


def substitution(position=8, wild_type="A", mutant="G") -> Modification:
    return Modification(kind=ModificationKind.SUBSTITUTION,
                        description=f"{wild_type}{position}{mutant}",
                        position=position, wild_type=wild_type, mutant=mutant)


def variant(name="TEST-1", parent="TESTPEPTIDE", modifications=None,
            outcomes=(), provenance=UNCITED, aliases=()) -> VariantRecord:
    return VariantRecord(
        name=name, parent=parent,
        modifications=tuple(modifications or [substitution()]),
        provenance=provenance, outcomes=tuple(outcomes), aliases=tuple(aliases))


class TestAMeasurementNeedsABaseline(unittest.TestCase):
    """
    A fold change survives being quoted into a summary. Its missing baseline
    does not, so the baseline is required at construction rather than checked
    at display time.
    """

    def test_no_comparator_is_refused(self):
        with self.assertRaises(EvidenceSchemaError) as caught:
            outcome(comparator="  ")
        self.assertIn("without a baseline", str(caught.exception))

    def test_no_assay_is_refused(self):
        with self.assertRaises(EvidenceSchemaError) as caught:
            outcome(assay="")
        self.assertIn("not comparable", str(caught.exception))

    def test_an_absolute_value_needs_a_unit(self):
        with self.assertRaises(EvidenceSchemaError):
            outcome(fold_change=None, value=12.0, unit="")

    def test_an_affinity_needs_the_thing_it_binds(self):
        with self.assertRaises(EvidenceSchemaError) as caught:
            outcome(measure=OutcomeMeasure.AFFINITY_KD, target="")
        self.assertIn("names no interaction", str(caught.exception))

    def test_an_affinity_with_a_target_is_accepted(self):
        self.assertTrue(outcome(measure=OutcomeMeasure.AFFINITY_KD, target="IGF1R"))

    def test_a_contradiction_between_direction_and_magnitude_is_refused(self):
        """UNCHANGED alongside a 12-fold change. One is wrong and which cannot
        be guessed here, so neither is silently preferred."""
        with self.assertRaises(EvidenceSchemaError) as caught:
            outcome(direction=Direction.UNCHANGED, fold_change=12.0)
        self.assertIn("which one cannot be guessed", str(caught.exception))

    def test_a_negative_fold_change_is_refused(self):
        with self.assertRaises(EvidenceSchemaError):
            outcome(fold_change=-2.0)

    def test_not_determined_is_kept_apart_from_unchanged(self):
        """
        A study that did not resolve the direction and one that found no change
        are different results. Collapsing them turns the first into the second.
        """
        self.assertIsNot(Direction.NOT_DETERMINED, Direction.UNCHANGED)
        undetermined = outcome(direction=Direction.NOT_DETERMINED, fold_change=None)
        self.assertFalse(undetermined.is_quantitative)
        self.assertIn("magnitude not recorded", undetermined.describe())


class TestNoCitationNoDirectExperimental(unittest.TestCase):
    """
    The rule is enforced through Provenance.max_tier, the same gate the
    biological-context layer uses, rather than re-implemented here where it
    could drift.
    """

    def test_a_cited_outcome_read_by_a_curator_reaches_direct_experimental(self):
        """
        Both halves are needed now: the tier is the weaker of what the source
        permits and what the extraction permits. A citation says the paper is
        real; it says nothing about whether the number was read correctly.
        """
        self.assertIs(
            outcome(provenance=CITED,
                    extraction=Extraction.CURATOR_READ_FULL_TEXT).tier,
            EvidenceTier.DIRECT_EXPERIMENTAL)

    def test_a_cited_outcome_nobody_is_recorded_as_having_read_does_not(self):
        self.assertIs(outcome(provenance=CITED).extraction, Extraction.UNRECORDED)
        self.assertIs(outcome(provenance=CITED).tier,
                      EvidenceTier.BIOCHEMICAL_PRINCIPLE)

    def test_an_uncited_outcome_caps_below_it(self):
        self.assertIs(outcome(provenance=UNCITED).tier,
                      EvidenceTier.BIOCHEMICAL_PRINCIPLE)

    def test_no_provenance_kind_reaches_direct_experimental_without_a_source(self):
        for kind in SourceKind:
            bare = Provenance(kind=kind)
            if bare.max_tier is EvidenceTier.DIRECT_EXPERIMENTAL:
                self.fail(f"{kind.value} reached DIRECT_EXPERIMENTAL with no citation "
                          f"or accession")

    def test_the_records_strongest_tier_is_its_weakest_link(self):
        record = variant(outcomes=[
            outcome(provenance=CITED, extraction=Extraction.CURATOR_READ_FULL_TEXT),
            outcome(provenance=UNCITED)])
        self.assertIs(record.strongest_tier, EvidenceTier.DIRECT_EXPERIMENTAL)


class TestADesignIsNotEvidence(unittest.TestCase):
    """
    A record of modifications with nothing measured is a design. The
    distinction exists because a table of modification names under a column
    headed Evidence acquires the column's meaning.
    """

    def test_a_record_with_no_outcomes_reports_no_evidence(self):
        self.assertFalse(variant().has_evidence)

    def test_a_design_cannot_support_a_claim_about_effect(self):
        match = EvidenceMatch(record=variant(),
                              kind=MatchKind.SAME_PEPTIDE_SAME_SUBSTITUTION)
        self.assertFalse(match.is_usable_as_evidence)

    def test_a_record_with_no_modification_is_refused(self):
        with self.assertRaises(EvidenceSchemaError) as caught:
            VariantRecord(name="X", parent="Y", modifications=(), provenance=UNCITED)
        self.assertIn("nothing for an outcome to be about", str(caught.exception))

    def test_a_measured_attributable_record_is_usable(self):
        match = EvidenceMatch(record=variant(outcomes=[outcome()]),
                              kind=MatchKind.SAME_PEPTIDE_SAME_SUBSTITUTION)
        self.assertTrue(match.is_usable_as_evidence)


class TestCombinationChangesAreConfounded(unittest.TestCase):
    """
    The standing example is a long-acting GLP-1 analogue: a backbone
    substitution, a sequence substitution and an attached fatty-acid chain, one
    measured half-life. Attributing it to the substitution is the readiest
    error in this domain, because the substitution is the part that looks like
    the other rows in a substitution table.
    """

    def combination(self):
        return variant(modifications=[
            Modification(kind=ModificationKind.BACKBONE_MODIFICATION,
                         description="backbone substitution at position 8", position=8),
            substitution(position=34, wild_type="K", mutant="R"),
            Modification(kind=ModificationKind.LIPIDATION,
                         description="fatty-acid chain at a named lysine", position=26),
        ], outcomes=[outcome(measure=OutcomeMeasure.PLASMA_HALF_LIFE,
                             fold_change=None, value=100.0, unit="h")])

    def test_more_than_one_change_is_marked_confounded(self):
        confounding = self.combination().confounding()
        self.assertTrue(confounding.is_confounded)
        self.assertEqual(len(confounding.changes), 3)

    def test_the_note_names_every_change_not_just_the_substitution(self):
        statement = self.combination().confounding().statement
        for change in ("backbone", "K34R", "fatty-acid"):
            self.assertIn(change, statement)

    def test_a_confounded_record_is_not_attributable(self):
        self.assertFalse(self.combination().is_attributable)

    def test_a_confounded_record_cannot_support_a_substitution_claim(self):
        match = EvidenceMatch(record=self.combination(),
                              kind=MatchKind.SAME_PEPTIDE_SAME_SUBSTITUTION)
        self.assertFalse(match.is_usable_as_evidence)

    def test_a_single_change_is_attributable_and_says_so(self):
        confounding = variant(outcomes=[outcome()]).confounding()
        self.assertFalse(confounding.is_confounded)
        self.assertIn("attributable", confounding.statement)

    def test_a_lipidation_at_a_named_position_is_not_a_point_change(self):
        """
        It has a position, which is exactly what makes it look like one. The
        chain's effects are not local to that residue, and treating it as a
        point change would let it be compared against a substitution table.
        """
        self.assertFalse(ModificationKind.LIPIDATION.is_point_change)
        self.assertTrue(ModificationKind.SUBSTITUTION.is_point_change)
        self.assertTrue(ModificationKind.NCAA_SUBSTITUTION.is_point_change)


class TestRetrievalRanksByHowCloselyItBears(unittest.TestCase):

    def store_with(self, records):
        loaded = VariantEvidenceStore()
        loaded._records = list(records)
        return loaded

    def test_the_exact_change_in_the_same_molecule_comes_first(self):
        records = [
            variant(name="OTHER-PEPTIDE", parent="SOMETHINGELSE",
                    modifications=[substitution()], outcomes=[outcome()]),
            variant(name="SAME-POSITION", parent="TESTPEPTIDE",
                    modifications=[substitution(mutant="V")], outcomes=[outcome()]),
            variant(name="EXACT", parent="TESTPEPTIDE",
                    modifications=[substitution()], outcomes=[outcome()]),
        ]
        result = self.store_with(records).for_substitution("TESTPEPTIDE", 8, "A", "G")
        self.assertEqual([m.record.name for m in result.matches][0], "EXACT")
        self.assertIs(result.matches[0].kind, MatchKind.SAME_PEPTIDE_SAME_SUBSTITUTION)

    def test_only_the_exact_change_transfers_without_an_argument(self):
        for kind in MatchKind:
            with self.subTest(kind=kind.value):
                self.assertEqual(kind.transfers_directly,
                                 kind is MatchKind.SAME_PEPTIDE_SAME_SUBSTITUTION)
                self.assertTrue(kind.caveat.strip())

    def test_the_same_change_elsewhere_is_returned_but_not_as_evidence(self):
        records = [variant(name="ELSEWHERE", parent="SOMETHINGELSE",
                           modifications=[substitution()], outcomes=[outcome()])]
        result = self.store_with(records).for_substitution("TESTPEPTIDE", 8, "A", "G")
        self.assertEqual(len(result.matches), 1)
        self.assertFalse(result.matches[0].is_usable_as_evidence)
        self.assertFalse(result.has_precedent)
        self.assertIn("argued equivalent", result.matches[0].kind.caveat)

    def test_the_name_is_matched_whole_never_as_a_substring(self):
        """
        The biological-context layer learned this one: matching "insulin" as a
        substring selects "Insulin-like growth factor 1", a different molecule
        with a different receptor.
        """
        from peptide_suite.core.variant_evidence import _same_molecule

        record = variant(name="IGF-1 variant", parent="Insulin-like growth factor 1",
                         outcomes=[outcome()])
        self.assertFalse(_same_molecule("insulin", record))
        self.assertTrue(_same_molecule("Insulin-like growth factor 1", record))

        # It still appears, because the SUBSTITUTION is the same one -- and
        # that is the point: it is returned as evidence about a different
        # peptide, never as evidence about this one.
        result = self.store_with([record]).for_substitution("insulin", 8, "A", "G")
        self.assertIs(result.matches[0].kind, MatchKind.SAME_SUBSTITUTION_OTHER_PEPTIDE)
        self.assertFalse(result.has_precedent)

        # A record sharing nothing but the name fragment is not returned at all.
        unrelated = variant(name="IGF-1 variant", parent="Insulin-like growth factor 1",
                            modifications=[substitution(position=40, mutant="W")],
                            outcomes=[outcome()])
        self.assertEqual(
            self.store_with([unrelated]).for_substitution("insulin", 8, "A", "G").matches,
            [])

    def test_an_alias_matches(self):
        records = [variant(name="V", parent="GLP-1 (7-37)", aliases=("GLP1",),
                           outcomes=[outcome()])]
        result = self.store_with(records).for_substitution("GLP1", 8, "A", "G")
        self.assertEqual(len(result.matches), 1)

    def test_best_tier_reflects_only_usable_matches(self):
        records = [variant(name="ELSEWHERE", parent="SOMETHINGELSE",
                           modifications=[substitution()],
                           outcomes=[outcome(provenance=CITED)])]
        result = self.store_with(records).for_substitution("TESTPEPTIDE", 8, "A", "G")
        self.assertIsNone(result.best_tier)


class TestTheEmptyStoreIsHonestAboutBeingEmpty(unittest.TestCase):

    def test_the_shipped_store_is_empty(self):
        self.assertTrue(store().is_empty)

    def test_an_empty_result_distinguishes_absence_from_disproof(self):
        result = precedent_for("GLP-1", 8, "A", "G")
        self.assertTrue(result.searched)
        self.assertTrue(result.store_is_empty)
        self.assertFalse(result.has_precedent)
        self.assertIn("not evidence that none exists", result.statement)

    def test_searched_and_not_searched_are_distinguishable(self):
        """They render identically as an empty list and mean opposite things."""
        unsearched = PrecedentResult(query="x", searched=False)
        self.assertFalse(unsearched.searched)
        self.assertTrue(precedent_for("GLP-1", 8, "A", "G").searched)

    def test_the_status_says_what_it_holds(self):
        status = store().status()
        self.assertEqual(status["records"], 0)
        self.assertEqual(status["records_with_measured_outcomes"], 0)
        self.assertEqual(status["statement"], EMPTY_STORE_STATEMENT)

    def test_an_unreadable_store_fails_loudly_rather_than_reading_as_empty(self):
        """
        An unreadable store and an empty one both produce "no precedent found",
        and only one of them is a bug.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "broken.json"
            path.write_text("{ this is not json")
            with self.assertRaises(EvidenceSchemaError):
                VariantEvidenceStore(path=path).records


class TestTheShippedFileInventsNothing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads(DATA_PATH.read_text(encoding="utf-8"))

    def test_no_variant_record_is_shipped(self):
        self.assertEqual(self.raw["variants"], [])

    def test_no_identifier_appears_in_any_record(self):
        """
        A fabricated identifier is worse than a missing one: it looks checkable
        and survives review. Scoped to the records, since the prose explains
        that none are claimed.
        """
        blob = json.dumps(self.raw["variants"])
        for tell in ("pmid", "doi", "accession", "PDB"):
            self.assertNotIn(tell.lower(), blob.lower())

    def test_the_file_says_why_it_is_empty(self):
        # The prose is stored as lines, so it is rejoined before matching:
        # asserting on the JSON blob tests where the line breaks fall.
        about = " ".join(
            line for block in self.raw["_about"].values()
            for line in (block if isinstance(block, list) else
                         block.values() if isinstance(block, dict) else [block])
        ).lower()
        self.assertIn("citation", about)
        self.assertIn("unreachable", about)
        self.assertIn("not evidence that no measurement exists", about)


class TestPrecedentDoesNotMoveTheMagnitude(unittest.TestCase):
    """
    The separation this repository is built on: confidence is whether the
    effect is real, magnitude is how much it matters. A published measurement
    of a substitution does not make its perturbation larger.
    """

    def test_the_recommendation_carries_precedent_without_a_score_change(self):
        from peptide_suite.workflows.optimize import OptimizeWorkflow
        _ctx, recs = OptimizeWorkflow().run(
            "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR",
            confirmed_goal="protease_resistance", auto_confirm=True)
        self.assertTrue(recs)
        for rec in recs:
            with self.subTest(rec=f"{rec.wild_type_aa}{rec.position}{rec.mutant_aa}"):
                self.assertIsNotNone(rec.experimental_precedent)
                self.assertTrue(rec.experimental_precedent["searched"])
                self.assertFalse(rec.has_experimental_precedent)
                # net_score is the magnitude axis and nothing about precedent
                # may appear in it.
                self.assertNotIn("precedent", json.dumps(rec.score_breakdown or {}))

    def test_precedent_orders_equal_scores_without_altering_them(self):
        """
        Ranking may put a measured substitution first. It may not make its
        number bigger, which is what would happen if precedent were folded into
        the score.
        """
        from peptide_suite.core import SubstitutionRecommendation
        from peptide_suite.workflows.optimize import OptimizeWorkflow

        measured = SubstitutionRecommendation(
            position=1, wild_type_aa="A", mutant_aa="G", net_score=0.4,
            net_recommendation="recommend",
            experimental_precedent={"has_precedent": True, "searched": True})
        computed = SubstitutionRecommendation(
            position=2, wild_type_aa="A", mutant_aa="V", net_score=0.4,
            net_recommendation="recommend",
            experimental_precedent={"has_precedent": False, "searched": True})

        ranked = OptimizeWorkflow()._rank_recommendations([computed, measured])
        self.assertEqual(ranked[0].mutant_aa, "G")
        self.assertEqual(ranked[0].net_score, 0.4)
        self.assertEqual(ranked[1].net_score, 0.4)


if __name__ == "__main__":
    unittest.main()


class TestTheLoaderRefusesWhatWouldBeBelieved(unittest.TestCase):
    """
    A row that is wrong in a way nobody notices is worse than a row that is
    missing: it gets trained on, displayed as precedent, and believed.
    """

    def rows(self, **overrides):
        row = {
            "parent": "GLP-1 (7-37)",
            "parent_sequence": "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG",
            "variant_name": "v", "modification_kind": "SUBSTITUTION",
            "position": "2", "wild_type": "A", "mutant": "G",
            "modification_description": "", "measure": "PROTEASE_STABILITY",
            "direction": "INCREASED", "fold_change": "4.0", "value": "", "unit": "",
            "comparator": "unmodified parent", "assay": "DPP-4 incubation",
            "target": "DPP4", "pmid": "12345678", "doi": "",
            "extraction": "CURATOR_READ_FULL_TEXT", "note": "",
        }
        row.update(overrides)
        return row

    def build(self, **overrides):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "load_variant_evidence", pathlib.Path("tools/load_variant_evidence.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, module.build_record(self.rows(**overrides))

    def test_a_well_formed_row_loads(self):
        _module, record = self.build()
        self.assertTrue(record.has_evidence)
        self.assertIs(record.strongest_tier, EvidenceTier.DIRECT_EXPERIMENTAL)

    def test_a_position_in_the_wrong_numbering_scheme_is_refused(self):
        """
        The check worth the most. GLP-1 appears in at least three numbering
        schemes; a position in the wrong one lands on the wrong residue and
        still looks entirely plausible, so nothing downstream reads as an
        error.
        """
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "load_variant_evidence", pathlib.Path("tools/load_variant_evidence.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaises(module.RowError) as caught:
            module.build_record(self.rows(position="8"))
        self.assertIn("numbering scheme", str(caught.exception))

    def test_a_row_with_no_citation_caps_below_direct_experimental(self):
        _module, record = self.build(pmid="", doi="")
        self.assertIs(record.strongest_tier, EvidenceTier.BIOCHEMICAL_PRINCIPLE)

    def test_a_row_with_no_measure_is_a_design(self):
        _module, record = self.build(measure="", direction="", fold_change="",
                                     comparator="", assay="", target="", pmid="")
        self.assertFalse(record.has_evidence)
