"""
PubMed gives citations. It does not give evidence.

These tests exist for the step between the two, because that is where the
pipeline wants to cut a corner: a search returns a paper, the paper has a
title, and the title very nearly says what happened. Turning that into an
outcome record is the one part nobody can automate, and the tests below are
what stop a later contributor from automating it anyway.
"""

import json
import pathlib
import tempfile
import unittest

from peptide_suite.core import EvidenceTier
from peptide_suite.core.biological_context import Provenance, SourceKind
from peptide_suite.core.literature import PubMedUnavailable, QueryTopic
from peptide_suite.core.variant_evidence import (
    Direction, Extraction, MeasuredOutcome, OutcomeMeasure,
)
from peptide_suite.core.variant_evidence_ingest import (
    Candidate, CurationRequired, INGEST_TOPICS, IngestResult, find_candidates,
    queue_path, save_queue, write_records,
)
from peptide_suite.tests.test_literature import FakeClient

CITED = Provenance(kind=SourceKind.PRIMARY_LITERATURE, pmid="11111111", year=2020)


def outcome(**kwargs) -> MeasuredOutcome:
    fields = dict(measure=OutcomeMeasure.PROTEASE_STABILITY,
                  direction=Direction.INCREASED, comparator="the parent peptide",
                  assay="synthetic, this test only", provenance=CITED, fold_change=3.0)
    fields.update(kwargs)
    return MeasuredOutcome(**fields)


class TestAGoodCitationDoesNotVouchForTheNumber(unittest.TestCase):
    """
    Provenance answers where a claim came from. It does not answer who read it
    out, and those are different failure modes: a PMID can be perfectly real
    while the number attached to it is a misreading.
    """

    def test_a_curator_read_outcome_reaches_direct_experimental(self):
        self.assertIs(outcome(extraction=Extraction.CURATOR_READ_FULL_TEXT).tier,
                      EvidenceTier.DIRECT_EXPERIMENTAL)

    def test_an_automated_parse_caps_below_it_despite_a_real_pmid(self):
        parsed = outcome(extraction=Extraction.AUTOMATED_PARSE)
        self.assertTrue(parsed.provenance.has_citation)
        self.assertIs(parsed.provenance.max_tier, EvidenceTier.DIRECT_EXPERIMENTAL)
        self.assertIs(parsed.tier, EvidenceTier.BIOCHEMICAL_PRINCIPLE)

    def test_an_unrecorded_extraction_caps_too(self):
        """
        The default. A record that does not say who extracted it has not
        established that anybody did.
        """
        self.assertIs(outcome().extraction, Extraction.UNRECORDED)
        self.assertIs(outcome().tier, EvidenceTier.BIOCHEMICAL_PRINCIPLE)

    def test_the_weaker_of_the_two_caps_wins(self):
        uncited_but_read = outcome(
            provenance=Provenance(kind=SourceKind.CURATED_UNVERIFIED),
            extraction=Extraction.CURATOR_READ_FULL_TEXT)
        self.assertIs(uncited_but_read.tier, EvidenceTier.BIOCHEMICAL_PRINCIPLE)

    def test_the_parse_caveat_names_the_inversion_it_guards_against(self):
        caveat = Extraction.AUTOMATED_PARSE.caveat
        self.assertIn("three-fold higher IC50", caveat)
        self.assertIn("PMID still", caveat)

    def test_an_abstract_read_says_to_check_the_two_fields_it_omits(self):
        caveat = Extraction.CURATOR_READ_ABSTRACT.caveat
        self.assertIn("assay", caveat)
        self.assertIn("comparator", caveat)

    def test_the_extraction_reaches_the_payload(self):
        payload = outcome(extraction=Extraction.AUTOMATED_PARSE).to_dict()
        self.assertEqual(payload["extraction"], "AUTOMATED_PARSE")
        self.assertTrue(payload["extraction_caveat"])
        self.assertEqual(payload["tier"], "BIOCHEMICAL_PRINCIPLE")


class TestASearchReturnsCandidatesNotRecords(unittest.TestCase):

    def test_a_live_search_yields_candidates(self):
        result = find_candidates("IGF-1", aliases=["somatomedin C"], gene="IGF1",
                                 client=FakeClient())
        self.assertTrue(result.succeeded)
        self.assertTrue(result.candidates)
        for candidate in result.candidates:
            with self.subTest(pmid=candidate.pmid):
                self.assertTrue(candidate.pmid)
                self.assertTrue(candidate.url.endswith(f"/{candidate.pmid}/"))

    def test_a_candidate_carries_no_outcome_field_at_all(self):
        """
        Not an empty comparator -- no comparator. A blank where a value belongs
        invites someone to fill it in from the title, and a structure that
        makes the wrong action easy will eventually have it taken.
        """
        payload = Candidate(pmid="1", title="t").to_dict()
        for absent in ("comparator", "assay", "direction", "fold_change", "measure"):
            self.assertNotIn(absent, payload)
        self.assertEqual(payload["status"], "AWAITING_CURATION")
        self.assertIn("read this paper", payload["what_is_missing"])

    def test_the_statement_says_a_candidate_is_not_a_record(self):
        result = find_candidates("IGF-1", gene="IGF1", client=FakeClient())
        self.assertIn("None of them is an evidence record yet", result.statement())

    def test_the_topics_are_the_variant_literature_not_the_structural_one(self):
        """
        Mixing in receptor and structure topics returns a queue of papers about
        the fold when what is wanted is a queue about the variants.
        """
        self.assertIn(QueryTopic.MUTAGENESIS, INGEST_TOPICS)
        self.assertIn(QueryTopic.ALANINE_SCAN, INGEST_TOPICS)
        self.assertNotIn(QueryTopic.STRUCTURE, INGEST_TOPICS)
        self.assertNotIn(QueryTopic.RECEPTOR, INGEST_TOPICS)


class TestAFailedSearchInventsNothing(unittest.TestCase):

    def test_an_unreachable_host_yields_no_candidates(self):
        result = find_candidates("IGF-1", gene="IGF1",
                                 client=FakeClient(fail="403 Forbidden"))
        self.assertFalse(result.succeeded)
        self.assertEqual(result.candidates, [])
        self.assertIn("403", result.error)
        self.assertIn("nothing was invented", result.statement())

    def test_an_unidentifiable_peptide_builds_no_query(self):
        result = find_candidates("", client=FakeClient())
        self.assertFalse(result.succeeded)
        self.assertIn("Nothing identifies this peptide", result.error)

    def test_an_empty_live_result_is_about_the_query_not_the_world(self):
        result = find_candidates("IGF-1", gene="IGF1", client=FakeClient(ids=()))
        self.assertEqual(result.candidates, [])
        self.assertIn("result about this query", result.statement())


class TestTheLastStepIsRefused(unittest.TestCase):

    def test_writing_records_raises_rather_than_being_absent(self):
        """
        Present so the omission is explicit. A contributor looking for the
        function that finishes the pipeline finds this instead of writing one.
        """
        with self.assertRaises(CurationRequired) as caught:
            write_records()
        message = str(caught.exception)
        self.assertIn("results section", message)
        self.assertIn("AUTOMATED_PARSE", message)
        self.assertIn("BIOCHEMICAL_PRINCIPLE", message)

    def test_the_queue_is_written_beside_the_store_never_into_it(self):
        """
        Candidates inside the store file would be one careless merge away from
        being read as records, and the distinction has to survive carelessness.
        """
        from peptide_suite.core.variant_evidence import DATA_PATH, VariantEvidenceStore

        with tempfile.TemporaryDirectory() as directory:
            result = find_candidates("IGF-1", gene="IGF1", client=FakeClient())
            path = save_queue([result], directory=pathlib.Path(directory))
            self.assertNotEqual(path, DATA_PATH)
            written = json.loads(path.read_text())
            self.assertIn("NOT evidence records", written["_about"])
            self.assertTrue(written["results"][0]["candidates"])

        # And the real store is untouched by any of it.
        self.assertTrue(VariantEvidenceStore().is_empty)

    def test_the_queue_file_holds_no_outcome_shaped_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            result = find_candidates("IGF-1", gene="IGF1", client=FakeClient())
            path = save_queue([result], directory=pathlib.Path(directory))
            blob = path.read_text().lower()
        for absent in ('"fold_change"', '"comparator"', '"assay"', '"direction"'):
            self.assertNotIn(absent, blob)


if __name__ == "__main__":
    unittest.main()
