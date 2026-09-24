"""
Retrieval provenance.

The bug these tests exist for: retrieve_homologs read three sequences out of a
bundled JSON fixture and returned the status string "retrieved 3 homologs from
NCBI". Conservation entropy computed over them was then reported as homolog
evidence. Two sibling methods invented a PubMed record (pmid "0000001", abstract
"This is a placeholder abstract for calibration testing") and a UniProt
annotation whose fields read "To be inferred from literature".

All three wrote their output to the cache, so on the next call it came back
with status "cached" and no marker at all. That is the part worth testing
hardest: a fabrication that is cached becomes indistinguishable from a
retrieval after exactly one run.
"""

import json
import tempfile
import contextlib
import json
import pathlib
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from peptide_suite.core.evidence_retrieval import (
    EvidenceRetriever, Retrieval, RetrievalSource,
)


def retriever() -> EvidenceRetriever:
    """A retriever with a cache directory that no other test can have touched."""
    return EvidenceRetriever(cache_dir=tempfile.mkdtemp())


class TestNothingIsFabricated(unittest.TestCase):

    def test_literature_returns_nothing_rather_than_an_invented_paper(self):
        result = retriever().retrieve_literature_context("IGF1")
        self.assertEqual(result.payload, [])
        self.assertIs(result.source, RetrievalSource.UNAVAILABLE)

    def test_uniprot_returns_nothing_rather_than_an_empty_shaped_record(self):
        """
        A dict whose fields read "To be inferred from literature" is worse than
        None: a caller sees populated fields and renders them.
        """
        result = retriever().retrieve_uniprot_data("IGF1")
        self.assertIsNone(result.payload)
        self.assertIs(result.source, RetrievalSource.UNAVAILABLE)

    def test_no_method_returns_a_fabricated_identifier(self):
        r = retriever()
        for result in (r.retrieve_literature_context("IGF1"),
                       r.retrieve_uniprot_data("IGF1"),
                       r.retrieve_homologs("NOTHING_HAS_THIS_NAME")):
            blob = json.dumps(result.payload)
            with self.subTest(service=result.service):
                for tell in ("0000001", "placeholder", "Mocked", "To be inferred",
                             "To be queried"):
                    self.assertNotIn(tell, blob)


@contextlib.contextmanager
def stub_panel(entries):
    """
    Point the fixture reader at a panel written for the test.

    These tests used to depend on the shipped panel carrying homolog lists.
    That coupled a guarantee about labelling to a file of biological data, so
    correcting the data broke the guarantee's tests -- and the panel's homolog
    lists turned out to be the same sequence repeated, which the conservation
    gate refused anyway. The guarantee is what matters and it is tested here
    against a panel this test owns.
    """
    directory = tempfile.mkdtemp()
    path = pathlib.Path(directory) / "test_panel.json"
    path.write_text(json.dumps(entries))
    original = EvidenceRetriever._fixture_homologs

    def patched(self, gene_name):
        key = re.sub(r"[^A-Za-z0-9]", "", gene_name or "").upper()
        for name, entry in entries.items():
            if re.sub(r"[^A-Za-z0-9]", "", name).upper() == key:
                return list(entry.get("homologs", []))
        return []

    EvidenceRetriever._fixture_homologs = patched
    try:
        yield
    finally:
        EvidenceRetriever._fixture_homologs = original
        shutil.rmtree(directory, ignore_errors=True)


PANEL_WITH_HOMOLOGS = {
    "BPC157": {
        "full_name": "stub entry",
        # Three distinct strings, so the conservation gate has something to act
        # on. Deliberately not presented as real orthologues anywhere.
        "homologs": ["GEPPPGKPADDAGLV", "GEPPPGKPADDAGLA", "GEPPPGKPTDDAGLV"],
    }
}


class TestFixturesAreNotCalledRetrievals(unittest.TestCase):

    def test_bundled_homologs_are_labelled_as_fixture(self):
        with stub_panel(PANEL_WITH_HOMOLOGS):
            result = retriever().retrieve_homologs("BPC157")
        self.assertIs(result.source, RetrievalSource.LOCAL_FIXTURE)
        self.assertTrue(result.payload)

    def test_a_fixture_does_not_claim_to_be_evidence(self):
        with stub_panel(PANEL_WITH_HOMOLOGS):
            self.assertFalse(retriever().retrieve_homologs("BPC157").is_evidence)

    def test_the_description_says_where_it_came_from(self):
        with stub_panel(PANEL_WITH_HOMOLOGS):
            described = retriever().retrieve_homologs("BPC157").describe()
        self.assertIn("not a retrieval", described)
        self.assertNotIn("from NCBI", described)

    def test_the_shipped_panel_bundles_no_homologs_and_says_so(self):
        # The panel's homolog lists each held one distinct sequence repeated,
        # which the conservation gate refused anyway. They are empty now, so
        # the honest source for a panel peptide is UNAVAILABLE, not a fixture.
        result = retriever().retrieve_homologs("IGF1")
        self.assertIs(result.source, RetrievalSource.UNAVAILABLE)
        self.assertEqual(result.payload, [])

    def test_an_unknown_peptide_yields_nothing_rather_than_the_query_back(self):
        result = retriever().retrieve_homologs("NOTHING_HAS_THIS_NAME")
        self.assertEqual(result.payload, [])
        self.assertIs(result.source, RetrievalSource.UNAVAILABLE)


class TestCacheCannotLaunderProvenance(unittest.TestCase):

    def test_a_fixture_is_never_written_to_the_cache(self):
        """
        The laundering step. Cache a fixture and it returns as a cache hit,
        which reads as "we fetched this once".
        """
        r = retriever()
        r.retrieve_homologs("IGF1")
        self.assertEqual(list(Path(r.cache_dir).glob("*.json")), [])

    def test_an_unmarked_cache_entry_is_ignored(self):
        """
        Retroactivity. A machine that ran the old code has poisoned entries on
        disk that look exactly like good ones; without this check it keeps
        serving them forever.
        """
        r = retriever()
        path = r._get_cache_path("IGF1_human", "homologs")
        path.write_text(json.dumps({"sequences": ["AAAA", "BBBB", "CCCC"]}))

        result = r.retrieve_homologs("IGF1")
        self.assertIsNot(result.source, RetrievalSource.CACHED)
        self.assertNotIn("AAAA", json.dumps(result.payload))

    def test_a_marked_live_entry_is_used(self):
        r = retriever()
        path = r._get_cache_path("IGF1_human", "homologs")
        path.write_text(json.dumps({
            "_source": RetrievalSource.LIVE.value,
            "_cached_utc": 0,
            "sequences": ["AAAA", "BBBB"],
        }))
        result = r.retrieve_homologs("IGF1")
        self.assertIs(result.source, RetrievalSource.CACHED)
        self.assertEqual(result.payload, ["AAAA", "BBBB"])

    def test_save_cache_refuses_a_non_live_source(self):
        r = retriever()
        path = r._get_cache_path("probe", "homologs")
        for source in (RetrievalSource.LOCAL_FIXTURE, RetrievalSource.UNAVAILABLE,
                       RetrievalSource.CACHED):
            with self.subTest(source=source.value):
                r._save_cache(path, {"sequences": ["X"]}, source)
                self.assertFalse(path.exists())


class TestWorkflowCarriesTheProvenance(unittest.TestCase):

    def test_fixture_homologs_produce_a_stated_caveat(self):
        """
        Conservation over fixture sequences is arithmetically identical and
        evidentially different, so the difference has to reach the output.

        The retrieval is injected rather than driven through the bundled panel:
        no panel entry currently carries enough homologs for conservation to
        run, so going through the data would test the panel's contents instead
        of the branch.
        """
        from peptide_suite.workflows.optimize import OptimizeWorkflow

        workflow = OptimizeWorkflow()
        sequence = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR"
        workflow.evidence_retriever.retrieve_homologs = lambda *a, **k: Retrieval(
            payload=[sequence[:-1] + "S", "HADGTFTSDVSSYLEGQAAKEFIAWLVKGR",
                     "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGK"],
            source=RetrievalSource.LOCAL_FIXTURE,
            detail="3 homolog(s) read from the bundled test panel.",
            service="peptide_suite/data/test_panel.json",
        )
        ctx, _recs = workflow.run(sequence, confirmed_goal="protease_resistance",
                                  auto_confirm=True)

        self.assertEqual(ctx.homolog_source, RetrievalSource.LOCAL_FIXTURE.value)
        self.assertTrue(ctx.conservation_available, "fixture did not yield enough sequences")
        self.assertTrue(
            any("bundled with" in n for n in ctx.data_notes),
            f"no fixture caveat in data_notes: {ctx.data_notes}",
        )

    def test_no_conservation_caveat_when_no_conservation_was_computed(self):
        """
        The regression this replaced: the caveat read "the conservation entropy
        below is computed over bundled sequences" on a run where conservation
        was never computed at all. A caveat about a number that does not exist
        is its own false statement.
        """
        from peptide_suite.workflows.optimize import OptimizeWorkflow
        ctx, _recs = OptimizeWorkflow().run(
            input_sequence_or_name="GEPPPGKPADDAGLV",
            confirmed_goal="protease_resistance", auto_confirm=True)

        # Which source failed to supply homologs is incidental here; that no
        # conservation was computed, and that nothing claimed otherwise, is the
        # regression. Pinning the source coupled this guard to the panel's
        # contents and broke when the panel's fabricated homologs were removed.
        self.assertIn(ctx.homolog_source,
                      (RetrievalSource.UNAVAILABLE.value, RetrievalSource.LOCAL_FIXTURE.value))
        self.assertFalse(ctx.conservation_available)
        self.assertFalse(
            any("entropy below is computed" in n for n in ctx.data_notes),
            f"claimed a conservation result that was never computed: {ctx.data_notes}",
        )

    def test_caller_supplied_homologs_are_labelled_as_such(self):
        from peptide_suite.workflows.optimize import OptimizeWorkflow
        ctx, _recs = OptimizeWorkflow().run(
            input_sequence_or_name="HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR",
            confirmed_goal="protease_resistance",
            auto_confirm=True,
            homologs=["HAEGTFTSDVSSYLEGQAAKEFIAWLVKGS", "HADGTFTSDVSSYLEGQAAKEFIAWLVKGR"],
        )
        self.assertEqual(ctx.homolog_source, "CALLER_SUPPLIED")


class TestIdentityReachesTheLookups(unittest.TestCase):
    """
    The name is not cosmetic: homolog retrieval, function inference and native
    context all key on it. A recognised peptide that stays "unnamed_peptide" is
    anonymous to every later step.
    """

    def test_a_recognised_sequence_gets_its_name(self):
        from peptide_suite.workflows.optimize import OptimizeWorkflow
        ctx, _ = OptimizeWorkflow().run(
            "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR",
            confirmed_goal="protease_resistance", auto_confirm=True)
        self.assertNotEqual(ctx.name, "unnamed_peptide")
        self.assertIn("GLP-1", ctx.name)

    def test_fixture_lookup_tolerates_punctuation_differences(self):
        """The panel is keyed BPC157; the reference set names it BPC-157."""
        from peptide_suite.core.evidence_retrieval import EvidenceRetriever
        with stub_panel(PANEL_WITH_HOMOLOGS):
            r = retriever()
            self.assertTrue(r._fixture_homologs("BPC-157"))
            self.assertEqual(r._fixture_homologs("BPC-157"), r._fixture_homologs("BPC157"))


class TestRetrievalContract(unittest.TestCase):

    def test_only_live_and_cached_count_as_evidence(self):
        for source, expected in (
            (RetrievalSource.LIVE, True),
            (RetrievalSource.CACHED, True),
            (RetrievalSource.LOCAL_FIXTURE, False),
            (RetrievalSource.UNAVAILABLE, False),
        ):
            with self.subTest(source=source.value):
                self.assertEqual(
                    Retrieval(payload=[], source=source, detail="").is_evidence, expected)


if __name__ == "__main__":
    unittest.main()
