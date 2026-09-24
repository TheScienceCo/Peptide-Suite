"""
PubMed retrieval.  [Phase 1, item 5]

The field this replaces returned an invented PMID and an abstract reading
"This is a placeholder abstract for calibration testing", cached it, and served
it back as a cache hit with no marker. So the tests that matter are the ones
about failure: what a search returns when it cannot reach the service, and
whether anything it returns can be mistaken for a retrieval.

No test here touches the network. The transport is replaced, which is also how
the LIVE path gets exercised in an environment that cannot reach eutils.
"""

import json
import tempfile
import unittest

from peptide_suite.core.evidence_retrieval import EvidenceRetriever, RetrievalSource
from peptide_suite.core.literature import (
    Article,
    LiteratureQuery,
    PubMedClient,
    PubMedUnavailable,
    QueryTopic,
    build_queries,
    search_literature,
)

# A shape-accurate esummary response. Values are obviously synthetic so no
# reader could mistake this fixture for a real record.
FAKE_SUMMARY = {
    "result": {
        "uids": ["11111111", "22222222"],
        "11111111": {
            "title": "A synthetic test record, not a real article",
            "fulljournalname": "Journal of Test Fixtures",
            "pubdate": "2020 Jan",
            "articleids": [{"idtype": "doi", "value": "10.0000/test.1"}],
        },
        "22222222": {
            "title": "A second synthetic test record",
            "source": "Test Letters",
            "pubdate": "2021",
            "articleids": [],
        },
    }
}


class FakeClient(PubMedClient):
    """A client whose transport is replaced rather than reaching NCBI."""

    def __init__(self, ids=("11111111", "22222222"), fail=None):
        super().__init__()
        self._ids = list(ids)
        self._fail = fail
        self.searches = []

    def search(self, term, retmax=8):
        if self._fail:
            raise PubMedUnavailable(self._fail)
        self.searches.append(term)
        return self._ids

    def summarise(self, pmids):
        if self._fail:
            raise PubMedUnavailable(self._fail)
        return [
            Article(pmid=p,
                    title=FAKE_SUMMARY["result"][p]["title"],
                    journal=FAKE_SUMMARY["result"][p].get("fulljournalname")
                    or FAKE_SUMMARY["result"][p].get("source", ""),
                    year=int(FAKE_SUMMARY["result"][p]["pubdate"][:4]),
                    doi=next((i["value"] for i in FAKE_SUMMARY["result"][p]["articleids"]
                              if i["idtype"] == "doi"), ""))
            for p in pmids if p in FAKE_SUMMARY["result"]
        ]


class TestQueriesAreBuiltFromIdentityNotSequence(unittest.TestCase):
    """
    Complete sequences almost never appear in abstracts, so searching for one
    is close to guaranteed to return nothing. Retrieval runs after identity
    resolution and asks what a paper would actually say.
    """

    def test_the_subject_clause_ors_the_names_together(self):
        queries = build_queries(name="Insulin-like growth factor 1",
                                aliases=["IGF-1", "somatomedin C"], gene="IGF1")
        self.assertTrue(queries)
        term = queries[0].term
        for token in ("Insulin-like growth factor 1", "IGF-1", "somatomedin C", "IGF1"):
            self.assertIn(token, term)
        self.assertIn(" OR ", term)

    def test_names_are_deduplicated_case_insensitively(self):
        queries = build_queries(name="IGF-1", aliases=["igf-1", "IGF-1"], gene="IGF1")
        self.assertEqual(queries[0].term.count("IGF-1"), 1)

    def test_each_topic_gets_its_own_query(self):
        topics = [QueryTopic.MUTAGENESIS, QueryTopic.STRUCTURE]
        queries = build_queries(name="IGF-1", topics=topics)
        self.assertEqual(len(queries), len(topics) + 1)
        for topic in topics:
            self.assertTrue(any(topic.value in q.term for q in queries))

    def test_the_receptor_query_uses_the_known_receptor(self):
        queries = build_queries(name="IGF-1", receptor="IGF1R",
                                topics=[QueryTopic.RECEPTOR])
        self.assertTrue(any("IGF1R" in q.term for q in queries))

    def test_sequence_search_is_the_fallback_for_unknown_peptides_only(self):
        known = build_queries(name="IGF-1", sequence="GPETLC")
        self.assertFalse(any(q.is_sequence_fallback for q in known))

        unknown = build_queries(sequence="MKWVTFISLL")
        self.assertEqual(len(unknown), 1)
        self.assertTrue(unknown[0].is_sequence_fallback)

    def test_nothing_identifiable_and_no_sequence_gives_no_query(self):
        self.assertEqual(build_queries(), [])

    def test_every_topic_explains_what_it_looks_for(self):
        for topic in QueryTopic:
            self.assertTrue(topic.describe.strip())


class TestTheLiveePathReturnsWhatItWasGiven(unittest.TestCase):

    def test_articles_come_back_with_their_identifiers(self):
        result = search_literature(FakeClient(), build_queries(name="IGF-1"))
        self.assertTrue(result.reachable)
        self.assertEqual({a.pmid for a in result.articles}, {"11111111", "22222222"})
        for article in result.articles:
            self.assertTrue(article.pmid)
            self.assertTrue(article.url.endswith(f"/{article.pmid}/"))

    def test_articles_are_deduplicated_across_topical_queries(self):
        # Nine topics all returning the same two records must not produce
        # eighteen articles.
        result = search_literature(
            FakeClient(), build_queries(name="IGF-1", topics=list(QueryTopic)))
        self.assertEqual(len(result.articles), 2)

    def test_the_queries_that_ran_are_recorded(self):
        # A search that found nothing and a search that never happened are
        # different, and an empty article list alone cannot tell them apart.
        queries = build_queries(name="IGF-1", topics=[QueryTopic.AFFINITY])
        result = search_literature(FakeClient(ids=[]), queries)
        self.assertTrue(result.reachable)
        self.assertEqual(result.articles, [])
        self.assertEqual(len(result.queries), len(queries))


class TestFailureInventsNothing(unittest.TestCase):

    def test_an_unreachable_service_returns_no_articles(self):
        result = search_literature(FakeClient(fail="simulated outage"),
                                   build_queries(name="IGF-1"))
        self.assertFalse(result.reachable)
        self.assertEqual(result.articles, [])
        self.assertIn("simulated outage", result.status)

    def test_a_partial_search_is_not_reported_as_a_complete_one(self):
        # Reporting the articles gathered before the failure under a success
        # flag would claim the remaining topics found nothing.
        class HalfFailing(FakeClient):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def search(self, term, retmax=8):
                self.calls += 1
                if self.calls > 1:
                    raise PubMedUnavailable("died partway")
                return self._ids

        result = search_literature(
            HalfFailing(), build_queries(name="IGF-1", topics=[QueryTopic.STRUCTURE]))
        self.assertFalse(result.reachable)
        self.assertEqual(result.articles, [])

    def test_an_http_error_does_not_open_the_circuit(self):
        # The host answered, so it is reachable. Only transport failures mean
        # "stop dialling".
        client = PubMedClient()
        self.assertFalse(client._circuit_open)

    def test_a_disabled_client_refuses_rather_than_returning_nothing_quietly(self):
        client = PubMedClient(enabled=False)
        with self.assertRaises(PubMedUnavailable):
            client._get("esearch.fcgi", {})


class TestRetrievalProvenance(unittest.TestCase):

    def retriever(self, pubmed=None):
        return EvidenceRetriever(cache_dir=tempfile.mkdtemp(), pubmed=pubmed)

    def test_a_live_retrieval_is_marked_live(self):
        result = self.retriever(FakeClient()).retrieve_literature_context("IGF1")
        self.assertIs(result.source, RetrievalSource.LIVE)
        self.assertTrue(result.payload)
        self.assertTrue(all(a["pmid"] for a in result.payload))

    def test_an_unreachable_service_is_marked_unavailable(self):
        result = self.retriever(FakeClient(fail="no route")).retrieve_literature_context("IGF1")
        self.assertIs(result.source, RetrievalSource.UNAVAILABLE)
        self.assertEqual(result.payload, [])
        self.assertIn("no route", result.detail)

    def test_a_cached_live_result_stays_distinguishable_from_a_fixture(self):
        retriever = self.retriever(FakeClient())
        first = retriever.retrieve_literature_context("IGF1")
        self.assertIs(first.source, RetrievalSource.LIVE)
        second = retriever.retrieve_literature_context("IGF1")
        self.assertIs(second.source, RetrievalSource.CACHED)
        self.assertIsNot(second.source, RetrievalSource.LOCAL_FIXTURE)

    def test_an_unavailable_result_is_never_cached(self):
        # Caching a failure would serve it back as a cache hit, which is how
        # the placeholder abstract survived in the first place.
        retriever = self.retriever(FakeClient(fail="no route"))
        retriever.retrieve_literature_context("IGF1")
        again = retriever.retrieve_literature_context("IGF1")
        self.assertIs(again.source, RetrievalSource.UNAVAILABLE)

    def test_nothing_returned_looks_like_a_placeholder(self):
        blob = json.dumps(
            self.retriever(FakeClient()).retrieve_literature_context("IGF1").payload)
        for tell in ("placeholder", "calibration testing", "0000001", "To be queried"):
            self.assertNotIn(tell, blob)


if __name__ == "__main__":
    unittest.main()
