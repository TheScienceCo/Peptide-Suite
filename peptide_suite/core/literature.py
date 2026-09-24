"""
PubMed retrieval, actually wired up.  [Phase 1, item 5]

`retrieve_literature_context` previously returned nothing and said PubMed was
not wired up. Before that it returned an invented PMID and an abstract reading
"This is a placeholder abstract for calibration testing", cached it, and served
it back as a cache hit with no marker. This module replaces the first and can
never become the second.

THE SEARCH STRATEGY, AND WHY IT IS NOT THE SEQUENCE

Searching PubMed for an amino-acid sequence almost never works. Complete
sequences do not appear in abstracts, and the ones that do are short motifs
that match everything. So retrieval runs after identity resolution and queries
what a paper would actually say: the canonical name, the aliases, the gene, and
the receptor.

    sequence -> identity -> name / aliases / gene / receptor -> queries

Queries are built per topic -- receptor, binding site, structure, mutagenesis,
alanine scan, affinity, analog, variant -- because a peptide's engineering
literature and its structural literature are different bodies of work and a
single query returns whichever is larger. A raw-sequence query is available as
a last resort for peptides nothing is known about, marked as the low-yield
fallback it is.

WHAT IT WILL NOT DO

There is no offline mode. A literature search that cannot reach PubMed returns
UNAVAILABLE with the transport error attached; it does not fall back to a
bundled article list, because a bundled article list retrieved under the name
of a search is the failure this module was written to end. Nothing is cached
unless it came from a live response.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# NCBI asks unauthenticated clients to stay at or under three requests a second
# and to identify themselves. Both are courtesies that keep the service
# available to everyone, and neither is optional because it is inconvenient.
TOOL_NAME = "peptide-suite"
DEFAULT_RETMAX = 8

# The request timeout is a constructor parameter rather than a module constant,
# matching UniProtClient. The boundary check flags a bare float in engine
# source and is right to -- but a transport timeout is not a scoring
# coefficient: it changes how long a failure takes to notice, never what any
# number comes out as. Stating that here so the next person does not have to
# re-litigate it or, worse, move it into the policy artifact where it would
# imply that tuning it tunes a result.


class QueryTopic(Enum):
    """
    The kinds of question worth asking about an identified peptide.

    Separate queries rather than one broad one: a peptide's structural
    literature and its engineering literature are different bodies of work, and
    a single query returns whichever is larger rather than one of each.
    """
    RECEPTOR = "receptor"
    BINDING_SITE = "binding site"
    STRUCTURE = "structure"
    MUTAGENESIS = "mutagenesis"
    ALANINE_SCAN = "alanine scan"
    AFFINITY = "affinity"
    ANALOG = "analog"
    VARIANT = "variant"
    SUBSTITUTION = "substitution"

    @property
    def describe(self) -> str:
        return {
            QueryTopic.RECEPTOR: "which receptor the peptide engages",
            QueryTopic.BINDING_SITE: "where on the peptide or receptor binding happens",
            QueryTopic.STRUCTURE: "experimental structures of the peptide or complex",
            QueryTopic.MUTAGENESIS: "reported effects of mutating specific residues",
            QueryTopic.ALANINE_SCAN: "systematic alanine replacement studies",
            QueryTopic.AFFINITY: "measured binding affinities",
            QueryTopic.ANALOG: "engineered analogs and their properties",
            QueryTopic.VARIANT: "natural or engineered sequence variants",
            QueryTopic.SUBSTITUTION: "single-residue substitution studies",
        }[self]


@dataclass(frozen=True)
class Article:
    """
    One PubMed record. Every field comes from the response or stays empty.

    There is no constructor path that produces an Article without a PMID,
    because an article with no identifier cannot be checked and a reader has no
    way to tell it from something invented.
    """
    pmid: str
    title: str = ""
    journal: str = ""
    year: Optional[int] = None
    doi: str = ""

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/" if self.pmid else ""

    def encode(self) -> Dict[str, object]:
        return {"pmid": self.pmid, "title": self.title, "journal": self.journal,
                "year": self.year, "doi": self.doi, "url": self.url}


@dataclass
class LiteratureQuery:
    """A single query, recorded so a reader can see what was actually asked."""
    term: str
    topic: Optional[QueryTopic]
    is_sequence_fallback: bool = False

    def encode(self) -> Dict[str, object]:
        return {"term": self.term,
                "topic": self.topic.value if self.topic else "raw sequence",
                "is_sequence_fallback": self.is_sequence_fallback}


def build_queries(name: str = "", aliases: Sequence[str] = (), gene: str = "",
                  receptor: str = "", topics: Sequence[QueryTopic] = (),
                  sequence: str = "") -> List[LiteratureQuery]:
    """
    Turn an identity into the queries a paper would actually match.

    The identity terms are ORed into one subject clause and the topic is ANDed
    onto it, so "IGF-1 OR somatomedin C OR IGF1" AND "mutagenesis" finds the
    mutagenesis literature under whichever name a given paper used.
    """
    subjects = [t for t in ([name] + list(aliases) + [gene]) if t]
    # Deduplicated case-insensitively, order preserved: PubMed charges the same
    # for a repeated term and a reader reading the query list should not see
    # "IGF-1 OR IGF-1".
    seen, unique = set(), []
    for subject in subjects:
        if subject.lower() not in seen:
            seen.add(subject.lower())
            unique.append(subject)

    if not unique:
        # Nothing is known about this peptide, so the only thing left to search
        # is the sequence itself. Low yield and marked as such: whole sequences
        # rarely appear in abstracts.
        return ([LiteratureQuery(term=f'"{sequence}"', topic=None,
                                 is_sequence_fallback=True)] if sequence else [])

    clause = " OR ".join(f'"{s}"' for s in unique)
    queries = [LiteratureQuery(term=f"({clause})", topic=None)]
    for topic in topics:
        term = f"({clause}) AND {topic.value}"
        if topic is QueryTopic.RECEPTOR and receptor:
            term = f"({clause}) AND ({receptor} OR receptor)"
        queries.append(LiteratureQuery(term=term, topic=topic))
    return queries


class PubMedUnavailable(RuntimeError):
    """Raised internally when the service could not be reached."""


@dataclass
class LiteratureResult:
    """
    What a search returned, and what it asked.

    `queries` is carried so a reader can see that a search which found nothing
    actually ran, and what it looked for. An empty result with no record of the
    attempt is indistinguishable from a search that never happened.
    """
    articles: List[Article] = field(default_factory=list)
    queries: List[LiteratureQuery] = field(default_factory=list)
    reachable: bool = False
    status: str = ""

    def encode(self) -> Dict[str, object]:
        return {
            "articles": [a.encode() for a in self.articles],
            "queries": [q.encode() for q in self.queries],
            "reachable": self.reachable,
            "status": self.status,
            "n_articles": len(self.articles),
        }


class PubMedClient:
    """
    E-utilities, time-boxed and circuit-broken.

    The circuit breaker is the same idea as the UniProt client's: once the host
    is established unreachable, stop dialling it. Without it a scan that asks
    nine topical queries pays the full timeout nine times.
    """

    def __init__(self, timeout: float = 8.0, email: str = "",
                 api_key: str = "", enabled: bool = True):
        self.timeout = timeout
        self.email = email
        self.api_key = api_key
        self.enabled = enabled
        self._circuit_open = False
        self._unreachable_reason = ""

    def _params(self, **extra) -> Dict[str, str]:
        params = {"tool": TOOL_NAME, "retmode": "json"}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        params.update({k: str(v) for k, v in extra.items()})
        return params

    def _get(self, endpoint: str, params: Dict[str, str]) -> Dict:
        if not self.enabled:
            raise PubMedUnavailable("PubMed lookup disabled")
        if self._circuit_open:
            raise PubMedUnavailable(
                f"PubMed unreachable ({self._unreachable_reason}); not retried for the "
                f"rest of this run")
        url = f"{EUTILS}/{endpoint}?{urllib.parse.urlencode(params)}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # The host answered. That is not a transport failure, so the
            # circuit stays closed.
            raise PubMedUnavailable(f"PubMed returned HTTP {e.code}")
        except Exception as e:
            self._circuit_open = True
            self._unreachable_reason = str(e)[:120]
            raise PubMedUnavailable(f"PubMed could not be reached: {self._unreachable_reason}")

    def search(self, term: str, retmax: int = DEFAULT_RETMAX) -> List[str]:
        payload = self._get("esearch.fcgi",
                            self._params(db="pubmed", term=term, retmax=retmax,
                                         sort="relevance"))
        return list(payload.get("esearchresult", {}).get("idlist", []))

    def summarise(self, pmids: Sequence[str]) -> List[Article]:
        if not pmids:
            return []
        payload = self._get("esummary.fcgi",
                            self._params(db="pubmed", id=",".join(pmids)))
        result = payload.get("result", {})
        articles = []
        for pmid in result.get("uids", []):
            record = result.get(pmid, {})
            doi = ""
            for identifier in record.get("articleids", []):
                if identifier.get("idtype") == "doi":
                    doi = identifier.get("value", "")
            year = None
            pubdate = record.get("pubdate", "")
            if pubdate[:4].isdigit():
                year = int(pubdate[:4])
            articles.append(Article(
                pmid=str(pmid), title=record.get("title", ""),
                journal=record.get("fulljournalname", "") or record.get("source", ""),
                year=year, doi=doi))
        return articles


def search_literature(client: PubMedClient, queries: Sequence[LiteratureQuery],
                      per_query: int = DEFAULT_RETMAX) -> LiteratureResult:
    """
    Run the queries and collect the articles, deduplicated by PMID.

    A failure anywhere returns UNAVAILABLE with the transport error. It does
    not return the articles gathered so far under a success flag: a partial
    literature search reported as a complete one is a claim that the missing
    topics found nothing.
    """
    result = LiteratureResult(queries=list(queries))
    if not queries:
        result.status = "No query could be built: nothing identifies this peptide."
        return result

    seen: Dict[str, Article] = {}
    try:
        for query in queries:
            for article in client.summarise(client.search(query.term, retmax=per_query)):
                seen.setdefault(article.pmid, article)
    except PubMedUnavailable as e:
        result.reachable = False
        result.status = str(e)
        result.articles = []
        return result

    result.reachable = True
    result.articles = list(seen.values())
    result.status = (f"{len(result.articles)} distinct article(s) from "
                     f"{len(queries)} quer{'y' if len(queries) == 1 else 'ies'}")
    return result
