"""
From PubMed to a curation queue, and the step nobody can automate.  [Phase 3]

Opening PubMed gets this repository verifiable citations. It does not get it
evidence records, and the gap between those two things is the whole reason this
module exists rather than a function that writes rows.

A `MeasuredOutcome` needs a comparator, an assay, a direction and a magnitude.
Those live in a paper's results, frequently in a figure, and an abstract that
mentions any of them usually mentions one. So a search returns candidates: real
PMIDs, real titles, and outcome fields that are empty because nobody has read
the paper yet. A candidate is not a record and cannot be used as one -- it has
no outcome at all, so `VariantRecord` treats it as a design, and the retrieval
engine already refuses to let a design support a claim about effect.

THE TEMPTING SHORTCUT is to parse the abstract. It works often enough to be
dangerous: "a three-fold higher IC50" and "three-fold more potent" are opposite
results in similar words, and the PMID checks out either way, so a misparse
survives exactly the review that would otherwise catch it. That is what
`Extraction` is for. A parsed outcome is admissible -- it caps at
BIOCHEMICAL_PRINCIPLE and says why -- and it can never be promoted by having a
good citation, because the citation was never the thing in doubt.

So the pipeline is: search (automatic), queue (automatic), read (human), record
(human). This module owns the first two and refuses the fourth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .literature import (
    Article, LiteratureResult, PubMedClient, QueryTopic, build_queries,
    search_literature,
)
from .variant_evidence import DATA_PATH, Extraction

#: The topics that bear on "somebody changed this residue and measured
#: something". The structural and receptor topics are deliberately absent: they
#: are the biological-context layer's business, and mixing them in here returns
#: a queue of papers about the fold when what is wanted is a queue of papers
#: about the variants.
INGEST_TOPICS = (
    QueryTopic.MUTAGENESIS,
    QueryTopic.ALANINE_SCAN,
    QueryTopic.VARIANT,
    QueryTopic.SUBSTITUTION,
    QueryTopic.ANALOG,
    QueryTopic.AFFINITY,
)


@dataclass(frozen=True)
class Candidate:
    """
    A paper that might contain a measurable outcome, and does not yet.

    Deliberately holds no outcome fields. An empty string where a comparator
    belongs invites someone to fill it in from the title, and a structure that
    makes the wrong action easy will eventually have it taken.
    """

    pmid: str
    title: str
    journal: str = ""
    year: Optional[int] = None
    doi: str = ""
    matched_topics: tuple = ()
    peptide: str = ""

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/" if self.pmid else ""

    def to_dict(self) -> Dict[str, Any]:
        return {"pmid": self.pmid, "title": self.title, "journal": self.journal,
                "year": self.year, "doi": self.doi, "url": self.url,
                "matched_topics": list(self.matched_topics), "peptide": self.peptide,
                "status": "AWAITING_CURATION",
                "what_is_missing": (
                    "No outcome has been recorded. Somebody has to read this paper and "
                    "fill in what was measured, against what comparator, in what assay, "
                    "and in which direction. None of those are in the title.")}


@dataclass
class IngestResult:
    """What a search found, and what it explicitly did not do with it."""

    peptide: str
    candidates: List[Candidate] = field(default_factory=list)
    literature: Optional[LiteratureResult] = None
    searched_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return not self.error and self.literature is not None and self.literature.reachable

    def statement(self) -> str:
        if self.error:
            return (f"No candidates for {self.peptide}: {self.error} Nothing was added "
                    f"to the store, and nothing was invented to stand in for what the "
                    f"search would have returned.")
        if not self.candidates:
            return (f"The search ran and returned no papers for {self.peptide} under "
                    f"the variant topics. That is a result about this query, not about "
                    f"whether such papers exist.")
        return (
            f"{len(self.candidates)} candidate paper(s) for {self.peptide}, queued for "
            f"curation. None of them is an evidence record yet: a candidate carries a "
            f"citation and no outcome, because what was measured, against what, and in "
            f"which direction are in the results section and not in the title. Reading "
            f"them is the step this module does not do.")

    def to_dict(self) -> Dict[str, Any]:
        return {"peptide": self.peptide, "searched_utc": self.searched_utc,
                "succeeded": self.succeeded, "error": self.error,
                "candidates": [c.to_dict() for c in self.candidates],
                "statement": self.statement(),
                "literature": self.literature.encode() if self.literature else None}


def find_candidates(peptide: str, aliases: Sequence[str] = (), gene: str = "",
                    client: Optional[PubMedClient] = None,
                    topics: Sequence[QueryTopic] = INGEST_TOPICS,
                    per_query: int = 20) -> IngestResult:
    """
    Search PubMed for papers that might hold a variant outcome.

    Returns candidates, never records. A failed search returns the transport
    error and an empty list rather than a shorter list under a success flag:
    reporting a partial search as a complete one is a claim that the topics
    that never ran found nothing.
    """
    queries = build_queries(name=peptide, aliases=aliases, gene=gene, topics=topics)
    if not queries:
        return IngestResult(peptide=peptide,
                            error="Nothing identifies this peptide, so no query "
                                  "could be built.")

    result = search_literature(client or PubMedClient(), queries, per_query=per_query)
    if not result.reachable:
        return IngestResult(peptide=peptide, literature=result,
                            error=result.status or "The search did not complete.")

    topic_names = tuple(t.value for t in topics)
    candidates = [
        Candidate(pmid=a.pmid, title=a.title, journal=a.journal, year=a.year,
                  doi=a.doi, matched_topics=topic_names, peptide=peptide)
        for a in result.articles if a.pmid]
    return IngestResult(peptide=peptide, candidates=candidates, literature=result)


class CurationRequired(RuntimeError):
    """Raised when something asks this module to write a record."""


def write_records(*_args, **_kwargs):
    """
    Turn candidates into evidence records. Not implemented, and raises.

    Present so that the omission is explicit rather than merely absent. A
    future contributor looking for the function that finishes the pipeline
    finds this instead of writing one, and finds the reason with it.
    """
    raise CurationRequired(
        "Candidates are not converted into evidence records automatically. A record "
        "needs the comparator, the assay, the direction and the magnitude, and those "
        "are in the results section -- an abstract that mentions any of them usually "
        "mentions one. Parsing them out is admissible but caps at "
        f"{Extraction.AUTOMATED_PARSE.max_tier.name} and must be recorded as "
        f"{Extraction.AUTOMATED_PARSE.value}, because 'a three-fold higher IC50' and "
        f"'three-fold more potent' are opposite results in similar words and the PMID "
        f"checks out either way. Write the records with an extraction method that says "
        f"who read the paper."
    )


def queue_path(directory: Optional[Path] = None) -> Path:
    """Where the curation queue lives: beside the store, not inside it."""
    return Path(directory or DATA_PATH.parent) / "variant_evidence_queue.json"


def save_queue(results: Sequence[IngestResult],
               directory: Optional[Path] = None) -> Path:
    """
    Write the candidate queue to disk, separately from the evidence store.

    A separate file on purpose. Candidates in the store file would be one
    careless merge away from being read as records, and the whole point of the
    distinction is that it survives carelessness.
    """
    path = queue_path(directory)
    path.write_text(json.dumps({
        "_about": (
            "Candidate papers awaiting curation. NOT evidence records, and not read by "
            "the evidence store: a candidate has a citation and no outcome. Kept in a "
            "separate file so that a careless merge cannot turn a queue into a store."),
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "results": [r.to_dict() for r in results],
    }, indent=2), encoding="utf-8")
    return path
