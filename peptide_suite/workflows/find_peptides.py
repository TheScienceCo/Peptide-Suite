"""
Workflow 2: "Find Peptides"

Input: a functional goal as keywords (e.g. "myelinating peptides").

What this is, stated plainly: a structured, expression-profile-informed
literature search. It surfaces and ranks peptides that existing sources already
associate with a function. It does not discover novel peptides and it is not a
substitute for wet-lab work.

Flow:
  Step 1  Identify cell types known to perform the function (GO / Cell Ontology)
  Step 2  Check whether the literature already answers the question. If it
          clearly does, say so and stop -- do not force the full workflow.
  Step 3  Otherwise, cross-reference expression profiles for those cell types
          against literature mentions and rank candidates by combined evidence.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from peptide_suite.core import (
    CellTypeHit,
    PeptideCandidate,
    FindPeptidesResult,
    EvidenceTier,
    ConfidenceLevel,
    evidence_weight,
)
from peptide_suite.core.confidence_scoring import ConfidenceScorer
from peptide_suite.core.evidence_retrieval import EvidenceRetriever

logger = logging.getLogger(__name__)

METHODOLOGY_NOTE = (
    "This is a structured, expression-profile-informed literature search, not a "
    "wet-lab discovery tool. It ranks peptides that existing sources already "
    "associate with the queried function. Every candidate requires independent "
    "verification against primary literature before use."
)


class FindPeptidesWorkflow:
    """Locate peptides associated with a functional goal."""

    def __init__(self):
        self.evidence = EvidenceRetriever()
        self.scorer = ConfidenceScorer()
        self.ontology = self._load_ontology()

    @staticmethod
    def _load_ontology() -> Dict:
        path = Path(__file__).parent.parent / "data" / "function_ontology.json"
        with open(path) as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("_")}

    def run(self, goal_keywords: str, force_full_workflow: bool = False) -> FindPeptidesResult:
        """
        Execute the Find Peptides workflow.

        Args:
            goal_keywords: Free-text functional goal, e.g. "myelinating peptides"
            force_full_workflow: Run Step 3 even when Step 2 finds a clear answer

        Returns:
            FindPeptidesResult
        """
        logger.info("=== Workflow 2: Find Peptides ===")
        logger.info(f"Query: {goal_keywords}")

        result = FindPeptidesResult(query=goal_keywords, methodology_note=METHODOLOGY_NOTE)

        # ---- Step 1: identify capable cell types -------------------------
        domain_key, domain = self._match_domain(goal_keywords)

        if domain is None:
            result.data_notes.append(
                f"No ontology match for '{goal_keywords}'. Live Gene Ontology and UniProt "
                f"keyword search are not wired up in this build, and the local fallback cache "
                f"does not cover this function. No cell types or candidates are reported "
                f"rather than guessing. Known domains in the local cache: "
                f"{', '.join(sorted(self.ontology))}."
            )
            logger.warning(f"No ontology match for '{goal_keywords}'")
            return result

        logger.info(f"Matched functional domain '{domain_key}' ({domain['go_term']})")

        source_label, source_tier = self._resolve_ontology_source(goal_keywords)

        for ct in domain["cell_types"]:
            result.cell_types.append(
                CellTypeHit(
                    name=ct["name"],
                    ontology_id=ct.get("ontology_id", ""),
                    relationship=ct.get("relationship", ""),
                    source=source_label,
                    confidence=evidence_weight(source_tier),
                )
            )

        logger.info(f"Step 1: {len(result.cell_types)} cell types identified")

        # ---- Step 2: check for an existing known answer ------------------
        established = domain.get("established_peptides", [])
        if established:
            result.known_answer_found = True
            for entry in established:
                result.known_answers.append(
                    self._build_candidate(
                        entry,
                        evidence_tier=EvidenceTier.HOMOLOG_EXPERIMENTAL,
                        literature_note=(
                            "Well-established association in the literature for this function."
                        ),
                        literature_score=0.85,
                        source_label=source_label,
                    )
                )

            logger.info(
                f"Step 2: {len(result.known_answers)} established peptides found — "
                f"the literature already answers this question"
            )

            if not force_full_workflow:
                result.data_notes.append(
                    f"The literature already provides a clear answer for "
                    f"'{goal_keywords}' ({len(result.known_answers)} established "
                    f"peptide(s)). The full expression-comparison workflow was not run, "
                    f"since forcing it here would add noise rather than information. "
                    f"Re-run with force_full_workflow=True to see ranked candidates anyway."
                )
                return result

        # ---- Step 3: rank candidates by combined evidence ----------------
        expression_data, expression_status = self.evidence.retrieve_expression_profiles(
            [ct.name for ct in result.cell_types]
        )

        if not expression_data:
            result.data_notes.append(
                f"Expression profiles NOT retrieved ({expression_status}). Human Protein "
                f"Atlas / GTEx retrieval is not wired up in this build, so expression "
                f"specificity is reported as unavailable rather than estimated, and ranking "
                f"below reflects literature association only."
            )
            logger.warning(f"No expression data: {expression_status}")

        for entry in domain.get("candidate_peptides", []):
            result.candidates.append(
                self._build_candidate(
                    entry,
                    evidence_tier=EvidenceTier.BIOCHEMICAL_PRINCIPLE,
                    literature_note="Reported association; weaker or more context-dependent than the established set.",
                    literature_score=0.55,
                    source_label=source_label,
                    expression_data=expression_data,
                )
            )

        result.candidates.sort(key=lambda c: c.combined_score, reverse=True)
        logger.info(f"Step 3: {len(result.candidates)} candidates ranked")

        return result

    # ---- internals -------------------------------------------------------

    def _match_domain(self, query: str) -> Tuple[Optional[str], Optional[Dict]]:
        """Match a free-text query against known functional domains."""
        q = query.lower()
        best_key, best_domain, best_len = None, None, 0

        for key, domain in self.ontology.items():
            for kw in domain.get("keywords", []):
                # Longest matching keyword wins, so "wound healing" beats "wound"
                if kw in q and len(kw) > best_len:
                    best_key, best_domain, best_len = key, domain, len(kw)

        return best_key, best_domain

    def _resolve_ontology_source(self, query: str) -> Tuple[str, EvidenceTier]:
        """
        Determine which resource answered, and at what evidence tier.

        Live GO/UniProt retrieval would raise this to DIRECT_EXPERIMENTAL; the
        local cache is explicitly a lower tier so the distinction stays visible
        in the output rather than being smoothed over.
        """
        go_hits, status = self.evidence.query_gene_ontology(query)
        if go_hits:
            return "Gene Ontology (live)", EvidenceTier.DIRECT_EXPERIMENTAL
        return "curated_local_seed (unverified)", EvidenceTier.BIOCHEMICAL_PRINCIPLE

    def _build_candidate(
        self,
        entry: Dict,
        evidence_tier: EvidenceTier,
        literature_note: str,
        literature_score: float,
        source_label: str,
        expression_data: Optional[Dict] = None,
    ) -> PeptideCandidate:
        """Assemble a candidate, leaving unretrieved scores as None rather than 0."""
        gene = entry.get("gene", "")

        expression_score = None
        expression_note = (
            "Expression specificity NOT retrieved — no Human Protein Atlas / GTEx query "
            "was performed, so no specificity value is reported."
        )
        if expression_data and gene in expression_data:
            expression_score = expression_data[gene]["specificity"]
            expression_note = expression_data[gene]["note"]

        # Combined score uses only the components that were actually computed.
        if expression_score is not None:
            combined = 0.6 * literature_score + 0.4 * expression_score
        else:
            combined = literature_score

        confidence = (
            ConfidenceLevel.HIGH
            if combined >= 0.7
            else ConfidenceLevel.MEDIUM
            if combined >= 0.4
            else ConfidenceLevel.LOW
        )

        citations = []
        if entry.get("uniprot"):
            citations.append(f"UniProt:{entry['uniprot']} (unverified — confirm at uniprot.org)")

        return PeptideCandidate(
            name=entry["name"],
            gene=gene,
            uniprot=entry.get("uniprot", ""),
            rationale=entry.get("rationale", ""),
            expression_note=expression_note,
            literature_note=f"{literature_note} Source: {source_label}.",
            expression_score=expression_score,
            literature_score=literature_score,
            combined_score=round(combined, 3),
            confidence=confidence,
            evidence_tier=evidence_tier,
            citations=citations,
            requires_verification=True,
        )


def format_find_result(result: FindPeptidesResult) -> str:
    """Render a FindPeptidesResult for terminal display."""
    lines = ["", "=" * 70, "WORKFLOW 2: FIND PEPTIDES", "=" * 70]
    lines.append(f"\nQuery: {result.query}")

    if result.cell_types:
        lines.append(f"\nSTEP 1 — Cell types implicated ({len(result.cell_types)}):")
        for ct in result.cell_types:
            ident = f" [{ct.ontology_id}]" if ct.ontology_id else ""
            lines.append(f"  • {ct.name}{ident}")
            lines.append(f"      {ct.relationship}")
            lines.append(f"      source: {ct.source}")

    if result.known_answer_found:
        lines.append(f"\nSTEP 2 — The literature already answers this:")
        for c in result.known_answers:
            lines.append(f"  • {c.name} ({c.gene}) — confidence {c.confidence.value}")
            lines.append(f"      {c.rationale}")
            for cit in c.citations:
                lines.append(f"      {cit}")
    else:
        lines.append("\nSTEP 2 — No established answer found; proceeding to ranking.")

    if result.candidates:
        lines.append(f"\nSTEP 3 — Ranked candidates ({len(result.candidates)}):")
        for i, c in enumerate(result.candidates, 1):
            lines.append(f"  {i}. {c.name} ({c.gene}) — score {c.combined_score:.2f}, {c.confidence.value}")
            lines.append(f"      {c.rationale}")
            lines.append(f"      literature: {c.literature_note}")
            lines.append(f"      expression: {c.expression_note}")

    if result.data_notes:
        lines.append("\nLIMITATIONS:")
        for note in result.data_notes:
            lines.append(f"  ! {note}")

    lines.append(f"\n{result.methodology_note}")
    return "\n".join(lines)
