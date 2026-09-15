"""
Evidence retrieval from external sources:
- NCBI Entrez API (homolog retrieval)
- UniProt API (annotations)
- PubMed E-utilities (literature)
- Human Protein Atlas / GTEx (expression)
- Gene Ontology (function->cell type mapping)

All external queries are wrapped in error handling and caching.
"""

import logging
import json
import time
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RetrievalSource(Enum):
    """
    Where a piece of evidence actually came from.

    This exists because a status string is not provenance. The previous version
    returned "retrieved 3 homologs from NCBI" for sequences read out of a
    bundled JSON fixture, and conservation entropy computed from them was then
    presented as homolog evidence. The difference between a database and a file
    in this repository is the difference between a finding and a fixture, and it
    has to be carried in a field that a caller cannot accidentally reword.
    """
    LIVE = "LIVE"                    # actually fetched from the named service
    CACHED = "CACHED"                # fetched live earlier and cached since
    LOCAL_FIXTURE = "LOCAL_FIXTURE"  # read from a file in this repository
    UNAVAILABLE = "UNAVAILABLE"      # not wired up, or the attempt failed


@dataclass
class Retrieval:
    """A retrieval result and the truth about where it came from."""
    payload: Any
    source: RetrievalSource
    detail: str
    service: str = ""

    @property
    def is_evidence(self) -> bool:
        """
        Whether this may be cited as external evidence.

        A fixture may be used -- it is real sequence data and useful for a
        demonstration -- but it is not evidence that anything outside this
        repository agrees with, so it does not qualify.
        """
        return self.source in (RetrievalSource.LIVE, RetrievalSource.CACHED)

    def describe(self) -> str:
        if self.source is RetrievalSource.LOCAL_FIXTURE:
            return (f"{self.detail} This is bundled fixture data, not a retrieval. "
                    f"Nothing outside this repository was consulted.")
        if self.source is RetrievalSource.UNAVAILABLE:
            return self.detail
        return self.detail


class EvidenceRetriever:
    """
    Centralized interface for literature, homolog, and expression data.
    Gracefully handles network failures and missing APIs.
    """

    def __init__(self, cache_dir: str = ".evidence_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # API keys (can be set via env var, otherwise optional)
        self.ncbi_api_key = None  # In production: os.getenv("NCBI_API_KEY")
        self.rate_limit_delay = 0.3  # NCBI rate limit

    def _get_cache_path(self, query_id: str, query_type: str) -> Path:
        """Get cache file path for a query."""
        safe_name = query_id.replace("/", "_").replace(" ", "_")
        return self.cache_dir / f"{query_type}_{safe_name}.json"

    def _load_cache(self, cache_path: Path) -> Optional[Dict]:
        """
        Load a cached result, but only one this version wrote.

        An entry without the LIVE marker was written by the version that cached
        fabricated and fixture data, so it cannot be distinguished from a real
        retrieval by inspection. Treating it as a cache miss is what makes the
        fix retroactive: without this, a machine that ever ran the old code
        keeps serving laundered provenance forever, and the poisoned entries are
        invisible because they look exactly like good ones.
        """
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, "r") as f:
                entry = json.load(f)
        except Exception as e:
            logger.warning(f"Cache load failed for {cache_path}: {e}")
            return None

        if not isinstance(entry, dict) or entry.get("_source") != RetrievalSource.LIVE.value:
            logger.warning(
                f"Ignoring cache entry {cache_path.name}: it carries no live-retrieval marker, "
                f"so it predates provenance tracking and its origin cannot be established."
            )
            return None
        return entry

    def _save_cache(self, cache_path: Path, data: Dict, source: RetrievalSource) -> None:
        """
        Save a result to cache, but only if it was actually retrieved.

        Caching anything else launders its provenance: a fixture written to the
        cache comes back on the next call as a cache hit, and a cache hit reads
        as "we fetched this once". After one run the fabricated and the real
        would have been indistinguishable, which is worse than never caching at
        all.
        """
        if source is not RetrievalSource.LIVE:
            logger.debug(f"Not caching {cache_path.name}: source is {source.value}, not LIVE")
            return
        try:
            with open(cache_path, "w") as f:
                json.dump({"_source": source.value, "_cached_utc": time.time(), **data}, f)
        except Exception as e:
            logger.warning(f"Cache save failed for {cache_path}: {e}")

    # ====== NCBI Entrez Queries ======

    def retrieve_homologs(
        self,
        gene_name: str,
        organism: str = "human",
        max_homologs: int = 30,
    ) -> Retrieval:
        """
        Retrieve homologous sequences.

        Returns a Retrieval, not a (list, string) pair, so the caller has to
        read the source off a field rather than off prose it is free to
        paraphrase. The paraphrase is how "read from a bundled fixture" became
        "retrieved 3 homologs from NCBI".
            - Sequences are aligned (gaps indicated by '-')
        """
        cache_key = f"{gene_name}_{organism}"
        cache_path = self._get_cache_path(cache_key, "homologs")

        cached = self._load_cache(cache_path)
        if cached:
            sequences = cached.get("sequences", [])
            logger.info(f"Using cached homologs for {gene_name} ({len(sequences)} sequences)")
            return Retrieval(
                payload=sequences,
                source=RetrievalSource.CACHED,
                detail=f"{len(sequences)} homolog(s) from a previous live NCBI retrieval",
                service="NCBI Entrez",
            )

        # Live retrieval is not wired up. Saying so is the whole point: the
        # previous version called a fixture reader here and reported its output
        # as an NCBI result.
        fixture = self._fixture_homologs(gene_name)
        if fixture:
            return Retrieval(
                payload=fixture,
                source=RetrievalSource.LOCAL_FIXTURE,
                detail=(f"{len(fixture)} homolog(s) read from the bundled test panel for "
                        f"{gene_name}."),
                service="peptide_suite/data/test_panel.json",
            )

        return Retrieval(
            payload=[],
            source=RetrievalSource.UNAVAILABLE,
            detail=("NCBI Entrez retrieval is not wired up in this build and no bundled "
                    "sequences exist for this peptide, so no homologs were obtained."),
            service="NCBI Entrez",
        )

    def _fixture_homologs(self, gene_name: str) -> List[str]:
        """
        Homologs bundled with the repository, for peptides in the test panel.

        Real sequences, useful for exercising the conservation path without a
        network. They are returned tagged as a fixture and never cached, so
        nothing downstream can mistake them for a database answer.
        """
        panel_path = Path(__file__).parent.parent / "data" / "test_panel.json"
        if not panel_path.exists():
            return []
        try:
            panel = json.loads(panel_path.read_text())
        except Exception as e:
            logger.debug(f"Could not load test panel: {e}")
            return []
        # Matched on alphanumerics only. The panel is keyed "BPC157" while the
        # reference set names the same peptide "BPC-157", so an exact match on
        # the raw name never fired and this whole branch was unreachable.
        def normalise(name: str) -> str:
            return "".join(c for c in name.upper() if c.isalnum())

        wanted = normalise(gene_name)
        for key, entry in panel.items():
            if normalise(key) == wanted and isinstance(entry, dict):
                return list(entry.get("homologs", []))
        return []

    def retrieve_literature_context(
        self, gene_name: str, keyword: str = ""
    ) -> Retrieval:
        """
        Retrieve relevant PubMed articles.

        Each article is {"pmid": ..., "title": ..., "abstract": ...}.
        """
        cache_key = f"{gene_name}_{keyword}"
        cache_path = self._get_cache_path(cache_key, "literature")

        cached = self._load_cache(cache_path)
        if cached:
            articles = cached.get("articles", [])
            return Retrieval(
                payload=articles,
                source=RetrievalSource.CACHED,
                detail=f"{len(articles)} article(s) from a previous live PubMed retrieval",
                service="PubMed E-utilities",
            )

        # There is no honest offline version of a literature search. The
        # previous version returned an invented PMID and an abstract reading
        # "This is a placeholder abstract for calibration testing", then cached
        # it, after which it came back as a cache hit with no marker at all.
        return Retrieval(
            payload=[],
            source=RetrievalSource.UNAVAILABLE,
            detail=("PubMed retrieval is not wired up in this build, so no literature was "
                    "consulted. Any claim below rests on sequence computation or on the "
                    "bundled reference set, not on a literature search."),
            service="PubMed E-utilities",
        )

    # ====== UniProt Queries ======

    def retrieve_uniprot_data(self, gene_name: str) -> Retrieval:
        """Retrieve UniProt protein data (annotations, tissue expression hints)."""
        cache_path = self._get_cache_path(gene_name, "uniprot")

        cached = self._load_cache(cache_path)
        if cached:
            return Retrieval(
                payload=cached,
                source=RetrievalSource.CACHED,
                detail="annotation from a previous live UniProt retrieval",
                service="UniProt",
            )

        # The previous version returned a dict whose fields read "To be inferred
        # from literature" and "To be queried from Human Protein Atlas". Those
        # are not annotations; shaped like annotations, they are worse than
        # nothing, because a caller sees populated fields.
        return Retrieval(
            payload=None,
            source=RetrievalSource.UNAVAILABLE,
            detail=("UniProt retrieval is not wired up in this build (and is blocked by the "
                    "container's egress policy), so no annotation was obtained. Identification "
                    "falls back to the local reference set."),
            service="UniProt",
        )

    # ====== Function Inference ======

    def infer_function_from_name(self, gene_or_peptide_name: str) -> Tuple[str, float]:
        """
        Infer primary function from gene/peptide name and cached data.

        Returns:
            Tuple of (inferred_function_string, confidence_0_to_1)
        """
        name_upper = gene_or_peptide_name.upper()

        # Hardcoded inference for common test cases
        known_functions = {
            "IGF1": ("Insulin-like growth factor 1 (IGF-1 receptor binding and mitogenic signaling)", 0.95),
            "INS": ("Insulin (glucose homeostasis via IR/IGF-1R binding)", 0.95),
            "GCG": ("Glucagon (blood glucose elevation via GCGR)", 0.9),
            "GLP1": ("GLP-1 agonist (incretin effect, glucose-dependent insulin secretion)", 0.9),
            "BPC157": (
                "Body Protection Compound 157 (wound healing, gastrointestinal repair, "
                "anti-inflammatory signaling)",
                0.8,
            ),
        }

        if name_upper in known_functions:
            return known_functions[name_upper]

        # Fallback: ask user to specify
        return (
            "Unknown peptide. Please specify target function (e.g., 'binding affinity', 'protease resistance').",
            0.0,
        )

    # ====== Gene Ontology / expression (Workflow 2) ======

    def query_gene_ontology(self, keywords: str) -> Tuple[List[Dict], str]:
        """
        Map a functional keyword query to GO biological process terms and the
        cell types annotated to them.

        Returns:
            Tuple of (hits, status_message). Returns an empty list when live
            retrieval is unavailable, so callers fall back to the local cache
            at a visibly lower evidence tier rather than silently pretending
            the ontology answered.
        """
        logger.info(f"Gene Ontology query for '{keywords}': live retrieval not wired up")
        return [], (
            "Gene Ontology API not wired up in this build — no live ontology query performed"
        )

    def retrieve_expression_profiles(
        self, cell_types: List[str]
    ) -> Tuple[Dict[str, Dict], str]:
        """
        Retrieve expression specificity for genes in the given cell types from
        Human Protein Atlas / GTEx.

        Returns:
            Tuple of (gene -> {specificity, note}, status_message). Returns an
            empty dict when unavailable; callers must report expression
            specificity as unretrieved rather than substituting a placeholder
            number.
        """
        logger.info(
            f"Expression profile query for {len(cell_types)} cell type(s): "
            f"live retrieval not wired up"
        )
        return {}, (
            "Human Protein Atlas / GTEx APIs not wired up in this build — "
            "no expression specificity retrieved"
        )

    def get_function_keywords(self) -> List[str]:
        """Return common functional keywords for 'Find Peptides' workflow."""
        return [
            "binding_affinity",
            "protease_resistance",
            "activation_signal",
            "antagonism",
            "tissue_repair",
            "anti_inflammatory",
            "mitogenic",
            "metabolic_regulation",
        ]
