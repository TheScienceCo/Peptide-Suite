"""
UniProt lookup.

A local reference cache of a dozen peptides will never recognise an arbitrary
protein, and curating a bigger one is the wrong answer to that problem. This
module queries UniProt directly so the tool can identify what was pasted.

Three routes, in descending reliability:

  1. ACCESSION   a FASTA header carries sp|P24043|... — look it up exactly
  2. PEPTIDE     find UniProt entries containing the sequence as a substring
  3. TEXT        search the protein name from the FASTA description

All network use is optional, time-boxed, and cached. When UniProt cannot be
reached the result says so explicitly and the caller falls back to local
inference, rather than the failure surfacing as "not recognised" — which would
blame the peptide for a network problem.
"""

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

UNIPROT_REST = "https://rest.uniprot.org"
PEPTIDE_SEARCH = "https://peptidesearch.uniprot.org/asyncrest"
USER_AGENT = "peptide-suite/1.2 (research tool)"

# FASTA headers in the db|accession|entry form, e.g.
#   >sp|P24043|LAMA2_HUMAN Laminin subunit alpha-2 OS=Homo sapiens ...
FASTA_DB_HEADER = re.compile(r"^(sp|tr)\|([A-Z0-9]+(?:-\d+)?)\|(\S+)\s*(.*)$", re.IGNORECASE)
ACCESSION_ONLY = re.compile(r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b")


@dataclass
class UniProtRecord:
    """What UniProt knows about an entry."""
    accession: str
    entry_name: str = ""
    protein_name: str = ""
    gene: str = ""
    organism: str = ""
    length: int = 0
    function: str = ""
    subcellular_location: str = ""
    keywords: List[str] = field(default_factory=list)
    features: List[Dict] = field(default_factory=list)
    sequence: str = ""
    url: str = ""

    @property
    def citation(self) -> str:
        return f"UniProt:{self.accession} ({self.url})"


@dataclass
class LookupResult:
    """Outcome of a lookup, including the case where nothing was reached."""
    record: Optional[UniProtRecord] = None
    route: str = ""            # "accession" | "peptide" | "text" | ""
    reachable: bool = True
    status: str = ""
    candidates: List[Dict] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.record is not None


def parse_fasta_header(raw: str) -> Dict[str, str]:
    """
    Pull the accession, entry name and description out of a pasted FASTA header.

    Returns empty strings rather than None so callers can treat this uniformly.
    """
    text = raw.strip()
    if not text.startswith(">"):
        return {"accession": "", "entry_name": "", "description": "", "organism": ""}

    header = text.split("\n", 1)[0][1:].strip()

    match = FASTA_DB_HEADER.match(header)
    if match:
        _db, accession, entry_name, description = match.groups()
        organism = ""
        os_match = re.search(r"OS=(.+?)(?:\s+[A-Z]{2}=|$)", description)
        if os_match:
            organism = os_match.group(1).strip()
        # Strip the trailing OS=/OX=/GN= key-value tail from the description
        clean_desc = re.split(r"\s+[A-Z]{2}=", description)[0].strip()
        return {
            "accession": accession.upper(),
            "entry_name": entry_name,
            "description": clean_desc,
            "organism": organism,
        }

    loose = ACCESSION_ONLY.search(header)
    return {
        "accession": loose.group(1) if loose else "",
        "entry_name": "",
        "description": header,
        "organism": "",
    }


class UniProtClient:
    """Time-boxed, cached UniProt queries that degrade gracefully offline."""

    def __init__(self, cache_dir: str = ".evidence_cache", timeout: float = 8.0,
                 enabled: bool = True):
        self.cache_dir = Path(cache_dir) / "uniprot"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.enabled = enabled
        self._unreachable_reason = ""
        # Once the host is established unreachable, stop dialling it. The class
        # promises to be "time-boxed, cached, and degrade gracefully offline",
        # and without this the time box is per call rather than per run: on a
        # network where the request hangs rather than refusing, every lookup
        # costs the full timeout, and a scan that identifies many sequences
        # pays it once each.
        self._circuit_open = False

    # ---- plumbing --------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:120]
        return self.cache_dir / f"{safe}.json"

    def _get_json(self, url: str, cache_key: str) -> Tuple[Optional[Dict], str]:
        cached = self._cache_path(cache_key)
        if cached.exists():
            try:
                return json.loads(cached.read_text()), "cached"
            except Exception:
                pass

        if not self.enabled:
            return None, "UniProt lookup disabled"

        if self._circuit_open:
            return None, (f"UniProt unreachable ({self._unreachable_reason}); not retried "
                          f"for the rest of this run")

        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            cached.write_text(json.dumps(data))
            return data, "retrieved"
        except urllib.error.HTTPError as e:
            return None, f"UniProt returned HTTP {e.code}"
        except urllib.error.URLError as e:
            # The host did not answer. Open the circuit: a second identical
            # failure tells us nothing and costs another timeout.
            self._unreachable_reason = str(e.reason)
            self._circuit_open = True
            return None, f"UniProt unreachable: {e.reason}"
        except Exception as e:  # timeouts, malformed JSON
            self._unreachable_reason = str(e)
            self._circuit_open = True
            return None, f"UniProt lookup failed: {e}"

    # ---- parsing ---------------------------------------------------------

    @staticmethod
    def _record_from_entry(entry: Dict) -> UniProtRecord:
        accession = entry.get("primaryAccession", "")

        desc = entry.get("proteinDescription", {})
        recommended = desc.get("recommendedName", {}).get("fullName", {}).get("value", "")
        submitted = ""
        if not recommended and desc.get("submissionNames"):
            submitted = desc["submissionNames"][0].get("fullName", {}).get("value", "")

        genes = entry.get("genes", [])
        gene = genes[0].get("geneName", {}).get("value", "") if genes else ""

        function, location = "", ""
        for comment in entry.get("comments", []):
            ctype = comment.get("commentType", "")
            if ctype == "FUNCTION" and not function:
                texts = comment.get("texts", [])
                if texts:
                    function = texts[0].get("value", "")
            elif ctype == "SUBCELLULAR LOCATION" and not location:
                locs = comment.get("subcellularLocations", [])
                names = [l.get("location", {}).get("value", "") for l in locs]
                location = "; ".join(n for n in names if n)

        features = [
            {
                "type": f.get("type", ""),
                "description": f.get("description", ""),
                "start": f.get("location", {}).get("start", {}).get("value"),
                "end": f.get("location", {}).get("end", {}).get("value"),
            }
            for f in entry.get("features", [])
            if f.get("type") in {"Domain", "Region", "Disulfide bond", "Glycosylation",
                                 "Modified residue", "Signal", "Propeptide", "Chain",
                                 "Peptide", "Active site", "Binding site"}
        ]

        return UniProtRecord(
            accession=accession,
            entry_name=entry.get("uniProtkbId", ""),
            protein_name=recommended or submitted,
            gene=gene,
            organism=entry.get("organism", {}).get("scientificName", ""),
            length=entry.get("sequence", {}).get("length", 0),
            function=function,
            subcellular_location=location,
            keywords=[k.get("name", "") for k in entry.get("keywords", [])],
            features=features,
            sequence=entry.get("sequence", {}).get("value", ""),
            url=f"https://www.uniprot.org/uniprotkb/{accession}",
        )

    # ---- routes ----------------------------------------------------------

    def by_accession(self, accession: str) -> LookupResult:
        """Exact lookup. The most reliable route when a FASTA header is present."""
        url = f"{UNIPROT_REST}/uniprotkb/{urllib.parse.quote(accession)}"
        data, status = self._get_json(url, f"acc_{accession}")

        if data is None:
            return LookupResult(reachable=False, status=status, route="accession")
        return LookupResult(record=self._record_from_entry(data),
                            route="accession", status=status)

    def by_text(self, query: str, organism: str = "") -> LookupResult:
        """Search by protein name. Used when a header has a description but no accession."""
        terms = [f'"{query}"']
        if organism:
            terms.append(f'organism_name:"{organism}"')
        q = " AND ".join(terms)
        url = (f"{UNIPROT_REST}/uniprotkb/search?query={urllib.parse.quote(q)}"
               f"&format=json&size=5")
        data, status = self._get_json(url, f"text_{query}_{organism}")

        if data is None:
            return LookupResult(reachable=False, status=status, route="text")

        results = data.get("results", [])
        if not results:
            return LookupResult(status=f"no UniProt entry matched '{query}'", route="text")

        candidates = [
            {
                "accession": r.get("primaryAccession", ""),
                "name": r.get("proteinDescription", {})
                         .get("recommendedName", {}).get("fullName", {}).get("value", ""),
                "organism": r.get("organism", {}).get("scientificName", ""),
            }
            for r in results[1:]
        ]
        return LookupResult(record=self._record_from_entry(results[0]),
                            route="text", status=status, candidates=candidates)

    def by_peptide(self, sequence: str, max_wait: float = 20.0) -> LookupResult:
        """
        Find entries containing this sequence as an exact substring.

        UniProt's peptide search is asynchronous: the POST returns a job
        location which is polled until it resolves. For a long sequence a
        representative window is searched instead, since the service is
        intended for peptide-length queries.
        """
        if not self.enabled:
            return LookupResult(reachable=False, status="UniProt lookup disabled", route="peptide")

        probe = sequence.upper()
        if len(probe) > 40:
            # A window from the middle avoids signal peptides and terminal tags
            mid = len(probe) // 2
            probe = probe[mid - 15:mid + 15]

        cache_key = f"pep_{probe}"
        cached = self._cache_path(cache_key)
        if cached.exists():
            try:
                accs = json.loads(cached.read_text()).get("accessions", [])
                if accs:
                    return self._first_of(accs, route="peptide", status="cached")
                return LookupResult(status="no UniProt entry contains this sequence",
                                    route="peptide")
            except Exception:
                pass

        payload = urllib.parse.urlencode({"peps": probe, "lEQi": "off", "spOnly": "off"}).encode()
        req = urllib.request.Request(
            PEPTIDE_SEARCH + "/", data=payload,
            headers={"User-Agent": USER_AGENT,
                     "Content-Type": "application/x-www-form-urlencoded"},
        )

        if self._circuit_open:
            return LookupResult(
                reachable=False, route="peptide",
                status=(f"UniProt unreachable ({self._unreachable_reason}); not retried "
                        f"for the rest of this run"))

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                job_url = resp.headers.get("Location") or resp.geturl()

            deadline = time.time() + max_wait
            body = ""
            while time.time() < deadline:
                poll = urllib.request.Request(job_url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(poll, timeout=self.timeout) as resp:
                    if resp.status == 200:
                        body = resp.read().decode("utf-8").strip()
                        break
                time.sleep(1.5)

            accessions = [a for a in re.split(r"[\s,]+", body) if a][:5]
            cached.write_text(json.dumps({"accessions": accessions}))

            if not accessions:
                return LookupResult(status="no UniProt entry contains this sequence",
                                    route="peptide")
            return self._first_of(accessions, route="peptide", status="retrieved")

        except urllib.error.URLError as e:
            self._unreachable_reason = str(e.reason)
            self._circuit_open = True
            return LookupResult(reachable=False, route="peptide",
                                status=f"UniProt peptide search unreachable: {e.reason}")
        except Exception as e:
            return LookupResult(reachable=False, route="peptide",
                                status=f"UniProt peptide search failed: {e}")

    def _first_of(self, accessions: List[str], route: str, status: str) -> LookupResult:
        first = self.by_accession(accessions[0])
        if not first.found:
            return LookupResult(reachable=first.reachable, route=route, status=first.status)
        first.route = route
        first.status = status
        first.candidates = [{"accession": a} for a in accessions[1:]]
        return first

    # ---- entry point -----------------------------------------------------

    def identify(self, sequence: str, raw_input: str = "") -> LookupResult:
        """
        Identify a pasted sequence, trying the routes in order of reliability.
        """
        header = parse_fasta_header(raw_input) if raw_input else {}

        if header.get("accession"):
            result = self.by_accession(header["accession"])
            if result.found or not result.reachable:
                return result

        result = self.by_peptide(sequence)
        if result.found:
            return result
        peptide_unreachable = not result.reachable

        if header.get("description"):
            text_result = self.by_text(header["description"], header.get("organism", ""))
            if text_result.found or not text_result.reachable:
                return text_result

        if peptide_unreachable:
            return result

        return LookupResult(
            status="UniProt was reached but no entry matched this sequence",
            route="peptide",
        )

    def connectivity(self) -> Tuple[bool, str]:
        """Check whether UniProt is reachable, for the self-test command."""
        data, status = self._get_json(f"{UNIPROT_REST}/uniprotkb/P01308", "acc_P01308")
        return data is not None, status
