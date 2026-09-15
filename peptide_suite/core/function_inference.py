"""
Sequence-based function inference.

The previous implementation matched a hardcoded list of gene *names*. A pasted
sequence has no name, so inference could never succeed for the primary input
mode. This module infers from the sequence itself, through four descending
levels of evidence, and always returns a usable default:

    0. UNIPROT IDENTIFICATION     the sequence is looked up in UniProt
    1. EXACT / NEAR-EXACT MATCH   the sequence is a known peptide
    2. BIOACTIVE MOTIF            it contains a characterised functional motif
    3. FAMILY SIGNATURE           its computed properties fit a known profile
    4. LIABILITY-DRIVEN DEFAULT   nothing is recognised, so the goal is chosen
                                  from the liabilities actually found in it

Level 4 is the floor, not a failure: even for a completely unrecognised
sequence, the Tier 0 liability scan produces a defensible goal. What changes
across levels is the confidence and the claim type, both of which are reported,
so a level-4 guess never presents itself as a level-1 identification.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .epistemics import Claim
from .physics_tiers import Tier0Sequence

logger = logging.getLogger(__name__)


@dataclass
class FunctionInference:
    """The inferred function of a peptide, with how it was arrived at."""
    inferred_function: str
    suggested_goal: str
    goal_reason: str
    confidence: float           # 0-1
    basis: str                  # Which level answered
    level: int                  # 0-4
    claim: Claim = None
    matched_name: str = ""
    parent_protein: str = ""
    native_context_note: str = ""
    alternatives: List[Dict] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    uniprot: object = None   # UniProtRecord when identification came from UniProt

    @property
    def is_identification(self) -> bool:
        """True only when the sequence was actually recognised, not merely profiled."""
        return self.level <= 2


class FunctionInferencer:
    """Infers likely function from sequence, never returning nothing."""

    def __init__(self, uniprot: Optional["UniProtClient"] = None):
        self.reference = self._load_reference()
        self.tier0 = Tier0Sequence()
        if uniprot is None:
            from .uniprot_client import UniProtClient
            uniprot = UniProtClient()
        self.uniprot = uniprot

    @staticmethod
    def _load_reference() -> Dict:
        path = Path(__file__).parent.parent / "data" / "reference_peptides.json"
        with open(path) as f:
            return json.load(f)

    # ---- level 1: known sequence ----------------------------------------

    @staticmethod
    def _identity(a: str, b: str) -> float:
        """Fractional identity over the shorter sequence, ungapped."""
        if not a or not b:
            return 0.0
        short, long = (a, b) if len(a) <= len(b) else (b, a)
        best = 0
        for offset in range(len(long) - len(short) + 1):
            matches = sum(1 for i, ch in enumerate(short) if long[offset + i] == ch)
            best = max(best, matches)
        return best / len(short)

    # Below this length a "contained peptide" hit is chance, not signal:
    # every tetrapeptide occurs somewhere in a long enough sequence.
    MIN_CONTAINMENT_LENGTH = 6

    def _match_known_peptide(self, seq: str) -> Optional[FunctionInference]:
        """
        Match against the reference set in both directions.

        A pasted sequence relates to a known peptide in four ways, and only one
        of them is equality. Checking only for equality — or only for similar
        length — misses the two most common real cases: a fragment of a known
        peptide, and a precursor or construct that contains one.
        """
        exact, contains, fragment_of, similar = None, [], [], []

        for name, entry in self.reference["exact_peptides"].items():
            ref = entry["sequence"].upper()

            if seq == ref:
                exact = (name, entry)
                break

            if len(ref) >= self.MIN_CONTAINMENT_LENGTH and ref in seq:
                contains.append((name, entry, seq.index(ref), len(ref)))
            elif len(seq) >= self.MIN_CONTAINMENT_LENGTH and seq in ref:
                fragment_of.append((name, entry, ref.index(seq), len(seq)))
            elif abs(len(seq) - len(ref)) <= max(len(ref) * 0.5, 8):
                identity = self._identity(seq, ref)
                if identity >= 0.80:
                    similar.append((name, entry, identity))

        if exact:
            name, entry = exact
            return self._peptide_hit(
                name, entry, confidence=0.95,
                basis=f"Exact sequence match to {name}",
                claim_text=f"Sequence matches {name}: {entry['function']}",
            )

        if contains:
            # The longest contained peptide is the most specific identification
            contains.sort(key=lambda h: h[3], reverse=True)
            name, entry, at, length = contains[0]
            others = [h[0] for h in contains[1:]]
            return self._peptide_hit(
                name, entry, confidence=0.88,
                basis=f"Contains the full {name} sequence at position {at + 1}-{at + length}",
                claim_text=f"Contains {name} ({length} residues at position {at + 1})",
                caveats=[
                    f"Your sequence is {len(seq)} residues and contains {name} ({length} residues) "
                    f"within it. The surrounding residues are not part of {name} and may change "
                    f"its behaviour — a contained peptide is not necessarily an active one.",
                ] + ([f"Also contains: {', '.join(others)}."] if others else []),
                extra={"contained_at": at + 1, "contained_length": length},
            )

        if fragment_of:
            fragment_of.sort(key=lambda h: h[3], reverse=True)
            name, entry, at, length = fragment_of[0]
            coverage = length / len(entry["sequence"])
            return self._peptide_hit(
                name, entry, confidence=0.70 + 0.2 * coverage,
                basis=(
                    f"Fragment of {name}: residues {at + 1}-{at + length} of "
                    f"{len(entry['sequence'])} ({coverage:.0%} coverage)"
                ),
                claim_text=f"Fragment of {name}, residues {at + 1}-{at + length}",
                caveats=[
                    f"This is a {coverage:.0%} fragment of {name}, not the whole peptide. The "
                    f"annotated function describes the full sequence; whether this fragment "
                    f"retains it depends on where the active region sits. Both termini here are "
                    f"excision artefacts.",
                ],
            )

        if similar:
            similar.sort(key=lambda h: h[2], reverse=True)
            name, entry, identity = similar[0]
            return self._peptide_hit(
                name, entry, confidence=0.70 + (identity - 0.80) * 1.2,
                basis=f"{identity:.0%} identity to {name}",
                claim_text=f"Sequence closely resembles {name}: {entry['function']}",
                caveats=[
                    f"Not an exact match ({identity:.0%} identity to {name}). The differences may "
                    f"be exactly the positions that matter, so treat the functional assignment as "
                    f"provisional.",
                ],
            )

        return None

    def _peptide_hit(self, name, entry, confidence, basis, claim_text,
                     caveats=None, extra=None) -> FunctionInference:
        return FunctionInference(
            inferred_function=entry["function"],
            suggested_goal=entry["suggested_goal"],
            goal_reason=entry["goal_reason"],
            confidence=round(min(0.95, confidence), 2),
            basis=basis,
            level=1,
            matched_name=name,
            parent_protein=entry.get("family", ""),
            claim=Claim.retrieved(
                claim_text,
                citations=[f"local reference cache entry '{name}' (unverified — confirm against UniProt)"],
                tested=False,
            ),
            caveats=caveats or [],
        )

    def resolve_name(self, query: str) -> Optional[Tuple[str, Dict]]:
        """
        Look up a reference peptide by name rather than by sequence.

        Someone who knows what they want should be able to type "GLP-1" instead
        of hunting down the sequence first.
        """
        q = query.strip().lower()
        if not q:
            return None

        peptides = self.reference["exact_peptides"]

        for name, entry in peptides.items():
            if name.lower() == q:
                return name, entry

        # Normalised comparison: "glp1" should find "GLP-1 (7-37)"
        def norm(s):
            return "".join(ch for ch in s.lower() if ch.isalnum())

        nq = norm(q)
        candidates = [(n, e) for n, e in peptides.items() if norm(n).startswith(nq)]
        if not candidates:
            candidates = [(n, e) for n, e in peptides.items() if nq and nq in norm(n)]
        if not candidates:
            candidates = [(n, e) for n, e in peptides.items() if nq and nq in norm(e.get("family", ""))]

        if candidates:
            # Shortest name is the least qualified, so the most canonical
            candidates.sort(key=lambda c: len(c[0]))
            return candidates[0]
        return None

    def lookup_protein_by_name(self, query: str) -> Optional[Dict]:
        """
        Resolve a name that is a protein rather than a reference peptide.

        Typing "laminin" should not dead-end: the function ontology knows about
        these even when no peptide sequence for them exists locally. Returns
        what is known plus the derived motifs that ARE analysable, so the answer
        is a route forward rather than a miss.
        """
        q = query.strip().lower()
        if not q:
            return None

        ontology_path = Path(__file__).parent.parent / "data" / "function_ontology.json"
        try:
            with open(ontology_path) as f:
                ontology = json.load(f)
        except Exception:
            return None

        for domain, spec in ontology.items():
            if domain.startswith("_"):
                continue
            for bucket in ("established_peptides", "candidate_peptides"):
                for entry in spec.get(bucket, []):
                    name = entry.get("name", "").lower()
                    gene = entry.get("gene", "").lower()
                    if q in name or (gene and q == gene) or (len(q) > 3 and q in name.replace("-", "")):
                        derived = [
                            {"motif": m, "function": e["function"]}
                            for m, e in self.reference["bioactive_motifs"].items()
                            if q in e.get("parent_protein", "").lower()
                        ]
                        return {
                            "name": entry["name"],
                            "gene": entry.get("gene", ""),
                            "uniprot": entry.get("uniprot", ""),
                            "rationale": entry.get("rationale", ""),
                            "biological_process": spec.get("go_name", domain),
                            "go_term": spec.get("go_term", ""),
                            "derived_motifs": derived,
                        }
        return None

    # ---- level 2: bioactive motif ---------------------------------------

    def _match_motif(self, seq: str) -> Optional[FunctionInference]:
        hits = []
        for motif, entry in self.reference["bioactive_motifs"].items():
            idx = seq.find(motif)
            if idx >= 0:
                hits.append((motif, idx, entry))

        if not hits:
            return None

        # Longer motifs are more specific, so they win.
        hits.sort(key=lambda h: len(h[0]), reverse=True)
        motif, idx, entry = hits[0]

        caveats = []
        if entry.get("min_length_warning"):
            caveats.append(entry["min_length_warning"])
        if len(motif) <= 4:
            caveats.append(
                f"'{motif}' is only {len(motif)} residues. A match this short occurs by chance "
                f"in roughly one in {20 ** len(motif):,} random positions, so on its own it is "
                f"weak evidence of function."
            )

        alternatives = [
            {
                "motif": m,
                "position": i + 1,
                "parent_protein": e["parent_protein"],
                "function": e["function"],
            }
            for m, i, e in hits[1:]
        ]

        return FunctionInference(
            inferred_function=(
                f"{entry['function']} (via the {motif} motif of {entry['parent_protein']})"
            ),
            suggested_goal=entry["suggested_goal"],
            goal_reason=entry["goal_reason"],
            confidence=0.75 if len(motif) >= 5 else 0.40,
            basis=f"Contains the {motif} motif at position {idx + 1} ({entry['parent_protein']})",
            level=2,
            matched_name=motif,
            parent_protein=entry["parent_protein"],
            native_context_note=entry.get("native_context_note", ""),
            alternatives=alternatives,
            claim=Claim.retrieved(
                f"{motif} at position {idx + 1} is a characterised motif of "
                f"{entry['parent_protein']}: {entry['function']}",
                citations=[f"local reference cache motif '{motif}' (unverified — confirm against primary literature)"],
                tested=False,
            ),
            caveats=caveats,
        )

    # ---- level 3: family signature --------------------------------------

    def _match_family_signature(self, seq: str) -> Optional[FunctionInference]:
        sigs = self.reference["family_signatures"]
        net_charge = self.tier0.charge_calc.net_charge(seq, ph=7.4)
        moment = self.tier0.windowed_hydrophobic_moment(seq)["max_moment"]
        cys = seq.count("C")
        gly_pro = (seq.count("G") + seq.count("P")) / len(seq) if seq else 0.0

        amp = sigs["cationic_amphipathic_amp"]
        if (net_charge >= amp["min_net_charge"]
                and moment >= amp["min_hydrophobic_moment"]
                and amp["min_length"] <= len(seq) <= amp["max_length"]):
            return FunctionInference(
                inferred_function=(
                    "Possible cationic amphipathic / host-defence peptide — membrane-active "
                    "profile. This is a property match, not an identification."
                ),
                suggested_goal=amp["suggested_goal"],
                goal_reason=amp["goal_reason"],
                confidence=0.40,
                basis=(
                    f"Net charge {net_charge:+.1f} at pH 7.4 and maximum hydrophobic moment "
                    f"{moment:.2f} over an 11-residue window fit the cationic amphipathic profile"
                ),
                level=3,
                claim=Claim.computed(
                    f"Net charge {net_charge:+.1f}, muH {moment:.2f} — matches the cationic "
                    f"amphipathic signature",
                    method="Tier 0 charge and hydrophobic moment", tier=0,
                ),
                caveats=[
                    "Many peptides share this physicochemical profile without being antimicrobial. "
                    "This narrows the possibilities; it does not identify the peptide."
                ],
            )

        dis = sigs["disulfide_rich"]
        if cys >= dis["min_cysteines"] and cys % 2 == 0:
            return FunctionInference(
                inferred_function=(
                    f"Possible disulfide-rich scaffold ({cys} cysteines) — defensin, knottin, or "
                    f"growth-factor-like fold. This is a composition match, not an identification."
                ),
                suggested_goal=dis["suggested_goal"],
                goal_reason=dis["goal_reason"],
                confidence=0.40,
                basis=f"{cys} cysteines in {len(seq)} residues, consistent with a crosslinked fold",
                level=3,
                claim=Claim.computed(
                    f"{cys} cysteines, even count — consistent with {cys // 2} disulfide bonds",
                    method="Tier 0 composition", tier=0,
                ),
                caveats=[
                    "Disulfide connectivity cannot be determined from sequence. Which cysteines "
                    "pair with which changes the fold entirely, and that requires a structure."
                ],
            )

        col = sigs["collagen_like"]
        if gly_pro >= col["min_gly_pro_fraction"]:
            return FunctionInference(
                inferred_function=(
                    f"Gly/Pro-rich composition ({gly_pro:.0%}) — collagen-like or structurally "
                    f"extended. This is a composition match, not an identification."
                ),
                suggested_goal=col["suggested_goal"],
                goal_reason=col["goal_reason"],
                confidence=0.35,
                basis=f"Gly+Pro make up {gly_pro:.0%} of the sequence",
                level=3,
                claim=Claim.computed(
                    f"Gly+Pro fraction {gly_pro:.2f}",
                    method="Tier 0 composition", tier=0,
                ),
                caveats=["Composition alone does not establish a collagen-like fold."],
            )

        return None

    def _describe_profile(self, seq: str) -> str:
        """
        A one-line physicochemical description built from computed Tier 0 values.

        Used when nothing was identified, so the output still says something true
        and specific about the sequence rather than only reporting a miss.
        """
        net = self.tier0.charge_calc.net_charge(seq, ph=7.4)
        moment = self.tier0.windowed_hydrophobic_moment(seq)["max_moment"]
        pi = self.tier0.isoelectric_point(seq)
        cys = seq.count("C")

        charge_word = "cationic" if net > 1 else "anionic" if net < -1 else "near-neutral"
        parts = [
            f"{len(seq)} residues",
            f"{charge_word} ({net:+.1f} at pH 7.4, pI {pi})",
        ]
        if moment >= 0.35:
            parts.append(f"amphipathic (max muH {moment:.2f})")
        if cys >= 2:
            parts.append(f"{cys} cysteines")

        return " · ".join(parts) + "."

    # ---- level 4: liability-driven default ------------------------------

    def _liability_default(self, seq: str) -> FunctionInference:
        """
        The floor. Nothing was recognised, so the goal comes from the liabilities
        actually present in the sequence rather than from a guess about function.
        """
        liabilities = self.tier0.scan_liabilities(seq)
        high = [l for l in liabilities if l.severity == "high"]

        # Lead with what was measured. "Not recognised" alone is useless to a
        # reader: the Tier 0 properties below are real numbers about their
        # sequence and are worth more than the absence of a database hit.
        profile = self._describe_profile(seq)

        if high:
            named = ", ".join(f"{l.motif} at {l.display_position}" for l in high[:3])
            return FunctionInference(
                inferred_function=(
                    f"{profile} Not matched to a named protein, so no functional assignment "
                    f"is claimed — but the properties above are computed from your sequence."
                ),
                suggested_goal="protease_resistance",
                goal_reason=(
                    f"Defaulting to protease resistance because the sequence carries "
                    f"{len(high)} high-severity liability motif(s) ({named}). These are concrete, "
                    f"sequence-derived findings that can be acted on without knowing the function."
                ),
                confidence=0.20,
                basis=f"{len(liabilities)} liability motif(s) found, {len(high)} high severity",
                level=4,
                claim=Claim.computed(
                    f"{len(high)} high-severity liabilities: {named}",
                    method="Tier 0 liability motif scan", tier=0,
                ),
                caveats=[
                    "No function was identified. The suggested goal reflects what is wrong with "
                    "the sequence, not what it is for. If you know the intended function, "
                    "override the selection.",
                ],
            )

        return FunctionInference(
            inferred_function=(
                f"{profile} Not matched to a named protein, and no high-severity liability "
                f"motifs were found — so there is no obvious first move from the sequence alone."
            ),
            suggested_goal="protease_resistance",
            goal_reason=(
                "Defaulting to protease resistance as the highest-confidence lane: it is scored "
                "by matching residues against documented protease specificities, which is a real "
                "motif match rather than an estimate. The other lanes need a receptor structure "
                "this build does not have."
            ),
            confidence=0.10,
            basis="No sequence, motif, or family match; no high-severity liabilities",
            level=4,
            claim=Claim.inferred(
                "Goal defaulted to protease resistance",
                inference_step=(
                    "Nothing about the sequence was recognised, so the goal was chosen for having "
                    "the strongest available evidence path rather than for fitting this peptide."
                ),
            ),
            caveats=[
                "This is a default, not an inference about this peptide. Override it if you know "
                "the intended function.",
            ],
        )

    # ---- entry point -----------------------------------------------------

    def _match_uniprot(self, seq: str, raw_input: str) -> Optional[FunctionInference]:
        """Level 0: ask UniProt what this actually is."""
        result = self.uniprot.identify(seq, raw_input=raw_input)

        if not result.found:
            if not result.reachable:
                # A network failure must not be reported as an unrecognised
                # peptide — that blames the input for an infrastructure problem.
                self._uniprot_note = (
                    f"UniProt could not be reached ({result.status}), so identification fell "
                    f"back to the local reference set, which covers only a handful of peptides. "
                    f"If this sequence is a known protein, that is why it was not recognised."
                )
            else:
                self._uniprot_note = ""
            return None

        rec = result.record
        goal, reason = self._goal_for_record(rec, seq)

        function = rec.function or rec.protein_name or "No functional annotation in the UniProt entry"
        if rec.protein_name and rec.function:
            function = f"{rec.protein_name} — {rec.function}"

        caveats = []
        if rec.length and len(seq) < rec.length * 0.9:
            caveats.append(
                f"The pasted sequence is {len(seq)} residues but {rec.accession} is "
                f"{rec.length}. This is a fragment of the full protein, so its termini are "
                f"excision artefacts and the annotated function describes the whole protein, "
                f"not necessarily this piece of it."
            )

        return FunctionInference(
            inferred_function=function,
            suggested_goal=goal,
            goal_reason=reason,
            confidence=0.95 if result.route == "accession" else 0.85,
            basis=(
                f"UniProt {rec.accession} ({rec.entry_name or rec.protein_name})"
                f" via {result.route} lookup"
                + (f", {rec.organism}" if rec.organism else "")
            ),
            level=0,
            matched_name=rec.protein_name or rec.accession,
            parent_protein=rec.protein_name,
            native_context_note=(
                f"Subcellular location: {rec.subcellular_location}."
                if rec.subcellular_location else ""
            ),
            claim=Claim.retrieved(
                f"{rec.protein_name or rec.accession}: {rec.function or 'annotated in UniProt'}",
                citations=[rec.citation],
                tested=True,
            ),
            caveats=caveats,
            uniprot=rec,
        )

    @staticmethod
    def _goal_for_record(rec, seq: str) -> Tuple[str, str]:
        """Choose a goal from what UniProt says about the protein."""
        blob = " ".join([rec.protein_name, rec.function, " ".join(rec.keywords)]).lower()

        if any(k in blob for k in ("extracellular matrix", "basement membrane", "cell adhesion",
                                   "laminin", "collagen", "integrin")):
            return "binding_affinity", (
                "UniProt annotates this as an extracellular matrix or adhesion protein. For this "
                "class, activity depends on receptor engagement and on how the motif is presented "
                "— density, spacing and valency — rather than on circulating half-life, so binding "
                "is the productive axis."
            )
        if any(k in blob for k in ("antimicrobial", "antibiotic", "defensin", "host defense")):
            return "protease_resistance", (
                "UniProt annotates antimicrobial activity. These peptides are typically "
                "protease-labile and lose activity on cleavage, so stability is the limiting "
                "property."
            )
        if any(k in blob for k in ("hormone", "receptor agonist", "secreted", "signaling")):
            return "protease_resistance", (
                "UniProt annotates this as a secreted signalling molecule. Circulating peptide "
                "hormones are typically exposure-limited, so proteolytic stability is usually "
                "the first thing worth fixing."
            )
        return "protease_resistance", (
            "No annotation clearly indicated a goal, so this defaults to the highest-confidence "
            "lane: protease liability is scored from documented specificities rather than "
            "estimated."
        )

    def infer(self, sequence: str, name: str = "", raw_input: str = "") -> FunctionInference:
        """
        Infer function from sequence, falling through the evidence levels.

        Always returns a FunctionInference with a usable suggested_goal.
        """
        seq = sequence.upper().strip()
        if not seq:
            raise ValueError("Cannot infer function from an empty sequence")

        self._uniprot_note = ""

        uniprot_hit = self._match_uniprot(seq, raw_input or name)
        if uniprot_hit is not None:
            logger.info(f"Function inference level 0: {uniprot_hit.basis}")
            return uniprot_hit

        for level_fn in (self._match_known_peptide, self._match_motif, self._match_family_signature):
            result = level_fn(seq)
            if result is not None:
                if self._uniprot_note:
                    result.caveats.append(self._uniprot_note)
                logger.info(f"Function inference level {result.level}: {result.basis}")
                return result

        result = self._liability_default(seq)
        if self._uniprot_note:
            result.caveats.insert(0, self._uniprot_note)
        logger.info(f"Function inference level 4 (default): {result.basis}")
        return result
