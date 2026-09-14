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

    def _match_known_peptide(self, seq: str) -> Optional[FunctionInference]:
        best, best_id = None, 0.0

        for name, entry in self.reference["exact_peptides"].items():
            ref = entry["sequence"].upper()
            if seq == ref:
                identity = 1.0
            elif abs(len(seq) - len(ref)) > max(len(ref) * 0.5, 8):
                continue
            else:
                identity = self._identity(seq, ref)

            if identity > best_id:
                best, best_id = (name, entry), identity

        if best is None or best_id < 0.80:
            return None

        name, entry = best
        exact = best_id >= 0.999

        return FunctionInference(
            inferred_function=entry["function"],
            suggested_goal=entry["suggested_goal"],
            goal_reason=entry["goal_reason"],
            confidence=0.95 if exact else 0.70 + (best_id - 0.80) * 1.2,
            basis=(
                f"Exact sequence match to {name}" if exact
                else f"{best_id:.0%} identity to {name}"
            ),
            level=1,
            matched_name=name,
            claim=Claim.retrieved(
                f"Sequence {'matches' if exact else 'closely resembles'} {name}: {entry['function']}",
                citations=[f"local reference cache entry '{name}' (unverified — confirm against UniProt)"],
                tested=False,
            ),
            caveats=(
                [] if exact else
                [f"Not an exact match ({best_id:.0%} identity to {name}). The differences may "
                 f"be exactly the positions that matter, so treat the functional assignment as "
                 f"provisional."]
            ),
        )

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
