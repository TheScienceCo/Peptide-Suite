"""
Sequence-based function inference.

The previous implementation matched a hardcoded list of gene *names*. A pasted
sequence has no name, so inference could never succeed for the primary input
mode. This module infers from the sequence itself, through four descending
levels of evidence, and always returns a usable default:

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
    level: int                  # 1-4
    claim: Claim = None
    matched_name: str = ""
    parent_protein: str = ""
    native_context_note: str = ""
    alternatives: List[Dict] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)

    @property
    def is_identification(self) -> bool:
        """True only when the sequence was actually recognised, not merely profiled."""
        return self.level <= 2


class FunctionInferencer:
    """Infers likely function from sequence, never returning nothing."""

    def __init__(self):
        self.reference = self._load_reference()
        self.tier0 = Tier0Sequence()

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

    # ---- level 4: liability-driven default ------------------------------

    def _liability_default(self, seq: str) -> FunctionInference:
        """
        The floor. Nothing was recognised, so the goal comes from the liabilities
        actually present in the sequence rather than from a guess about function.
        """
        liabilities = self.tier0.scan_liabilities(seq)
        high = [l for l in liabilities if l.severity == "high"]

        if high:
            named = ", ".join(f"{l.motif} at {l.display_position}" for l in high[:3])
            return FunctionInference(
                inferred_function=(
                    "Not recognised. No functional assignment is claimed — the sequence does not "
                    "match a known peptide, a characterised motif, or a family profile."
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
                "Not recognised. No functional assignment is claimed, and no high-severity "
                "liability motifs were found either."
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

    def infer(self, sequence: str, name: str = "") -> FunctionInference:
        """
        Infer function from sequence, falling through the evidence levels.

        Always returns a FunctionInference with a usable suggested_goal.
        """
        seq = sequence.upper().strip()
        if not seq:
            raise ValueError("Cannot infer function from an empty sequence")

        for level_fn in (self._match_known_peptide, self._match_motif, self._match_family_signature):
            result = level_fn(seq)
            if result is not None:
                logger.info(f"Function inference level {result.level}: {result.basis}")
                return result

        result = self._liability_default(seq)
        logger.info(f"Function inference level 4 (default): {result.basis}")
        return result
