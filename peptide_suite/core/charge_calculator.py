"""
Henderson-Hasselbalch charge state calculations per residue.
Computes pKa-dependent ionization at specified pH.

Equation #11 reference: pH = pKa + log([A-]/[HA])
Rearranged: alpha (fraction charged) = 1 / (1 + 10^(pKa - pH))
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ChargeState:
    """Computed charge state for a residue."""
    residue: str  # 1-letter code
    position: int
    ph: float
    effective_charge: float  # -1, 0, or +1 mostly, can be fractional
    is_ionizable: bool
    pka: float
    confidence: str  # "high" if pKa well-characterized, "low" if context-dependent


class ChargeCalculator:
    """
    Computes residue ionization state using Henderson-Hasselbalch.
    Uses literature pKa values for free amino acids as baseline.
    Adjusts based on local context (buried vs surface).
    """

    # pKa values are measured literature quantities, so they live in reference
    # data with their sources and their spread across those sources, not in this
    # module. The burial shift applied on top of them is a modelling choice and
    # comes from the policy.
    _REFERENCE = None

    @classmethod
    def reference(cls) -> Dict:
        if cls._REFERENCE is None:
            path = Path(__file__).parent.parent / "data" / "pka_reference.json"
            cls._REFERENCE = json.loads(path.read_text())
        return cls._REFERENCE

    @classmethod
    def sidechain_pka(cls) -> Dict[str, float]:
        return {aa: entry["pka"] for aa, entry in cls.reference()["sidechain"].items()}

    @classmethod
    def pka_entry(cls, aa: str) -> Optional[Dict]:
        return cls.reference()["sidechain"].get(aa)

    @classmethod
    def terminal_pka(cls, which: str) -> float:
        return cls.reference()["terminal"][which]["pka"]

    @staticmethod
    def burial_shift() -> float:
        from ..runtime import threshold
        return threshold("chemistry.burial_pka_shift")

    def __init__(self):
        pass

    def charge_at_ph(
        self,
        residue: str,
        ph: float = 7.4,
        position: int = 0,
        is_buried: bool = False,
        sequence_length: int = 20,
    ) -> ChargeState:
        """
        Calculate effective charge of a residue at given pH.

        Args:
            residue: 1-letter amino acid code
            ph: pH to calculate at (default 7.4 physiological)
            position: position in sequence (0-indexed)
            is_buried: True if residue is likely buried (no pKa pdb yet, heuristic)
            sequence_length: for rough surface/burial heuristic

        Returns:
            ChargeState object
        """
        residue = residue.upper()

        # Determine pKa
        if residue not in self.sidechain_pka():
            # Non-ionizable residue
            return ChargeState(
                residue=residue,
                position=position,
                ph=ph,
                effective_charge=0.0,
                is_ionizable=False,
                pka=None,
                confidence="high",
            )

        pka = self.sidechain_pka()[residue]

        entry = self.pka_entry(residue)

        if is_buried:
            pka += self.burial_shift()

        # Confidence tracks the pKa's own spread, not just burial. A histidine
        # on the surface is still the least determined charge in the peptide,
        # because its cited pKa spans the range that decides whether it is
        # charged at physiological pH at all.
        confidence = "low" if (is_buried or entry.get("context_dependent")) else "high"

        # Henderson-Hasselbalch: alpha = 1 / (1 + 10^(pKa - pH))
        # This gives fraction ionized (proton-free state)
        exponent = pka - ph
        alpha = 1.0 / (1.0 + 10.0 ** exponent)

        # Determine charge based on residue type
        # For acidic residues (D, E): charge is -alpha (deprotonated = negative)
        # For basic residues (H, K, R, Y): charge is +alpha (deprotonated = positive)
        # For C: charge is -alpha if deprotonated
        if residue in ["D", "E", "C"]:
            # Acidic: when protonated (low pH), neutral; when deprotonated (high pH), negative
            # So effective charge = -(alpha)
            effective_charge = -alpha
        elif residue in ["H", "K", "R"]:
            # Basic: when protonated (low pH), positive; when deprotonated (high pH), neutral
            # So effective charge = +(1 - alpha) = +(protonated fraction)
            effective_charge = (1.0 - alpha)
        elif residue == "Y":
            # Tyrosine is acidic like D/E/C. It was previously zeroed below
            # alpha = 0.2, which put a step discontinuity in the titration curve
            # -- the charge jumped from 0 to -0.2 across a hundredth of a pH
            # unit. Its ionization at physiological pH is small, and small is
            # what the equation already says.
            effective_charge = -alpha
        else:
            effective_charge = 0.0

        return ChargeState(
            residue=residue,
            position=position,
            ph=ph,
            effective_charge=effective_charge,
            is_ionizable=True,
            pka=pka,
            confidence=confidence,
        )

    def sequence_charge_profile(
        self,
        sequence: str,
        ph: float = 7.4,
        surface_burial_estimate: Dict[int, bool] = None,
    ) -> Dict[int, ChargeState]:
        """
        Compute charge state for every residue in a sequence.

        Args:
            sequence: Amino acid sequence (1-letter codes)
            ph: pH
            surface_burial_estimate: Dict mapping position -> is_buried (optional)
                                     if None, assumes all surface

        Returns:
            Dict mapping position -> ChargeState
        """
        if surface_burial_estimate is None:
            surface_burial_estimate = {}

        charges = {}
        for i, aa in enumerate(sequence):
            is_buried = surface_burial_estimate.get(i, False)
            charges[i] = self.charge_at_ph(
                aa,
                ph=ph,
                position=i,
                is_buried=is_buried,
                sequence_length=len(sequence),
            )

        return charges

    def terminal_charges(self, sequence: str, ph: float = 7.4) -> Dict[str, float]:
        """
        Charge contributed by the free N-terminal amino and C-terminal carboxyl.

        These are not side chains and are easy to overlook, but on a short
        peptide they are frequently the dominant charges: a 5-mer of neutral
        residues still carries +1 and -1 at physiological pH. Omitting them also
        makes the isoelectric point undefined for any peptide with no ionizable
        side chains, which is not a real property of such a peptide.
        """
        if not sequence:
            return {"n_terminus": 0.0, "c_terminus": 0.0}

        # N-terminal amino group: protonated (+1) below its pKa
        n_term = 1.0 / (1.0 + 10.0 ** (ph - self.terminal_pka('n_terminus')))
        # C-terminal carboxyl: deprotonated (-1) above its pKa
        c_term = -1.0 / (1.0 + 10.0 ** (self.terminal_pka('c_terminus') - ph))

        return {"n_terminus": n_term, "c_terminus": c_term}

    def net_charge(
        self,
        sequence: str,
        ph: float = 7.4,
        surface_burial_estimate: Dict[int, bool] = None,
        include_termini: bool = True,
    ) -> float:
        """
        Calculate net charge of the entire peptide.

        Includes the free termini by default. Pass include_termini=False only
        when modelling a peptide that is capped at both ends.
        """
        charges = self.sequence_charge_profile(sequence, ph, surface_burial_estimate)
        total = sum(cs.effective_charge for cs in charges.values())

        if include_termini:
            term = self.terminal_charges(sequence, ph)
            total += term["n_terminus"] + term["c_terminus"]

        return total

    def charge_distribution_summary(
        self, charges: Dict[int, ChargeState]
    ) -> Dict:
        """Summarize charge distribution (e.g., count of + positions, - positions)."""
        positive = sum(1 for cs in charges.values() if cs.effective_charge > 0.2)
        negative = sum(1 for cs in charges.values() if cs.effective_charge < -0.2)
        neutral = len(charges) - positive - negative

        return {
            "positive_residues": positive,
            "negative_residues": negative,
            "neutral_residues": neutral,
            "net_charge": sum(cs.effective_charge for cs in charges.values()),
        }
