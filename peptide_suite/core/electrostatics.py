"""
Multi-pH protonation.  [Addendum 2 section 6b]

The isoelectric point is one number summarising a whole titration curve, and it
answers a question nobody designing a peptide actually has. pI says where net
charge crosses zero; it does not say which residue carries the charge, and it
does not say whether that changes between the compartments the peptide passes
through.

So this module reports per-residue protonation at each pH the policy names --
physiological, interstitial, endosomal -- and identifies the residues whose
state actually moves across that window. A residue whose protonation is
constant across the range is not a design handle no matter how charged it is;
a residue that flips is, and there are usually very few of them.

Everything here is computed from the sequence and a literature pKa table. No
structure is required and none is assumed, so nothing in this module is gated.
What a pKa table cannot know -- the shift a real environment imposes on a
specific residue -- is reported as the spread on the input rather than folded
silently into the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .charge_calculator import ChargeCalculator

# The compartments the policy names, in the order a systemically dosed peptide
# meets them. Labels are the engine's; the pH values are the policy's.
COMPARTMENTS = (
    ("physiological", "chemistry.reference_ph"),
    ("interstitial", "chemistry.interstitial_ph"),
    ("endosomal", "chemistry.endosomal_ph"),
)


@dataclass
class ResiduePoint:
    """One residue's state at one pH."""
    ph: float
    compartment: str
    protonated_fraction: float
    effective_charge: float


@dataclass
class ResidueTitration:
    """How one ionizable residue behaves across the whole pH window."""
    position: int          # zero-based
    display_position: int  # one-based, what a chemist writes
    residue: str
    pka: float
    group: str
    cited_range: List[float]
    context_dependent: bool
    points: List[ResiduePoint] = field(default_factory=list)
    note: str = ""

    @property
    def protonation_swing(self) -> float:
        """Largest change in protonated fraction across the window."""
        fractions = [p.protonated_fraction for p in self.points]
        return max(fractions) - min(fractions) if fractions else 0.0

    @property
    def charge_swing(self) -> float:
        charges = [p.effective_charge for p in self.points]
        return max(charges) - min(charges) if charges else 0.0

    def source_disagreement(self) -> float:
        """
        How much the cited spread on this pKa actually moves the answer.

        The largest change in protonated fraction, at any pH examined, between
        taking the low end of the cited range and taking the high end.
        """
        lo, hi = self.cited_range
        if not self.points:
            return 0.0
        return max(
            abs(_protonated_fraction(hi, point.ph) - _protonated_fraction(lo, point.ph))
            for point in self.points
        )

    def uncertainty_note(self, threshold: float) -> str:
        """
        What the cited spread does to this residue's numbers, when it does
        anything.

        Computed rather than guessed from proximity. An earlier version flagged
        any pKa within a unit of the window, which fired on every glutamate in
        the sequence: their cited range is 4.1-4.5, and at pH 5.5 and above
        both ends give a protonated fraction near zero, so the disagreement
        changes nothing. Three identical warnings that change nothing are how a
        reader learns to skip warnings.
        """
        disagreement = self.source_disagreement()
        if disagreement < threshold:
            return ""
        lo, hi = self.cited_range
        return (
            f"Sources disagree on this pKa ({lo}-{hi}). Across the pH range examined that "
            f"disagreement moves the protonated fraction by up to "
            f"{disagreement * 100:.0f} points, so these figures follow the single value "
            f"{self.pka} and another source's value would change them materially."
        )


@dataclass
class ElectrostaticProfile:
    """Per-residue protonation across the compartment series, plus net charge."""
    sequence: str
    compartments: Dict[str, float]                 # label -> pH
    net_charge: Dict[str, float]                   # label -> net charge
    titrations: List[ResidueTitration]
    terminal_charges: Dict[str, Dict[str, float]]  # label -> {n_terminus, c_terminus}
    switchable: List[ResidueTitration]
    switch_threshold: float
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.titrations:
            return ("No ionizable side chains. The peptide's charge across this pH range "
                    "comes entirely from its termini.")
        if not self.switchable:
            return (f"{len(self.titrations)} ionizable side chain(s), none of which change "
                    f"protonation state materially between pH "
                    f"{max(self.compartments.values())} and {min(self.compartments.values())}. "
                    f"Charge is effectively constant across these compartments.")
        names = ", ".join(f"{t.residue}{t.display_position}" for t in self.switchable)
        return (f"{len(self.switchable)} of {len(self.titrations)} ionizable side chain(s) "
                f"change protonation materially across this pH range: {names}. These are the "
                f"positions where compartment matters.")


def _protonated_fraction(pka: float, ph: float) -> float:
    """
    Fraction of the group holding its proton, from Henderson-Hasselbalch.

    Reported rather than charge because it is the same quantity for an acid and
    a base -- charge then follows from which of the two it is. Keeping them
    separate is what stops an acid and a base with the same pKa looking like
    different physics.
    """
    return 1.0 / (1.0 + 10.0 ** (ph - pka))


def compute_profile(
    sequence: str,
    *,
    switch_threshold: Optional[float] = None,
    include_termini: bool = True,
    buried_positions: Optional[Dict[int, bool]] = None,
) -> ElectrostaticProfile:
    """
    Titrate a sequence across the policy's compartment series.

    `switch_threshold` is the change in protonated fraction that counts as
    material. It defaults to the policy's material-contribution cutoff, because
    "material" is one idea and should not have two numbers.
    """
    from ..runtime import threshold

    sequence = sequence.upper()
    buried_positions = buried_positions or {}
    if switch_threshold is None:
        switch_threshold = threshold("confidence.material_contribution_cutoff")

    calculator = ChargeCalculator()
    compartments = {label: threshold(key) for label, key in COMPARTMENTS}

    notes: List[str] = []
    seen = set(compartments.values())
    if len(seen) < len(compartments):
        notes.append(
            "Two or more compartments in the active policy are set to the same pH, so the "
            "comparison across them cannot show a difference."
        )

    titrations: List[ResidueTitration] = []
    for index, residue in enumerate(sequence):
        entry = calculator.pka_entry(residue)
        if entry is None:
            continue

        pka = entry["pka"]
        if buried_positions.get(index):
            pka += calculator.burial_shift()

        titration = ResidueTitration(
            position=index,
            display_position=index + 1,
            residue=residue,
            pka=pka,
            group=entry["group"],
            cited_range=list(entry["cited_range"]),
            context_dependent=entry.get("context_dependent", False),
            note=entry.get("note", ""),
        )
        for label, ph in compartments.items():
            state = calculator.charge_at_ph(
                residue, ph=ph, position=index,
                is_buried=buried_positions.get(index, False),
                sequence_length=len(sequence),
            )
            titration.points.append(ResiduePoint(
                ph=ph,
                compartment=label,
                protonated_fraction=_protonated_fraction(pka, ph),
                effective_charge=state.effective_charge,
            ))
        titrations.append(titration)

    net_charge = {
        label: calculator.net_charge(
            sequence, ph=ph,
            surface_burial_estimate=buried_positions,
            include_termini=include_termini,
        )
        for label, ph in compartments.items()
    }
    terminal_charges = {
        label: calculator.terminal_charges(sequence, ph=ph)
        for label, ph in compartments.items()
    } if include_termini else {}

    switchable = [t for t in titrations if t.protonation_swing >= switch_threshold]
    switchable.sort(key=lambda t: t.protonation_swing, reverse=True)

    if include_termini:
        notes.append(
            "Net charge includes the free termini. Pass include_termini=False for a "
            "peptide capped at both ends; on a short sequence the termini are often the "
            "largest charges present."
        )
    if not buried_positions:
        notes.append(
            "Every residue is treated as solvent-exposed. No structure was supplied, so no "
            "burial shift was applied to any pKa. A buried ionizable group can differ from "
            "the value used here by more than the whole pH range examined."
        )

    return ElectrostaticProfile(
        sequence=sequence,
        compartments=compartments,
        net_charge=net_charge,
        titrations=titrations,
        terminal_charges=terminal_charges,
        switchable=switchable,
        switch_threshold=switch_threshold,
        notes=notes,
    )
