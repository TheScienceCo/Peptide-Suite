"""
Synthetic feasibility gate.  [Addendum 2 section 10, build step 6h]

A system reasoning purely from biophysics misses this class of constraint
entirely, and the worked example is the clearest statement of why.

Semaglutide's Lys34 to Arg is not a biophysical optimization. Native GLP-1
carries lysines at 26 and 34, and with both present you cannot regioselectively
acylate: the coupling chemistry does not distinguish two primary amines in the
same molecule. Arg34 exists to make Lys26 the only nucleophile. No amount of
quantum featurization generates that substitution, and a biophysics-only scorer
will mis-rank it if proposed -- it looks like a conservative charge-preserving
swap with no obvious benefit, because its benefit is not in the physics of the
peptide at all. It is in the chemistry of making it.

Everything here is computed from the sequence. There is no gating and no
UNRESOLVED, because none of these checks needs a structure: aspartimide risk is
a dipeptide motif, oxidation liability is a residue count, and regioselectivity
is an arithmetic fact about how many nucleophiles of a given kind are present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Severity(Enum):
    BLOCKING = "blocking"      # the chemistry does not work as specified
    HIGH = "high"
    MODERATE = "moderate"
    NOTE = "note"


@dataclass
class FeasibilityFlag:
    """
    One synthesis or stability liability.

    `remedy` is required in spirit: a flag that names a problem without naming
    what is usually done about it sends the reader to the literature for
    something the field settled decades ago.
    """
    code: str
    severity: Severity
    positions: List[int]
    description: str
    remedy: str = ""
    basis: str = ""

    def summary(self) -> str:
        where = f" at {', '.join(str(p) for p in self.positions)}" if self.positions else ""
        return f"[{self.severity.value}] {self.code}{where}: {self.description}"


# Residue-level oxidation liabilities. Counted rather than scored: how much a
# given Met matters depends on formulation and storage, which this module has
# no access to, so it reports presence and location.
_OXIDATION_PRONE = {"M": "methionine (sulfoxide)",
                    "W": "tryptophan (oxindole, kynurenine)",
                    "C": "cysteine (disulfide scrambling, sulfonic acid)"}

# Sequence motifs. Each is a real, named degradation route with a standard
# mitigation, which is why they are worth flagging at all.
_MOTIFS = [
    ("aspartimide", re.compile(r"DG"), Severity.HIGH,
     "Asp-Gly is the classic aspartimide sequence. Under the repeated base treatment "
     "of Fmoc SPPS the aspartyl side chain cyclises onto the backbone nitrogen, and "
     "the ring then opens to a mixture including the beta-peptide and the D-epimer.",
     "Backbone amide protection on the glycine (Hmb or Dmb), or a bulky Asp ester."),
    ("deamidation", re.compile(r"NG"), Severity.MODERATE,
     "Asn-Gly deamidates through the same succinimide chemistry, converting Asn to a "
     "mixture of Asp and isoAsp. This is a storage liability rather than a synthesis "
     "one, and it changes the molecule's charge.",
     "Substitute the Asn or the following Gly if activity permits, or control "
     "formulation pH."),
    ("dg_sequence_isoaspartate", re.compile(r"DS|DN"), Severity.NOTE,
     "Asp followed by a small residue is a slower aspartimide route than Asp-Gly, "
     "worth noting when the position is otherwise flexible.",
     "Usually tolerable; watch it if the peptide is stored in solution."),
]


def _pyroglutamate_risk(sequence: str) -> Optional[FeasibilityFlag]:
    """
    N-terminal Gln cyclises to pyroglutamate, capping the peptide.

    Flagged as a liability here rather than as the useful modification it is
    elsewhere: in the native-contact module a deliberate pGlu is ESSENTIAL and
    PROTECTIVE, while here it is a spontaneous side reaction on a peptide not
    designed for it. Same chemistry, opposite intent, and the difference is
    whether it was chosen.
    """
    if not sequence.startswith("Q"):
        return None
    return FeasibilityFlag(
        code="n_terminal_gln_cyclization", severity=Severity.MODERATE, positions=[1],
        description=("An N-terminal glutamine cyclises spontaneously to pyroglutamate, "
                     "capping the alpha-amino group. The product is a different molecule "
                     "with a blocked N-terminus and it forms without being asked."),
        remedy=("Use pyroglutamate deliberately if the cap is wanted, or change the "
                "N-terminal residue if a free amine is required."),
        basis="sequence: N-terminal Q")


def _cysteine_flags(sequence: str) -> List[FeasibilityFlag]:
    """
    Free thiols and disulfide pairing ambiguity.

    The count matters more than the positions: one cysteine cannot pair, two
    have one possible pairing, and four have three. Ambiguity is a combinatorial
    fact, and a peptide with six cysteines has fifteen possible pairings that
    the synthesis must be designed to choose between.
    """
    positions = [i + 1 for i, aa in enumerate(sequence) if aa == "C"]
    if not positions:
        return []
    count = len(positions)

    if count == 1:
        return [FeasibilityFlag(
            code="unpaired_cysteine", severity=Severity.MODERATE, positions=positions,
            description=("A single cysteine cannot form an intramolecular disulfide, so it "
                         "stays a free thiol: prone to oxidation and to intermolecular "
                         "dimerisation."),
            remedy="Cap it, pair it with an introduced partner, or substitute it.",
            basis="sequence: cysteine count")]

    if count % 2 == 1:
        return [FeasibilityFlag(
            code="odd_cysteine_count", severity=Severity.HIGH, positions=positions,
            description=(f"{count} cysteines: an odd number, so at least one free thiol "
                         f"remains however they pair."),
            remedy="Decide which is free and protect it explicitly.",
            basis="sequence: cysteine count")]

    # Double factorial: (n-1)!! distinct perfect matchings on n cysteines.
    pairings = 1
    for k in range(count - 1, 0, -2):
        pairings *= k
    if count == 2:
        return []
    return [FeasibilityFlag(
        code="disulfide_pairing_ambiguity", severity=Severity.HIGH, positions=positions,
        description=(f"{count} cysteines admit {pairings} distinct disulfide pairings. "
                     f"Air oxidation gives a mixture, and the intended isomer is one of "
                     f"them rather than the product."),
        remedy=("Orthogonal protecting groups (Trt/Acm/Mob) to direct the pairings, with "
                "the order of deprotection specified."),
        basis="sequence: cysteine count")]


def _aggregation_prone_stretches(sequence: str) -> List[FeasibilityFlag]:
    """
    Stretches that aggregate on-resin during SPPS.

    Beta-branched and bulky hydrophobic runs cause the growing chain to
    self-associate, and coupling yields collapse. This is a synthesis failure,
    not a formulation one: the peptide never gets made.
    """
    flags = []
    prone = set("VIYFWLT")
    run_start = None
    for i, aa in enumerate(sequence):
        if aa in prone:
            run_start = i if run_start is None else run_start
        else:
            if run_start is not None and i - run_start >= 5:
                flags.append(FeasibilityFlag(
                    code="spps_aggregation_stretch", severity=Severity.MODERATE,
                    positions=list(range(run_start + 1, i + 1)),
                    description=(f"A {i - run_start}-residue run of beta-branched and bulky "
                                 f"hydrophobic residues. On-resin aggregation here collapses "
                                 f"coupling yields."),
                    remedy=("Pseudoproline dipeptides, backbone amide protection, or a "
                            "different resin and solvent system."),
                    basis="sequence: hydrophobic run length"))
            run_start = None
    if run_start is not None and len(sequence) - run_start >= 5:
        flags.append(FeasibilityFlag(
            code="spps_aggregation_stretch", severity=Severity.MODERATE,
            positions=list(range(run_start + 1, len(sequence) + 1)),
            description=(f"A {len(sequence) - run_start}-residue hydrophobic run at the "
                         f"C-terminal end."),
            remedy="Pseudoproline dipeptides or backbone amide protection.",
            basis="sequence: hydrophobic run length"))
    return flags


def regioselectivity_conflict(sequence: str, chemistry: str) -> Optional[FeasibilityFlag]:
    """
    Whether a conjugation chemistry has more than one target in this sequence.

    This is the semaglutide check. Amine-directed acylation cannot distinguish
    two lysines, so a peptide with two of them cannot be regioselectively
    acylated at one -- and the fix is to remove the other lysine, which is a
    synthesis-driven substitution no biophysical argument would produce.
    """
    targets = {
        "amine": ("K", "lysine epsilon-amine"),
        "acylation": ("K", "lysine epsilon-amine"),
        "lipidation": ("K", "lysine epsilon-amine"),
        "thiol": ("C", "cysteine thiol"),
        "maleimide": ("C", "cysteine thiol"),
        "carboxyl": ("E", "glutamate side-chain carboxyl"),
    }
    entry = targets.get((chemistry or "").lower())
    if entry is None:
        return None
    residue, label = entry
    positions = [i + 1 for i, aa in enumerate(sequence) if aa == residue]
    if len(positions) < 2:
        return None

    remedy = (f"Remove the competing {residue} by substitution, leaving one nucleophile. "
              f"This is what semaglutide's Lys34 to Arg does: native GLP-1 carries lysines "
              f"at 26 and 34, and Arg34 exists so that Lys26 is the only site the acylation "
              f"chemistry can reach.")
    return FeasibilityFlag(
        code="regioselectivity_conflict", severity=Severity.BLOCKING, positions=positions,
        description=(f"{len(positions)} {label} groups are present, and {chemistry} "
                     f"chemistry cannot distinguish between them. The reaction gives a "
                     f"mixture of positional isomers, not the intended conjugate."),
        remedy=remedy,
        basis=f"sequence: count of {residue}")


def screen(sequence: str, conjugation_chemistry: str = "") -> List[FeasibilityFlag]:
    """
    Every feasibility flag for a sequence, optionally for a conjugation chemistry.

    Ordered by severity so a blocking conflict is not buried under three notes.
    """
    sequence = (sequence or "").upper()
    flags: List[FeasibilityFlag] = []

    for code, pattern, severity, description, remedy in _MOTIFS:
        positions = [m.start() + 1 for m in pattern.finditer(sequence)]
        if positions:
            flags.append(FeasibilityFlag(code=code, severity=severity, positions=positions,
                                         description=description, remedy=remedy,
                                         basis=f"sequence motif: {pattern.pattern}"))

    for residue, label in _OXIDATION_PRONE.items():
        if residue == "C":
            continue  # covered in more detail by the cysteine flags
        positions = [i + 1 for i, aa in enumerate(sequence) if aa == residue]
        if positions:
            flags.append(FeasibilityFlag(
                code="oxidation_liability", severity=Severity.NOTE, positions=positions,
                description=f"{len(positions)} {label} residue(s): an oxidation liability "
                            f"in storage and, for Trp, during cleavage.",
                remedy="Scavengers in the cleavage cocktail; inert headspace on storage.",
                basis=f"sequence: count of {residue}"))

    flags.extend(_cysteine_flags(sequence))
    flags.extend(_aggregation_prone_stretches(sequence))

    pyro = _pyroglutamate_risk(sequence)
    if pyro:
        flags.append(pyro)

    if conjugation_chemistry:
        conflict = regioselectivity_conflict(sequence, conjugation_chemistry)
        if conflict:
            flags.append(conflict)

    order = {Severity.BLOCKING: 0, Severity.HIGH: 1, Severity.MODERATE: 2, Severity.NOTE: 3}
    return sorted(flags, key=lambda f: (order[f.severity], f.code))


@dataclass
class ImmunogenicityScreen:
    """
    MHC-II binding over the windows a proposal alters.

    Not implemented, and the stub raises rather than returning a score. A
    plausible-looking immunogenicity number with no allele set and no predictor
    behind it is precisely the fabricated value the output contract forbids, and
    an unimplemented screen returning "low risk" is worse than no screen.
    """
    altered_windows: List[str] = field(default_factory=list)

    def run(self):
        raise NotImplementedError(
            "MHC-II binding prediction is not wired up in this build. It needs a predictor "
            "and a declared allele set, and both are part of the result: a binding score "
            "without the alleles it was computed over is not interpretable. Until then, "
            "immunogenicity is reported as UNKNOWN rather than estimated."
        )

    @staticmethod
    def windows_for(sequence: str, position: int, width: int = 15) -> List[str]:
        """
        The peptide windows a substitution at `position` would change.

        Computed even though the screen is not, because the windows are a
        sequence fact and having them ready is what makes wiring a predictor in
        later a small job rather than a redesign.
        """
        sequence = (sequence or "").upper()
        index = position - 1
        if not (0 <= index < len(sequence)):
            return []
        start = max(0, index - width + 1)
        end = min(len(sequence), index + width)
        return [sequence[i:i + width]
                for i in range(start, min(end, len(sequence) - width + 1))]
