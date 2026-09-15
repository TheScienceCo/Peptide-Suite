"""
Peptide sequence management, validation, and basic property computation.
Handles loading from sequences, NCBI lookups, and property calculations.
"""

import logging
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

CANONICAL_AAS = set("ACDEFGHIKLMNPQRSTVWY")

# Codes that appear in real database sequences and must not cause a rejection.
# Rejecting these means a user cannot paste a genuine UniProt or RefSeq entry,
# which is the most common thing anyone will try to do.
AMBIGUITY_CODES = {
    "X": "any amino acid (position unresolved)",
    "B": "Asp or Asn (unresolved)",
    "Z": "Glu or Gln (unresolved)",
    "J": "Leu or Ile (unresolved)",
    "U": "selenocysteine",
    "O": "pyrrolysine",
}

ACCEPTED_AAS = CANONICAL_AAS | set(AMBIGUITY_CODES)

# Past some length a sequence is a protein, not a peptide. The distinction
# changes what the analysis can honestly claim, so it is surfaced rather than
# ignored. Where the line falls is a policy question.
def peptide_length_ceiling() -> int:
    from ..runtime import int_threshold
    return int_threshold("identification.peptide_length_ceiling")


class PeptideManager:
    """Loads, validates, and manages peptide sequences."""

    def __init__(self):
        pass

    def clean_sequence(self, raw: str) -> Tuple[str, List[str]]:
        """
        Strip a pasted sequence down to residues, tolerating real-world formats.

        Handles the shapes sequences actually arrive in: FASTA, whitespace-blocked,
        line-numbered (the NCBI and EMBL display formats), lowercase, and
        containing ambiguity codes. Returns the residues plus a list of notes
        describing what was cleaned away, so nothing is removed silently.
        """
        notes: List[str] = []
        text = raw.strip()

        if text.startswith(">"):
            lines = text.split("\n")
            notes.append(f"FASTA header removed: {lines[0][1:].strip()[:70]}")
            text = "\n".join(lines[1:])

        if any(ch.isdigit() for ch in text):
            notes.append("Position numbers removed (line-numbered sequence format)")
        if "-" in text or "." in text:
            notes.append("Alignment gap characters removed")
        if "*" in text:
            notes.append("Stop codon marker removed")

        kept = []
        dropped = set()
        for ch in text.upper():
            if ch in ACCEPTED_AAS:
                kept.append(ch)
            elif ch.isspace() or ch.isdigit() or ch in "-.*":
                continue
            else:
                dropped.add(ch)

        if dropped:
            notes.append(
                f"Unrecognised character(s) removed: {', '.join(sorted(dropped))}"
            )

        sequence = "".join(kept)

        present_ambiguity = sorted(set(sequence) & set(AMBIGUITY_CODES))
        if present_ambiguity:
            notes.append(
                "Contains ambiguity codes: "
                + "; ".join(f"{c} = {AMBIGUITY_CODES[c]}" for c in present_ambiguity)
                + ". These positions are kept but cannot be scored for charge, "
                "hydrophobicity, or liability motifs."
            )

        return sequence, notes

    def validate_sequence(self, sequence: str) -> Tuple[bool, str]:
        """
        Validate an already-cleaned amino acid sequence.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not sequence:
            return False, (
                "No amino acid residues found. Paste a protein or peptide sequence — "
                "FASTA, numbered, or plain text all work."
            )

        invalid = set(sequence) - ACCEPTED_AAS
        if invalid:
            return False, f"Invalid amino acid(s): {', '.join(sorted(invalid))}"

        if len(sequence) < 5:
            return False, f"Sequence too short ({len(sequence)} residues; minimum 5)"

        return True, ""

    def classify_length(self, sequence: str) -> Dict:
        """
        Distinguish a peptide from a full-length protein.

        This matters because most of the analysis in this suite assumes a
        peptide. Running a per-position substitution scan over a 3,000-residue
        protein produces roughly 57,000 candidates and answers a question nobody
        asked; the useful move on a protein is to locate the relevant domain
        first and analyse that.
        """
        n = len(sequence)
        if n <= peptide_length_ceiling():
            return {"is_protein": False, "length": n, "note": ""}

        return {
            "is_protein": True,
            "length": n,
            "note": (
                f"This is {n} residues — a full-length protein, not a peptide. "
                f"A per-position substitution scan would generate {n * 19:,} candidates "
                f"and rank them against each other, which is not a meaningful question at "
                f"this scale. Sequence-level properties (composition, charge, liability "
                f"motifs, domain-level signals) are still computed and reported. To run the "
                f"optimisation workflows, identify the bioactive region first and paste that."
            ),
        }

    def load_sequence(self, input_str: str) -> Tuple[str, str]:
        """
        Load a sequence from a pasted string.

        Returns:
            Tuple of (sequence, name)
        """
        name = "unnamed_peptide"
        text = input_str.strip()

        if text.startswith(">"):
            header = text.split("\n", 1)[0][1:].strip()
            # FASTA headers are commonly db|accession|entry description
            parts = [p for p in header.split("|") if p]
            name = parts[-1].split()[0] if parts else (header.split()[0] if header else name)

        sequence, _notes = self.clean_sequence(input_str)

        is_valid, error = self.validate_sequence(sequence)
        if not is_valid:
            raise ValueError(error)

        return sequence, name

    def basic_properties(self, sequence: str) -> Dict:
        """Compute basic sequence-level properties."""
        sequence = sequence.upper()

        # Amino acid composition
        composition = {}
        for aa in CANONICAL_AAS:
            composition[aa] = sequence.count(aa)

        # Basic stats
        length = len(sequence)
        charged_count = sequence.count("D") + sequence.count("E") + sequence.count("K") + sequence.count("R")

        return {
            "length": length,
            "composition": composition,
            "charged_residues": charged_count,
            "percent_charged": round(100 * charged_count / length, 1),
            "aromatic_count": sequence.count("F") + sequence.count("W") + sequence.count("Y"),
            "proline_count": sequence.count("P"),
            "cysteine_count": sequence.count("C"),
        }

    def get_motif_positions(self, sequence: str, motif: str) -> List[int]:
        """
        Find all occurrences of a motif in the sequence.

        Args:
            sequence: Peptide sequence
            motif: Motif string (e.g., "RXR" where X is any AA)

        Returns:
            List of 0-indexed start positions
        """
        sequence = sequence.upper()
        motif = motif.upper()

        # Simple regex-like matching with X as wildcard
        positions = []
        for i in range(len(sequence) - len(motif) + 1):
            match = True
            for j, m_char in enumerate(motif):
                if m_char != "X" and sequence[i + j] != m_char:
                    match = False
                    break
            if match:
                positions.append(i)

        return positions

    def six_frame_translation(self, dna_sequence: str) -> List[Tuple[str, int]]:
        """
        Translate DNA to protein in all 6 reading frames.

        Returns:
            List of (protein_sequence, frame_number) tuples
        """
        # Simplified codon table (production would use full NCBI table)
        codon_table = {
            "TTT": "F",
            "TTC": "F",
            "TTA": "L",
            "TTG": "L",
            "CTT": "L",
            "CTC": "L",
            "CTA": "L",
            "CTG": "L",
            "ATT": "I",
            "ATC": "I",
            "ATA": "I",
            "ATG": "M",
            "GTT": "V",
            "GTC": "V",
            "GTA": "V",
            "GTG": "V",
            "TCT": "S",
            "TCC": "S",
            "TCA": "S",
            "TCG": "S",
            "CCT": "P",
            "CCC": "P",
            "CCA": "P",
            "CCG": "P",
            "ACT": "T",
            "ACC": "T",
            "ACA": "T",
            "ACG": "T",
            "GCT": "A",
            "GCC": "A",
            "GCA": "A",
            "GCG": "A",
            "TAT": "Y",
            "TAC": "Y",
            "TAA": "*",
            "TAG": "*",
            "CAT": "H",
            "CAC": "H",
            "CAA": "Q",
            "CAG": "Q",
            "AAT": "N",
            "AAC": "N",
            "AAA": "K",
            "AAG": "K",
            "GAT": "D",
            "GAC": "D",
            "GAA": "E",
            "GAG": "E",
            "TGT": "C",
            "TGC": "C",
            "TGA": "*",
            "TGG": "W",
            "CGT": "R",
            "CGC": "R",
            "CGA": "R",
            "CGG": "R",
            "AGT": "S",
            "AGC": "S",
            "AGA": "R",
            "AGG": "R",
            "GGT": "G",
            "GGC": "G",
            "GGA": "G",
            "GGG": "G",
        }

        dna_sequence = dna_sequence.upper().replace("U", "T")
        translations = []

        # Forward frames (0, 1, 2)
        for frame in range(3):
            protein = ""
            for i in range(frame, len(dna_sequence) - 2, 3):
                codon = dna_sequence[i : i + 3]
                if len(codon) == 3:
                    aa = codon_table.get(codon, "X")
                    if aa == "*":
                        break
                    protein += aa
            if protein:
                translations.append((protein, frame))

        # Reverse complement frames (3, 4, 5)
        complement = str.maketrans("ACGT", "TGCA")
        rev_complement = dna_sequence.translate(complement)[::-1]

        for frame in range(3):
            protein = ""
            for i in range(frame, len(rev_complement) - 2, 3):
                codon = rev_complement[i : i + 3]
                if len(codon) == 3:
                    aa = codon_table.get(codon, "X")
                    if aa == "*":
                        break
                    protein += aa
            if protein:
                translations.append((protein, frame + 3))

        return translations
