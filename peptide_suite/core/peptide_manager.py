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


class PeptideManager:
    """Loads, validates, and manages peptide sequences."""

    def __init__(self):
        pass

    def validate_sequence(self, sequence: str) -> Tuple[bool, str]:
        """
        Validate amino acid sequence.

        Returns:
            Tuple of (is_valid, error_message)
        """
        sequence = sequence.upper().replace(" ", "").replace("\n", "")

        if not sequence:
            return False, "Empty sequence"

        # Check for non-standard characters
        invalid = set(sequence) - CANONICAL_AAS
        if invalid:
            return False, f"Invalid amino acid(s): {', '.join(sorted(invalid))}"

        if len(sequence) < 5:
            return False, "Peptide too short (minimum 5 residues)"

        if len(sequence) > 200:
            logger.warning(f"Large peptide ({len(sequence)} AA) may be slower to analyze")

        return True, ""

    def load_sequence(self, input_str: str) -> Tuple[str, str]:
        """
        Load sequence from string (raw sequence or FASTA-like format).

        Returns:
            Tuple of (sequence, name)
        """
        input_str = input_str.strip()

        if input_str.startswith(">"):
            # FASTA format
            lines = input_str.split("\n")
            name = lines[0][1:].strip()  # Remove '>' and trim
            sequence = "".join(lines[1:]).replace(" ", "").replace("\n", "").upper()
        else:
            # Raw sequence
            name = "unnamed_peptide"
            sequence = input_str.replace(" ", "").replace("\n", "").upper()

        is_valid, error = self.validate_sequence(sequence)
        if not is_valid:
            raise ValueError(f"Invalid sequence: {error}")

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
