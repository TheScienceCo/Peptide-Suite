"""
Tests for sequence input handling.

Rejecting a real database sequence is the worst possible failure for this tool,
because it surfaces as "not recognised" and blames the user's peptide for a
parsing problem. These tests pin the formats sequences actually arrive in.
"""

import unittest

from peptide_suite.core.charge_calculator import ChargeCalculator
from peptide_suite.core.peptide_manager import PeptideManager, peptide_length_ceiling
from peptide_suite.core.physics_tiers import Tier0Sequence
from peptide_suite.core.uniprot_client import parse_fasta_header

GLP1 = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG"


class TestRealWorldPasteFormats(unittest.TestCase):

    def setUp(self):
        self.pm = PeptideManager()

    def test_plain_sequence(self):
        seq, _ = self.pm.load_sequence(GLP1)
        self.assertEqual(seq, GLP1)

    def test_lowercase(self):
        seq, _ = self.pm.load_sequence(GLP1.lower())
        self.assertEqual(seq, GLP1)

    def test_whitespace_blocked(self):
        seq, _ = self.pm.load_sequence("HAEGTFTSDV SSYLEGQAAK EFIAWLVKGR G")
        self.assertEqual(seq, GLP1)

    def test_line_numbered_format(self):
        """NCBI and EMBL display formats interleave position numbers."""
        raw = "        1 haegtftsdv ssylegqaak\n       21 efiawlvkgr g"
        seq, _ = self.pm.load_sequence(raw)
        self.assertEqual(seq, GLP1)

    def test_fasta_with_wrapped_lines(self):
        raw = ">sp|P01275|GLP1 Glucagon\nHAEGTFTSDVSSYLEG\nQAAKEFIAWLVKGRG"
        seq, name = self.pm.load_sequence(raw)
        self.assertEqual(seq, GLP1)
        self.assertEqual(name, "GLP1")

    def test_alignment_gaps_removed(self):
        seq, _ = self.pm.load_sequence("HAEG--TFTSDV.SSYLEGQAAKEFIAWLVKGRG")
        self.assertEqual(seq, "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG")

    def test_stop_codon_marker_removed(self):
        seq, _ = self.pm.load_sequence(GLP1 + "*")
        self.assertEqual(seq, GLP1)

    def test_ambiguity_codes_are_kept_not_rejected(self):
        """X, U, B, Z appear in real UniProt entries and must not be a hard error."""
        for code in ("X", "U", "B", "Z", "J", "O"):
            raw = GLP1[:10] + code + GLP1[10:]
            seq, _ = self.pm.load_sequence(raw)
            self.assertIn(code, seq, f"{code} was dropped or rejected")

    def test_cleaning_reports_what_it_removed(self):
        """Nothing is stripped silently."""
        _, notes = self.pm.clean_sequence(">hdr\n  1 HAEGTFTSDV\n 11 SSYLEGQAAK*")
        joined = " ".join(notes).lower()
        self.assertIn("fasta header", joined)
        self.assertIn("position numbers", joined)
        self.assertIn("stop codon", joined)

    def test_empty_input_gives_an_actionable_message(self):
        with self.assertRaises(ValueError) as ctx:
            self.pm.load_sequence("12345 \n 678")
        self.assertIn("No amino acid residues", str(ctx.exception))


class TestProteinVersusPeptide(unittest.TestCase):

    def setUp(self):
        self.pm = PeptideManager()

    def test_peptide_is_not_flagged(self):
        self.assertFalse(self.pm.classify_length(GLP1)["is_protein"])

    def test_full_length_protein_is_flagged(self):
        result = self.pm.classify_length("A" * 3100)
        self.assertTrue(result["is_protein"])
        self.assertIn("full-length protein", result["note"])

    def test_boundary(self):
        ceiling = peptide_length_ceiling()
        self.assertFalse(self.pm.classify_length("A" * ceiling)["is_protein"])
        self.assertTrue(self.pm.classify_length("A" * (ceiling + 1))["is_protein"])


class TestTerminalCharges(unittest.TestCase):
    """
    The free N-terminal amino and C-terminal carboxyl were defined but never
    used, so net charge and pI were wrong for every peptide — badly wrong for
    short ones, where the termini can be the only charges present.
    """

    def setUp(self):
        self.cc = ChargeCalculator()
        self.t0 = Tier0Sequence()

    def test_peptide_with_no_ionizable_side_chains_is_still_zwitterionic(self):
        neutral = "MTTTTTTSSSSNNQQAA"
        terminals = self.cc.terminal_charges(neutral, ph=7.4)
        self.assertGreater(terminals["n_terminus"], 0.9)
        self.assertLess(terminals["c_terminus"], -0.9)

    def test_isoelectric_point_is_defined_without_ionizable_side_chains(self):
        """Previously returned 0.0, which is not a real pI for any peptide."""
        pi = self.t0.isoelectric_point("MTTTTTTSSSSNNQQAA")
        self.assertGreater(pi, 4.0)
        self.assertLess(pi, 9.0)

    def test_basic_and_acidic_peptides_have_expected_pi(self):
        self.assertGreater(self.t0.isoelectric_point("KKKKK"), 9.0)
        self.assertLess(self.t0.isoelectric_point("EEEEE"), 5.0)

    def test_termini_can_be_excluded_for_capped_peptides(self):
        with_termini = self.cc.net_charge(GLP1, ph=7.4, include_termini=True)
        without = self.cc.net_charge(GLP1, ph=7.4, include_termini=False)
        self.assertNotAlmostEqual(with_termini, without, places=3)


class TestFastaHeaderParsing(unittest.TestCase):

    def test_uniprot_header_yields_accession_and_organism(self):
        h = (">sp|P24043|LAMA2_HUMAN Laminin subunit alpha-2 "
             "OS=Homo sapiens OX=9606 GN=LAMA2 PE=1 SV=4")
        parsed = parse_fasta_header(h)
        self.assertEqual(parsed["accession"], "P24043")
        self.assertEqual(parsed["entry_name"], "LAMA2_HUMAN")
        self.assertEqual(parsed["description"], "Laminin subunit alpha-2")
        self.assertEqual(parsed["organism"], "Homo sapiens")

    def test_trembl_header(self):
        parsed = parse_fasta_header(">tr|A0A024R161|A0A024R161_HUMAN Some protein")
        self.assertEqual(parsed["accession"], "A0A024R161")

    def test_bare_accession_is_found(self):
        self.assertEqual(parse_fasta_header(">P01308 insulin")["accession"], "P01308")

    def test_non_fasta_input_returns_empty_fields(self):
        parsed = parse_fasta_header(GLP1)
        self.assertEqual(parsed["accession"], "")


if __name__ == "__main__":
    unittest.main()
