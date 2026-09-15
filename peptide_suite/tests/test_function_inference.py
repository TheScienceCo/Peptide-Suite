"""
Tests for sequence-based function inference.

The property that matters most: inference never returns nothing. The previous
implementation matched gene names, so a pasted sequence — the primary input
mode — could never be recognised and the user was left to pick a goal unaided.
"""

import unittest

from peptide_suite.core.epistemics import ClaimType
from peptide_suite.core.function_inference import FunctionInferencer
from peptide_suite.core.uniprot_client import UniProtClient

GLP1 = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG"
LAMININ_IKVAV = "CSRARKQAASIKVAVSADR"
LAMININ_YIGSR = "CDPGYIGSRCDA"
NONSENSE = "MTTTTTTSSSSNNQQAA"


class TestAlwaysReturnsAGoal(unittest.TestCase):
    """No input should leave the user with nothing selected."""

    def setUp(self):
        # UniProt disabled: these tests cover local inference and must not
        # depend on network reachability.
        self.fi = FunctionInferencer(uniprot=UniProtClient(enabled=False))
        self.valid_goals = {"protease_resistance", "binding_affinity", "generic_improvement"}

    def test_every_sequence_yields_a_usable_goal(self):
        for seq in [GLP1, LAMININ_IKVAV, NONSENSE, "AAAAAAAAAA",
                    "KWKLFKKIGAVLKVLTTGLPALIS", "CCCCCCGGGG", "MG"]:
            inference = self.fi.infer(seq)
            self.assertIn(inference.suggested_goal, self.valid_goals, f"failed on {seq}")
            self.assertTrue(inference.goal_reason, f"no reason given for {seq}")

    def test_unrecognised_sequence_still_gets_a_goal_and_a_reason(self):
        r = self.fi.infer(NONSENSE)
        self.assertEqual(r.level, 4)
        self.assertTrue(r.suggested_goal)
        self.assertFalse(r.is_identification)
        self.assertTrue(r.caveats, "a level-4 default must carry a caveat")

    def test_empty_sequence_raises_rather_than_guessing(self):
        with self.assertRaises(ValueError):
            self.fi.infer("")


class TestEvidenceLevels(unittest.TestCase):

    def setUp(self):
        # UniProt disabled: these tests cover local inference and must not
        # depend on network reachability.
        self.fi = FunctionInferencer(uniprot=UniProtClient(enabled=False))

    def test_exact_match_is_level_one(self):
        r = self.fi.infer(GLP1)
        self.assertEqual(r.level, 1)
        self.assertGreaterEqual(r.confidence, 0.9)
        self.assertEqual(r.suggested_goal, "protease_resistance")
        self.assertTrue(r.is_identification)

    def test_near_match_is_level_one_but_caveated(self):
        variant = "HGEGTFTSDVSSYLEGQAAKEFIAWLVKGRG"   # one substitution
        r = self.fi.infer(variant)
        self.assertEqual(r.level, 1)
        self.assertLess(r.confidence, 0.95)
        self.assertTrue(r.caveats)

    def test_motif_match_is_level_two(self):
        r = self.fi.infer(LAMININ_IKVAV)
        self.assertEqual(r.level, 2)
        self.assertEqual(r.matched_name, "IKVAV")
        self.assertIn("Laminin", r.parent_protein)
        self.assertEqual(r.suggested_goal, "binding_affinity")

    def test_motif_match_reports_native_context(self):
        """A motif excised from a large ECM protein must say so."""
        r = self.fi.infer(LAMININ_YIGSR)
        self.assertTrue(r.native_context_note)
        self.assertIn("fragment", r.native_context_note.lower())

    def test_short_motif_carries_a_chance_match_caveat(self):
        r = self.fi.infer("GGGRGDSPGGG")
        self.assertEqual(r.matched_name, "RGD")
        self.assertTrue(any("residues" in c for c in r.caveats))
        self.assertLess(r.confidence, 0.5)

    def test_family_signature_is_level_three(self):
        cationic_amphipathic = "KWKLFKKIGAVLKVLTTGLPALIS"
        r = self.fi.infer(cationic_amphipathic)
        self.assertEqual(r.level, 3)
        self.assertFalse(r.is_identification)
        self.assertIn("not an identification", r.inferred_function.lower())

    def test_confidence_decreases_with_level(self):
        levels = [
            self.fi.infer(GLP1),
            self.fi.infer(LAMININ_IKVAV),
            self.fi.infer("KWKLFKKIGAVLKVLTTGLPALIS"),
            self.fi.infer(NONSENSE),
        ]
        confidences = [r.confidence for r in levels]
        self.assertEqual(confidences, sorted(confidences, reverse=True))

    def test_level_four_is_not_presented_as_identification(self):
        r = self.fi.infer(NONSENSE)
        self.assertFalse(r.is_identification)
        self.assertIn("not matched", r.inferred_function.lower())

    def test_level_four_still_reports_measured_properties(self):
        """
        An unrecognised sequence must still say something true about itself.
        "Not recognised" alone tells the user nothing they can act on.
        """
        r = self.fi.infer(NONSENSE)
        self.assertIn("residues", r.inferred_function)
        self.assertIn("pH 7.4", r.inferred_function)


class TestClaimProvenance(unittest.TestCase):

    def setUp(self):
        # UniProt disabled: these tests cover local inference and must not
        # depend on network reachability.
        self.fi = FunctionInferencer(uniprot=UniProtClient(enabled=False))

    def test_sequence_match_is_a_retrieved_claim(self):
        self.assertEqual(self.fi.infer(GLP1).claim.claim_type, ClaimType.RETRIEVED)

    def test_family_signature_is_a_computed_claim(self):
        r = self.fi.infer("KWKLFKKIGAVLKVLTTGLPALIS")
        self.assertEqual(r.claim.claim_type, ClaimType.COMPUTED)
        self.assertTrue(r.claim.method)

    def test_reference_matches_are_flagged_as_unverified(self):
        """The local cache is not an authority and must not present as one."""
        r = self.fi.infer(GLP1)
        self.assertTrue(any("unverified" in c for c in r.claim.citations))
        self.assertFalse(r.claim.experimentally_tested)


class TestReferenceDataIntegrity(unittest.TestCase):

    def setUp(self):
        # UniProt disabled: these tests cover local inference and must not
        # depend on network reachability.
        self.fi = FunctionInferencer(uniprot=UniProtClient(enabled=False))

    def test_every_reference_peptide_infers_to_itself(self):
        for name, entry in self.fi.reference["exact_peptides"].items():
            r = self.fi.infer(entry["sequence"])
            self.assertEqual(r.level, 1, f"{name} did not self-match")
            self.assertEqual(r.matched_name, name)

    def test_reference_sequences_are_valid_amino_acids(self):
        canonical = set("ACDEFGHIKLMNPQRSTVWY")
        for name, entry in self.fi.reference["exact_peptides"].items():
            invalid = set(entry["sequence"].upper()) - canonical
            self.assertFalse(invalid, f"{name} contains non-canonical residues: {invalid}")

    def test_every_reference_entry_declares_a_goal_and_a_reason(self):
        for name, entry in self.fi.reference["exact_peptides"].items():
            self.assertTrue(entry.get("suggested_goal"), f"{name} has no suggested goal")
            self.assertTrue(entry.get("goal_reason"), f"{name} has no goal reason")

    def test_bpc157_is_a_pentadecapeptide(self):
        """
        BPC-157 is named as a pentadecapeptide throughout the literature. This
        pins the length so a longer sequence cannot be filed under this name
        without the test failing and forcing the question.
        """
        seq = self.fi.reference["exact_peptides"]["BPC-157"]["sequence"]
        self.assertEqual(len(seq), 15, f"BPC-157 should be 15 residues, found {len(seq)}")


if __name__ == "__main__":
    unittest.main()


class TestNameResolution(unittest.TestCase):
    """
    Typing a name must work. Many peptide and protein names are themselves valid
    amino acid strings — "laminin" is L-A-M-I-N-I-N, "oxytocin" is
    O-X-Y-T-O-C-I-N — so a sequence-first parser silently analyses a short
    peptide nobody asked about instead of looking the name up.
    """

    def setUp(self):
        self.fi = FunctionInferencer(uniprot=UniProtClient(enabled=False))

    def test_common_names_resolve(self):
        for query, expected in [
            ("GLP-1", "GLP-1 (7-37)"),
            ("glp1", "GLP-1 (7-37)"),
            ("oxytocin", "Oxytocin"),
            ("Substance P", "Substance P"),
            ("LL-37", "LL-37"),
            ("bpc157", "BPC-157"),
            ("melittin", "Melittin"),
        ]:
            hit = self.fi.resolve_name(query)
            self.assertIsNotNone(hit, f"'{query}' did not resolve")
            self.assertEqual(hit[0], expected)

    def test_names_that_are_also_valid_sequences_still_resolve(self):
        """The whole point: these parse as peptides, so they must be checked as names first."""
        for query in ("oxytocin", "melittin", "laminin"):
            self.assertTrue(
                self.fi.resolve_name(query) or self.fi.lookup_protein_by_name(query),
                f"'{query}' resolved as neither a peptide name nor a protein name",
            )

    def test_protein_name_resolves_with_analysable_motifs(self):
        hit = self.fi.lookup_protein_by_name("laminin")
        self.assertIsNotNone(hit)
        self.assertEqual(hit["gene"], "LAMA2")
        motifs = {m["motif"] for m in hit["derived_motifs"]}
        self.assertIn("IKVAV", motifs)
        self.assertIn("YIGSR", motifs)

    def test_unknown_name_resolves_to_nothing(self):
        self.assertIsNone(self.fi.resolve_name("zzzznotapeptide"))
        self.assertIsNone(self.fi.lookup_protein_by_name("zzzznotapeptide"))


class TestBidirectionalSequenceMatching(unittest.TestCase):
    """A pasted sequence relates to a known peptide in four ways, not just one."""

    def setUp(self):
        self.fi = FunctionInferencer(uniprot=UniProtClient(enabled=False))

    def test_exact(self):
        r = self.fi.infer(GLP1)
        self.assertEqual(r.matched_name, "GLP-1 (7-37)")
        self.assertIn("Exact", r.basis)

    def test_query_containing_a_known_peptide(self):
        """An expression construct with tags around a known peptide."""
        construct = "MKTIIALSYIFCLVFA" + GLP1 + "GRRRSHHHHHH"
        r = self.fi.infer(construct)
        self.assertEqual(r.matched_name, "GLP-1 (7-37)")
        self.assertIn("Contains", r.basis)
        self.assertTrue(any("contains" in c.lower() for c in r.caveats))

    def test_fragment_of_a_known_peptide(self):
        r = self.fi.infer(GLP1[4:22])
        self.assertEqual(r.matched_name, "GLP-1 (7-37)")
        self.assertIn("Fragment", r.basis)
        self.assertTrue(any("fragment" in c.lower() for c in r.caveats))

    def test_short_fragments_do_not_match_by_chance(self):
        """A tetrapeptide occurs in something by chance; that is not identification."""
        r = self.fi.infer("GTFTS")
        self.assertNotEqual(r.level, 1)

    def test_database_covers_the_common_peptides(self):
        """Regression guard: the reference set must stay large enough to be useful."""
        self.assertGreaterEqual(len(self.fi.reference["exact_peptides"]), 40)
        for expected in ("Oxytocin", "Substance P", "LL-37", "Melittin",
                         "Angiotensin II", "Bradykinin", "Somatostatin-14"):
            self.assertIn(expected, self.fi.reference["exact_peptides"])
