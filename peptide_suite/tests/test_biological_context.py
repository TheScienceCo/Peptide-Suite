"""
Known biology, retrieved before anything is predicted.  [Phase 1]

The failure this layer exists to prevent: the system would rank substitutions
in IGF-1 against a binding-affinity goal without ever naming IGF1R. Not because
it disagreed about the receptor -- because nothing was responsible for saying
what the molecule was.

The tests that matter are the ones about what the layer refuses to say. It is
easy to make a context object that always has an answer.
"""

import json
import unittest
from pathlib import Path

from peptide_suite.core import EvidenceTier
from peptide_suite.core.biological_context import (
    BiologicalContext,
    InteractionType,
    MoleculeForm,
    Provenance,
    SourceKind,
    retrieve,
    unknown_context,
)

MATURE_IGF1 = "GPETLCGAELVDALQFVCGDRGFYFNKPTGYGSSSRRAPQTGIVDECCFRSCDLRRLEMYCAPLKPAKSA"
GLP1 = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG"
UNKNOWN = "MKWVTFISLLFLFSSAYSRGVFRR"


class TestTheIGF1AcceptanceCase(unittest.TestCase):
    """
    The state the product must no longer permit: recognised IGF-1, a
    binding-affinity goal selected, and no IGF1R anywhere in the output.
    """

    def setUp(self):
        self.context = retrieve(MATURE_IGF1)

    def test_the_canonical_mature_sequence_is_recognised(self):
        self.assertTrue(self.context.is_established)
        self.assertEqual(self.context.name, "Insulin-like growth factor 1")
        self.assertEqual(self.context.gene, "IGF1")
        self.assertEqual(len(MATURE_IGF1), 70)

    def test_it_is_recognised_as_the_mature_peptide_not_the_precursor(self):
        self.assertIs(self.context.form, MoleculeForm.MATURE_PEPTIDE)
        self.assertTrue(self.context.sequence_matches_mature)
        self.assertIn("E-peptide", self.context.precursor_of)

    def test_igf1r_is_exposed_as_an_established_receptor(self):
        genes = [r.target_gene for r in self.context.primary_receptors]
        self.assertIn("IGF1R", genes)

    def test_the_insulin_receptor_is_secondary_not_primary(self):
        # Cross-reactivity is a real and design-relevant fact, and reporting it
        # at the same level as the principal receptor would misdescribe the
        # molecule.
        insr = [r for r in self.context.receptors if r.target_gene == "INSR"]
        self.assertEqual(len(insr), 1)
        self.assertFalse(insr[0].is_primary)

    def test_the_binding_proteins_are_not_filed_as_receptors(self):
        igfbp = [r for r in self.context.receptors if r.target_gene == "IGFBP1-6"]
        self.assertEqual(len(igfbp), 1)
        self.assertIs(igfbp[0].interaction_type, InteractionType.BINDING_PROTEIN)

    def test_the_bcad_domain_organisation_is_exposed(self):
        names = [r.name for r in self.context.regions]
        for domain in ("B domain", "C domain", "A domain", "D domain"):
            self.assertIn(domain, names)

    def test_the_regions_tile_the_mature_sequence_without_gaps_or_overlap(self):
        spans = sorted((r.start, r.end) for r in self.context.regions)
        self.assertEqual(spans[0][0], 1)
        self.assertEqual(spans[-1][1], len(MATURE_IGF1))
        for (_, end), (start, _) in zip(spans, spans[1:]):
            self.assertEqual(start, end + 1, f"regions do not tile: {spans}")

    def test_the_three_disulfides_land_on_cysteines(self):
        # A disulfide table that disagrees with the sequence beside it means
        # one of them is wrong. This is the cheap place to notice.
        self.assertEqual(len(self.context.disulfides), 3)
        self.assertEqual(self.context.disulfide_inconsistencies(), [])

    def test_no_binding_interface_is_manufactured(self):
        # Residue-level contacts come from structures and mutagenesis. None was
        # retrieved, so none is asserted -- and in particular none is invented
        # from hydrophobicity.
        self.assertEqual(self.context.interfaces, [])


class TestProvenanceCapsWhatMayBeClaimed(unittest.TestCase):
    """
    The rule "no citation, no direct-experimental designation" is enforced in
    `Provenance.max_tier` rather than left to each caller to remember.
    """

    def test_a_curated_record_cannot_be_experimental_evidence(self):
        curated = Provenance(kind=SourceKind.CURATED_UNVERIFIED, needs_verification=True)
        self.assertIs(curated.max_tier, EvidenceTier.BIOCHEMICAL_PRINCIPLE)
        self.assertFalse(curated.has_citation)

    def test_literature_without_a_citation_is_not_direct_experimental(self):
        uncited = Provenance(kind=SourceKind.PRIMARY_LITERATURE)
        self.assertIsNot(uncited.max_tier, EvidenceTier.DIRECT_EXPERIMENTAL)

    def test_literature_with_a_pmid_is(self):
        cited = Provenance(kind=SourceKind.PRIMARY_LITERATURE, pmid="12345678")
        self.assertIs(cited.max_tier, EvidenceTier.DIRECT_EXPERIMENTAL)

    def test_a_structure_needs_an_accession_to_count(self):
        self.assertIsNot(Provenance(kind=SourceKind.EXPERIMENTAL_STRUCTURE).max_tier,
                         EvidenceTier.DIRECT_EXPERIMENTAL)
        self.assertIs(Provenance(kind=SourceKind.EXPERIMENTAL_STRUCTURE,
                                 accession="1ABC").max_tier,
                      EvidenceTier.DIRECT_EXPERIMENTAL)

    def test_every_igf1_claim_is_marked_unverified(self):
        # Nothing in this environment can reach UniProt, so nothing curated has
        # been checked against it, and the record says so on every field rather
        # than in a header a reader may not reach.
        context = retrieve(MATURE_IGF1)
        self.assertTrue(context.needs_verification)
        for receptor in context.receptors:
            self.assertTrue(receptor.provenance.needs_verification)
            self.assertIsNot(receptor.tier, EvidenceTier.DIRECT_EXPERIMENTAL)

    def test_an_unmeasured_affinity_is_absent_rather_than_null(self):
        # A null affinity is one `?? 0` away from being read as a number.
        for receptor in retrieve(MATURE_IGF1).receptors:
            self.assertFalse(receptor.has_measured_affinity)
            self.assertNotIn("affinity", receptor.encode())


class TestAnUnknownPeptideGetsNoBiography(unittest.TestCase):

    def test_nothing_is_established(self):
        context = retrieve(UNKNOWN)
        self.assertFalse(context.is_established)
        self.assertEqual(context.receptors, [])
        self.assertEqual(context.regions, [])
        self.assertEqual(context.name, "")

    def test_the_absence_is_stated_rather_than_left_as_an_empty_list(self):
        # "We have no record of this" and "we looked and it has no receptors"
        # render identically if the only difference is an empty list.
        context = retrieve(UNKNOWN)
        self.assertIn("No established peptide identity", context.summary())
        self.assertTrue(context.retrieval_notes)

    def test_no_receptor_is_inferred_from_family_resemblance(self):
        # A near-miss of a glucagon-superfamily peptide: one substitution from
        # GLP-1. It must not acquire GLP1R.
        near_miss = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRA"
        self.assertNotEqual(near_miss, GLP1)
        context = retrieve(near_miss)
        self.assertFalse(context.is_established)
        self.assertEqual([r.target_gene for r in context.receptors], [])


class TestFormsAreNotConflated(unittest.TestCase):

    def test_a_fragment_is_reported_as_a_fragment(self):
        fragment = MATURE_IGF1[:20]
        context = retrieve(fragment, name="IGF-1")
        self.assertTrue(context.is_established)
        self.assertIs(context.form, MoleculeForm.FRAGMENT)
        self.assertFalse(context.sequence_matches_mature)
        self.assertTrue(any("fragment" in n for n in context.retrieval_notes))

    def test_a_name_match_on_a_different_sequence_says_so(self):
        context = retrieve("WWWWWWWWWW", name="IGF-1")
        self.assertTrue(any("differs from the mature" in n for n in context.retrieval_notes))

    def test_every_form_describes_itself(self):
        for form in MoleculeForm:
            self.assertTrue(form.describe.strip())


class TestNameMatchingIsNotSubstringMatching(unittest.TestCase):

    def test_insulin_does_not_select_insulin_like_growth_factor(self):
        # The substring trap: "insulin" is inside "Insulin-like growth factor 1".
        context = retrieve("", name="insulin")
        self.assertFalse(context.is_established)


class TestTheCuratedFileKeepsItsOwnRules(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.raw = json.loads((root / "data" / "biological_context.json").read_text())

    def test_no_record_carries_an_unsourced_citation(self):
        # Nothing here was verifiable from this environment, so no citation is
        # claimed. A fabricated PMID is worse than a missing one: it looks
        # checkable and survives review. Scoped to the records -- the header
        # prose says the word while explaining that none are claimed.
        blob = json.dumps(self.raw["peptides"]).lower()
        for tell in ('"pmid"', '"doi"', '"pdb"', '"structures"'):
            self.assertNotIn(tell, blob)

    def test_every_peptide_declares_a_form(self):
        for name, record in self.raw["peptides"].items():
            with self.subTest(peptide=name):
                self.assertIn(record["form"], [f.value for f in MoleculeForm])

    def test_every_receptor_declares_an_interaction_type(self):
        valid = {t.value for t in InteractionType}
        for name, record in self.raw["peptides"].items():
            for receptor in record.get("receptors", []):
                with self.subTest(peptide=name, target=receptor["target_gene"]):
                    self.assertIn(receptor["interaction_type"], valid)

    def test_exactly_one_primary_receptor_per_peptide(self):
        for name, record in self.raw["peptides"].items():
            primaries = [r for r in record.get("receptors", []) if r.get("is_primary")]
            with self.subTest(peptide=name):
                self.assertEqual(len(primaries), 1,
                                 "a peptide with two principal receptors needs both "
                                 "justified, not silently listed")

    def test_curated_disulfides_land_on_cysteines(self):
        for name, record in self.raw["peptides"].items():
            sequence = record.get("mature_sequence", "")
            for bond in record.get("disulfides", []):
                with self.subTest(peptide=name, bond=(bond["first"], bond["second"])):
                    self.assertEqual(sequence[bond["first"] - 1], "C")
                    self.assertEqual(sequence[bond["second"] - 1], "C")


class TestTheCorrectedReferenceData(unittest.TestCase):
    """
    The panel held a 33-residue sequence under the name of a 70-residue peptide,
    an isolated chain under the name of a two-chain molecule, and an uncited
    variant effect at a position whose residue did not match.
    """

    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.panel = json.loads((root / "data" / "test_panel.json").read_text())
        cls.reference = json.loads(
            (root / "data" / "reference_peptides.json").read_text())["exact_peptides"]

    def test_the_panel_igf1_is_the_canonical_mature_peptide(self):
        self.assertEqual(self.panel["IGF1"]["sequence"], MATURE_IGF1)
        self.assertEqual(self.panel["IGF1"]["form"], "MATURE_PEPTIDE")

    def test_the_reference_set_can_identify_mature_igf1(self):
        self.assertEqual(self.reference["IGF-1"]["sequence"], MATURE_IGF1)

    def test_the_insulin_a_chain_is_not_called_insulin(self):
        entry = self.panel["INS"]
        self.assertEqual(entry["form"], "ISOLATED_CHAIN")
        self.assertIn("A-chain", entry["full_name"])
        self.assertIn("not insulin", entry["function"])

    def test_the_unverifiable_entry_is_flagged_rather_than_relabelled(self):
        entry = self.panel["BPC157"]
        self.assertTrue(entry["needs_verification"])
        self.assertEqual(entry["form"], "UNKNOWN")
        self.assertEqual(entry["function"], "",
                         "an entry whose identity is unestablished must not carry a "
                         "function annotation")

    def test_the_uncited_variant_effect_is_gone(self):
        for name, entry in self.panel.items():
            if name.startswith("_"):
                continue
            with self.subTest(entry=name):
                self.assertNotIn("known_var_effects", entry)

    def test_no_entry_presents_repeated_copies_as_homologs(self):
        for name, entry in self.panel.items():
            if name.startswith("_"):
                continue
            homologs = entry.get("homologs", [])
            with self.subTest(entry=name):
                self.assertEqual(len(homologs), len(set(homologs)))


if __name__ == "__main__":
    unittest.main()


class TestLabelsDoNotOverstateCapability(unittest.TestCase):
    """
    A label reading "Binding affinity" implies a computed dissociation
    constant. What is computed is the size of a physicochemical perturbation,
    and the direction of the effect is explicitly not determined.
    """

    @classmethod
    def setUpClass(cls):
        # The catalogue, not the HTTP layer that serves it. Reading it through
        # `peptide_suite.api` imported FastAPI, which the boundary CI job does
        # not install on purpose -- so this class passed locally and failed
        # there. See TestTheSuiteNeedsNoThirdPartyPackages below.
        from peptide_suite.core.goal_catalog import goal_catalog
        cls.payload = goal_catalog()
        cls.goals = {g["id"]: g for g in cls.payload["goals"]}

    def test_the_binding_label_disclaims_a_predicted_constant(self):
        label = self.goals["binding_affinity"]["label"]
        self.assertNotEqual(label, "Binding affinity")
        self.assertIn("not a predicted Kd", label)

    def test_every_goal_declares_what_it_does_not_compute(self):
        for goal_id, goal in self.goals.items():
            with self.subTest(goal=goal_id):
                self.assertTrue(goal["computes"])
                self.assertTrue(goal["does_not_compute"])
                self.assertIn(goal["strongest_evidence"], [t.name for t in EvidenceTier])

    def test_the_binding_lane_disclaims_every_affinity_constant(self):
        disclaimed = " ".join(self.goals["binding_affinity"]["does_not_compute"]).lower()
        for constant in ("kd", "ki", "ic50", "ec50"):
            self.assertIn(constant, disclaimed)
        self.assertIn("direction is not determined", disclaimed)

    def test_the_four_reasoning_kinds_are_distinguished(self):
        kinds = self.payload["binding_reasoning_kinds"]
        self.assertEqual(
            [k["kind"] for k in kinds],
            ["sequence_derived", "experimental_mutation", "structure_supported",
             "learned_model"])

    def test_only_the_sequence_derived_kind_is_available_here(self):
        # Presenting all four as one "binding affinity prediction" is the
        # overstatement this structure exists to prevent.
        kinds = {k["kind"]: k["available"] for k in self.payload["binding_reasoning_kinds"]}
        self.assertTrue(kinds["sequence_derived"])
        self.assertFalse(kinds["experimental_mutation"])
        self.assertFalse(kinds["structure_supported"])
        self.assertFalse(kinds["learned_model"])

    def test_every_unavailable_kind_says_why(self):
        """
        Substance, not a magic phrase. This asserted the literal string "Not
        available", which passed while the text was hardcoded and broke the
        moment the evidence-store kind started computing its own reason -- a
        test enforcing wording rather than the guarantee the wording carried.
        """
        for kind in self.payload["binding_reasoning_kinds"]:
            if not kind["available"]:
                with self.subTest(kind=kind["kind"]):
                    why = kind["what_it_is_not"]
                    self.assertGreater(len(why.split()), 8,
                                       "an unavailable capability needs a reason, not a "
                                       "restatement that it is unavailable")
                    self.assertNotEqual(why.strip().lower(), "not available.")
