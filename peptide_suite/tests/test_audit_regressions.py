"""
Regressions found by auditing the built system rather than by reading it.

Each of these was working code that produced a wrong or expensive answer on an
input nobody had tried. They are grouped here because what they have in common
is how they were found: running every entry point against the inputs it will
actually meet, instead of the ones it was written for.
"""

import time
import unittest

from peptide_suite.core.contact_classifier import GoldenCases, matches_any_alias
from peptide_suite.core.partner_module import PartnerCases
from peptide_suite.core.uniprot_client import UniProtClient
from peptide_suite.workflows.transform import MIN_DESIGNABLE_LENGTH, TransformWorkflow


class TestGoldenCaseMatchingIsNotSubstringMatching(unittest.TestCase):
    """
    Substring matching attributed insulin's B28/B29 dimer interface to the
    A-chain and to proinsulin. The A-chain does not carry that interface, so a
    SCAFFOLD classification and its design consequences were being attached to
    the wrong molecule -- and in proinsulin the region is held by the C-peptide.
    """

    def test_the_a_chain_does_not_inherit_a_b_chain_contact(self):
        for name in ("insulin A", "Insulin A-chain", "insulin a chain"):
            with self.subTest(peptide=name):
                self.assertEqual(GoldenCases.for_peptide(name), [])

    def test_the_precursor_does_not_inherit_the_mature_contact(self):
        self.assertEqual(GoldenCases.for_peptide("proinsulin"), [])

    def test_a_different_protein_containing_the_name_does_not_match(self):
        self.assertEqual(GoldenCases.for_peptide("insulin-like growth factor"), [])

    def test_the_molecule_itself_still_matches(self):
        for name in ("insulin", "Insulin", "human insulin", "insulin B-chain"):
            with self.subTest(peptide=name):
                self.assertTrue(GoldenCases.for_peptide(name))

    def test_neutral_qualifiers_do_not_block_a_match(self):
        """"human oxytocin" is oxytocin; "insulin A-chain" is not insulin."""
        self.assertTrue(GoldenCases.for_peptide("human oxytocin"))
        self.assertTrue(GoldenCases.for_peptide("synthetic ghrelin"))

    def test_chain_is_not_a_neutral_qualifier(self):
        self.assertFalse(matches_any_alias("insulin A-chain", ["insulin"]))

    def test_partner_cases_use_the_same_rule(self):
        self.assertTrue(PartnerCases.for_peptide("magainin 2"))
        self.assertEqual(PartnerCases.for_peptide("premagainin"), [])
        self.assertEqual(PartnerCases.for_peptide("oxytocin"), [])

    def test_every_case_declares_what_it_applies_to(self):
        """
        Matching is only as safe as the alias list. A case with none would
        silently never match, which is a quieter failure than a wrong match but
        still a failure.
        """
        import json
        from pathlib import Path
        root = Path(__file__).resolve().parents[1] / "data"
        for filename, block in (("native_context_golden.json", "cases"),
                                ("partner_peptides.json", "cases")):
            data = json.loads((root / filename).read_text())
            for key, entry in data[block].items():
                with self.subTest(file=filename, case=key):
                    self.assertTrue(entry.get("applies_to"),
                                    f"{key} declares no applies_to list")


class TestGarbageInputIsNotIdentified(unittest.TestCase):
    """
    The worst find of the audit, and reachable from the UI by typing a stray
    character. Typing "!!!!" came back as thyrotropin-releasing hormone at 0.95
    confidence.

    The mechanism: the name query is normalised by stripping non-alphanumerics,
    so "!!!!" became "", and `norm(name).startswith("")` is true for every
    peptide in the reference set. Every peptide matched, the shortest name won,
    and TRH is the shortest name. Two of the three candidate filters guarded
    against an empty query; the first did not.
    """

    def setUp(self):
        from peptide_suite.core.function_inference import FunctionInferencer
        self.inferencer = FunctionInferencer()

    def test_punctuation_resolves_to_nothing(self):
        for query in ("!!!!", "?", "...", "@@@", "-", "###", "   "):
            with self.subTest(query=query):
                self.assertIsNone(self.inferencer.resolve_name(query))

    def test_an_empty_normalised_query_matches_nothing(self):
        """The specific mechanism, pinned so a refactor cannot reintroduce it."""
        self.assertIsNone(self.inferencer.resolve_name("!@#$%^&*()"))

    def test_a_single_character_is_not_a_name(self):
        """
        Prefix-matching one character lands on whichever name sorts shortest,
        which is a coin flip presented as an identification. "a" returned A4G47.
        """
        for query in ("a", "b", "z", "ab", "gl"):
            with self.subTest(query=query):
                self.assertIsNone(self.inferencer.resolve_name(query))

    def test_the_protein_lookup_has_the_same_floor(self):
        """Typing "a" returned a confident BDNF match from the ontology."""
        for query in ("a", "in", "e"):
            with self.subTest(query=query):
                self.assertIsNone(self.inferencer.lookup_protein_by_name(query))

    def test_real_names_still_resolve(self):
        """A guard that blocks real queries has traded one failure for another."""
        for query, expected in (("GLP-1", "GLP-1"), ("glp1", "GLP-1"),
                                ("oxytocin", "Oxytocin"), ("bpc157", "BPC-157"),
                                ("TRH", "TRH"), ("A4G47", "A4G47")):
            with self.subTest(query=query):
                resolved = self.inferencer.resolve_name(query)
                self.assertIsNotNone(resolved, f"{query} no longer resolves")
                self.assertIn(expected, resolved[0])

    def test_real_protein_names_still_resolve(self):
        for query in ("laminin", "bdnf", "neuregulin"):
            with self.subTest(query=query):
                self.assertIsNotNone(self.inferencer.lookup_protein_by_name(query))

    def test_garbage_reaches_the_workflow_as_nothing(self):
        """
        End to end through the workflow, which is the path the UI uses. A
        confident identification with no sequence behind it is the shape of the
        failure, so both halves are checked.
        """
        from peptide_suite.workflows.optimize import OptimizeWorkflow
        for query in ("!!!!", "a", "..."):
            with self.subTest(query=query):
                context, recommendations = OptimizeWorkflow().run(
                    query, confirmed_goal="protease_resistance", auto_confirm=True)
                self.assertEqual(context.sequence, "")
                self.assertEqual(recommendations, [])


class TestUniProtDoesNotRedialAnUnreachableHost(unittest.TestCase):
    """
    The class promises to be time-boxed and to degrade gracefully offline. It
    recorded the unreachable reason and then never read it, so the time box was
    per call rather than per run: on a network that hangs rather than refusing,
    every lookup cost the full timeout.
    """

    def test_the_circuit_starts_closed(self):
        self.assertFalse(UniProtClient()._circuit_open)

    def test_a_transport_failure_opens_the_circuit(self):
        import urllib.error
        client = UniProtClient(cache_dir="/nonexistent-cache-dir-for-test")
        client._cache_path = lambda key: __import__("pathlib").Path("/nonexistent/x.json")

        def boom(*a, **kw):
            raise urllib.error.URLError("simulated outage")

        import urllib.request
        original = urllib.request.urlopen
        urllib.request.urlopen = boom
        try:
            client._get_json("https://example.invalid/x", "k1")
            self.assertTrue(client._circuit_open)
            # Second call must not reach the network at all.
            calls = []
            urllib.request.urlopen = lambda *a, **kw: calls.append(1)
            _data, status = client._get_json("https://example.invalid/y", "k2")
            self.assertEqual(calls, [], "an open circuit still dialled the host")
            self.assertIn("not retried", status)
        finally:
            urllib.request.urlopen = original

    def test_repeated_identification_does_not_repeat_the_cost(self):
        """
        The end-to-end version. The first lookup may pay for the failure; the
        rest must not.
        """
        from peptide_suite.core.function_inference import FunctionInferencer
        inferencer = FunctionInferencer()
        timings = []
        for probe in ("NOTAREALSEQA", "NOTAREALSEQB", "NOTAREALSEQC"):
            start = time.perf_counter()
            inferencer.infer(probe)
            timings.append(time.perf_counter() - start)
        if not inferencer.uniprot._circuit_open:
            self.skipTest("UniProt was reachable in this environment")
        self.assertLess(max(timings[1:]), 0.1,
                        f"later lookups still paying network cost: {timings}")


class TestTransformStatesWhatItsLengthMeans(unittest.TestCase):
    """
    A 331-residue protein came back with peptide-shaped proposals and nothing
    said the framing was wrong, and a one-residue input produced a ranked
    proposal to cap both ends of a single amino acid.
    """

    def test_a_protein_is_flagged(self):
        notes = TransformWorkflow().run("M" + "AGKLVIFEDST" * 30)["length_notes"]
        self.assertTrue(any("protein, not a peptide" in n for n in notes))

    def test_a_too_short_input_is_flagged(self):
        notes = TransformWorkflow().run("A")["length_notes"]
        self.assertTrue(any("no meaningful design space" in n for n in notes))

    def test_a_normal_peptide_is_not_flagged(self):
        self.assertEqual(
            TransformWorkflow().run("HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR")["length_notes"], [])

    def test_the_boundary_is_consistent_with_the_constant(self):
        short = TransformWorkflow().run("A" * (MIN_DESIGNABLE_LENGTH - 1))["length_notes"]
        ok = TransformWorkflow().run("A" * MIN_DESIGNABLE_LENGTH)["length_notes"]
        self.assertTrue(short)
        self.assertEqual(ok, [])

    def test_the_analysis_still_runs_rather_than_refusing(self):
        """
        Stated, not refused: the physics is still real, it is the design
        framing that does not transfer.
        """
        result = TransformWorkflow().run("A")
        self.assertTrue(result["length_notes"])
        self.assertIn("physics", result)


if __name__ == "__main__":
    unittest.main()


class TestNoLookupAnswersADegenerateQuery(unittest.TestCase):
    """
    The bug class, swept rather than the instance patched.

    Typing "!!!!" returned TRH at 0.95 confidence because one filter matched an
    empty normalised query. The same shape -- a lookup that returns something
    confident for input carrying no information -- could exist in any of the
    name-keyed lookups, so all of them are checked together and any new one
    should be added here.
    """

    DEGENERATE = ["", " ", "!", "!!!!", "?", "...", "@#$", "-", "a", "z", "ab", "1"]

    def _lookups(self):
        from peptide_suite.core.class_b1 import is_class_b1
        from peptide_suite.core.function_inference import FunctionInferencer
        from peptide_suite.core.holdout import GoldenSet
        from peptide_suite.core.ncaa_registry import NCAARegistry, detect_noncanonical
        from peptide_suite.core.synthetic_feasibility import regioselectivity_conflict

        inferencer = FunctionInferencer()
        return {
            "resolve_name": inferencer.resolve_name,
            "lookup_protein_by_name": inferencer.lookup_protein_by_name,
            "GoldenCases.for_peptide": GoldenCases.for_peptide,
            "PartnerCases.for_peptide": PartnerCases.for_peptide,
            "GoldenSet.motifs_of": GoldenSet.motifs_of,
            "GoldenSet.drugs_carrying": GoldenSet.drugs_carrying,
            "detect_noncanonical": detect_noncanonical,
            "is_class_b1": is_class_b1,
            "ncaa_status_permits_scoring":
                lambda q: NCAARegistry.status(q).permits_scoring,
            "regioselectivity_conflict":
                lambda q: regioselectivity_conflict("HAEGTFKAAK", q),
        }

    @staticmethod
    def _is_confident(result) -> bool:
        if result is None or result is False:
            return False
        if isinstance(result, (list, tuple, dict, set, str)) and len(result) == 0:
            return False
        return True

    def test_no_lookup_returns_a_confident_answer_to_degenerate_input(self):
        for name, fn in self._lookups().items():
            for query in self.DEGENERATE:
                with self.subTest(lookup=name, query=query):
                    try:
                        result = fn(query)
                    except (ValueError, TypeError):
                        continue  # an explicit refusal is a correct answer
                    self.assertFalse(
                        self._is_confident(result),
                        f"{name}({query!r}) answered with {result!r:.70}")
