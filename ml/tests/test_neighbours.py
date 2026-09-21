"""
Nearest examples and the out-of-distribution warning.  [Addendum 3, section 6]

The warning has to survive two ways of being useless: a verdict from a
reference set too small to have a spread, and a distance measured across two
unrelated spaces. Both are refused rather than reported.
"""

import unittest

from ml.embeddings.encoder import DeterministicEncoder, PositionalOneHotEncoder
from ml.embeddings.explorer import single_substitution_variants
from ml.explain.neighbours import (
    MIN_REFERENCE,
    NeighbourError,
    nearest_examples,
    out_of_distribution,
)

REF = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG"
FAR = "W" * len(REF)


def cloud(encoder, n=60):
    pairs = single_substitution_variants(REF)[:n]
    return [encoder.encode(s) for _, s in pairs], [l for l, _ in pairs]


class TestNearestExamples(unittest.TestCase):

    def setUp(self):
        self.encoder = PositionalOneHotEncoder(max_length=len(REF))
        self.reference, self.labels = cloud(self.encoder)

    def test_neighbours_come_back_ranked_and_closest_first(self):
        query = self.encoder.encode(REF)
        found = nearest_examples(query, self.reference, self.labels, k=5)
        self.assertEqual(len(found), 5)
        self.assertEqual([n.rank for n in found], [1, 2, 3, 4, 5])
        distances = [n.distance for n in found]
        self.assertEqual(distances, sorted(distances))

    def test_a_sequence_in_the_reference_set_is_its_own_nearest(self):
        query = self.reference[7]
        found = nearest_examples(query, self.reference, self.labels, k=1)
        self.assertEqual(found[0].sequence, query.sequence)
        self.assertAlmostEqual(found[0].distance, 0.0)

    def test_labels_must_match(self):
        with self.assertRaises(NeighbourError):
            nearest_examples(self.encoder.encode(REF), self.reference, self.labels[:-1])

    def test_two_spaces_are_refused(self):
        other = DeterministicEncoder(max_length=len(REF)).encode(REF)
        with self.assertRaises(NeighbourError) as cm:
            nearest_examples(other, self.reference, self.labels)
        self.assertIn("no meaning", str(cm.exception))


class TestOutOfDistribution(unittest.TestCase):

    def setUp(self):
        self.encoder = PositionalOneHotEncoder(max_length=len(REF))
        self.reference, self.labels = cloud(self.encoder)

    def test_a_reference_set_too_small_to_have_a_spread_is_refused(self):
        with self.assertRaises(NeighbourError) as cm:
            out_of_distribution(self.encoder.encode(REF),
                                self.reference[:MIN_REFERENCE - 1],
                                self.labels[:MIN_REFERENCE - 1])
        self.assertIn("too few", str(cm.exception))

    def test_a_remote_sequence_is_flagged(self):
        report = out_of_distribution(self.encoder.encode(FAR), self.reference,
                                     self.labels, query_label="poly-W")
        self.assertTrue(report.is_outside)
        self.assertEqual(report.percentile, 1.0)
        self.assertIn("extrapolation", report.verdict)

    def test_a_member_of_the_cloud_is_not_flagged(self):
        report = out_of_distribution(self.reference[3], self.reference, self.labels)
        self.assertFalse(report.is_outside)
        self.assertAlmostEqual(report.distance_to_nearest, 0.0)

    def test_the_verdict_is_stated_in_the_reference_set_s_own_terms(self):
        report = out_of_distribution(self.encoder.encode(FAR), self.reference, self.labels)
        self.assertIn("%", report.verdict)
        self.assertGreater(report.reference_median_nn_distance, 0)

    def test_the_caveat_names_the_kind_of_space(self):
        report = out_of_distribution(self.encoder.encode(FAR), self.reference, self.labels)
        self.assertFalse(report.encoder_has_learned_content)
        self.assertIn("no learned content", report.caveat)


class TestDegeneracyIsReportedNotSmoothed(unittest.TestCase):
    """
    The composition encoder maps different sequences to one vector. That drags
    the comparison distribution to zero and makes every query look remote, so
    the collapse is stated rather than quietly absorbed.
    """

    def test_composition_collapses_a_substitution_scan_and_says_so(self):
        # The collapse needs the same wild-type residue at two positions, so the
        # scan has to reach far enough along the sequence for one to repeat.
        encoder = DeterministicEncoder(max_length=len(REF))
        reference, labels = cloud(encoder, n=300)
        report = out_of_distribution(encoder.encode(FAR), reference, labels)
        self.assertGreater(report.n_coincident, 0)
        self.assertIn("exactly on top of", report.verdict)

    def test_the_positional_encoder_has_nothing_to_report(self):
        encoder = PositionalOneHotEncoder(max_length=len(REF))
        reference, labels = cloud(encoder, n=300)
        report = out_of_distribution(encoder.encode(FAR), reference, labels)
        self.assertEqual(report.n_coincident, 0)
        self.assertNotIn("exactly on top of", report.verdict)

    def test_the_projection_counts_distinct_positions(self):
        from ml.embeddings.explorer import project
        composition = DeterministicEncoder(max_length=len(REF))
        reference, labels = cloud(composition, n=120)
        projection = project(reference, labels, REF)
        self.assertLess(projection.distinct_positions, len(reference))
        self.assertTrue(any("distinct positions" in w for w in projection.warnings))

        positional = PositionalOneHotEncoder(max_length=len(REF))
        reference, labels = cloud(positional, n=120)
        projection = project(reference, labels, REF)
        self.assertEqual(projection.distinct_positions, len(reference))
        self.assertFalse(any("distinct positions" in w for w in projection.warnings))


if __name__ == "__main__":
    unittest.main()
