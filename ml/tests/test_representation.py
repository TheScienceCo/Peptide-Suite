"""
The representation explorer.  [Addendum 3, section 2]

A two-dimensional scatter of high-dimensional vectors is easy to draw and easy
to over-read. These tests hold the three things that keep it honest: the
explained variance is reported, the encoder that made the space is named, and
two different spaces are never mixed.
"""

import unittest

from ml.embeddings.encoder import (
    DeterministicEncoder,
    EncoderUnavailable,
    ESM2Encoder,
    PositionalOneHotEncoder,
)
from ml.embeddings.explorer import (
    MIN_POINTS,
    N_COMPONENTS,
    ProjectionError,
    VariantClass,
    classify_variant,
    project,
    single_substitution_variants,
)

REF = "HAEGTFTSDVSSYLEGQ"


def variants(n, encoder=None):
    encoder = encoder or DeterministicEncoder(max_length=len(REF))
    pairs = [("wild type", REF)] + single_substitution_variants(REF)[:n]
    return [encoder.encode(s) for _, s in pairs], [l for l, _ in pairs]


class TestVariantClassification(unittest.TestCase):

    def test_the_reference_is_its_own_class(self):
        self.assertEqual(classify_variant(REF, REF), (VariantClass.WILD_TYPE, 0))

    def test_one_difference_is_a_single_substitution(self):
        mutant = "A" + REF[1:]
        self.assertEqual(classify_variant(REF, mutant), (VariantClass.SINGLE, 1))

    def test_two_differences_are_multi(self):
        mutant = "W" + REF[1:-1] + "W"
        self.assertEqual(sum(1 for a, b in zip(REF, mutant) if a != b), 2)
        self.assertEqual(classify_variant(REF, mutant), (VariantClass.MULTI, 2))

    def test_a_different_length_gets_no_substitution_count(self):
        klass, n = classify_variant(REF, REF + "GG")
        self.assertIs(klass, VariantClass.UNRELATED)
        self.assertIsNone(n, "a count from an alignment this module did not compute "
                             "would be a number with no derivation behind it")

    def test_the_variant_generator_covers_the_grid(self):
        generated = single_substitution_variants(REF)
        self.assertEqual(len(generated), len(REF) * 19)
        label, sequence = generated[0]
        self.assertEqual(label, f"{REF[0]}1{'A' if REF[0] != 'A' else 'C'}")
        self.assertEqual(len(sequence), len(REF))
        self.assertEqual(sum(1 for a, b in zip(REF, sequence) if a != b), 1)


class TestWhatTheProjectionReports(unittest.TestCase):

    def test_explained_variance_is_reported_per_component(self):
        embeddings, labels = variants(40)
        p = project(embeddings, labels, REF)
        self.assertEqual(len(p.explained_variance_ratio), N_COMPONENTS)
        self.assertLessEqual(p.cumulative_explained, 1.0 + 1e-9)
        self.assertGreaterEqual(p.explained_variance_ratio[0], p.explained_variance_ratio[1],
                                "components come out ordered by the variance they carry")

    def test_the_encoder_that_made_the_space_travels_with_it(self):
        embeddings, labels = variants(40)
        p = project(embeddings, labels, REF)
        self.assertEqual(p.encoder_model, DeterministicEncoder.model)
        self.assertFalse(p.encoder_has_learned_content)
        self.assertIn("composition", p.interpretation.lower())

    def test_the_interpretation_describes_the_encoder_actually_used(self):
        # Branching on the encoder kind alone told a reader looking at a
        # positional projection that the axes were components of composition.
        comp_e, comp_l = variants(40, DeterministicEncoder(max_length=len(REF)))
        pos_e, pos_l = variants(40, PositionalOneHotEncoder(max_length=len(REF)))
        composition = project(comp_e, comp_l, REF)
        positional = project(pos_e, pos_l, REF)
        self.assertIn("composition", composition.interpretation.lower())
        self.assertNotIn("position-aware", composition.interpretation.lower())
        self.assertIn("position-aware", positional.interpretation.lower())
        self.assertNotIn("principal components of amino-acid composition",
                         positional.interpretation.lower())

    def test_an_unlearned_space_says_so(self):
        embeddings, labels = variants(40)
        p = project(embeddings, labels, REF)
        self.assertTrue(any("no learned content" in w for w in p.warnings))

    def test_the_two_kinds_of_collapse_are_counted_separately(self):
        # One is the encoder mapping distinct sequences to one vector; the
        # other is two dimensions not being enough to keep distinct vectors
        # apart. Conflating them would blame the wrong thing.
        encoder = PositionalOneHotEncoder(max_length=len(REF))
        embeddings, labels = variants(150, encoder)
        p = project(embeddings, labels, REF)
        self.assertEqual(p.distinct_positions, len(embeddings))
        self.assertLessEqual(p.distinct_projected, p.distinct_positions)
        if p.distinct_projected < len(p.points):
            self.assertTrue(any("distinct spots" in w for w in p.warnings))

    def test_a_weak_projection_warns_before_it_is_read(self):
        # A single-substitution cloud in one-hot space has no dominant
        # direction, so the two components carry little. The picture is still
        # drawn; what must not happen is drawing it silently.
        encoder = PositionalOneHotEncoder(max_length=len(REF))
        embeddings, labels = variants(120, encoder)
        p = project(embeddings, labels, REF)
        self.assertLess(p.cumulative_explained, 0.5)
        self.assertTrue(any("of the variation" in w for w in p.warnings))

    def test_the_projection_is_stable_between_runs(self):
        embeddings, labels = variants(40)
        first = project(embeddings, labels, REF)
        second = project(embeddings, labels, REF)
        for a, b in zip(first.points, second.points):
            self.assertAlmostEqual(a.x, b.x)
            self.assertAlmostEqual(a.y, b.y)

    def test_the_sign_convention_is_fixed(self):
        # An SVD's singular vectors are sign-arbitrary. Left alone, the plot
        # mirrors itself between runs on the same data and looks like a
        # different result.
        embeddings, labels = variants(40)
        p = project(embeddings, labels, REF)
        for axis in ("x", "y"):
            values = [getattr(pt, axis) for pt in p.points]
            extreme = max(values, key=abs)
            self.assertGreater(extreme, 0, f"{axis} axis is not sign-anchored")


class TestRefusals(unittest.TestCase):

    def test_two_spaces_are_never_mixed(self):
        composition = DeterministicEncoder(max_length=len(REF))
        positional = PositionalOneHotEncoder(max_length=len(REF))
        embeddings = [composition.encode(REF), positional.encode(REF),
                      composition.encode("A" + REF[1:]), positional.encode("A" + REF[1:])]
        with self.assertRaises(ProjectionError) as cm:
            project(embeddings, list("abcd"), REF)
        self.assertIn("unrelated spaces", str(cm.exception))

    def test_too_few_points_is_refused_not_drawn(self):
        embeddings, labels = variants(MIN_POINTS - 2)
        self.assertLess(len(embeddings), MIN_POINTS)
        with self.assertRaises(ProjectionError) as cm:
            project(embeddings, labels, REF)
        self.assertIn("too few", str(cm.exception))

    def test_a_set_with_no_variation_is_refused(self):
        encoder = DeterministicEncoder(max_length=len(REF))
        embeddings = [encoder.encode(REF) for _ in range(MIN_POINTS + 2)]
        with self.assertRaises(ProjectionError) as cm:
            project(embeddings, [str(i) for i in range(len(embeddings))], REF)
        self.assertIn("no variation", str(cm.exception))

    def test_labels_must_match_the_embeddings(self):
        embeddings, labels = variants(10)
        with self.assertRaises(ProjectionError):
            project(embeddings, labels[:-1], REF)


class TestTheEncodersStaySeparate(unittest.TestCase):

    def test_composition_cannot_see_where_a_substitution_happened(self):
        encoder = DeterministicEncoder(max_length=len(REF))
        # The same swap (S->W) made at two different positions.
        first_s = REF.index("S")
        second_s = REF.index("S", first_s + 1)
        a = REF[:first_s] + "W" + REF[first_s + 1:]
        b = REF[:second_s] + "W" + REF[second_s + 1:]
        first, second = encoder.encode(a), encoder.encode(b)
        self.assertNotEqual(a, b)
        self.assertEqual(first.pooled, second.pooled,
                         "composition is order-blind; this is a limitation to state, "
                         "not one to hide")

    def test_the_positional_encoder_does_see_it(self):
        encoder = PositionalOneHotEncoder(max_length=len(REF))
        first_s = REF.index("S")
        second_s = REF.index("S", first_s + 1)
        a = REF[:first_s] + "W" + REF[first_s + 1:]
        b = REF[:second_s] + "W" + REF[second_s + 1:]
        self.assertNotEqual(encoder.encode(a).pooled, encoder.encode(b).pooled)

    def test_the_two_deterministic_encoders_are_not_interchangeable(self):
        composition = DeterministicEncoder(max_length=len(REF)).encode(REF)
        positional = PositionalOneHotEncoder(max_length=len(REF)).encode(REF)
        self.assertFalse(composition.comparable_with(positional))
        self.assertNotEqual(composition.model, positional.model)

    def test_neither_deterministic_encoder_claims_learned_content(self):
        for encoder in (DeterministicEncoder(), PositionalOneHotEncoder()):
            self.assertFalse(encoder.kind.has_learned_content)

    def test_the_pretrained_encoder_still_refuses_rather_than_falling_back(self):
        with self.assertRaises(EncoderUnavailable):
            ESM2Encoder().encode(REF)


if __name__ == "__main__":
    unittest.main()
