"""
The semi-empirical engine, and the line it must not cross.
[Addendum 2 section 6, step 6d]

An engine is installed now, so the interesting tests are no longer "does it
raise". They are: does running a calculation successfully still fail to earn a
residue its place in the registry, and does the level of theory travel with
every number that depends on it.

These tests do not run a calculation. A geometry optimisation is tens of
seconds and the suite is 1.2 seconds; the calculation is exercised by
`tools/qm/` and its artifact is checked in `test_ncaa_registry.py`. What is
checked here is the accounting around it, which is where the mistakes live.
"""

import unittest

from peptide_suite.core.qm_engine import (
    EngineUnavailable,
    LevelOfTheory,
    OptimisedGeometry,
    engine_status,
    qm_stage_status,
)


def geometry(level=LevelOfTheory.GFN2_XTB, converged=True):
    return OptimisedGeometry(
        symbols=("C", "H", "H", "H", "H"),
        positions_angstrom=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)),
        energy_hartree=-40.5,
        level=level,
        converged=converged,
        steps=12,
        max_force_ev_per_angstrom=0.01,
        force_threshold_ev_per_angstrom=0.02,
        starting_energy_hartree=-40.4,
    )


class TestLevelOfTheoryDecidesWhatIsLicensed(unittest.TestCase):
    """
    The distinction the module exists for: a real electronic-structure
    calculation is not automatically the right one.
    """

    def test_only_hf_631gd_licenses_resp_charges(self):
        licensing = [l for l in LevelOfTheory if l.licenses_resp_charges]
        self.assertEqual(licensing, [LevelOfTheory.HF_631Gd])

    def test_gfn2_xtb_is_quantum_and_still_does_not_license_charges(self):
        # Both halves matter. Calling it "not QM" would understate it; calling
        # it sufficient would be the error.
        self.assertTrue(LevelOfTheory.GFN2_XTB.is_quantum)
        self.assertFalse(LevelOfTheory.GFN2_XTB.licenses_resp_charges)

    def test_a_better_method_is_not_automatically_an_acceptable_one(self):
        # B3LYP/6-31G** is a better method than HF/6-31G* and is still wrong
        # here: the force field's Lennard-Jones terms were fitted against
        # HF/6-31G*'s overpolarisation.
        self.assertTrue(LevelOfTheory.B3LYP_631Gdp.is_quantum)
        self.assertFalse(LevelOfTheory.B3LYP_631Gdp.licenses_resp_charges)

    def test_force_field_levels_are_not_quantum(self):
        self.assertFalse(LevelOfTheory.MMFF94.is_quantum)
        self.assertFalse(LevelOfTheory.GFN_FF.is_quantum)

    def test_every_level_says_what_it_is_for(self):
        for level in LevelOfTheory:
            self.assertTrue(level.role.strip())


class TestAGeometryCarriesItsOwnCaveats(unittest.TestCase):

    def test_convergence_is_separate_from_the_coordinates(self):
        # An unconverged optimisation returns coordinates that look exactly
        # like converged ones, so the flag is what distinguishes them.
        unconverged = geometry(converged=False)
        self.assertFalse(unconverged.converged)
        self.assertEqual(len(unconverged.positions_angstrom), 5)
        self.assertIn("must not be used", unconverged.caveat)

    def test_the_threshold_travels_with_the_result(self):
        # "Optimised" without a force threshold is not a claim.
        self.assertEqual(geometry().force_threshold_ev_per_angstrom, 0.02)

    def test_a_converged_pre_optimisation_still_says_it_is_not_a_substitute(self):
        self.assertIn("not a substitute", geometry().caveat)
        self.assertIn("usable for the ESP stage",
                      geometry(level=LevelOfTheory.HF_631Gd).caveat)

    def test_the_description_names_the_level_and_the_state(self):
        described = geometry().describe()
        self.assertIn("GFN2-xTB", described)
        self.assertIn("converged", described)
        self.assertIn("CH4", described)

    def test_an_unconverged_geometry_says_so_loudly(self):
        self.assertIn("DID NOT CONVERGE", geometry(converged=False).describe())

    def test_relaxation_is_reported_against_the_starting_point(self):
        self.assertAlmostEqual(geometry().relaxation_hartree, -0.1)

    def test_the_hartree_conversion_is_sourced_not_transcribed(self):
        """
        A unit conversion is not a tunable coefficient, so the boundary check
        flagging it was a false positive -- but copying the CODATA value here
        would freeze one revision of it into this file while the library doing
        the arithmetic uses another. It is imported from ASE instead, and the
        module must contain no transcribed copy.
        """
        import inspect
        from peptide_suite.core import qm_engine
        source = inspect.getsource(qm_engine)
        self.assertIn("from ase.units import Hartree", source)
        self.assertNotIn("27.21", source, "the conversion was transcribed after all")


class TestRefusals(unittest.TestCase):

    def test_a_level_with_no_engine_is_refused_by_name(self):
        from peptide_suite.core.qm_engine import optimise
        with self.assertRaises(EngineUnavailable) as caught:
            optimise(["C"], [[0, 0, 0]], level=LevelOfTheory.HF_631Gd)
        self.assertIn("HF/6-31G*", str(caught.exception))

    def test_engine_status_reports_what_each_engine_licenses(self):
        status = engine_status()
        for name in ("rdkit", "ase", "xtb", "psi4"):
            self.assertIn(name, status)
            self.assertTrue(status[name]["licenses"])
            if not status[name]["available"]:
                self.assertTrue(status[name]["reason"],
                                f"{name} is unavailable and gives no reason")

    def test_the_out_of_process_interpreter_is_never_guessed(self):
        # A path assembled by convention would eventually find an interpreter
        # and run the wrong one.
        import os
        saved = os.environ.pop("PEPTIDE_SUITE_QM_PYTHON", None)
        try:
            status = qm_stage_status()
            self.assertFalse(status["available"])
            self.assertIn("PEPTIDE_SUITE_QM_PYTHON", status["reason"])
        finally:
            if saved is not None:
                os.environ["PEPTIDE_SUITE_QM_PYTHON"] = saved


if __name__ == "__main__":
    unittest.main()
