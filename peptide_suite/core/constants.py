"""
Physical constants.

EXEMPT from the engine/policy boundary. [Addendum 1 section 1, confirmed exemption]

Everything in this module is exact by SI definition. Since the 2019 redefinition
the kilogram, ampere, kelvin and mole are fixed by assigning exact values to h,
e, k_B and N_A, so these carry zero uncertainty and cannot be tuned. A value
that cannot vary is not a coefficient, and putting it in a policy artifact would
invite someone to change something that is true by definition.

The exemption is narrow and the boundary matters more than the convenience.
What does NOT belong here, and why:

    pH 7.4              A modelling convention. Plasma pH is a 7.35-7.45 range
                        and the choice of a working value is exactly the kind of
                        decision the policy artifact exists to hold.

    Side-chain pKa      Measured model-compound values that differ between
                        sources and shift substantially in a folded context.
                        These are reference data with provenance, not constants.

    Hydrophobicity      Published scales (Kyte-Doolittle, Eisenberg). Tables
    scales              produced by a method, replaceable by another table.

    Helix geometry      ~100 degrees per residue is an idealisation; real values
                        vary with the structure being described.

If a value could reasonably be measured differently tomorrow, it is not here.
"""

from typing import Final

# --- SI defining constants (exact, 2019 redefinition) ----------------------

SPEED_OF_LIGHT_M_PER_S: Final[int] = 299_792_458
PLANCK_J_S: Final[float] = 6.626_070_15e-34
ELEMENTARY_CHARGE_C: Final[float] = 1.602_176_634e-19
BOLTZMANN_J_PER_K: Final[float] = 1.380_649e-23
AVOGADRO_PER_MOL: Final[float] = 6.022_140_76e23

# --- Exact by definition, derived from the above ---------------------------

GAS_CONSTANT_J_PER_MOL_K: Final[float] = BOLTZMANN_J_PER_K * AVOGADRO_PER_MOL
FARADAY_C_PER_MOL: Final[float] = ELEMENTARY_CHARGE_C * AVOGADRO_PER_MOL

# --- Exact unit conversions ------------------------------------------------

JOULES_PER_CALORIE: Final[float] = 4.184          # thermochemical calorie, exact
KCAL_PER_MOL_PER_J_PER_MOL: Final[float] = 1.0 / (JOULES_PER_CALORIE * 1000.0)
ZERO_CELSIUS_IN_KELVIN: Final[float] = 273.15     # exact


def rt_kcal_per_mol(temperature_c: float) -> float:
    """
    RT in kcal/mol at a given temperature.

    The temperature is an argument rather than a constant because the
    temperature an assay was run at is a property of that assay, recorded in
    its provenance metadata, not a fixed value of the system.
    """
    kelvin = temperature_c + ZERO_CELSIUS_IN_KELVIN
    return GAS_CONSTANT_J_PER_MOL_K * kelvin * KCAL_PER_MOL_PER_J_PER_MOL
