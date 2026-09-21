"""
Declared policy keys.  [Addendum 1 section 2]

Key names are public; their values are not and live only in a policy artifact.
This module is the engine's statement of what it requires a policy pack to
supply. It contains no values.

Entries carry a semantic type, units and a valid range so a pack can be
validated on load. They deliberately carry no description: the schema must not
reveal why a family exists or what magnitude to expect from it.
"""

from typing import Dict, NamedTuple, Tuple


class KeySpec(NamedTuple):
    semantic_type: str
    units: str
    valid_range: Tuple[float, float]


UNIT = (0.0, 1.0)
SIGNED = (-1.0, 1.0)


# Weighted feature families.
FEATURE_FAMILIES: Dict[str, KeySpec] = {
    "evidence.tier_direct_experimental":   KeySpec("bounded_unit", "dimensionless", UNIT),
    "evidence.tier_homolog_experimental":  KeySpec("bounded_unit", "dimensionless", UNIT),
    "evidence.tier_biochemical_principle": KeySpec("bounded_unit", "dimensionless", UNIT),
    "evidence.tier_inference_only":        KeySpec("bounded_unit", "dimensionless", UNIT),

    "objective.potency":                   KeySpec("bounded_unit", "dimensionless", UNIT),
    "objective.functional_selectivity":    KeySpec("bounded_unit", "dimensionless", UNIT),
    "objective.proteolytic_half_life":     KeySpec("bounded_unit", "dimensionless", UNIT),
    "objective.albumin_fcrn":              KeySpec("bounded_unit", "dimensionless", UNIT),
    "objective.aggregation":               KeySpec("bounded_unit", "dimensionless", UNIT),
    "objective.solubility":                KeySpec("bounded_unit", "dimensionless", UNIT),
    "objective.immunogenicity":            KeySpec("bounded_unit", "dimensionless", UNIT),
    "objective.synthesizability":          KeySpec("bounded_unit", "dimensionless", UNIT),

    "discovery.literature_component":      KeySpec("bounded_unit", "dimensionless", UNIT),
    "discovery.expression_component":      KeySpec("bounded_unit", "dimensionless", UNIT),
}


# Thresholds and cutoffs.
THRESHOLDS: Dict[str, KeySpec] = {
    "confidence.high_cutoff":                     KeySpec("bounded_unit", "dimensionless", UNIT),
    "confidence.medium_cutoff":                   KeySpec("bounded_unit", "dimensionless", UNIT),
    "confidence.material_contribution_cutoff":    KeySpec("bounded_unit", "dimensionless", UNIT),

    "structure_gate.min_plddt":                   KeySpec("dimensionless", "pLDDT", (0.0, 100.0)),
    "structure_gate.max_interface_pae":           KeySpec("angstrom", "angstrom", (0.0, 35.0)),
    "structure_gate.min_iptm":                    KeySpec("bounded_unit", "dimensionless", UNIT),
    "structure_gate.min_seed_count":              KeySpec("count", "seeds", (1.0, 100.0)),

    "parameter_budget.measured_ratio_divisor":    KeySpec("count", "dimensionless", (1.0, 1000.0)),

    "conservation.conserved_entropy_cutoff":      KeySpec("dimensionless", "bits", (0.0, 4.5)),
    "conservation.variable_entropy_cutoff":       KeySpec("dimensionless", "bits", (0.0, 4.5)),
    "conservation.tolerance_sigmoid_steepness":   KeySpec("dimensionless", "dimensionless", (0.0, 20.0)),
    "conservation.tolerance_sigmoid_inflection":  KeySpec("dimensionless", "bits", (0.0, 4.5)),
    "conservation.min_distinct_sequences":        KeySpec("count", "sequences", (2.0, 1000.0)),

    "identification.min_containment_length":      KeySpec("count", "residues", (3.0, 50.0)),
    "identification.peptide_length_ceiling":      KeySpec("count", "residues", (10.0, 1000.0)),
    "identification.exact_match_confidence":      KeySpec("bounded_unit", "dimensionless", UNIT),
    "identification.containment_confidence":      KeySpec("bounded_unit", "dimensionless", UNIT),
    "identification.similarity_floor":            KeySpec("bounded_unit", "dimensionless", UNIT),

    "preorganization.propensity_to_helicity_scale": KeySpec("dimensionless", "fraction_per_kcal", (0.0, 5.0)),

    "chemistry.reference_ph":                     KeySpec("ph", "pH", (0.0, 14.0)),
    "chemistry.interstitial_ph":                  KeySpec("ph", "pH", (0.0, 14.0)),
    "chemistry.endosomal_ph":                     KeySpec("ph", "pH", (0.0, 14.0)),
    "chemistry.burial_pka_shift":                 KeySpec("ph", "pH units", (-5.0, 5.0)),

    "class_b1.restricted_zone_residues":          KeySpec("count", "residues", (1.0, 30.0)),

    "conformer.max_ensemble_length":              KeySpec("count", "residues", (5.0, 100.0)),
    "nbo.proline_rich_fraction":                  KeySpec("fraction", "fraction", (0.0, 1.0)),
}


def required_keys() -> Dict[str, Dict[str, KeySpec]]:
    return {"feature_families": dict(FEATURE_FAMILIES), "thresholds": dict(THRESHOLDS)}


def free_parameter_count() -> int:
    """
    Parameters the policy is permitted to tune. Logged against the measured-data
    budget on every eval run.
    """
    return len(FEATURE_FAMILIES) + len(THRESHOLDS)
