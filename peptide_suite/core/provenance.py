"""
Provenance tier schema and licensing enforcement.  [Addendum 2, step 6a]

No biophysical quantity enters this system as a bare number. Every value carries
a provenance tier and the metadata that tier requires, and the tier determines
what the value is permitted to do.

The licensing table is the anti-overfitting mechanism, so it is enforced here in
code rather than described in documentation. A violation raises; it does not
warn. The reason is that the failure mode being prevented is silent: a QM
descriptor quietly reaching a learner produces a model that looks excellent on
retrospective holdouts for exactly the wrong reason, and nothing in the output
would reveal it.

PUBLIC / PRIVATE BOUNDARY (per Addendum 1, engine/policy separation):
    public   — tier definitions, required metadata, the licensing table,
               capability checks, feature-family names, gate thresholds,
               the aggregate-term registry's NAMES, parameter-budget accounting
    private  — the numeric weight attached to any feature family or aggregate
               term, which is supplied by the policy pack through
               `PolicyPack`, never hardcoded here

This module deliberately contains no weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------

class ProvenanceTier(Enum):
    """Where a number came from. Ordering here is documentation, not precedence."""
    MEASURED = "MEASURED"
    MEASURED_STRUCTURE = "MEASURED_STRUCTURE"
    PREDICTED_STRUCTURE = "PREDICTED_STRUCTURE"
    CLASSICAL_SIM = "CLASSICAL_SIM"
    SEMIEMPIRICAL = "SEMIEMPIRICAL"
    QM = "QM"
    LITERATURE_ASSERTED = "LITERATURE_ASSERTED"
    HEURISTIC = "HEURISTIC"


# Metadata each tier must carry. A quantity missing any of these cannot be
# constructed — the point is that an unreproducible number is not admissible,
# and the cheapest place to enforce that is at the boundary.
REQUIRED_METADATA: Dict[ProvenanceTier, FrozenSet[str]] = {
    ProvenanceTier.MEASURED: frozenset({
        "assay_type", "readout", "construct", "species",
        "temperature_c", "buffer", "citation",
    }),
    ProvenanceTier.MEASURED_STRUCTURE: frozenset({
        "pdb_id", "method", "resolution_angstrom",
        "peptide_resolved", "peptide_b_factor",
    }),
    ProvenanceTier.PREDICTED_STRUCTURE: frozenset({
        "model", "model_version", "plddt", "pae", "iptm", "seed_count",
    }),
    ProvenanceTier.CLASSICAL_SIM: frozenset({
        "force_field", "water_model", "sampling_time_ns", "replicate_count",
    }),
    ProvenanceTier.SEMIEMPIRICAL: frozenset({
        "method", "solvation_model", "ensemble_size", "energy_window_kcal",
    }),
    ProvenanceTier.QM: frozenset({
        "level_of_theory", "solvation_model", "truncation_scheme",
    }),
    ProvenanceTier.LITERATURE_ASSERTED: frozenset({
        "citation", "claim_string",
    }),
    ProvenanceTier.HEURISTIC: frozenset({
        "formula_reference",
    }),
}

# Readouts admissible as a regression target. A mixed pile of Kd, IC50 and EC50
# is not one quantity, so the readout is recorded and checked rather than assumed.
VALID_READOUTS: FrozenSet[str] = frozenset({"Kd", "Ki", "IC50", "EC50", "pKd", "pKi", "pIC50", "pEC50"})


# ---------------------------------------------------------------------------
# Capabilities and the licensing table
# ---------------------------------------------------------------------------

class Capability(Enum):
    """What a number is permitted to do."""
    REGRESSION_TARGET = "regression_target"      # may be fitted against
    SCORE_WEIGHTED = "score_weighted"            # may carry a learned weight directly
    SCORE_AGGREGATE_ONLY = "score_aggregate_only"  # may enter scoring only via a declared aggregate
    LEARNER_RAW_INPUT = "learner_raw_input"      # raw per-atom/per-residue values may reach a learner
    GATE = "gate"                                # may block or allow a proposal
    EXPLAIN = "explain"                          # may appear in a rationale


LICENSING: Dict[ProvenanceTier, FrozenSet[Capability]] = {
    # Experimental measurement: the only thing that may be fitted against.
    ProvenanceTier.MEASURED: frozenset({
        Capability.REGRESSION_TARGET, Capability.SCORE_WEIGHTED,
        Capability.LEARNER_RAW_INPUT, Capability.GATE, Capability.EXPLAIN,
    }),
    # An experimental structure is a measurement of geometry, not of affinity,
    # so it informs features but is not itself a target.
    ProvenanceTier.MEASURED_STRUCTURE: frozenset({
        Capability.SCORE_WEIGHTED, Capability.LEARNER_RAW_INPUT,
        Capability.GATE, Capability.EXPLAIN,
    }),
    # Everything here is additionally conditional on the confidence gate; see
    # `check_capability`, which refuses all use when the gate has not passed.
    ProvenanceTier.PREDICTED_STRUCTURE: frozenset({
        Capability.SCORE_WEIGHTED, Capability.LEARNER_RAW_INPUT,
        Capability.GATE, Capability.EXPLAIN,
    }),
    ProvenanceTier.CLASSICAL_SIM: frozenset({
        Capability.SCORE_WEIGHTED, Capability.LEARNER_RAW_INPUT,
        Capability.GATE, Capability.EXPLAIN,
    }),
    # QM and semiempirical reach the scoring function only through a bounded,
    # named set of aggregated terms. Raw per-atom output may not be fed to a
    # learner: thousands of candidate descriptors against a few hundred measured
    # affinities is how a scorer memorises the holdout.
    ProvenanceTier.SEMIEMPIRICAL: frozenset({
        Capability.SCORE_AGGREGATE_ONLY, Capability.GATE, Capability.EXPLAIN,
    }),
    ProvenanceTier.QM: frozenset({
        Capability.SCORE_AGGREGATE_ONLY, Capability.GATE, Capability.EXPLAIN,
    }),
    # A claim from a paper may gate and may explain; it carries no number of
    # ours, so it may never be numerically weighted.
    ProvenanceTier.LITERATURE_ASSERTED: frozenset({
        Capability.GATE, Capability.EXPLAIN,
    }),
    ProvenanceTier.HEURISTIC: frozenset({
        Capability.SCORE_WEIGHTED, Capability.LEARNER_RAW_INPUT,
        Capability.GATE, Capability.EXPLAIN,
    }),
}


class LicenseViolation(Exception):
    """Raised when a quantity is used in a way its provenance tier forbids."""


class UnresolvedValue(Exception):
    """Raised when an UNRESOLVED quantity is used as a number."""


# ---------------------------------------------------------------------------
# UNRESOLVED
# ---------------------------------------------------------------------------

class _Unresolved:
    """
    The value of a quantity that could not be determined.

    Distinct from None and from 0: "not determined" and "determined to be zero"
    are different statements, and a gate that failed must not surface as a small
    number. Arithmetic on it raises rather than propagating silently.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNRESOLVED"

    def __bool__(self) -> bool:
        return False

    def _refuse(self, *_args, **_kwargs):
        raise UnresolvedValue(
            "UNRESOLVED cannot be used as a number. A quantity that could not be "
            "determined must be reported as unresolved, not substituted with a value."
        )

    __add__ = __radd__ = __sub__ = __rsub__ = _refuse
    __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _refuse
    __float__ = __int__ = __lt__ = __le__ = __gt__ = __ge__ = _refuse


UNRESOLVED = _Unresolved()


# ---------------------------------------------------------------------------
# Confidence gates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StructureGate:
    """
    Confidence thresholds a predicted complex must clear before any feature may
    be computed on it.

    Thresholds are public (this is engine, not policy). They are deliberately
    conservative: the cost of refusing a usable structure is a missing feature,
    while the cost of accepting an unusable one is a confidently wrong number
    with no downstream signal that anything went wrong.
    """
    min_plddt: float = 70.0
    max_pae: float = 5.0          # angstrom, interface-restricted
    min_iptm: float = 0.6
    min_seed_count: int = 3

    def evaluate(self, metadata: Dict[str, Any]) -> "GateResult":
        failures: List[str] = []

        plddt = metadata.get("plddt")
        if plddt is None or plddt < self.min_plddt:
            failures.append(f"pLDDT {plddt} < {self.min_plddt}")

        pae = metadata.get("pae")
        if pae is None or pae > self.max_pae:
            failures.append(f"interface PAE {pae} > {self.max_pae}")

        iptm = metadata.get("iptm")
        if iptm is None or iptm < self.min_iptm:
            failures.append(f"ipTM {iptm} < {self.min_iptm}")

        seeds = metadata.get("seed_count")
        if seeds is None or seeds < self.min_seed_count:
            failures.append(f"seed count {seeds} < {self.min_seed_count}")

        return GateResult(passed=not failures, failures=failures)


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: List[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        return "; ".join(self.failures)


DEFAULT_STRUCTURE_GATE = StructureGate()


# ---------------------------------------------------------------------------
# Quantity
# ---------------------------------------------------------------------------

@dataclass
class Quantity:
    """
    A biophysical value inseparable from where it came from.

    Construct through the tier-specific classmethods rather than directly: they
    enforce the metadata contract, and a quantity that cannot satisfy its tier's
    contract should fail to exist rather than exist unlabelled.
    """
    name: str
    value: Any                       # float, or UNRESOLVED
    tier: ProvenanceTier
    metadata: Dict[str, Any] = field(default_factory=dict)
    units: str = ""
    feature_family: str = ""
    gate: Optional[GateResult] = None

    def __post_init__(self):
        missing = REQUIRED_METADATA[self.tier] - set(self.metadata)
        if missing:
            raise ValueError(
                f"{self.tier.value} quantity '{self.name}' is missing required metadata: "
                f"{', '.join(sorted(missing))}. This tier cannot be recorded without it, "
                f"because the value would not be reproducible or auditable."
            )

        if self.tier is ProvenanceTier.MEASURED:
            readout = self.metadata.get("readout")
            if readout not in VALID_READOUTS:
                raise ValueError(
                    f"MEASURED quantity '{self.name}' declares readout '{readout}', which is not "
                    f"one of {sorted(VALID_READOUTS)}. Mixed readouts are not a single quantity "
                    f"and must not be pooled."
                )

        if self.tier is ProvenanceTier.PREDICTED_STRUCTURE:
            if self.gate is None:
                self.gate = DEFAULT_STRUCTURE_GATE.evaluate(self.metadata)
            # Checked whether or not the gate was supplied: a pre-evaluated
            # failing gate must void the value just as an unevaluated one does.
            # A feature computed on a structure that failed its gate is not a
            # low-confidence number, it is not a number.
            if not self.gate.passed:
                self.value = UNRESOLVED
                logger.info(
                    f"'{self.name}': structure gate failed ({self.gate.reason}); "
                    f"value set to UNRESOLVED"
                )

    @property
    def resolved(self) -> bool:
        return self.value is not UNRESOLVED

    @property
    def capabilities(self) -> FrozenSet[Capability]:
        if self.tier is ProvenanceTier.PREDICTED_STRUCTURE and (
            self.gate is None or not self.gate.passed
        ):
            return frozenset()
        return LICENSING[self.tier]

    def permits(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def require(self, capability: Capability) -> "Quantity":
        """Assert a capability, raising with the reason if the tier forbids it."""
        check_capability(self, capability)
        return self

    def as_float(self) -> float:
        if not self.resolved:
            raise UnresolvedValue(
                f"'{self.name}' is UNRESOLVED"
                + (f" ({self.gate.reason})" if self.gate and self.gate.failures else "")
            )
        return float(self.value)

    # -- tier-specific constructors -----------------------------------------

    @classmethod
    def measured(cls, name: str, value: float, *, assay_type: str, readout: str,
                 construct: str, species: str, temperature_c: float, buffer: str,
                 citation: str, units: str = "", feature_family: str = "") -> "Quantity":
        return cls(name=name, value=value, tier=ProvenanceTier.MEASURED, units=units,
                   feature_family=feature_family,
                   metadata={"assay_type": assay_type, "readout": readout,
                             "construct": construct, "species": species,
                             "temperature_c": temperature_c, "buffer": buffer,
                             "citation": citation})

    @classmethod
    def measured_structure(cls, name: str, value: Any, *, pdb_id: str, method: str,
                           resolution_angstrom: float, peptide_resolved: bool,
                           peptide_b_factor: Optional[float], units: str = "",
                           feature_family: str = "") -> "Quantity":
        return cls(name=name, value=value, tier=ProvenanceTier.MEASURED_STRUCTURE,
                   units=units, feature_family=feature_family,
                   metadata={"pdb_id": pdb_id, "method": method,
                             "resolution_angstrom": resolution_angstrom,
                             "peptide_resolved": peptide_resolved,
                             "peptide_b_factor": peptide_b_factor})

    @classmethod
    def predicted_structure(cls, name: str, value: Any, *, model: str, model_version: str,
                            plddt: float, pae: float, iptm: float, seed_count: int,
                            units: str = "", feature_family: str = "",
                            gate: Optional[StructureGate] = None) -> "Quantity":
        metadata = {"model": model, "model_version": model_version, "plddt": plddt,
                    "pae": pae, "iptm": iptm, "seed_count": seed_count}
        evaluated = (gate or DEFAULT_STRUCTURE_GATE).evaluate(metadata)
        return cls(name=name, value=value, tier=ProvenanceTier.PREDICTED_STRUCTURE,
                   units=units, feature_family=feature_family,
                   metadata=metadata, gate=evaluated)

    @classmethod
    def classical_sim(cls, name: str, value: Any, *, force_field: str, water_model: str,
                      sampling_time_ns: float, replicate_count: int, units: str = "",
                      feature_family: str = "") -> "Quantity":
        return cls(name=name, value=value, tier=ProvenanceTier.CLASSICAL_SIM,
                   units=units, feature_family=feature_family,
                   metadata={"force_field": force_field, "water_model": water_model,
                             "sampling_time_ns": sampling_time_ns,
                             "replicate_count": replicate_count})

    @classmethod
    def semiempirical(cls, name: str, value: Any, *, method: str, solvation_model: str,
                      ensemble_size: int, energy_window_kcal: float, units: str = "",
                      feature_family: str = "") -> "Quantity":
        return cls(name=name, value=value, tier=ProvenanceTier.SEMIEMPIRICAL,
                   units=units, feature_family=feature_family,
                   metadata={"method": method, "solvation_model": solvation_model,
                             "ensemble_size": ensemble_size,
                             "energy_window_kcal": energy_window_kcal})

    @classmethod
    def qm(cls, name: str, value: Any, *, level_of_theory: str, solvation_model: str,
           truncation_scheme: str, units: str = "", feature_family: str = "") -> "Quantity":
        return cls(name=name, value=value, tier=ProvenanceTier.QM, units=units,
                   feature_family=feature_family,
                   metadata={"level_of_theory": level_of_theory,
                             "solvation_model": solvation_model,
                             "truncation_scheme": truncation_scheme})

    @classmethod
    def literature_asserted(cls, name: str, claim_string: str, *, citation: str,
                            feature_family: str = "") -> "Quantity":
        # Carries no number of ours by construction.
        return cls(name=name, value=UNRESOLVED, tier=ProvenanceTier.LITERATURE_ASSERTED,
                   feature_family=feature_family,
                   metadata={"citation": citation, "claim_string": claim_string})

    @classmethod
    def heuristic(cls, name: str, value: float, *, formula_reference: str,
                  units: str = "", feature_family: str = "") -> "Quantity":
        return cls(name=name, value=value, tier=ProvenanceTier.HEURISTIC, units=units,
                   feature_family=feature_family,
                   metadata={"formula_reference": formula_reference})


def check_capability(quantity: Quantity, capability: Capability) -> None:
    """Raise LicenseViolation if this quantity may not do this."""
    if capability in quantity.capabilities:
        return

    if quantity.tier is ProvenanceTier.PREDICTED_STRUCTURE and (
        quantity.gate is None or not quantity.gate.passed
    ):
        raise LicenseViolation(
            f"'{quantity.name}' is PREDICTED_STRUCTURE whose confidence gate did not pass "
            f"({quantity.gate.reason if quantity.gate else 'gate not evaluated'}). No feature "
            f"may be computed on it and no capability is granted. Emit UNRESOLVED instead."
        )

    granted = ", ".join(sorted(c.value for c in quantity.capabilities)) or "none"
    raise LicenseViolation(
        f"'{quantity.name}' is {quantity.tier.value} and may not be used for "
        f"'{capability.value}'. Permitted: {granted}. "
        + _LICENSE_RATIONALE.get((quantity.tier, capability), "")
    )


_LICENSE_RATIONALE = {
    (ProvenanceTier.QM, Capability.REGRESSION_TARGET):
        "Only MEASURED may be a regression target; fitting to computed values fits the method.",
    (ProvenanceTier.QM, Capability.SCORE_WEIGHTED):
        "QM enters scoring only through a declared aggregate term, never as a directly weighted feature.",
    (ProvenanceTier.QM, Capability.LEARNER_RAW_INPUT):
        "Raw per-atom QM output may not reach a learner: the descriptor count dwarfs the measured data.",
    (ProvenanceTier.SEMIEMPIRICAL, Capability.SCORE_WEIGHTED):
        "Semiempirical enters scoring only through a declared aggregate term.",
    (ProvenanceTier.SEMIEMPIRICAL, Capability.LEARNER_RAW_INPUT):
        "Raw semiempirical output may not reach a learner, for the same reason as QM.",
    (ProvenanceTier.LITERATURE_ASSERTED, Capability.SCORE_WEIGHTED):
        "A literature claim carries no number of ours and may gate or explain, never be weighted.",
    (ProvenanceTier.LITERATURE_ASSERTED, Capability.REGRESSION_TARGET):
        "A claim is not a measurement; only MEASURED may be a regression target.",
    (ProvenanceTier.MEASURED_STRUCTURE, Capability.REGRESSION_TARGET):
        "A structure measures geometry, not affinity; only MEASURED may be a regression target.",
    (ProvenanceTier.HEURISTIC, Capability.REGRESSION_TARGET):
        "A rule-based descriptor is not a measurement.",
    (ProvenanceTier.CLASSICAL_SIM, Capability.REGRESSION_TARGET):
        "Simulation output is not a measurement; fitting to it fits the force field.",
    (ProvenanceTier.PREDICTED_STRUCTURE, Capability.REGRESSION_TARGET):
        "A predicted structure is not a measurement.",
}


# ---------------------------------------------------------------------------
# Aggregate terms: the only route by which QM/SEMIEMPIRICAL reach scoring
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AggregateTerm:
    """
    A named, bounded reduction of QM or semiempirical output.

    The NAME and the reduction are public engine; the WEIGHT attached to the
    term lives in the policy pack. The bound matters as much as the name: an
    unbounded aggregate over per-contact energies reintroduces the dimensionality
    the licensing rule exists to prevent.
    """
    name: str
    source_tiers: FrozenSet[ProvenanceTier]
    reduction: str              # e.g. "sum", "mean", "max_abs", "count_above_threshold"
    max_inputs: int             # hard bound on how many raw values may be reduced
    description: str = ""

    def validate_inputs(self, quantities: Iterable[Quantity]) -> None:
        quantities = list(quantities)
        if len(quantities) > self.max_inputs:
            raise LicenseViolation(
                f"Aggregate term '{self.name}' accepts at most {self.max_inputs} inputs; "
                f"{len(quantities)} were supplied. The bound is the mechanism that keeps a "
                f"per-contact decomposition from becoming a high-dimensional feature set."
            )
        for q in quantities:
            if q.tier not in self.source_tiers:
                raise LicenseViolation(
                    f"Aggregate term '{self.name}' accepts "
                    f"{sorted(t.value for t in self.source_tiers)}; "
                    f"'{q.name}' is {q.tier.value}."
                )


class AggregateTermRegistry:
    """
    The declared set of aggregate terms. Anything not registered here cannot
    carry QM or semiempirical signal into the scoring function.
    """

    def __init__(self):
        self._terms: Dict[str, AggregateTerm] = {}

    def declare(self, term: AggregateTerm) -> AggregateTerm:
        if term.name in self._terms:
            raise ValueError(f"Aggregate term '{term.name}' is already declared")
        self._terms[term.name] = term
        return term

    def get(self, name: str) -> AggregateTerm:
        if name not in self._terms:
            raise LicenseViolation(
                f"'{name}' is not a declared aggregate term. QM and semiempirical values may "
                f"enter scoring only through the declared set: "
                f"{sorted(self._terms) or '(none declared)'}."
            )
        return self._terms[name]

    @property
    def names(self) -> List[str]:
        return sorted(self._terms)

    def __len__(self) -> int:
        return len(self._terms)


# ---------------------------------------------------------------------------
# Parameter budget
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReceptorComplex:
    """
    A receptor identity that is complete enough to pool measurements under.

    A receptor gene alone is underspecified where accessory proteins determine
    pharmacology: CLR with RAMP1 is the CGRP receptor and CLR with RAMP2 is AM1,
    same gene, different ligand preference. Affinities measured against
    different accessory complexes are not the same quantity, so the identifier
    carries the accessory subunits and two complexes that differ in them will
    not compare equal.
    """
    receptor: str
    accessory: Tuple[str, ...] = ()
    species: str = ""

    @property
    def identifier(self) -> str:
        parts = [self.receptor]
        if self.accessory:
            parts.append("+".join(sorted(self.accessory)))
        if self.species:
            parts.append(f"({self.species})")
        return "".join(p if i == 0 else f"+{p}" if not p.startswith("(") else p
                       for i, p in enumerate(parts))

    def poolable_with(self, other: "ReceptorComplex") -> bool:
        return (self.receptor == other.receptor
                and set(self.accessory) == set(other.accessory))

    def __str__(self) -> str:
        return self.identifier


@dataclass
class ParameterBudget:
    """
    Free parameters must not exceed N_measured / divisor.

    The budget is computed over the POOLED measurement set rather than per
    class: the policy is one artifact with one parameter count, so counting it
    against a single class would understate the data required.

    Pooling is only legitimate across complexes that are the same quantity.
    `add_pool` refuses to merge measurements from complexes that differ in
    accessory subunits, since those are different pharmacology recorded under a
    shared receptor name.
    """
    divisor: float
    pools: Dict[str, int] = field(default_factory=dict)
    free_parameters: int = 0
    _complexes: Dict[str, "ReceptorComplex"] = field(default_factory=dict)

    def add_pool(self, complex_: "ReceptorComplex", n_measured: int) -> "ParameterBudget":
        key = complex_.identifier
        for existing_key, existing in self._complexes.items():
            if existing_key == key:
                continue
            if existing.receptor == complex_.receptor and not existing.poolable_with(complex_):
                raise LicenseViolation(
                    f"Refusing to pool '{key}' with '{existing_key}': same receptor, different "
                    f"accessory subunits. These are different pharmacology and their affinities "
                    f"are not the same quantity."
                )
        self._complexes[key] = complex_
        self.pools[key] = self.pools.get(key, 0) + n_measured
        return self

    @property
    def n_measured(self) -> int:
        return sum(self.pools.values())

    @property
    def budget(self) -> float:
        return self.n_measured / self.divisor

    @property
    def within_budget(self) -> bool:
        return self.free_parameters <= self.budget

    @property
    def utilisation(self) -> float:
        return self.free_parameters / self.budget if self.budget else float("inf")

    def require(self) -> "ParameterBudget":
        if not self.within_budget:
            raise LicenseViolation(
                f"Parameter budget exceeded: {self.free_parameters} free parameters against "
                f"{self.n_measured} pooled measured values across {len(self.pools)} complex(es) "
                f"(budget {self.budget:.1f}). Adding a feature family requires removing one or "
                f"acquiring data."
            )
        return self

    def report_line(self) -> str:
        status = "WITHIN BUDGET" if self.within_budget else "OVER BUDGET"
        pools = ", ".join(f"{k}={v}" for k, v in sorted(self.pools.items())) or "none"
        return (
            f"PARAMETER BUDGET: {self.free_parameters} free parameters / {self.n_measured} "
            f"pooled measured values = budget {self.budget:.1f}, "
            f"utilisation {self.utilisation:.0%} — {status}\n"
            f"  pools: {pools}"
        )


# ---------------------------------------------------------------------------
# Policy pack interface (weights live behind this; see Addendum 1)
# ---------------------------------------------------------------------------

class PolicyPack:
    """
    Interface to the private scoring policy.

    This engine holds feature-family names, licensing and gates. The numeric
    weights attached to those families are supplied by the policy pack and are
    never hardcoded here. The base class raises so that an unloaded policy fails
    loudly rather than scoring with implicit zeros.

    NOTE: the concrete loading and protection mechanism is specified in
    Addendum 1, which I have not seen. This is a placeholder interface shaped to
    the separation Addendum 2 describes; confirm it against Addendum 1 before
    anything depends on its exact signature.
    """

    def weight_for(self, feature_family: str) -> float:
        raise NotImplementedError(
            "No policy pack is loaded. Weights are private policy and are not part of "
            "this engine; load a policy pack before scoring."
        )

    def declared_aggregate_terms(self) -> List[str]:
        raise NotImplementedError("No policy pack is loaded.")


# ---------------------------------------------------------------------------
# Traceability
# ---------------------------------------------------------------------------

@dataclass
class ClaimTrace:
    """
    The provenance of every number behind one output claim.

    Required by the rule that any output claim must be traceable to the
    provenance tier of every number that produced it.
    """
    claim: str
    quantities: List[Quantity] = field(default_factory=list)

    @property
    def tiers(self) -> Set[ProvenanceTier]:
        return {q.tier for q in self.quantities}

    @property
    def weakest_tier(self) -> Optional[ProvenanceTier]:
        """
        The tier that limits what this claim can assert.

        Ordered by what each tier licenses rather than by any notion of quality:
        a claim is only as assertable as its least-licensed input.
        """
        order = [
            ProvenanceTier.MEASURED,
            ProvenanceTier.MEASURED_STRUCTURE,
            ProvenanceTier.CLASSICAL_SIM,
            ProvenanceTier.HEURISTIC,
            ProvenanceTier.PREDICTED_STRUCTURE,
            ProvenanceTier.SEMIEMPIRICAL,
            ProvenanceTier.QM,
            ProvenanceTier.LITERATURE_ASSERTED,
        ]
        present = [t for t in order if t in self.tiers]
        return present[-1] if present else None

    @property
    def has_unresolved(self) -> bool:
        return any(not q.resolved for q in self.quantities)

    def render(self) -> str:
        if not self.quantities:
            return f"{self.claim} [no supporting quantities recorded]"
        parts = [f"{q.name}={q.value if q.resolved else 'UNRESOLVED'} ({q.tier.value})"
                 for q in self.quantities]
        suffix = " — contains UNRESOLVED inputs" if self.has_unresolved else ""
        return f"{self.claim} :: " + "; ".join(parts) + suffix


def licensing_table_markdown() -> str:
    """Render the licensing table, for the eval report header and the README."""
    caps = list(Capability)
    lines = ["| Tier | " + " | ".join(c.value for c in caps) + " |",
             "|---|" + "---|" * len(caps)]
    for tier in ProvenanceTier:
        row = ["yes" if c in LICENSING[tier] else "—" for c in caps]
        lines.append(f"| {tier.value} | " + " | ".join(row) + " |")
    lines.append("")
    lines.append("PREDICTED_STRUCTURE grants none of the above until its confidence gate passes.")
    return "\n".join(lines)
