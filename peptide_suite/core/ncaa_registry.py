"""
Non-canonical residue parameterization — a hard gate.
[Addendum 2 section 6, build step 6d]

Aib, Nle, Cha, N-methyls, beta-amino acids, D-residues, staples, fatty-acid
linkers, gamma-Glu and OEG spacers: none of these exist in ff19SB, CHARMM36m or
any standard library. A force field asked to simulate one either refuses or
silently substitutes something else, and the second failure mode is the
dangerous one.

So: no proposal involving an unparameterized residue may be scored, simulated,
or ranked. The proposal engine draws only from the registry. A proposal for a
residue outside it is emitted as a research request carrying the cost of
parameterizing it, never as a recommendation.

The registry is empty. That is its correct current state, not an omission: this
project has parameterized nothing, so every non-canonical proposal is a research
request today. Adding an entry is a claim that the work was done.

This is the honest justification for the QM arm, and it belongs in the README as
such. For non-canonical chemistry QM is not an enhancement that sharpens an
existing number. It is the entry ticket to having a number at all.

The gate here is independent of the structure-template gate in
structure_template.py. A proposal can have an excellent bound structure and
still be unrankable because its chemistry has no parameters, and the two
refusals have different remedies: one needs a structure, the other needs weeks
of QM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

_DATA = Path(__file__).parent.parent / "data"


class ParameterizationStage(Enum):
    """
    The pipeline, in order. Every stage is required; none is optional.

    The ordering is a dependency chain rather than a preference: charges are
    fitted to the electrostatic potential, which is computed on the optimised
    geometry, and the torsion fit is done with those charges already in place.
    Running them out of order produces parameters that are individually
    reasonable and jointly wrong.
    """
    GEOMETRY_OPTIMIZATION = "geometry_optimization"
    ESP = "electrostatic_potential"
    RESP_CHARGES = "resp_charges"
    TORSION_SCANS = "torsion_scans"
    FORCE_FIELD_FIT = "force_field_fit"
    EXPERIMENTAL_VALIDATION = "experimental_validation"

    @property
    def description(self) -> str:
        return {
            ParameterizationStage.GEOMETRY_OPTIMIZATION:
                "Optimise the capped residue geometry at a stated level of theory.",
            ParameterizationStage.ESP:
                "Compute the electrostatic potential on a grid around that geometry.",
            ParameterizationStage.RESP_CHARGES:
                "Fit restrained electrostatic-potential charges to it.",
            ParameterizationStage.TORSION_SCANS:
                "Scan every rotatable dihedral, relaxed, with the fitted charges in place.",
            ParameterizationStage.FORCE_FIELD_FIT:
                "Fit the scanned profiles to GAFF2 or OpenFF functional forms.",
            ParameterizationStage.EXPERIMENTAL_VALIDATION:
                "Compare the result against experimental conformational data for this "
                "residue. Without this stage the parameters reproduce the QM, which is "
                "not the same as reproducing the molecule.",
        }[self]


PIPELINE: tuple = tuple(ParameterizationStage)


class ResidueStatus(Enum):
    CANONICAL = "canonical"                # in the standard library already
    PARAMETERIZED = "parameterized"        # every pipeline stage done and validated
    IN_PROGRESS = "in_progress"            # some stages genuinely run, not all
    UNPARAMETERIZED = "unparameterized"    # recognised, nothing run
    UNRECOGNISED = "unrecognised"          # not even identifiable

    @property
    def permits_scoring(self) -> bool:
        return self in (ResidueStatus.CANONICAL, ResidueStatus.PARAMETERIZED)

    @property
    def describe(self) -> str:
        return {
            ResidueStatus.CANONICAL: "already in the standard library",
            ResidueStatus.PARAMETERIZED: "parameterized and validated by this project",
            ResidueStatus.IN_PROGRESS: (
                "partly parameterized: some stages have genuinely run and their artifacts "
                "are recorded, but the pipeline is incomplete, so this residue still may "
                "not be scored, simulated or ranked"),
            ResidueStatus.UNPARAMETERIZED: "recognised, with no parameterization work done",
            ResidueStatus.UNRECOGNISED: "not in the catalogue",
        }[self]


class ParameterizationError(Exception):
    """Raised when an unparameterized residue is used as if it were parameterized."""


@dataclass(frozen=True)
class CatalogueEntry:
    """What is known about a non-canonical residue, short of parameters."""
    code: str
    full_name: str
    residue_class: str
    absent_from: List[str]
    torsion_scan_driver: Optional[str]
    side_chain_rotatable_dihedrals: Optional[int]
    note: str
    parameters_transfer_from_l_enantiomer: bool = False


@dataclass
class ParameterizationCost:
    """
    What parameterizing this residue would take.

    Deliberately not a number of hours or dollars. Those depend on hardware,
    level of theory and how much of the work is already automated, so a figure
    here would be invented and would then travel as though it were estimated.
    What is stated instead is checkable: which stages remain, and what the
    expensive one scales with.
    """
    residue: str
    stages_outstanding: List[ParameterizationStage]
    scan_driver: Optional[str]
    known_dihedral_count: Optional[int]
    notes: List[str] = field(default_factory=list)

    @property
    def is_estimable(self) -> bool:
        """Whether the dominant cost can be stated at all, rather than guessed."""
        return self.scan_driver is not None

    def summary(self) -> str:
        if not self.stages_outstanding:
            return f"{self.residue} is fully parameterized; no work outstanding."
        stages = len(self.stages_outstanding)
        if self.known_dihedral_count == 0 and self.scan_driver:
            scale = (f"No side-chain dihedral to scan; the scan cost is carried by "
                     f"{self.scan_driver}.")
        elif self.known_dihedral_count:
            scale = (f"Torsion scanning covers {self.known_dihedral_count} side-chain "
                     f"dihedral(s) ({self.scan_driver}).")
        elif self.scan_driver:
            # Some drivers are written as full clauses ("scales with chain
            # length"), so the prefix is only added when it is not already there.
            driver = self.scan_driver
            scale = (driver[0].upper() + driver[1:] + "."
                     if driver.lower().startswith("scales with")
                     else f"Torsion-scan cost scales with {driver}.")
        else:
            scale = ("The torsion-scan cost cannot be stated without the specific "
                     "construct, so no estimate is given rather than a guessed one.")
        return (f"{stages} of {len(PIPELINE)} pipeline stages outstanding for "
                f"{self.residue}. {scale}")


@dataclass
class ResearchRequest:
    """
    A proposal that cannot be a recommendation.

    Emitted in place of a ranked proposal, never alongside one. It carries no
    score, because scoring it is the thing the gate forbids -- and a research
    request with a score next to it would be read as a recommendation with a
    caveat, which is the same failure wearing a different label.
    """
    residue: str
    proposal: str
    status: ResidueStatus
    cost: ParameterizationCost
    catalogue: Optional[CatalogueEntry] = None

    @property
    def rationale(self) -> str:
        """
        Why this particular residue blocks the proposal.

        Deliberately narrow. The shared framing -- that a research request is not
        a recommendation, and that the system will not manufacture a basis it
        does not have -- is stated once where the requests are presented.
        Repeating it per residue produced three near-identical paragraphs on a
        single linker and taught the reader to skip all of them.
        """
        if self.status is ResidueStatus.UNRECOGNISED:
            return (
                f"Not in the non-canonical catalogue, so the system cannot even say what "
                f"parameterizing it would involve."
            )
        if self.status is ResidueStatus.IN_PROGRESS:
            done = len(PIPELINE) - len(self.cost.stages_outstanding)
            return (
                f"Parameterization has started: {done} of {len(PIPELINE)} stages have run "
                f"and their artifacts are recorded. That is real progress and it licenses "
                f"nothing -- an incomplete pipeline yields no usable parameters, so this "
                f"is still a research request rather than a recommendation."
            )
        absent = (", ".join(self.catalogue.absent_from)
                  if self.catalogue and self.catalogue.absent_from
                  else "standard force fields")
        full_name = self.catalogue.full_name if self.catalogue else "non-canonical chemistry"
        # The code is already the heading, so the full name is only worth
        # repeating when it says something the code does not.
        name = "" if full_name.lower().startswith(self.residue.lower()) else f"{full_name}. "
        return f"{name}No parameters in {absent}; absent from the project registry."


class NCAARegistry:
    """
    The parameterized-residue registry, and the catalogue behind it.

    Two files, deliberately separate. The catalogue says what a residue is; the
    registry says whether anyone has done the work. Merging them would let
    recognition be mistaken for readiness, which is the exact confusion the gate
    exists to prevent.
    """

    _catalogue: Optional[Dict[str, CatalogueEntry]] = None
    _registry: Optional[Dict[str, dict]] = None

    @classmethod
    def catalogue(cls) -> Dict[str, CatalogueEntry]:
        if cls._catalogue is None:
            raw = json.loads((_DATA / "ncaa_catalogue.json").read_text())
            cls._catalogue = {
                code: CatalogueEntry(
                    code=code,
                    full_name=entry["full_name"],
                    residue_class=entry["class"],
                    absent_from=list(entry.get("absent_from", [])),
                    torsion_scan_driver=entry.get("torsion_scan_driver"),
                    side_chain_rotatable_dihedrals=entry.get("side_chain_rotatable_dihedrals"),
                    note=entry.get("note", ""),
                    parameters_transfer_from_l_enantiomer=entry.get(
                        "parameters_transfer_from_l_enantiomer", False),
                )
                for code, entry in raw["residues"].items()
            }
        return cls._catalogue

    @classmethod
    def registry(cls) -> Dict[str, dict]:
        if cls._registry is None:
            raw = json.loads((_DATA / "parameterized_residues.json").read_text())
            cls._registry = dict(raw.get("residues", {}))
        return cls._registry

    @classmethod
    def completed_stages(cls, residue: str) -> set:
        record = cls.registry().get(residue) or {}
        return {ParameterizationStage(s) for s in record.get("stages_completed", [])}

    @classmethod
    def usable(cls) -> Dict[str, dict]:
        """
        The residues that may actually be scored: every stage complete.

        Separate from `registry()`, which also holds partial records. A record
        exists to make progress checkable; only a complete one licenses
        anything, and conflating the two would let a half-parameterized residue
        into a ranking on the strength of having a file.
        """
        return {code: record for code, record in cls.registry().items()
                if cls.completed_stages(code) >= set(PIPELINE)}

    @classmethod
    def is_empty(cls) -> bool:
        """Whether anything at all may be scored. Partial records do not count."""
        return not cls.usable()

    @classmethod
    def status(cls, residue: str) -> ResidueStatus:
        if residue in cls.usable():
            return ResidueStatus.PARAMETERIZED
        if cls.completed_stages(residue):
            return ResidueStatus.IN_PROGRESS
        entry = cls.catalogue().get(residue)
        if entry is None:
            return ResidueStatus.UNRECOGNISED
        # A D-enantiomer of a canonical residue is the one case needing no new
        # parameters: the force field's functional form is achiral, so the terms
        # are the L values and the inversion rides on the C-alpha improper.
        if entry.parameters_transfer_from_l_enantiomer:
            return ResidueStatus.CANONICAL
        return ResidueStatus.UNPARAMETERIZED

    @classmethod
    def cost(cls, residue: str) -> ParameterizationCost:
        entry = cls.catalogue().get(residue)
        status = cls.status(residue)

        if status in (ResidueStatus.PARAMETERIZED, ResidueStatus.CANONICAL):
            return ParameterizationCost(
                residue=residue, stages_outstanding=[], scan_driver=None,
                known_dihedral_count=None,
                notes=([entry.note] if entry and entry.note else []),
            )

        completed = set()
        record = cls.registry().get(residue)
        if record:
            completed = {ParameterizationStage(s) for s in record.get("stages_completed", [])}

        notes = [entry.note] if entry and entry.note else []
        if status is ResidueStatus.UNRECOGNISED:
            notes.append(
                "Not in the catalogue, so even the stage list is a guess. Add it to "
                "ncaa_catalogue.json before requesting parameterization."
            )

        return ParameterizationCost(
            residue=residue,
            stages_outstanding=[s for s in PIPELINE if s not in completed],
            scan_driver=entry.torsion_scan_driver if entry else None,
            known_dihedral_count=entry.side_chain_rotatable_dihedrals if entry else None,
            notes=notes,
        )

    @classmethod
    def require_parameterized(cls, residue: str) -> None:
        """Raise unless this residue may be scored. The gate, as an assertion."""
        status = cls.status(residue)
        if not status.permits_scoring:
            raise ParameterizationError(
                f"{residue} is {status.value} and may not be scored, simulated or ranked. "
                f"{cls.cost(residue).summary()}"
            )

    @classmethod
    def research_request(cls, residue: str, proposal: str) -> ResearchRequest:
        return ResearchRequest(
            residue=residue,
            proposal=proposal,
            status=cls.status(residue),
            cost=cls.cost(residue),
            catalogue=cls.catalogue().get(residue),
        )


# Recognising non-canonical chemistry in a proposal's own words. Ordered so that
# the more specific pattern wins: "gamma-Glu-2xOEG" names two linkers, and
# matching only the first would understate what needs parameterizing.
_PATTERNS = [
    ("Aib", re.compile(r"\bAib\b|alpha-aminoisobutyric", re.I)),
    ("Nle", re.compile(r"\bNle\b|norleucine", re.I)),
    ("Cha", re.compile(r"\bCha\b|cyclohexylalanine", re.I)),
    ("N-methyl", re.compile(r"N-methylat|\bN-Me\b|N-methyl", re.I)),
    ("D-residue", re.compile(r"\bD-(?:Ala|Arg|Asn|Asp|Cys|Gln|Glu|His|Ile|Leu|Lys|Met|"
                             r"Phe|Pro|Ser|Thr|Trp|Tyr|Val)\b|\bD-amino acid\b|"
                             r"\bD-residue\b|\bD-enantiomer\b", re.I)),
    ("staple", re.compile(r"\bstaple[sd]?\b|stapling|ring-closing metathesis", re.I)),
    ("gamma-Glu", re.compile(r"gamma-Glu|γGlu|γ-Glu", re.I)),
    # No leading \b on OEG: it appears inside tokens like "2xOEG", where 'x' and
    # 'O' are both word characters and the boundary never matches. Caught on the
    # engine's own gamma-Glu-2xOEG proposal.
    ("OEG", re.compile(r"OEG|AEEA|oligoethylene glycol|ethylene glycol", re.I)),
    ("fatty-acid", re.compile(r"\bC1[0-9]\b|\bC[0-9] diacid\b|diacid|acylat|fatty.acid|"
                              r"lipidat|palmitoyl|myristoyl", re.I)),
    ("beta-amino-acid", re.compile(r"\bbeta-amino acid\b|β-amino acid|\bbeta-homo", re.I)),
]


def detect_noncanonical(text: str) -> List[str]:
    """
    Non-canonical chemistry named in a proposal.

    Text matching is a blunt instrument, and it is used here in the direction
    where bluntness is safe: a false positive routes a proposal to a research
    request, which is recoverable, while a false negative lets unparameterized
    chemistry be ranked, which is the failure the gate exists to prevent.
    """
    return [code for code, pattern in _PATTERNS if pattern.search(text or "")]


def gate_proposal(text: str) -> List[ResearchRequest]:
    """
    Research requests for every unparameterized residue a proposal names.

    An empty list means the proposal may be scored and ranked normally.
    """
    requests = []
    for code in detect_noncanonical(text):
        if not NCAARegistry.status(code).permits_scoring:
            requests.append(NCAARegistry.research_request(code, text))
    return requests


# ---------------------------------------------------------------------------
# The QM pipeline, stubbed
# ---------------------------------------------------------------------------

class PipelineNotImplemented(NotImplementedError):
    """
    Raised by every stage of the parameterization pipeline.

    Each stage names what it would need and what running it would produce, so
    the stub is a specification rather than a placeholder. It raises rather than
    returning a default, because a stage that returned something plausible would
    let the registry fill with parameters nobody computed -- which is the one
    failure this whole module is built to prevent.
    """


@dataclass(frozen=True)
class StageRequirement:
    """What a pipeline stage needs before it could run."""
    stage: ParameterizationStage
    inputs: List[str]
    produces: str
    external_tooling: List[str]

    def describe(self) -> str:
        return (f"{self.stage.value}: needs {', '.join(self.inputs)}; produces "
                f"{self.produces}; requires {', '.join(self.external_tooling)}.")


STAGE_REQUIREMENTS: Dict[ParameterizationStage, StageRequirement] = {
    ParameterizationStage.GEOMETRY_OPTIMIZATION: StageRequirement(
        stage=ParameterizationStage.GEOMETRY_OPTIMIZATION,
        inputs=["capped residue structure", "declared level of theory", "solvation model"],
        produces="an optimised geometry and its energy",
        external_tooling=["a QM package (Psi4, ORCA, Gaussian)"],
    ),
    ParameterizationStage.ESP: StageRequirement(
        stage=ParameterizationStage.ESP,
        inputs=["optimised geometry", "grid specification"],
        produces="the electrostatic potential sampled around the molecule",
        external_tooling=["the same QM package"],
    ),
    ParameterizationStage.RESP_CHARGES: StageRequirement(
        stage=ParameterizationStage.RESP_CHARGES,
        inputs=["ESP grid", "restraint weights", "equivalence constraints"],
        produces="per-atom partial charges summing to the formal charge",
        external_tooling=["RESP fitting (antechamber, resp, or psiRESP)"],
    ),
    ParameterizationStage.TORSION_SCANS: StageRequirement(
        stage=ParameterizationStage.TORSION_SCANS,
        inputs=["fitted charges", "the list of rotatable dihedrals", "scan spacing"],
        produces="relaxed energy profiles for each dihedral",
        external_tooling=["the QM package, driven by a scan harness (e.g. TorsionDrive)"],
    ),
    ParameterizationStage.FORCE_FIELD_FIT: StageRequirement(
        stage=ParameterizationStage.FORCE_FIELD_FIT,
        inputs=["torsion profiles", "target functional form"],
        produces="GAFF2 or OpenFF torsion parameters and the residual of the fit",
        external_tooling=["ForceBalance or an equivalent fitting tool"],
    ),
    ParameterizationStage.EXPERIMENTAL_VALIDATION: StageRequirement(
        stage=ParameterizationStage.EXPERIMENTAL_VALIDATION,
        inputs=["fitted parameters", "experimental conformational data for this residue"],
        produces="a comparison, and an explicit statement of what is not reproduced",
        external_tooling=["MD sampling, plus a literature source for the experimental data"],
    ),
}


@dataclass(frozen=True)
class StageResult:
    """
    What one stage produced, and at what level of theory.

    Returned rather than written anywhere. Running a stage never touches the
    registry: a record is a claim that work was done and validated, and it is
    made by a person invoking the recording tool, not as a side effect of a
    calculation succeeding. That separation is the whole reason a residue
    cannot acquire parameters by accident.
    """
    stage: ParameterizationStage
    residue: str
    level_of_theory: str
    satisfies_stage: bool
    artifact: Dict
    note: str

    def describe(self) -> str:
        verdict = ("satisfies the stage" if self.satisfies_stage
                   else "does NOT satisfy the stage")
        return (f"{self.stage.value} for {self.residue} at {self.level_of_theory}: "
                f"{verdict}. {self.note}")


def run_stage(stage: ParameterizationStage, residue: str, smiles: Optional[str] = None,
              **kwargs) -> "StageResult":
    """
    Run one pipeline stage, where an engine for it exists.

    GFN2-xTB is installed, so the geometry stage runs for real -- and returns a
    result that says it does not satisfy the stage, because GFN2-xTB is not the
    level of theory the charge derivation downstream requires. A pre-optimised
    geometry is a genuine contribution to the stage that follows and is not the
    stage itself, and the distinction is carried in the return value rather than
    in a comment.

    Every other stage raises and names the engine it is missing. The engines are
    checked at call time, not at import: this module is the gate and must stay
    importable in an environment with no computational chemistry installed at
    all.
    """
    requirement = STAGE_REQUIREMENTS[stage]

    if stage is ParameterizationStage.GEOMETRY_OPTIMIZATION:
        if not smiles:
            raise ValueError(
                f"Optimising {residue} needs its capped structure as SMILES. Without one "
                f"there is nothing to optimise, and guessing a structure for a residue "
                f"whose chemistry is the thing in question would be the whole error."
            )
        from .qm_engine import LevelOfTheory, build_from_smiles, optimise

        symbols, positions = build_from_smiles(smiles, seed=kwargs.get("seed", 0xF00D))
        geometry = optimise(symbols, positions,
                            fmax=kwargs.get("fmax", 0.02),
                            max_steps=kwargs.get("max_steps", 500))
        return StageResult(
            stage=stage,
            residue=residue,
            level_of_theory=geometry.level.value,
            # False deliberately, and this is the load-bearing line of the
            # module: the calculation ran, converged, and still does not tick
            # the stage off, because the stage is defined by the level of
            # theory its output has to feed.
            satisfies_stage=geometry.level.licenses_resp_charges,
            artifact={
                "level_of_theory": geometry.level.value,
                "energy_hartree": geometry.energy_hartree,
                "converged": geometry.converged,
                "steps": geometry.steps,
                "max_force_ev_per_angstrom": geometry.max_force_ev_per_angstrom,
                "force_threshold_ev_per_angstrom": geometry.force_threshold_ev_per_angstrom,
                "symbols": list(geometry.symbols),
                "positions_angstrom": [list(row) for row in geometry.positions_angstrom],
                "formula": geometry.formula,
            },
            note=(f"{geometry.describe()} {LevelOfTheory.GFN2_XTB.value} is "
                  f"{LevelOfTheory.GFN2_XTB.role}, so this is a starting point for the "
                  f"HF/6-31G* optimisation rather than a replacement for it."),
        )

    raise PipelineNotImplemented(
        f"No engine is wired up for this stage in this process. "
        f"Stage {requirement.describe()} "
        f"Psi4 is conda-only and the analysis engine must stay pip-installable, so the "
        f"HF/6-31G* stages run out of process -- see tools/qm/run_qm_stages.py. Until "
        f"every stage has run for {residue}, that residue stays out of the usable "
        f"registry and every proposal using it is a research request."
    )


def pipeline_specification() -> List[str]:
    """The pipeline as an ordered, readable list. Used by the API and the README."""
    return [f"{i}. {STAGE_REQUIREMENTS[stage].describe()}" for i, stage in enumerate(PIPELINE, 1)]
