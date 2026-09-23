"""
The semi-empirical engine, wired up for real.  [Addendum 2 section 6, step 6d]

Step 6d stubbed the parameterization pipeline because no engine was installed.
One now is: GFN2-xTB, through the `xtb` Python bindings, driven by ASE's BFGS
optimiser. This module is the boundary between "we have a method" and "we have
parameters", and the whole point of it is that those are not the same thing.

WHAT GFN2-xTB IS, AND WHAT IT IS NOT

It is a real electronic-structure method: a tight-binding approximation to DFT,
parameterized across the periodic table, that produces energies, gradients and
optimised geometries good enough to be worth having. Running it is not a
simulation of running something better.

It is not the level of theory the pipeline's charge stage requires. RESP charges
for a GAFF2/AMBER-compatible residue are conventionally fitted to an HF/6-31G*
electrostatic potential, and that convention is not arbitrary: HF/6-31G*
overpolarises by roughly the amount a fixed-charge force field needs to mimic
condensed-phase polarisation. Charges fitted to a GFN2-xTB potential would be
better-converged and *wrong for the force field they are going into*, which is
worse than having none.

So a GFN2-xTB geometry is recorded here as what it is -- a pre-optimisation --
and it does not tick off the pipeline's geometry stage. It shortens the stage
that follows, because a high-level optimiser started from an xtb geometry
converges in far fewer steps than one started from a force-field guess. That is
a real contribution and it is a smaller one than completing the stage.

The consequence is deliberate and is the reason this module exists in this
shape: **running everything here does not put a residue in the registry.** The
registry still requires the full pipeline, and one stage run at a lower level of
theory than specified is not one stage done.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

def ev_per_hartree() -> float:
    """
    The eV-to-Hartree conversion, taken from ASE rather than written down here.

    ASE reports energies in eV; the QM literature and every level-of-theory
    comparison in this project are in Hartree, so the conversion happens once,
    at the boundary.

    Sourced rather than inlined for two reasons. The engine/policy boundary
    forbids a bare numeric literal in engine code, and this one would be a real
    false positive -- a unit conversion is not a tunable coefficient. But it is
    also not exact by SI definition the way the constants in `constants.py`
    are: the Hartree depends on measured quantities and its CODATA value has
    uncertainty in the last digits. Copying it here would freeze one CODATA
    revision into this file; importing it tracks whatever revision the library
    doing the arithmetic is using, which is the one that matters for the
    numbers actually being converted.
    """
    from ase.units import Hartree
    return float(Hartree)


class EngineUnavailable(RuntimeError):
    """Raised when a calculation is requested and its engine is not installed."""


class LevelOfTheory(Enum):
    """
    Methods this project can distinguish between, ordered by what they license.

    An enum rather than a string because the difference between GFN2-xTB and
    HF/6-31G* decides whether a result may become a RESP charge, and a free-text
    method field makes that decision by string comparison somewhere downstream.
    """
    MMFF94 = "MMFF94"
    GFN_FF = "GFN-FF"
    GFN2_XTB = "GFN2-xTB"
    HF_631Gd = "HF/6-31G*"
    B3LYP_631Gdp = "B3LYP/6-31G**"

    @property
    def is_quantum(self) -> bool:
        """Whether the method solves an electronic-structure problem at all."""
        return self not in (LevelOfTheory.MMFF94, LevelOfTheory.GFN_FF)

    @property
    def licenses_resp_charges(self) -> bool:
        """
        Whether an ESP from this method may be fitted to RESP charges for a
        GAFF2/AMBER-compatible residue.

        Only HF/6-31G*. Not because better methods are worse, but because the
        force field's non-bonded terms were fitted against HF/6-31G*'s
        overpolarisation, and a charge set that does not share it is
        inconsistent with the Lennard-Jones parameters it will be used beside.
        """
        return self is LevelOfTheory.HF_631Gd

    @property
    def role(self) -> str:
        if self.licenses_resp_charges:
            return "the charge-derivation level this pipeline requires"
        if self.is_quantum:
            return "a pre-optimisation level: real electronic structure, wrong reference for RESP"
        return "a force-field level: useful for cleaning a guessed geometry, nothing more"


@dataclass(frozen=True)
class OptimisedGeometry:
    """
    A converged geometry and everything needed to judge it.

    `converged` is carried separately from the coordinates because an
    unconverged optimisation still returns coordinates, and they look exactly
    like converged ones.
    """
    symbols: Tuple[str, ...]
    positions_angstrom: Tuple[Tuple[float, float, float], ...]
    energy_hartree: float
    level: LevelOfTheory
    converged: bool
    steps: int
    max_force_ev_per_angstrom: float
    force_threshold_ev_per_angstrom: float
    starting_energy_hartree: float
    note: str = ""

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    @property
    def relaxation_hartree(self) -> float:
        return self.energy_hartree - self.starting_energy_hartree

    @property
    def caveat(self) -> str:
        """
        What this geometry may and may not be used for.

        A property of the result rather than something the optimiser writes
        into it, so an OptimisedGeometry built anywhere -- a test, a cached
        artifact, a reconstruction from JSON -- carries the same warning.
        """
        if not self.converged:
            return (f"Did not reach {self.force_threshold_ev_per_angstrom} eV/A in "
                    f"{self.steps} steps. The coordinates are whatever the optimiser last "
                    f"held and must not be used as a geometry.")
        if self.level.licenses_resp_charges:
            return "Converged at the charge-derivation level; usable for the ESP stage."
        return (f"Converged at {self.level.value}, which is {self.level.role}. A better "
                f"starting point for the charge-derivation level than a force-field "
                f"guess; not a substitute for one.")

    @property
    def formula(self) -> str:
        counts: Dict[str, int] = {}
        for symbol in self.symbols:
            counts[symbol] = counts.get(symbol, 0) + 1
        return "".join(f"{s}{counts[s] if counts[s] > 1 else ''}"
                       for s in sorted(counts, key=lambda s: (s != "C", s != "H", s)))

    def describe(self) -> str:
        state = "converged" if self.converged else "DID NOT CONVERGE"
        return (f"{self.formula} ({self.n_atoms} atoms) {state} at {self.level.value} in "
                f"{self.steps} steps; E = {self.energy_hartree:.6f} Eh, relaxation "
                f"{self.relaxation_hartree * 1000:.2f} mEh, residual force "
                f"{self.max_force_ev_per_angstrom:.4f} eV/A "
                f"(threshold {self.force_threshold_ev_per_angstrom}). {self.caveat}")


def engine_status() -> Dict[str, object]:
    """
    Which calculation engines are actually importable, and what each licenses.

    Reported rather than assumed: the whole module is a statement about what can
    be computed here, and that answer changes with the environment.
    """
    status: Dict[str, object] = {}

    def probe(name: str, importer, licenses: str):
        try:
            importer()
        except Exception as exc:                       # pragma: no cover - env dependent
            status[name] = {"available": False, "reason": str(exc)[:200],
                            "licenses": licenses}
        else:
            status[name] = {"available": True, "reason": "", "licenses": licenses}

    probe("rdkit", lambda: __import__("rdkit.Chem", fromlist=["Chem"]),
          "building a 3D structure from SMILES")
    probe("ase", lambda: __import__("ase.optimize", fromlist=["BFGS"]),
          "driving a geometry optimisation")
    probe("xtb", lambda: __import__("xtb.ase.calculator", fromlist=["XTB"]),
          "GFN2-xTB energies, gradients and pre-optimised geometries")
    probe("psi4", lambda: __import__("psi4"),
          "HF/6-31G* ESP, which is what RESP charges require")
    return status


def require(engine: str) -> None:
    status = engine_status()[engine]
    if not status["available"]:                        # type: ignore[index]
        raise EngineUnavailable(
            f"{engine} is not installed, so {status['licenses']} cannot be done here. "  # type: ignore[index]
            f"Import failed with: {status['reason']}"  # type: ignore[index]
        )


def build_from_smiles(smiles: str, seed: int = 0xF00D
                      ) -> Tuple[List[str], List[Tuple[float, float, float]]]:
    """
    A 3D structure from SMILES, cleaned with MMFF94 before anything quantum.

    The seed is fixed and reported: ETKDG is stochastic, and an embedding that
    cannot be reproduced makes every energy downstream unreproducible too.
    """
    require("rdkit")
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse the SMILES {smiles!r}.")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        raise ValueError(f"RDKit could not embed a 3D conformer for {smiles!r}.")
    AllChem.MMFFOptimizeMolecule(mol)

    conformer = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = [tuple(conformer.GetAtomPosition(i))       # type: ignore[misc]
                 for i in range(mol.GetNumAtoms())]
    return symbols, positions


def optimise(
    symbols: Sequence[str],
    positions: Sequence[Sequence[float]],
    level: LevelOfTheory = LevelOfTheory.GFN2_XTB,
    fmax: float = 0.02,
    max_steps: int = 500,
) -> OptimisedGeometry:
    """
    Relax a structure and report whether it actually converged.

    `fmax` is a force threshold in eV/A and it travels with the result, because
    "optimised" without one means nothing: the same molecule at 0.05 and at
    0.005 are different claims about the same coordinates.
    """
    if level is not LevelOfTheory.GFN2_XTB:
        raise EngineUnavailable(
            f"{level.value} is not wired up here. Only {LevelOfTheory.GFN2_XTB.value} has an "
            f"engine installed, and it is {LevelOfTheory.GFN2_XTB.role}."
        )
    require("ase")
    require("xtb")

    import numpy as np
    from ase import Atoms
    from ase.optimize import BFGS
    from xtb.ase.calculator import XTB

    atoms = Atoms(symbols=list(symbols), positions=[list(p) for p in positions])
    atoms.calc = XTB(method="GFN2-xTB")
    per_hartree = ev_per_hartree()
    starting = atoms.get_potential_energy() / per_hartree

    optimiser = BFGS(atoms, logfile=None)
    converged = bool(optimiser.run(fmax=fmax, steps=max_steps))
    forces = atoms.get_forces()
    max_force = float(np.max(np.linalg.norm(forces, axis=1)))

    return OptimisedGeometry(
        symbols=tuple(atoms.get_chemical_symbols()),
        positions_angstrom=tuple(tuple(float(x) for x in row)     # type: ignore[misc]
                                 for row in atoms.get_positions()),
        energy_hartree=float(atoms.get_potential_energy() / per_hartree),
        level=level,
        converged=converged,
        steps=int(optimiser.get_number_of_steps()),
        max_force_ev_per_angstrom=max_force,
        force_threshold_ev_per_angstrom=fmax,
        starting_energy_hartree=float(starting),
    )


# ---------------------------------------------------------------------------
# The out-of-process half
# ---------------------------------------------------------------------------

QM_PYTHON_ENV_VAR = "PEPTIDE_SUITE_QM_PYTHON"


def qm_interpreter() -> Optional[str]:
    """
    The interpreter that has Psi4, if one has been pointed at.

    Psi4 is conda-only and the analysis engine must stay pip-installable, so the
    HF/6-31G* stages run in a separate interpreter and this is how it is found.
    Returned as None rather than guessed: a path assembled by convention would
    eventually find *an* interpreter and run the wrong one.
    """
    import os
    return os.environ.get(QM_PYTHON_ENV_VAR) or None


def qm_stage_status() -> Dict[str, object]:
    """Whether the out-of-process stages can run, and what to set if not."""
    interpreter = qm_interpreter()
    if not interpreter:
        return {
            "available": False,
            "reason": (f"{QM_PYTHON_ENV_VAR} is not set. The HF/6-31G* stages need an "
                       f"interpreter with Psi4 and the resp package; point this variable "
                       f"at one."),
        }
    import subprocess
    try:
        proof = subprocess.run(
            [interpreter, "-c", "import psi4, resp; print(psi4.__version__)"],
            capture_output=True, text=True, timeout=180)
    except Exception as exc:                            # pragma: no cover - env dependent
        return {"available": False, "reason": f"{interpreter} could not be run: {exc}"}
    if proof.returncode != 0:
        return {"available": False,
                "reason": f"{interpreter} has no working Psi4/resp: {proof.stderr.strip()[:200]}"}
    return {"available": True, "reason": "", "interpreter": interpreter,
            "psi4_version": proof.stdout.strip()}
