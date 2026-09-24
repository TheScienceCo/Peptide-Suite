"""
The charge-derivation half of the parameterization pipeline.
[Addendum 2 section 6, step 6d]

Runs in a QM environment (Psi4 + resp), which is a different interpreter from
the one the analysis pipeline uses -- Psi4 is conda-only and the engine must
stay installable with pip. So this is a script invoked across a process
boundary rather than an import, and it speaks JSON in both directions.

Stages covered, all at the level of theory the pipeline actually specifies:

  GEOMETRY_OPTIMIZATION   HF/6-31G*, started from the GFN2-xTB geometry
  ESP                     HF/6-31G* on a Connolly surface, 4 shells
  RESP_CHARGES            two-stage restrained fit, methyls equivalenced

HF/6-31G* is not chosen because it is the best available method -- Psi4 here
would happily run B3LYP or MP2. It is chosen because the force field these
charges are going into had its Lennard-Jones terms fitted against HF/6-31G*'s
overpolarisation, and a better-converged charge set from a better method is
inconsistent with them. Matching the reference matters more than the method's
own accuracy, which is a thing that is easy to get backwards.

Usage:  python run_qm_stages.py input_geometry.json output_result.json
"""

import json
import sys
import time

import numpy as np
import psi4
import resp

# The conventional RESP settings: four Connolly shells, the standard two-stage
# restraint weights. Written out rather than left to the library's defaults so
# the artifact can state what was actually used.
ESP_OPTIONS = {
    "VDW_SCALE_FACTORS": [1.4, 1.6, 1.8, 2.0],
    "VDW_POINT_DENSITY": 1.0,
    "RESP_A": 0.0005,        # stage 1 hyperbolic restraint
    "RESP_B": 0.1,           # restraint tightness
    "BASIS_ESP": "6-31G*",
    "METHOD_ESP": "scf",
}
STAGE2_A = 0.001             # stage 2 restrains harder, on the equivalenced set


ANGSTROM_PER_BOHR = 0.52917721092


def grid_point_count(grid_file):
    return int(sum(1 for line in open(grid_file) if line.strip()))


def esp_rrms(charges, grid_file, esp_file):
    """
    How well the fitted charges reproduce the QM potential they were fitted to.

    The number a RESP fit is judged by, and the one a reader needs in order to
    know whether the charges describe the molecule or merely satisfy the
    restraints. Recomputed from the written grid rather than trusted to an
    index into the fitter's output.
    """
    grid = np.loadtxt(grid_file)                       # Angstrom
    v_qm = np.loadtxt(esp_file)                        # atomic units
    if grid.ndim == 1:
        grid = grid.reshape(1, 3)
    # The fitted potential at each grid point, in atomic units: distances must
    # therefore be in Bohr, while the grid file is written in Angstrom.
    coords = np.array(CURRENT_GEOMETRY) / ANGSTROM_PER_BOHR
    points = grid / ANGSTROM_PER_BOHR
    d = np.linalg.norm(points[:, None, :] - coords[None, :, :], axis=2)
    v_fit = (np.asarray(charges)[None, :] / d).sum(axis=1)
    return float(np.sqrt(((v_qm - v_fit) ** 2).sum() / (v_qm ** 2).sum()))


CURRENT_GEOMETRY = []


def psi4_molecule(symbols, positions, charge=0, multiplicity=1):
    body = "\n".join(f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}"
                     for s, p in zip(symbols, positions))
    return psi4.geometry(f"{charge} {multiplicity}\n{body}\nno_reorient\nno_com\n")


def deficiencies(symbols, charges):
    """
    What this fit is known to get wrong, stated with the fit rather than left
    for a reader to notice.

    A single-conformer RESP sees chemically equivalent groups in different
    electrostatic environments and gives them different charges. For a residue
    whose two substituents are identical by constitution -- Aib's two Cbeta
    methyls -- that difference is an artifact of the conformer, not chemistry,
    and the standard remedy is a multi-conformer fit with explicit equivalencing.
    """
    found = []
    carbons = [i for i, s in enumerate(symbols) if s == "C"]
    spread = max(charges[i] for i in carbons) - min(charges[i] for i in carbons)
    found.append(
        f"Single-conformer fit. Chemically equivalent groups are not constrained to equal "
        f"charges across substituents, only within each methyl, so constitutionally "
        f"identical substituents can differ. Carbon charges span {spread:.3f} e. A "
        f"multi-conformer fit with explicit inter-group equivalencing is the standard "
        f"remedy and has not been run."
    )
    return found


def main(in_path, out_path):
    spec = json.load(open(in_path))
    symbols, positions = spec["symbols"], spec["positions"]

    psi4.set_memory(spec.get("memory", "6 GB"))
    psi4.set_num_threads(int(spec.get("threads", 4)))
    psi4.core.set_output_file(out_path + ".psi4.log", False)

    result = {
        "residue": spec["residue"],
        "smiles": spec["smiles"],
        "symbols": symbols,
        "formal_charge": spec.get("charge", 0),
        "level_of_theory": "HF/6-31G*",
        "psi4_version": psi4.__version__,
        "started_from": spec.get("started_from", "unspecified"),
        "stages": {},
    }

    # ---- geometry optimisation ------------------------------------------
    mol = psi4_molecule(symbols, positions, spec.get("charge", 0))
    started = time.time()
    energy, wfn = psi4.optimize("scf/6-31G*", molecule=mol, return_wfn=True)
    optimised = np.array(mol.geometry()) * psi4.constants.bohr2angstroms
    result["stages"]["geometry_optimization"] = {
        "level_of_theory": "HF/6-31G*",
        "energy_hartree": float(energy),
        "seconds": round(time.time() - started, 1),
        "n_basis_functions": int(wfn.basisset().nbf()),
        "positions_angstrom": [[float(x) for x in row] for row in optimised],
        "converged": True,      # psi4.optimize raises rather than returning unconverged
    }

    # ---- ESP + two-stage RESP -------------------------------------------
    # Rebuilt at the optimised geometry: fitting charges to an ESP computed on
    # a different geometry from the one the charges describe is a quiet way to
    # get a plausible, wrong answer.
    global CURRENT_GEOMETRY
    CURRENT_GEOMETRY = optimised.tolist()
    fitted = psi4_molecule(symbols, optimised.tolist(), spec.get("charge", 0))
    started = time.time()

    options = dict(ESP_OPTIONS)
    stage1 = resp.resp([fitted], options)

    stage2_options = dict(options)
    stage2_options["RESP_A"] = STAGE2_A
    stage2_options["grid"] = ["1_default_grid.dat"]
    stage2_options["esp"] = ["1_default_grid_esp.dat"]
    resp.set_stage2_constraint(fitted, stage1[1], stage2_options)
    stage2 = resp.resp([fitted], stage2_options)

    charges = [float(q) for q in stage2[1]]

    # Fit quality, computed rather than read off the library's return value.
    # The first version of this script reported `stage1[0][1]` as an RRMS; that
    # index is the ESP charge of the second atom. It looked like a plausible
    # residual (0.86) and was a carbonyl carbon's unrestrained charge -- the
    # exact failure this project exists to prevent, committed in its own tool.
    rrms = esp_rrms(charges, "1_default_grid.dat", "1_default_grid_esp.dat")
    result["stages"]["electrostatic_potential"] = {
        "level_of_theory": "HF/6-31G*",
        "surface": "Connolly, shells at "
                   + ", ".join(str(f) for f in ESP_OPTIONS["VDW_SCALE_FACTORS"])
                   + " x vdW radius",
        "point_density_per_A2": ESP_OPTIONS["VDW_POINT_DENSITY"],
        "seconds": round(time.time() - started, 1),
    }
    result["stages"]["resp_charges"] = {
        "model": "two-stage RESP",
        "restraint_a_stage1": ESP_OPTIONS["RESP_A"],
        "restraint_a_stage2": STAGE2_A,
        "restraint_b": ESP_OPTIONS["RESP_B"],
        "charges": charges,
        "charge_sum": round(sum(charges), 8),
        "esp_rrms": rrms,
        "esp_rrms_note": (
            "Relative RMS of the fitted potential against the QM potential over the grid: "
            "sqrt(sum (V_qm - V_fit)^2 / sum V_qm^2). Computed here from the grid and ESP "
            "files, not taken from the fitter's return value."),
        "n_grid_points": grid_point_count("1_default_grid.dat"),
        "equivalencing": "methyl and methylene hydrogens equivalenced in stage 2",
        "known_deficiencies": deficiencies(symbols, charges),
    }

    json.dump(result, open(out_path, "w"), indent=1)
    print(f"wrote {out_path}")
    print(f"  HF/6-31G* E = {energy:.8f} Eh")
    print(f"  RESP charges sum to {sum(charges):+.6f} (formal {spec.get('charge', 0)})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
