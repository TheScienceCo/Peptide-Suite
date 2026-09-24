"""
Write a parameterization record from a QM artifact.  [Addendum 2 step 6d]

Deliberately a separate, explicit act. Running a calculation never writes to the
registry: a record is a claim that work was done, and a claim should be made by
someone invoking a tool, not as a side effect of a computation returning without
an exception.

The tool records only stages whose artifact is actually present, and it refuses
to mark a stage complete on the strength of a stage before it. It will happily
write a record with three of six stages done -- that record makes the residue
IN_PROGRESS, which licenses nothing.

Usage:  python record_parameterization.py qm_artifact.json [--apply]
"""

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[2] / "peptide_suite" / "data" / "parameterized_residues.json"

# Which pipeline stage each artifact key evidences. A stage is recorded only if
# its own artifact is here; nothing is inferred from a neighbouring stage.
STAGE_FOR_ARTIFACT = {
    "geometry_optimization": "geometry_optimization",
    "electrostatic_potential": "electrostatic_potential",
    "resp_charges": "resp_charges",
}

# The level of theory a stage's artifact must have been produced at to count.
# A GFN2-xTB geometry is a real calculation and does not satisfy the geometry
# stage, because the stage exists to feed an HF/6-31G* charge derivation.
REQUIRED_LEVEL = {
    "geometry_optimization": "HF/6-31G*",
    "electrostatic_potential": "HF/6-31G*",
}


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build_record(artifact: dict) -> dict:
    stages = artifact.get("stages", {})
    completed, rejected = [], []

    for key, stage in STAGE_FOR_ARTIFACT.items():
        payload = stages.get(key)
        if not payload:
            continue
        required = REQUIRED_LEVEL.get(stage)
        actual = payload.get("level_of_theory")
        if required and actual != required:
            rejected.append(
                f"{stage}: ran at {actual}, which is not {required}. A calculation at the "
                f"wrong level of theory for what the stage feeds is not that stage."
            )
            continue
        if payload.get("converged") is False:
            rejected.append(f"{stage}: did not converge, so its output is not a result.")
            continue
        completed.append(stage)

    resp = stages.get("resp_charges", {})
    geometry = stages.get("geometry_optimization", {})

    return {
        "force_field": "GAFF2-compatible (charges only; no bonded terms fitted yet)",
        "stages_completed": completed,
        "stages_rejected": rejected,
        "geometry_method": geometry.get("level_of_theory", ""),
        "geometry_energy_hartree": geometry.get("energy_hartree"),
        "geometry_started_from": artifact.get("started_from", ""),
        "esp_method": (f"{stages.get('electrostatic_potential', {}).get('level_of_theory', '')}"
                       f" on a {stages.get('electrostatic_potential', {}).get('surface', '')}"),
        "charge_model": (f"{resp.get('model', '')}, a={resp.get('restraint_a_stage1')}"
                         f"/{resp.get('restraint_a_stage2')}, b={resp.get('restraint_b')}"),
        "charges": resp.get("charges"),
        "charge_sum": resp.get("charge_sum"),
        "esp_rrms": resp.get("esp_rrms"),
        "n_grid_points": resp.get("n_grid_points"),
        "symbols": artifact.get("symbols"),
        "smiles": artifact.get("smiles"),
        "known_deficiencies": resp.get("known_deficiencies", []),
        "torsion_scans": "",
        "experimental_validation": "",
        "validated_against": (
            "Nothing. The charges reproduce an HF/6-31G* electrostatic potential, which is "
            "what they were fitted to. No comparison against experimental conformational "
            "data has been made, and until one is these parameters describe a calculation "
            "rather than a molecule."),
        "parameterized_by": "Peptide-Suite, tools/qm/run_qm_stages.py",
        "psi4_version": artifact.get("psi4_version", ""),
        "date": date.today().isoformat(),
        "git_commit": git_commit(),
        "files": "artifact recorded inline; no force-field files produced yet",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    parser.add_argument("--apply", action="store_true",
                        help="write the record. Without it, print and change nothing.")
    args = parser.parse_args()

    artifact = json.loads(Path(args.artifact).read_text())
    residue = artifact["residue"]
    record = build_record(artifact)

    print(f"{residue}: {len(record['stages_completed'])} stage(s) recorded as complete")
    for stage in record["stages_completed"]:
        print(f"   accepted  {stage}")
    for reason in record["stages_rejected"]:
        print(f"   REJECTED  {reason}")
    if not args.apply:
        print("\nDry run. Pass --apply to write it.")
        return

    data = json.loads(REGISTRY.read_text())
    data["residues"][residue] = record
    REGISTRY.write_text(json.dumps(data, indent=2) + "\n")
    print(f"\nwrote {REGISTRY}")


if __name__ == "__main__":
    main()
