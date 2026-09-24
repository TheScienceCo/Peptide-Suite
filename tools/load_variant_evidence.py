#!/usr/bin/env python3
"""
Load curated variant records from a CSV into the evidence store.

Validation is the point. A row that is wrong in a way nobody notices is worse
than a row that is missing: it will be trained on, displayed as precedent, and
believed. So every rejection names the row, the column and what would fix it,
and `--dry-run` reports them all without writing anything.

The check worth the most is the numbering one. GLP-1 appears in the literature
in at least three numbering schemes and IGF-1 in two, and a position in the
wrong scheme lands on the wrong residue while still looking entirely plausible
-- nothing downstream reads as an error. So a row that supplies a parent
sequence must have `sequence[position - 1] == wild_type`, and is rejected when
it does not.

See docs/TRAINING_DATA.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from peptide_suite.core.biological_context import Provenance, SourceKind   # noqa: E402
from peptide_suite.core.variant_evidence import (                          # noqa: E402
    DATA_PATH, Direction, EvidenceSchemaError, Extraction, MeasuredOutcome,
    Modification, ModificationKind, OutcomeMeasure, VariantRecord,
)

REQUIRED = ("parent", "variant_name", "modification_kind")


class RowError(Exception):
    """A rejection, phrased so the person fixing the CSV knows what to change."""


def _enum(raw: str, enum, column: str):
    value = (raw or "").strip().upper()
    if not value:
        raise RowError(f"{column} is empty; one of: {', '.join(m.name for m in enum)}")
    try:
        return enum[value]
    except KeyError:
        raise RowError(f"{column}='{raw}' is not recognised; one of: "
                       f"{', '.join(m.name for m in enum)}")


def _number(raw: str, column: str):
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        raise RowError(f"{column}='{raw}' is not a number")


def _int(raw: str, column: str):
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        raise RowError(f"{column}='{raw}' is not a whole number")


def _check_numbering(row: Dict[str, str], position, wild_type: str) -> None:
    """
    The position must land on the residue the row claims is there.

    Skipped, with no complaint, when no parent sequence is given -- the check
    cannot run and refusing the row would demand a column that is optional.
    That is why the column is worth filling in.
    """
    sequence = "".join((row.get("parent_sequence") or "").split()).upper()
    if not sequence or position is None or not wild_type:
        return
    if position < 1 or position > len(sequence):
        raise RowError(
            f"position={position} is outside parent_sequence, which is "
            f"{len(sequence)} residues long")
    actual = sequence[position - 1]
    if actual != wild_type.upper():
        raise RowError(
            f"position={position} holds '{actual}' in parent_sequence, not "
            f"'{wild_type}'. This is almost always a numbering scheme mismatch -- "
            f"positions here are 1-indexed into parent_sequence, not into the "
            f"paper's numbering. Put the paper's numbering in `note`.")


def build_record(row: Dict[str, str]) -> VariantRecord:
    for column in REQUIRED:
        if not (row.get(column) or "").strip():
            raise RowError(f"{column} is required")

    kind = _enum(row.get("modification_kind", ""), ModificationKind, "modification_kind")
    position = _int(row.get("position", ""), "position")
    wild_type = (row.get("wild_type") or "").strip().upper()
    mutant = (row.get("mutant") or "").strip().upper()

    if kind.is_point_change and not (position and wild_type and mutant):
        raise RowError(
            f"modification_kind={kind.name} is a point change, so it needs position, "
            f"wild_type and mutant")
    _check_numbering(row, position, wild_type)

    description = (row.get("modification_description") or "").strip()
    if not description:
        description = (f"{wild_type}{position}{mutant}" if kind.is_point_change
                       else kind.name.replace("_", " ").lower())

    modification = Modification(kind=kind, description=description, position=position,
                                wild_type=wild_type, mutant=mutant)

    outcomes: Tuple[MeasuredOutcome, ...] = ()
    if (row.get("measure") or "").strip():
        pmid = (row.get("pmid") or "").strip()
        doi = (row.get("doi") or "").strip()
        provenance = Provenance(
            kind=SourceKind.PRIMARY_LITERATURE if (pmid or doi)
            else SourceKind.CURATED_UNVERIFIED,
            pmid=pmid, doi=doi, needs_verification=not (pmid or doi))
        try:
            outcomes = (MeasuredOutcome(
                measure=_enum(row["measure"], OutcomeMeasure, "measure"),
                direction=_enum(row.get("direction", ""), Direction, "direction"),
                comparator=(row.get("comparator") or "").strip(),
                assay=(row.get("assay") or "").strip(),
                provenance=provenance,
                fold_change=_number(row.get("fold_change", ""), "fold_change"),
                value=_number(row.get("value", ""), "value"),
                unit=(row.get("unit") or "").strip(),
                target=(row.get("target") or "").strip(),
                note=(row.get("note") or "").strip(),
                extraction=_enum(row.get("extraction", "UNRECORDED") or "UNRECORDED",
                                 Extraction, "extraction"),
            ),)
        except EvidenceSchemaError as e:
            raise RowError(str(e))

    try:
        return VariantRecord(
            name=row["variant_name"].strip(), parent=row["parent"].strip(),
            modifications=(modification,), outcomes=outcomes,
            parent_sequence="".join((row.get("parent_sequence") or "").split()).upper(),
            provenance=Provenance(kind=SourceKind.CURATED_UNVERIFIED,
                                  needs_verification=True))
    except EvidenceSchemaError as e:
        raise RowError(str(e))


def load(path: Path) -> Tuple[List[VariantRecord], List[str]]:
    records: List[VariantRecord] = []
    problems: List[str] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for n, row in enumerate(csv.DictReader(handle), start=2):   # 1 is the header
            name = (row.get("variant_name") or "?").strip()
            if "DELETE-ME" in name.upper() or "EXAMPLE" in name.upper():
                problems.append(f"row {n}: skipped the template's example row ({name})")
                continue
            try:
                records.append(build_record(row))
            except RowError as e:
                problems.append(f"row {n} ({name}): {e}")
    return records, problems


def summarise(records: List[VariantRecord]) -> str:
    with_outcome = [r for r in records if r.has_evidence]
    parents = {r.parent.strip().upper() for r in records}
    lines = [
        f"{len(records)} record(s) accepted across {len(parents)} parent molecule(s).",
        f"{len(with_outcome)} carry a measured outcome; "
        f"{len(records) - len(with_outcome)} are designs with no result.",
    ]
    quantitative = [r for r in with_outcome
                    if any(o.is_quantitative for o in r.outcomes)]
    lines.append(f"{len(quantitative)} carry a magnitude and could be regression rows.")
    if len(parents) < 4:
        lines.append(
            f"NOTE: {len(parents)} parent molecule(s) is too few to hold one out. "
            f"Splits group whole parents, so variants of one peptide cannot be "
            f"divided between train and test -- see docs/TRAINING_DATA.md.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="validate and report; write nothing")
    parser.add_argument("--append", action="store_true",
                        help="add to the existing store instead of replacing it")
    parser.add_argument("--store", type=Path, default=DATA_PATH)
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"No such file: {args.csv}", file=sys.stderr)
        return 2

    records, problems = load(args.csv)
    for problem in problems:
        print(f"REJECTED  {problem}", file=sys.stderr)
    print(summarise(records))

    if not records:
        print("Nothing to write.", file=sys.stderr)
        return 1 if problems else 0
    if args.dry_run:
        print("--dry-run: the store was not modified.")
        return 1 if problems else 0

    existing = json.loads(args.store.read_text(encoding="utf-8"))
    payload = [r.to_dict() for r in records]
    # to_dict is the display shape; the store reads its own. Re-serialise via the
    # round-trip the loader already validated rather than inventing a third form.
    written = [_storable(r) for r in records]
    existing["variants"] = (existing.get("variants", []) + written) if args.append else written
    args.store.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"Wrote {len(written)} record(s) to {args.store}.")
    return 1 if problems else 0


def _storable(record: VariantRecord) -> Dict[str, Any]:
    """The on-disk shape, which `_record_from_dict` reads back."""
    return {
        "name": record.name,
        "parent": record.parent,
        "parent_sequence": record.parent_sequence,
        "provenance": {"kind": record.provenance.kind.name,
                       "needs_verification": record.provenance.needs_verification},
        "modifications": [
            {"kind": m.kind.name, "description": m.description, "position": m.position,
             "wild_type": m.wild_type, "mutant": m.mutant}
            for m in record.modifications],
        "outcomes": [
            {"measure": o.measure.name, "direction": o.direction.name,
             "comparator": o.comparator, "assay": o.assay, "fold_change": o.fold_change,
             "value": o.value, "unit": o.unit, "target": o.target, "note": o.note,
             "extraction": o.extraction.name,
             "provenance": {"kind": o.provenance.kind.name, "pmid": o.provenance.pmid,
                            "doi": o.provenance.doi,
                            "needs_verification": o.provenance.needs_verification}}
            for o in record.outcomes],
    }


if __name__ == "__main__":
    raise SystemExit(main())
