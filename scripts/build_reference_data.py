#!/usr/bin/env python3
"""Build the application's static reference data from the checked-in CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs/aist_head_missing_face_csv/missing_face_measurements_combined.csv"
DEFAULT_OUTPUT = ROOT / "app/data/reference_data.json"
SOURCE_TITLE = "\u65e5\u672c\u4eba\u982d\u90e8\u5bf8\u6cd5\u30c7\u30fc\u30bf\u30d9\u30fc\u30b92001"
SOURCE_URL = "https://www.airc.aist.go.jp/dhrt/head/"
GROUPS = ("young_male", "young_female", "older_male", "older_female")
MEASUREMENTS = (
    (5, "interpupillary_breadth"),
    (13, "interocular_breadth"),
    (14, "biectocanthion_breadth"),
    (15, "mouth_breadth"),
)


class ReferenceDataError(ValueError):
    """Raised when the source CSV cannot safely produce the reference model."""


def _positive_decimal(value: str | None, field: str, row_number: int) -> float:
    if value is None or not value.strip():
        raise ReferenceDataError(f"row {row_number}: {field} is empty")
    try:
        number = Decimal(value.strip())
    except InvalidOperation as exc:
        raise ReferenceDataError(f"row {row_number}: {field} is not numeric: {value!r}") from exc
    if not number.is_finite() or number <= 0:
        raise ReferenceDataError(f"row {row_number}: {field} must be a finite positive number")
    result = float(number)
    if not math.isfinite(result) or result <= 0:
        raise ReferenceDataError(f"row {row_number}: {field} is outside the supported numeric range")
    return result


def build_reference_data(csv_path: Path) -> dict[str, Any]:
    measurement_keys = dict(MEASUREMENTS)
    expected_ids = set(measurement_keys)
    expected_groups = set(GROUPS)
    records: dict[tuple[int, str], dict[str, Any]] = {}

    try:
        stream = csv_path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ReferenceDataError(f"cannot read CSV {csv_path}: {exc}") from exc
    with stream:
        reader = csv.DictReader(stream)
        required = {"measurement_id", "reference_group", "unit", "n", "mean", "sd"}
        if reader.fieldnames is None:
            raise ReferenceDataError("CSV has no header")
        missing_columns = required - set(reader.fieldnames)
        if missing_columns:
            raise ReferenceDataError(f"CSV is missing required columns: {', '.join(sorted(missing_columns))}")
        for row_number, row in enumerate(reader, start=2):
            raw_id = (row.get("measurement_id") or "").strip()
            if not raw_id:
                raise ReferenceDataError(f"row {row_number}: measurement_id is empty")
            try:
                measurement_id = int(raw_id)
            except ValueError as exc:
                raise ReferenceDataError(f"row {row_number}: invalid measurement_id {raw_id!r}") from exc
            if measurement_id not in expected_ids:
                continue
            group = (row.get("reference_group") or "").strip()
            if group not in expected_groups:
                raise ReferenceDataError(f"row {row_number}: unexpected reference_group {group!r} for measurement {measurement_id}")
            key = (measurement_id, group)
            if key in records:
                raise ReferenceDataError(f"row {row_number}: duplicate measurement_id/reference_group {key}")
            unit = (row.get("unit") or "").strip()
            if unit != "mm":
                raise ReferenceDataError(f"row {row_number}: expected unit 'mm', got {unit!r}")
            raw_n = (row.get("n") or "").strip()
            if not re.fullmatch(r"[0-9]+", raw_n):
                raise ReferenceDataError(f"row {row_number}: n must be a positive integer, got {raw_n!r}")
            n = int(raw_n)
            if n <= 0:
                raise ReferenceDataError(f"row {row_number}: n must be a positive integer")
            mean = _positive_decimal(row.get("mean"), "mean", row_number)
            sd = _positive_decimal(row.get("sd"), "sd", row_number)
            records[key] = {"mean_mm": mean, "sd_mm": sd, "n": n}

    expected = {(measurement_id, group) for measurement_id in expected_ids for group in expected_groups}
    missing = expected - records.keys()
    if missing:
        formatted = ", ".join(f"({measurement_id}, {group})" for measurement_id, group in sorted(missing))
        raise ReferenceDataError(f"CSV is missing required records: {formatted}")

    reference_groups: dict[str, Any] = {}
    for group in GROUPS:
        measurements: dict[str, Any] = {}
        for measurement_id, measurement_key in MEASUREMENTS:
            record = records[(measurement_id, group)]
            measurements[measurement_key] = {"measurement_id": measurement_id, **record}
        reference_groups[group] = {"measurements": measurements}
    return {
        "model_version": "summary_delta_v1",
        "source": {"title": SOURCE_TITLE, "url": SOURCE_URL},
        "reference_groups": reference_groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="source CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="generated JSON path")
    args = parser.parse_args()
    try:
        data = build_reference_data(args.input)
    except ReferenceDataError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    args.output.write_text(serialized, encoding="utf-8", newline="\n")
    print(f"Generated {args.output} ({len(data['reference_groups'])} groups, {sum(len(g['measurements']) for g in data['reference_groups'].values())} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
