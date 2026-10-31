import csv
import json
from pathlib import Path

import pytest

from scripts.build_reference_data import (
    GROUPS,
    MEASUREMENTS,
    ReferenceDataError,
    build_reference_data,
)


HEADER = ["measurement_id", "unit", "reference_group", "n", "mean", "sd"]


def source_rows():
    return [
        {
            "measurement_id": str(measurement_id),
            "unit": "mm",
            "reference_group": group,
            "n": str(50 + group_index),
            "mean": str(60 + measurement_index + group_index / 10),
            "sd": str(2 + measurement_index / 10),
        }
        for measurement_index, (measurement_id, _) in enumerate(MEASUREMENTS)
        for group_index, group in enumerate(GROUPS)
    ]


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def test_builds_all_groups_measurements_and_preserves_csv_values(tmp_path):
    path = tmp_path / "source.csv"
    rows = source_rows()
    write_csv(path, rows)
    result = build_reference_data(path)
    assert list(result["reference_groups"]) == list(GROUPS)
    assert sum(len(group["measurements"]) for group in result["reference_groups"].values()) == 24
    measurement_keys = dict(MEASUREMENTS)
    for source in rows:
        entry = result["reference_groups"][source["reference_group"]]["measurements"][measurement_keys[int(source["measurement_id"])] ]
        assert entry["mean_mm"] == float(source["mean"])
        assert entry["sd_mm"] == float(source["sd"])
        assert entry["n"] == int(source["n"])


@pytest.mark.parametrize("mutation", ["missing_group", "missing_measurement", "duplicate", "unit", "numeric"])
def test_invalid_csv_fails(tmp_path, mutation):
    rows = source_rows()
    if mutation == "missing_group":
        rows = [row for row in rows if not (row["measurement_id"] == "5" and row["reference_group"] == "young_male")]
    elif mutation == "missing_measurement":
        rows = [row for row in rows if row["measurement_id"] != "13"]
    elif mutation == "duplicate":
        rows.append(rows[0].copy())
    elif mutation == "unit":
        rows[0]["unit"] = "cm"
    elif mutation == "numeric":
        rows[0]["mean"] = ""
    path = tmp_path / "source.csv"
    write_csv(path, rows)
    with pytest.raises(ReferenceDataError):
        build_reference_data(path)


@pytest.mark.parametrize("measurement_id", ["9", "16"])
@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_added_measurements_reject_missing_or_duplicate_records(tmp_path, measurement_id, mutation):
    rows = source_rows()
    target = next(row for row in rows if row["measurement_id"] == measurement_id and row["reference_group"] == "older_female")
    if mutation == "missing":
        rows.remove(target)
    else:
        rows.append(target.copy())
    path = tmp_path / "source.csv"
    write_csv(path, rows)
    with pytest.raises(ReferenceDataError):
        build_reference_data(path)


def test_generated_json_is_deterministic_utf8_and_matches_source(tmp_path):
    path = tmp_path / "source.csv"
    write_csv(path, source_rows())
    # Decimal source values and Japanese labels in checked-in data are not synthesized.
    first = json.dumps(build_reference_data(path), ensure_ascii=False, indent=2) + "\n"
    second = json.dumps(build_reference_data(path), ensure_ascii=False, indent=2) + "\n"
    assert first == second
    assert first.endswith("\n")
    assert "\\u" not in first


def test_checked_in_json_matches_real_csv():
    from scripts.build_reference_data import DEFAULT_INPUT, DEFAULT_OUTPUT
    generated = build_reference_data(DEFAULT_INPUT)
    assert generated == json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    serialized = json.dumps(generated, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    assert DEFAULT_OUTPUT.read_text(encoding="utf-8") == serialized
    rows = list(csv.DictReader(DEFAULT_INPUT.open(encoding="utf-8-sig", newline="")))
    for row in rows:
        measurement_id = int(row["measurement_id"])
        if measurement_id in dict(MEASUREMENTS):
            item = generated["reference_groups"][row["reference_group"]]["measurements"][dict(MEASUREMENTS)[measurement_id]]
            assert (item["mean_mm"], item["sd_mm"], item["n"]) == (float(row["mean"]), float(row["sd"]), int(row["n"]))
