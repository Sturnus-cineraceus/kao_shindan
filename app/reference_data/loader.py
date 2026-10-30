"""Read and validate the generated public reference statistics."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

MEASUREMENT_KEYS = (
    "interpupillary_breadth",
    "interocular_breadth",
    "biectocanthion_breadth",
    "mouth_breadth",
)
EXPECTED_GROUPS = ("young_male", "young_female", "older_male", "older_female")
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "reference_data.json"


class ReferenceDataError(ValueError):
    """Invalid or unavailable reference statistics."""


@dataclass(frozen=True)
class MeasurementSummary:
    mean_mm: float
    sd_mm: float
    n: int


@dataclass(frozen=True)
class ReferenceGroup:
    reference_group: str
    measurements: dict[str, MeasurementSummary]


@dataclass(frozen=True)
class ReferenceData:
    model_version: str
    groups: dict[str, ReferenceGroup]

    def get_group(self, reference_group: str) -> ReferenceGroup:
        try:
            return self.groups[reference_group]
        except (KeyError, TypeError) as exc:
            raise ReferenceDataError(f"Unknown reference_group: {reference_group!r}") from exc


def _positive_finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReferenceDataError(f"{field} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ReferenceDataError(f"{field} must be a finite positive number")
    return result


def parse_reference_data(payload: Any) -> ReferenceData:
    if not isinstance(payload, dict):
        raise ReferenceDataError("Reference data root must be an object")
    version = payload.get("model_version")
    if not isinstance(version, str) or not version:
        raise ReferenceDataError("Missing model_version")
    groups_payload = payload.get("reference_groups")
    if not isinstance(groups_payload, dict) or set(groups_payload) != set(EXPECTED_GROUPS):
        raise ReferenceDataError("Reference data must contain exactly the four supported groups")
    groups: dict[str, ReferenceGroup] = {}
    for group_name in EXPECTED_GROUPS:
        group_payload = groups_payload[group_name]
        measurements_payload = group_payload.get("measurements") if isinstance(group_payload, dict) else None
        if not isinstance(measurements_payload, dict) or set(measurements_payload) != set(MEASUREMENT_KEYS):
            raise ReferenceDataError(f"{group_name} must contain exactly the required measurements")
        measurements: dict[str, MeasurementSummary] = {}
        for key in MEASUREMENT_KEYS:
            entry = measurements_payload[key]
            if not isinstance(entry, dict):
                raise ReferenceDataError(f"{group_name}.{key} must be an object")
            mean = _positive_finite(entry.get("mean_mm"), f"{group_name}.{key}.mean_mm")
            sd = _positive_finite(entry.get("sd_mm"), f"{group_name}.{key}.sd_mm")
            n = entry.get("n")
            if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
                raise ReferenceDataError(f"{group_name}.{key}.n must be a positive integer")
            measurements[key] = MeasurementSummary(mean, sd, n)
        groups[group_name] = ReferenceGroup(group_name, measurements)
    return ReferenceData(version, groups)


@lru_cache(maxsize=4)
def load_reference_data(path: str | Path = DEFAULT_DATA_PATH) -> ReferenceData:
    """Load validated generated data; callers can pass another file for controlled use."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceDataError(f"Could not read reference data from {source}") from exc
    return parse_reference_data(payload)
