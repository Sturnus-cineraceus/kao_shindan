from dataclasses import fields
import math

import pytest

from app.feature_extraction import (
    FaceFeatures,
    FaceMeasurements,
    FeatureExtractionError,
    build_features,
    extract_measurements,
)
from app.landmarks import FaceLandmark
from app.statistics.models import FEATURE_KEYS


def _landmarks(count: int = 478) -> list[FaceLandmark]:
    return [FaceLandmark(0.0, 0.0, 123.0) for _ in range(count)]


def _set_pair(points, a, b, p1, p2):
    points[a] = FaceLandmark(*p1, 999.0)
    points[b] = FaceLandmark(*p2, -999.0)


def test_extracts_horizontal_vertical_and_diagonal_pixel_distances():
    points = _landmarks()
    _set_pair(points, 468, 473, (0.2, 0.5), (0.7, 0.5))
    _set_pair(points, 133, 362, (0.1, 0.1), (0.1, 0.5))
    _set_pair(points, 33, 263, (0.0, 0.0), (0.3, 0.4))
    _set_pair(points, 61, 291, (0.0, 0.0), (0.3, 0.4))

    result = extract_measurements(points, 1000, 500)

    assert result.interpupillary_breadth == pytest.approx(500)
    assert result.interocular_breadth == pytest.approx(200)
    # dx=300px and dy=200px, so the non-square image gives sqrt(130000).
    assert result.biectocanthion_breadth == pytest.approx(math.sqrt(130000))
    assert result.mouth_breadth == pytest.approx(math.sqrt(130000))


def test_uses_only_the_four_specified_landmark_pairs():
    points = _landmarks()
    pairs = ((468, 473), (133, 362), (33, 263), (61, 291))
    for offset, (a, b) in enumerate(pairs, start=1):
        _set_pair(points, a, b, (0.0, 0.0), (offset / 100, 0.0))

    measured = extract_measurements(points, 1000, 500)
    assert measured == FaceMeasurements(10, 20, 30, 40)


def test_rejects_missing_required_landmark():
    with pytest.raises(FeatureExtractionError, match="missing"):
        extract_measurements(_landmarks(473), 100, 100)


@pytest.mark.parametrize("width,height", [(0, 10), (10, 0), (-1, 10), (10, -1)])
def test_rejects_non_positive_dimensions(width, height):
    with pytest.raises(FeatureExtractionError, match="dimensions"):
        extract_measurements(_landmarks(), width, height)


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_coordinates(coordinate):
    points = _landmarks()
    points[468] = FaceLandmark(coordinate, 0.0, 0.0)
    with pytest.raises(FeatureExtractionError, match="finite"):
        extract_measurements(points, 100, 100)


def test_rejects_non_finite_distance_result():
    points = _landmarks()
    _set_pair(points, 468, 473, (-1e308, 0.0), (1e308, 0.0))
    with pytest.raises(FeatureExtractionError, match="not finite"):
        extract_measurements(points, 2, 100)


def test_builds_three_ratios_in_statistics_contract_order():
    measurements = FaceMeasurements(50, 30, 100, 40)
    features = build_features(measurements)

    assert features == FaceFeatures(0.5, 0.3, 0.4)
    assert features.as_tuple() == (0.5, 0.3, 0.4)
    assert tuple(field.name for field in fields(FaceFeatures)) == FEATURE_KEYS
    assert tuple(features.as_dict()) == FEATURE_KEYS


@pytest.mark.parametrize("denominator", [0.0, -1.0])
def test_rejects_zero_or_negative_outer_canthal_denominator(denominator):
    with pytest.raises(FeatureExtractionError, match="positive"):
        build_features(FaceMeasurements(1, 1, denominator, 1))


def test_rejects_non_finite_measurement_and_ratio_overflow():
    with pytest.raises(FeatureExtractionError, match="finite"):
        build_features(FaceMeasurements(float("nan"), 1, 2, 1))
    with pytest.raises(FeatureExtractionError, match="not finite"):
        build_features(FaceMeasurements(1.0, 1.0, float.fromhex("0x0.0000000000001p-1022"), 1.0))
