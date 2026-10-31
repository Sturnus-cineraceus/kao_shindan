import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi, chi2

from app.reference_data import (
    MEASUREMENT_KEYS,
    EXPECTED_GROUPS,
    ReferenceDataError,
    load_reference_data,
    parse_reference_data,
)
from app.statistics import FEATURE_KEYS, ModelError, build_reference_model, compare_features
from app.statistics.models import DENOMINATOR_KEY, NUMERATOR_KEYS, _volume_zone_score, _volume_zone_distribution


ROOT = Path(__file__).resolve().parents[1]


def test_reference_data_loads_and_validates_all_groups_and_measurements():
    data = load_reference_data()
    assert set(data.groups) == set(EXPECTED_GROUPS)
    for group in data.groups.values():
        assert set(group.measurements) == set(MEASUREMENT_KEYS)
        for entry in group.measurements.values():
            assert np.isfinite(entry.mean_mm) and entry.mean_mm > 0
            assert np.isfinite(entry.sd_mm) and entry.sd_mm > 0
            assert entry.n > 0
    with pytest.raises(ReferenceDataError):
        data.get_group("unknown")


def _synthetic_reference():
    return parse_reference_data({
        "model_version": "synthetic",
        "reference_groups": {
            "young_male": {"measurements": {
                "interpupillary_breadth": {"mean_mm": 4.0, "sd_mm": 0.5, "n": 10},
                "interocular_breadth": {"mean_mm": 2.0, "sd_mm": 0.4, "n": 10},
                "biectocanthion_breadth": {"mean_mm": 5.0, "sd_mm": 0.25, "n": 10},
                "mouth_breadth": {"mean_mm": 3.0, "sd_mm": 0.3, "n": 10},
            }},
            **{name: {"measurements": {
                key: {"mean_mm": 1.0, "sd_mm": 0.1, "n": 10} for key in MEASUREMENT_KEYS
            }} for name in EXPECTED_GROUPS[1:]},
        },
    })


def test_delta_mean_variance_and_shared_denominator_covariance():
    data = _synthetic_reference()
    model = build_reference_model("young_male", data)
    y = 5.0
    sd_y = 0.25
    means = [4.0, 2.0, 3.0]
    sds = [0.5, 0.4, 0.3]
    expected_mean = np.array(means) / y
    expected_variance = np.array(sds) ** 2 / y**2 + np.array(means) ** 2 * sd_y**2 / y**4
    expected_covariance = np.outer(means, means) * sd_y**2 / y**4
    np.fill_diagonal(expected_covariance, expected_variance)
    np.testing.assert_allclose(model.mean, expected_mean)
    np.testing.assert_allclose(np.diag(model.covariance), expected_variance)
    np.testing.assert_allclose(model.covariance, expected_covariance)
    assert model.covariance.shape == (3, 3)
    np.testing.assert_allclose(model.covariance, model.covariance.T)
    assert np.all(np.diag(model.covariance) > 0)
    assert model.features == FEATURE_KEYS
    assert NUMERATOR_KEYS == ("interpupillary_breadth", "interocular_breadth", "mouth_breadth")
    assert DENOMINATOR_KEY == "biectocanthion_breadth"


@pytest.mark.parametrize("group", EXPECTED_GROUPS)
def test_each_reference_group_builds_and_calculates(group):
    model = build_reference_model(group)
    result = compare_features(group, dict(zip(FEATURE_KEYS, model.mean)))
    assert result.reference_group == group
    assert result.model_mode == "summary_delta"
    assert result.reference_model_version == "summary_delta_v1"
    assert np.isfinite(result.distance)
    assert np.isfinite(result.score)
    assert np.isfinite(result.volume_zone_score)


def test_mean_vector_has_zero_z_and_distance_and_score_near_100():
    model = build_reference_model("young_male")
    result = compare_features("young_male", dict(zip(FEATURE_KEYS, model.mean)))
    assert result.distance == pytest.approx(0.0, abs=1e-12)
    assert result.score == pytest.approx(100.0)
    assert result.volume_zone_score == pytest.approx(0.0)
    assert result.features == dict(zip(FEATURE_KEYS, model.mean))
    assert list(result.z_scores) == list(FEATURE_KEYS)
    assert all(value == pytest.approx(0.0, abs=1e-12) for value in result.z_scores.values())


def test_distance_increase_lowers_score_and_uses_correlated_distance():
    model = build_reference_model("young_female")
    near = compare_features("young_female", model.mean + np.array([0.01, 0.01, 0.01]))
    far = compare_features("young_female", model.mean + np.array([0.1, 0.1, 0.1]))
    assert far.distance > near.distance > 0
    assert far.score < near.score < 100
    delta = np.array([0.1, 0.1, 0.1])
    expected = np.sqrt(delta @ np.linalg.solve(model.covariance, delta))
    assert far.distance == pytest.approx(expected)


@pytest.mark.parametrize("distance", [0.0, 0.5, np.sqrt(2), 3.0, 10.0])
def test_core3_legacy_scores_and_distribution(distance):
    model = build_reference_model("young_male")
    features = model.mean + distance * np.linalg.cholesky(model.covariance)[:, 0]
    result = compare_features("young_male", features)
    assert result.dimension == 3
    assert result.used_features == FEATURE_KEYS
    assert result.score == pytest.approx(100 * chi2.sf(result.distance ** 2, df=3))
    assert result.volume_zone_score == pytest.approx(100 * chi.pdf(result.distance, df=3) / chi.pdf(np.sqrt(2), df=3))
    distribution = result.volume_zone_distribution
    assert distribution.mode_distance == pytest.approx(np.sqrt(2))
    assert len(distribution.points) == 101
    assert distribution.points[0].distance == 0
    assert all(a.distance < b.distance for a, b in zip(distribution.points, distribution.points[1:]))
    assert all(np.isfinite(p.relative_density) and 0 <= p.relative_density <= 100 for p in distribution.points)
    assert distribution.points[-1].distance >= result.distance
    assert _volume_zone_score(distribution.mode_distance, result.dimension) == pytest.approx(100)


@pytest.mark.parametrize("dimension", [3, 4, 5, 8])
def test_distribution_uses_dimension(dimension):
    distribution = _volume_zone_distribution(1, dimension)
    assert distribution.mode_distance == pytest.approx(np.sqrt(dimension - 1))
    assert _volume_zone_score(distribution.mode_distance, dimension) == pytest.approx(100)
    assert distribution.points[-1].distance >= chi.ppf(0.999, df=dimension)
    assert distribution.points[25].relative_density == pytest.approx(
        100 * chi.pdf(distribution.points[25].distance, df=dimension) / chi.pdf(distribution.mode_distance, df=dimension))


@pytest.mark.parametrize("dimension", [0, 1, 2])
def test_volume_zone_rejects_dimensions_below_core3(dimension):
    with pytest.raises(ModelError):
        _volume_zone_score(1, dimension)
    with pytest.raises(ModelError):
        _volume_zone_distribution(1, dimension)


def test_distribution_rejects_invalid_calculation(monkeypatch):
    monkeypatch.setattr(chi, 'ppf', lambda *args, **kwargs: float('nan'))
    with pytest.raises(ModelError):
        _volume_zone_distribution(1, 3)


@pytest.mark.parametrize("distance", [0, 0.1, 1, np.sqrt(2), 2, 10, 100])
def test_volume_zone_score_is_finite_and_bounded(distance):
    value = _volume_zone_score(distance)
    assert np.isfinite(value)
    assert 0 <= value <= 100


def test_volume_zone_score_at_mode_and_tails():
    assert _volume_zone_score(0) == pytest.approx(0)
    assert _volume_zone_score(np.sqrt(2)) == pytest.approx(100)
    assert _volume_zone_score(10) == pytest.approx(0, abs=1e-12)
    model = build_reference_model("young_male")
    # Cholesky column produces a displacement at the requested Mahalanobis distance.
    features = model.mean + np.sqrt(2) * np.linalg.cholesky(model.covariance)[:, 0]
    result = compare_features("young_male", features)
    assert result.distance == pytest.approx(np.sqrt(2))
    assert result.volume_zone_score == pytest.approx(100)


@pytest.mark.parametrize("distance", [float("nan"), float("inf"), -1])
def test_volume_zone_rejects_invalid_distances(distance):
    with pytest.raises(ModelError):
        _volume_zone_score(distance)


@pytest.mark.parametrize("bad", [
    [float("nan"), 0.2, 0.3], [float("inf"), 0.2, 0.3], [0.2, 0.3], [0.2, 0.3, 0.4, 0.5],
])
def test_rejects_invalid_feature_vectors(bad):
    with pytest.raises(ModelError):
        compare_features("young_male", bad)


def test_rejects_unknown_group_and_malformed_feature_mapping():
    with pytest.raises(ModelError):
        compare_features("unknown", [0.1, 0.2, 0.3])
    with pytest.raises(ModelError):
        compare_features("young_male", {"wrong": 0.1})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0, -1])
def test_reference_loader_rejects_bad_summary_values(bad):
    path = ROOT / "app" / "data" / "reference_data.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["reference_groups"]["young_male"]["measurements"]["mouth_breadth"]["sd_mm"] = bad
    with pytest.raises(ReferenceDataError):
        parse_reference_data(payload)
