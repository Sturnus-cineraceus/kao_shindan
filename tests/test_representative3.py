import math

import numpy as np
import pytest
from scipy.stats import chi, chi2

from app.statistics.models import (
    REPRESENTATIVE3_FEATURES, build_representative3_model, compare_features,
)
from app.reference_data import load_reference_data


def test_representative3_covariance_and_distributions():
    model = build_representative3_model('young_female')
    assert model.features == REPRESENTATIVE3_FEATURES
    assert model.covariance.shape == (3, 3)
    assert np.allclose(model.covariance, model.covariance.T)
    assert np.all(np.diag(model.covariance) > 0)
    assert np.all(model.covariance[np.triu_indices(3, 1)] > 0)
    group = load_reference_data().get_group('young_female')
    width = group.measurements['bizygomatic_breadth']
    inputs = [group.measurements[key] for key in ('morphologic_face_height', 'biectocanthion_breadth', 'mouth_breadth')]
    expected = np.array([[a.mean_mm * b.mean_mm * width.sd_mm**2 / width.mean_mm**4
                          + (a.sd_mm**2 / width.mean_mm**2 if i == j else 0)
                          for j, b in enumerate(inputs)] for i, a in enumerate(inputs)])
    np.testing.assert_allclose(model.covariance, expected, rtol=1e-12, atol=0)
    mean_result = compare_features('young_female', model.mean, calculation_profile='representative3')
    assert mean_result.distance == 0
    assert mean_result.dimension == 3
    assert mean_result.used_features == REPRESENTATIVE3_FEATURES
    result = compare_features('young_female', model.mean + model.standard_deviations, calculation_profile='representative3')
    assert result.score == pytest.approx(100 * chi2.sf(result.distance**2, df=3))
    assert result.volume_zone_distribution.mode_distance == pytest.approx(math.sqrt(2))
    assert result.volume_zone_score == pytest.approx(100 * chi.pdf(result.distance, df=3) / chi.pdf(math.sqrt(2), df=3))


def test_legacy_default_and_explicit_match():
    values = [0.6, 0.3, 0.45]
    assert compare_features('young_female', values) == compare_features('young_female', values, calculation_profile='legacy')
