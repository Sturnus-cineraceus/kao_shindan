import numpy as np
import pytest

from app.feature_extraction.models import FaceMeasurements
from app.feature_extraction.service import extract_face_width_ratios, FeatureExtractionError
from app.statistics.models import OPTIONAL_FEATURE_ORDER, build_reference_model, compare_features, validate_additional_features, ModelError
from app.landmarks import FaceLandmark


def points():
    result = [FaceLandmark(0, 0, 0) for _ in range(478)]
    result[168] = FaceLandmark(.2, .1, 0)
    result[152] = FaceLandmark(.3, .7, 0)
    result[234] = FaceLandmark(.1, .4, 0)
    result[454] = FaceLandmark(.8, .5, 0)
    return result


def test_all_features_covariance_and_order():
    selected = tuple(reversed(OPTIONAL_FEATURE_ORDER))
    assert validate_additional_features(selected) == OPTIONAL_FEATURE_ORDER
    model = build_reference_model('young_female', additional_features=selected)
    assert model.features[3:] == OPTIONAL_FEATURE_ORDER
    assert model.covariance.shape == (8, 8)
    np.testing.assert_allclose(model.covariance, model.covariance.T)
    assert np.all(np.diag(model.covariance) > 0)
    assert model.covariance[3, 4] > 0
    assert model.covariance[0, 4] > 0
    assert model.covariance[0, 6] < 0
    result = compare_features('young_female', model.mean, additional_features=selected)
    assert result.dimension == 8 and result.used_features == model.features
    assert result.distance == pytest.approx(0)
    assert result.regularization_used


@pytest.mark.parametrize('selected', [('unknown',), ('face_aspect_ratio', 'face_aspect_ratio')])
def test_invalid_features(selected):
    with pytest.raises(ModelError):
        validate_additional_features(selected)


def test_photo_ratios_and_zero_width():
    p = points()
    measurements = FaceMeasurements(100, 50, 150, 120)
    result = extract_face_width_ratios(p, 1000, 500, measurements, OPTIONAL_FEATURE_ORDER)
    width = np.hypot(700, 50)
    assert result['mouth_to_face_width_ratio'] == pytest.approx(120 / width)
    assert result['face_aspect_ratio'] == pytest.approx(np.hypot(100, 300) / width)
    p[454] = p[234]
    with pytest.raises(FeatureExtractionError):
        extract_face_width_ratios(p, 1000, 500, measurements, OPTIONAL_FEATURE_ORDER)
