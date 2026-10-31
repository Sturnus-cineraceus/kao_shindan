import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import chi, chi2

from app.analysis import AnalysisError, AnalysisService
from app.feature_extraction.service import extract_face_aspect_ratio, FeatureExtractionError
from app.landmarks import FaceLandmark
from app.reference_data import load_reference_data
from app.statistics.models import build_reference_model, compare_features, FEATURE_KEYS


def points():
    result = [FaceLandmark(0, 0, 0) for _ in range(478)]
    result[168] = FaceLandmark(.2, .1, 0)
    result[152] = FaceLandmark(.3, .7, 0)
    result[234] = FaceLandmark(.1, .4, 0)
    result[454] = FaceLandmark(.8, .5, 0)
    return result


def test_aspect_pixel_coordinates():
    assert extract_face_aspect_ratio(points(), 1000, 500) == pytest.approx(math.hypot(100, 300) / math.hypot(700, 50))


@pytest.mark.parametrize('index', [168, 152, 234, 454])
@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_aspect_rejects_nonfinite(index, value):
    p = points()
    p[index] = FaceLandmark(value, 0, 0)
    with pytest.raises(FeatureExtractionError):
        extract_face_aspect_ratio(p, 1000, 500)


@pytest.mark.parametrize('index', [168, 152, 234, 454])
def test_aspect_rejects_missing(index):
    with pytest.raises(FeatureExtractionError):
        extract_face_aspect_ratio(points()[:index], 1000, 500)


def test_aspect_rejects_zero_width():
    p = points()
    p[454] = p[234]
    with pytest.raises(FeatureExtractionError):
        extract_face_aspect_ratio(p, 1000, 500)


def test_four_dimensional_model_and_statistics():
    selected = ('face_aspect_ratio',)
    core = build_reference_model('young_female')
    model = build_reference_model('young_female', additional_features=selected)
    group = load_reference_data().get_group('young_female')
    h, w = (group.measurements[k] for k in ('morphologic_face_height', 'bizygomatic_breadth'))
    assert model.features == FEATURE_KEYS + selected
    assert model.mean.shape == (4,) and model.covariance.shape == (4, 4)
    assert model.mean[3] == pytest.approx(h.mean_mm / w.mean_mm)
    assert model.covariance[3, 3] == pytest.approx(h.sd_mm**2 / w.mean_mm**2 + h.mean_mm**2 * w.sd_mm**2 / w.mean_mm**4)
    np.testing.assert_array_equal(model.covariance[:3, :3], core.covariance)
    np.testing.assert_array_equal(model.covariance[:3, 3], 0)
    assert np.all(np.diag(model.covariance) > 0)
    result = compare_features('young_female', model.mean + model.standard_deviations, additional_features=selected)
    assert result.dimension == 4
    assert result.score == pytest.approx(100 * chi2.sf(result.distance**2, 4))
    assert result.volume_zone_score == pytest.approx(100 * chi.pdf(result.distance, 4) / chi.pdf(math.sqrt(3), 4))
    assert result.volume_zone_distribution.mode_distance == pytest.approx(math.sqrt(3))
    assert len(result.volume_zone_distribution.points) == 101


def test_service_rejects_invalid_selection():
    service = AnalysisService(SimpleNamespace())
    for selection in [('unknown',), ('face_aspect_ratio', 'face_aspect_ratio')]:
        with pytest.raises(AnalysisError, match='invalid_feature_selection'):
            service.analyze(b'', 'young_female', selection)
