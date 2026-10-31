import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.analysis import AnalysisError, AnalysisService
from app.image_processing import decode_image
from app.landmarks import FaceLandmarkerService
from app.main import app
from app.validation import CheckResult, ValidationResult, ValidationConfig, validate
from app.feature_extraction import extract_measurements, build_features
from app.statistics import FEATURE_KEYS

FIXTURE = Path(__file__).parent / 'fixtures' / 'portrait_female_young_adult_01.png'
MODEL = Path('/opt/kao-cal/models/face_landmarker.task')


class StubLandmarker:
    def analyze(self, image):
        return SimpleNamespace(face_count=1, faces=((),), facial_transformation_matrices=())


@pytest.fixture
def client(monkeypatch):
    from app.feature_extraction import FaceMeasurements
    import app.analysis as analysis
    monkeypatch.setattr(analysis, 'validate', lambda *_: ValidationResult((CheckResult('face_count', True),), analysis_landmarks=()))
    monkeypatch.setattr(analysis, 'extract_measurements', lambda *_: FaceMeasurements(60, 30, 100, 45))
    with TestClient(app) as client:
        app.state.analysis_service = AnalysisService(StubLandmarker())
        yield client


def post(client, data=None, group='young_female'):
    files = {} if data is None else {'image': ('face.png', data, 'image/png')}
    form = {} if group is None else {'reference_group': group}
    return client.post('/analyze', files=files, data=form)


def test_api_success(client):
    response = post(client, FIXTURE.read_bytes())
    assert response.status_code == 200
    result = response.json()
    assert result['reference_group'] == 'young_female'
    assert 0 <= result['score'] <= 100
    assert 0 <= result['volume_zone_score'] <= 100
    assert {'distance', 'z_scores', 'features', 'model_mode', 'regularization_used'} <= result.keys()
    assert result['used_features'] == list(FEATURE_KEYS)
    assert result['profile_id'] == 'core3'
    assert result['dimension'] == 3
    assert result['volume_zone_distribution']['mode_distance'] == pytest.approx(math.sqrt(2))
    assert len(result['volume_zone_distribution']['points']) == 101
    assert result['reference_model_version'] and result['algorithm_version']
    assert 'faces' not in result


def test_optional_api(client, monkeypatch):
    import app.analysis as analysis
    from app.landmarks import FaceLandmark
    landmarks = [FaceLandmark(.5, .5, 0) for _ in range(478)]
    landmarks[168] = FaceLandmark(.5, .2, 0)
    landmarks[152] = FaceLandmark(.5, .8, 0)
    landmarks[234] = FaceLandmark(.2, .5, 0)
    landmarks[454] = FaceLandmark(.8, .5, 0)
    monkeypatch.setattr(analysis, 'validate', lambda *a, **kw: ValidationResult((CheckResult('face_count', True),), analysis_landmarks=landmarks))
    def request(features):
        files = [('image', ('face.png', FIXTURE.read_bytes(), 'image/png')),
                 ('reference_group', (None, 'young_female'))]
        files.extend(('additional_feature', (None, feature)) for feature in features)
        return client.post('/analyze', files=files)
    result = request(['face_aspect_ratio'])
    assert result.status_code == 200
    body = result.json()
    assert body['profile_id'] == 'core3+face_aspect_ratio' and body['dimension'] == 4
    assert body['used_features'] == list(FEATURE_KEYS) + ['face_aspect_ratio']
    assert 'face_aspect_ratio' in body['features'] and 'face_aspect_ratio' in body['z_scores']
    from app.statistics.models import OPTIONAL_FEATURE_ORDER
    first = request(list(OPTIONAL_FEATURE_ORDER))
    second = request(list(reversed(OPTIONAL_FEATURE_ORDER)))
    assert first.status_code == second.status_code == 200
    assert first.json()['profile_id'] == second.json()['profile_id']
    assert first.json()['used_features'] == list(FEATURE_KEYS) + list(OPTIONAL_FEATURE_ORDER)
    assert first.json()['dimension'] == 8
    for selection in (['unknown'], ['face_aspect_ratio', 'face_aspect_ratio']):
        response = request(selection)
        assert response.status_code == 400 and response.json()['error']['code'] == 'invalid_feature_selection'


def test_calculation_profiles(client, monkeypatch):
    import app.analysis as analysis
    from app.landmarks import FaceLandmark
    landmarks = [FaceLandmark(.5, .5, 0) for _ in range(478)]
    for index, x, y in ((168, .5, .2), (152, .5, .8), (234, .2, .5), (454, .8, .5)):
        landmarks[index] = FaceLandmark(x, y, 0)
    monkeypatch.setattr(analysis, 'validate', lambda *a, **kw: ValidationResult((CheckResult('face_count', True),), analysis_landmarks=landmarks))
    def request(profile=None, features=()):
        files = [('image', ('face.png', FIXTURE.read_bytes(), 'image/png')), ('reference_group', (None, 'young_female'))]
        if profile is not None:
            files.append(('calculation_profile', (None, profile)))
        files.extend(('additional_feature', (None, feature)) for feature in features)
        return client.post('/analyze', files=files)
    default, legacy = request(), request('legacy')
    assert default.status_code == legacy.status_code == 200
    assert default.json() == legacy.json()
    assert request('legacy', ('face_aspect_ratio',)).status_code == 200
    result = request('representative3')
    assert result.status_code == 200
    body = result.json()
    assert body['profile_id'] == 'representative3' and body['dimension'] == 3
    assert body['used_features'] == ['face_aspect_ratio', 'biectocanthion_to_face_width_ratio', 'mouth_to_face_width_ratio']
    assert set(body['features']) == set(body['z_scores']) == set(body['used_features'])
    for profile, features, code in [('unknown', (), 'invalid_calculation_profile'), ('representative3', ('face_aspect_ratio',), 'features_not_allowed_for_profile')]:
        response = request(profile, features)
        assert response.status_code == 400 and response.json()['error']['code'] == code


def test_validation_response_has_safe_multiple_issues(client, monkeypatch):
    import app.analysis as analysis
    monkeypatch.setattr(analysis, 'validate', lambda *_: ValidationResult((
        CheckResult('yaw', False, 'yaw_out_of_range', 22.0),
        CheckResult('blur', False, 'blurred', 4.0),
    )))
    response = post(client, FIXTURE.read_bytes())
    body = response.json()['error']
    assert [i['code'] for i in body['issues']] == ['HEAD_YAW_TOO_LARGE', 'IMAGE_BLURRED']
    assert '22' not in response.text and '4.0' not in response.text
    assert 'checks' not in body


def test_no_face_suppresses_dependent_issues(client, monkeypatch):
    import app.analysis as analysis
    monkeypatch.setattr(analysis, 'validate', lambda *_: ValidationResult((
        CheckResult('face_count', False, 'no_face'),
    )))
    response = post(client, FIXTURE.read_bytes())
    assert [x['code'] for x in response.json()['error']['issues']] == ['FACE_NOT_FOUND']


@pytest.mark.parametrize('image,group,code', [(None, 'young_male', 'invalid_request'), (b'x', None, 'invalid_request'),
    (b'x', 'unknown', 'invalid_reference_group'), (b'x', 'young_male', 'invalid_image')])
def test_api_invalid_inputs(client, image, group, code):
    response = post(client, image, group)
    assert response.status_code == 400
    assert response.json()['error']['code'] == code


@pytest.mark.parametrize('reason,code', [('no_face', 'no_face'), ('multiple_faces', 'multiple_faces'), ('blurred', 'validation_failed')])
def test_validation_stops_scoring(client, monkeypatch, reason, code):
    import app.analysis as analysis
    monkeypatch.setattr(analysis, 'validate', lambda *_: ValidationResult((CheckResult('check', False, reason),)))
    monkeypatch.setattr(analysis, 'compare_features', lambda *a, **kw: pytest.fail('scoring must not run'))
    response = post(client, FIXTURE.read_bytes())
    assert response.status_code == 422
    assert response.json()['error']['code'] == code
    if reason == 'blurred':
        assert response.json()['error']['issues'][0]['code'] == 'IMAGE_BLURRED'
    else:
        assert 'checks' not in response.json()['error']


def test_real_pipeline():
    if not MODEL.is_file():
        pytest.skip('Model asset is provisioned by the Docker image')
    landmarker = FaceLandmarkerService(MODEL)
    try:
        result = AnalysisService(landmarker).analyze(FIXTURE.read_bytes(), 'young_female')
        assert 0 <= result.score <= 100
        assert 'HEAD_PITCH_WARNING' in [warning.code for warning in result.quality_warnings]
        extended = AnalysisService(landmarker).analyze(FIXTURE.read_bytes(), 'young_female', ('face_aspect_ratio',))
        assert extended.profile_id == 'core3+face_aspect_ratio' and extended.dimension == 4
        assert extended.features['face_aspect_ratio'] > 0
    finally:
        landmarker.close()



def test_landmarker_reuses_and_closes_model(tmp_path):
    import numpy as np
    from app.image_processing import DecodedImage
    model = tmp_path / 'model.task'
    model.touch()
    calls = []
    class Task:
        def detect(self, image):
            return SimpleNamespace(face_landmarks=[], facial_transformation_matrixes=[])
        def close(self):
            calls.append('closed')
    def factory(path):
        calls.append('created')
        return Task()
    service = FaceLandmarkerService(model, factory)
    image = DecodedImage(np.zeros((2, 2, 3), dtype=np.uint8), 2, 2)
    service.analyze(image)
    service.analyze(image)
    service.close()
    assert calls == ['created', 'closed']


def test_service_passes_corrected_landmarks_to_extraction(monkeypatch):
    import app.analysis as analysis
    from app.feature_extraction import FaceMeasurements
    from app.landmarks import FaceLandmark
    corrected = (FaceLandmark(.3, .4, 0),)
    monkeypatch.setattr(analysis, 'validate', lambda *_: ValidationResult((CheckResult('face_count', True),), analysis_landmarks=corrected))
    received = []
    def extract(points, width, height):
        received.append(points)
        return FaceMeasurements(60, 30, 100, 45)
    monkeypatch.setattr(analysis, 'extract_measurements', extract)
    AnalysisService(StubLandmarker()).analyze(FIXTURE.read_bytes(), 'young_female')
    assert received == [corrected]


def test_api_returns_safe_quality_warnings(client, monkeypatch):
    import app.analysis as analysis
    from app.validation import QualityWarning
    warning = QualityWarning('HEAD_PITCH_WARNING', '顔の向きにより計測誤差が増える可能性があります。')
    monkeypatch.setattr(analysis, 'validate', lambda *_: ValidationResult((CheckResult('pitch', True, value=16.2, threshold=20),), (warning,), ()))
    response = post(client, FIXTURE.read_bytes())
    assert response.status_code == 200
    assert response.json()['quality_warnings'] == [{'code': warning.code, 'message': warning.message}]
    assert '16.2' not in response.text
    assert 'analysis_landmarks' not in response.text
