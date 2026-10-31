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

FIXTURE = Path(__file__).parent / 'fixtures' / 'portrait_female_young_adult_01.png'
MODEL = Path('/opt/kao-cal/models/face_landmarker.task')


class StubLandmarker:
    def analyze(self, image):
        return SimpleNamespace(face_count=1, faces=((),), facial_transformation_matrices=())


@pytest.fixture
def client(monkeypatch):
    from app.feature_extraction import FaceMeasurements
    import app.analysis as analysis
    monkeypatch.setattr(analysis, 'validate', lambda *_: ValidationResult((CheckResult('face_count', True),)))
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
    assert result['reference_model_version'] and result['algorithm_version']
    assert 'faces' not in result


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
        image = decode_image(FIXTURE.read_bytes())
        landmarks = landmarker.analyze(image)
        assert landmarks.face_count == 1
        quality = validate(image, landmarks)
        if not quality.passed:
            with pytest.raises(AnalysisError) as failure:
                AnalysisService(landmarker).analyze(FIXTURE.read_bytes(), 'young_female')
            assert failure.value.code == 'validation_failed'
        quality = validate(image, landmarks, ValidationConfig(pitch=180, roll=180))
        assert quality.passed, quality.reasons
        features = build_features(extract_measurements(landmarks.faces[0], image.width, image.height))
        assert all(math.isfinite(x) and x > 0 for x in features.as_tuple())
        from app.statistics import compare_features
        result = compare_features('young_female', features.as_dict())
        assert result.reference_group == 'young_female'
        assert result.reference_model_version and 0 <= result.score <= 100
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
