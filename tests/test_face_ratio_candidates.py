import math
from pathlib import Path

import pytest
from app.landmarks import FaceLandmark, FaceLandmarkerService
from app.image_processing import decode_image
from app.validation import validate
from scripts.inspect_face_ratio_candidates import measure_candidates


def points():
    p = [FaceLandmark(0, 0, 0) for _ in range(455)]
    for i, x, y in [(234, .1, .5), (454, .9, .5), (127, .2, .5), (356, .8, .5), (168, .5, .2), (6, .5, .3), (152, .5, .8)]:
        p[i] = FaceLandmark(x, y, 0)
    return p


def test_candidate_distances_and_ratios_on_non_square_image():
    distances, ratios = measure_candidates(points(), 1000, 500)
    assert distances['width_candidate_234_454'] == pytest.approx(800)
    assert distances['width_candidate_127_356'] == pytest.approx(600)
    assert distances['height_candidate_168_152'] == pytest.approx(300)
    assert distances['height_candidate_6_152'] == pytest.approx(250)
    assert ratios[('height_candidate_168_152', 'width_candidate_234_454')] == pytest.approx(0.375)
    assert ratios[('height_candidate_6_152', 'width_candidate_127_356')] == pytest.approx(250 / 600)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_rejects_nonfinite(value):
    p = points()
    p[6] = FaceLandmark(value, 0, 0)
    with pytest.raises(ValueError):
        measure_candidates(p, 1000, 500)


def test_rejects_missing_and_zero_distance():
    with pytest.raises(ValueError):
        measure_candidates(points()[:454], 1000, 500)
    p = points()
    p[454] = p[234]
    with pytest.raises(ValueError):
        measure_candidates(p, 1000, 500)


def test_fixture_candidates():
    model = Path('/opt/kao-cal/models/face_landmarker.task')
    if not model.is_file():
        pytest.skip('model provided by Docker')
    image = decode_image((Path(__file__).parent / 'fixtures/portrait_female_young_adult_01.png').read_bytes())
    service = FaceLandmarkerService(model)
    try:
        result = validate(image, service.analyze(image))
    finally:
        service.close()
    assert result.passed
    distances, ratios = measure_candidates(result.analysis_landmarks, image.width, image.height)
    assert all(math.isfinite(v) and v > 0 for v in (*distances.values(), *ratios.values()))
