import math
from dataclasses import replace

import cv2
import numpy as np
import pytest

from app.image_processing import DecodedImage
from app.landmarks import FaceLandmark, LandmarkResult
from app.validation import validate
from app.validation.config import ValidationConfig
from app.validation.service import FACE_OVAL, _pose, _quality


def image(rgb=None):
    if rgb is None:
        grid = (np.indices((100, 100))[1] // 4) % 2 * 100 + 80
        rgb = np.repeat(grid[:, :, None], 3, axis=2).astype(np.uint8)
    return DecodedImage(rgb, rgb.shape[1], rgb.shape[0])


def points():
    p = [FaceLandmark(.5, .5, 0) for _ in range(478)]
    for i in FACE_OVAL:
        p[i] = FaceLandmark(.25, .5, 0)
    p[FACE_OVAL[1]] = FaceLandmark(.75, .5, 0)
    for i, x, y in [(61,.25,.5),(291,.75,.5),(13,.5,.5),(14,.5,.52),
                     (33,.25,.4),(133,.5,.4),(159,.4,.4),(145,.4,.45),
                     (263,.75,.4),(362,.5,.4),(386,.6,.4),(374,.6,.45)]:
        p[i] = FaceLandmark(x, y, 0)
    return p


def result(p=None, matrix=None, count=1):
    if p is None:
        p = points()
    if matrix is None:
        matrix = np.eye(4)
    return LandmarkResult(100, 100, count, (tuple(p),) if count else (), (matrix,))


def check(res, name):
    return next(c for c in res.checks if c.name == name)


def rotate(yaw=0, pitch=0, roll=0):
    y, p, r = map(math.radians, (yaw, pitch, roll))
    ry = np.array([[math.cos(y),0,math.sin(y)],[0,1,0],[-math.sin(y),0,math.cos(y)]])
    rx = np.array([[1,0,0],[0,math.cos(p),-math.sin(p)],[0,math.sin(p),math.cos(p)]])
    rz = np.array([[math.cos(r),-math.sin(r),0],[math.sin(r),math.cos(r),0],[0,0,1]])
    m = np.eye(4)
    m[:3,:3] = rz @ ry @ rx
    return m


@pytest.mark.parametrize('count,reason', [(0,'no_face'),(1,None),(2,'multiple_faces')])
def test_face_count(count, reason):
    res = validate(image(), result(count=count))
    assert check(res, 'face_count').reason == reason
    if count != 1:
        assert not any(c.name in ('mouth', 'yaw', 'required_landmarks') for c in res.checks)


@pytest.mark.parametrize('x,passed', [(.299,False),(.30,True),(.301,True)])
def test_face_width(x, passed):
    p = points()
    for i in FACE_OVAL:
        p[i] = FaceLandmark(.3, .5, 0)
    p[FACE_OVAL[1]] = FaceLandmark(.3 + x, .5, 0)
    assert check(validate(image(), result(p)), 'face_size').passed == passed


@pytest.mark.parametrize('x,passed', [(.019,False),(.02,True),(.98,True),(.981,False)])
def test_edge(x, passed):
    p = points()
    p[468] = FaceLandmark(x, .5, 0)
    assert check(validate(image(), result(p)), 'edge').passed == passed


@pytest.mark.parametrize('coord', [float('nan'), float('inf'), -.01, 1.01])
def test_invalid_required_point(coord):
    p = points()
    p[468] = FaceLandmark(coord, .5, 0)
    res = validate(image(), result(p))
    assert 'required_landmark_missing' in res.reasons
    assert not any(c.name == 'mouth' for c in res.checks)


def test_missing_point():
    res = validate(image(), result(points()[:473]))
    assert 'required_landmark_missing' in res.reasons


def test_optional_landmark_is_only_required_when_selected():
    p = points()
    p[168] = FaceLandmark(float('nan'), .5, 0)
    assert check(validate(image(), result(p)), 'required_landmarks').passed
    selected = validate(image(), result(p), additional_required_landmarks=frozenset((168, 152, 234, 454)))
    assert not check(selected, 'required_landmarks').passed
    assert 168 in check(selected, 'required_landmarks').missing_ids


@pytest.mark.parametrize('dy,passed', [(.04,True),(.041,False)])
def test_mouth(dy, passed):
    p = points()
    p[13] = FaceLandmark(.5, 0, 0)
    p[14] = FaceLandmark(.5, dy, 0)
    assert check(validate(image(), result(p)), 'mouth').passed == passed


@pytest.mark.parametrize('side,dy,passed', [('right_eye',.024,False),('left_eye',.024,False),('right_eye',.025,True),('left_eye',.025,True)])
def test_eyes(side, dy, passed):
    p = points()
    a, b = (159,145) if side == 'right_eye' else (386,374)
    p[a] = FaceLandmark(.4, 0, 0)
    p[b] = FaceLandmark(.4, dy, 0)
    assert check(validate(image(), result(p)), side).passed == passed


def test_zero_denominators_fail():
    p = points()
    p[291] = p[61]
    p[133] = p[33]
    p[362] = p[263]
    res = validate(image(), result(p))
    assert all(not check(res, n).passed for n in ('mouth', 'right_eye', 'left_eye'))


def test_zero_outer_eye_width_fails():
    p = points()
    p[263] = p[33]
    assert 'invalid_measurement' in validate(image(), result(p)).reasons


@pytest.mark.parametrize('variance,passed', [(79.999, False), (80.0, True), (80.001, True)])
def test_blur_threshold(monkeypatch, variance, passed):
    class Laplacian:
        def var(self):
            return variance

    monkeypatch.setattr(cv2, 'Laplacian', lambda *args, **kwargs: Laplacian())
    assert check(validate(image(), result()), 'blur').passed == passed


@pytest.mark.parametrize('axis,angle,passed,warning', [
    (axis, sign * angle, passed, warning)
    for axis, cases in (
        ('yaw', [(10, True, False), (10.01, True, True), (15, True, True), (15.01, False, False)]),
        ('pitch', [(10, True, False), (10.01, True, True), (20, True, True), (20.01, False, False)]),
        ('roll', [(0, True, False), (10, True, False), (20, True, False), (30, True, False), (30.01, False, False)]),
    ) for angle, passed, warning in cases for sign in (-1, 1)
])
def test_pose_thresholds(axis, angle, passed, warning):
    res = validate(image(), result(matrix=rotate(**{axis: angle})))
    assert check(res, axis).passed == passed
    assert bool(res.warnings) == warning
    if warning:
        assert res.warnings[0].code == 'HEAD_' + axis.upper() + '_WARNING'


def test_euler_convention_and_scale():
    m = rotate(yaw=-7, pitch=4, roll=3)
    m[:3,:3] *= 2
    assert _pose(m, ValidationConfig()) == pytest.approx((-7, 4, 3))


@pytest.mark.parametrize('matrix', [None, [[1,2]], np.full((4,4),np.nan), np.diag([1,1,2,1]), np.diag([-1,1,1,1]), rotate(yaw=90)])
def test_invalid_matrix(matrix):
    lm = replace(result(), facial_transformation_matrices=() if matrix is None else (matrix,))
    assert 'invalid_transform_matrix' in validate(image(), lm).reasons


def test_quality_and_clipping_boundaries():
    def quality(values):
        rgb = np.repeat(np.array(values, dtype=np.uint8).reshape(10,10,1), 3, axis=2)
        return {c.name:c for c in _quality(image(rgb), ValidationConfig())}
    assert not quality([39]*100)['dark'].passed
    assert quality([40]*100)['dark'].passed
    assert quality([215]*100)['bright'].passed
    assert not quality([216]*100)['bright'].passed
    assert not quality([5]*15 + [100]*85)['black_clipping'].passed
    assert quality([5]*14 + [100]*86)['black_clipping'].passed
    assert not quality([250]*15 + [100]*85)['white_clipping'].passed
    assert quality([250]*14 + [100]*86)['white_clipping'].passed
    assert not quality([100]*100)['blur'].passed
    assert check(validate(image(), result()), 'blur').passed


def test_resize_and_source_unchanged():
    rgb = np.zeros((1001, 2, 3), dtype=np.uint8)
    rgb[:, :, 0] = 90
    original = rgb.copy()
    im = image(rgb)
    checks = _quality(im, ValidationConfig())
    resized = cv2.resize(rgb, (1, 512), interpolation=cv2.INTER_AREA)
    assert checks[1].value == float(cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).mean())
    np.testing.assert_array_equal(rgb, original)


def test_success():
    assert validate(image(), result()).passed


@pytest.mark.parametrize('angle', [10, -10, 20, -20, 30, -30])
def test_roll_correction_and_distance_invariance(angle):
    from app.validation.service import _normalize_roll
    from app.feature_extraction import extract_measurements, build_features
    original = tuple(points())
    # Use a non-square image and the same eye midpoint for the synthetic rotation.
    rotated = _normalize_roll(original, 800, 400, -angle)
    corrected = _normalize_roll(rotated, 800, 400, angle)
    for expected, actual in zip(original, corrected):
        assert (actual.x, actual.y, actual.z) == pytest.approx((expected.x, expected.y, expected.z))
    before = extract_measurements(rotated, 800, 400)
    after = extract_measurements(corrected, 800, 400)
    from dataclasses import astuple
    assert astuple(before) == pytest.approx(astuple(after))
    assert build_features(before).as_tuple() == pytest.approx(build_features(after).as_tuple())


def test_edge_uses_original_coordinates():
    p = points()
    p[468] = FaceLandmark(.03, .03, 0)
    res = validate(image(), result(p, rotate(roll=30)))
    assert check(res, 'edge').passed
    assert res.analysis_landmarks[468].x < 0
