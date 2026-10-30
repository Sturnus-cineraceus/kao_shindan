from __future__ import annotations

import math

import cv2
import numpy as np

from app.feature_extraction.service import pixel_distance
from app.image_processing import DecodedImage
from app.landmarks import LandmarkResult

from .config import ValidationConfig
from .models import CheckResult, ValidationResult


FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
REQUIRED = frozenset(FACE_OVAL) | {468, 473, 133, 362, 33, 263, 61, 291,
                                  13, 14, 159, 145, 386, 374}


def _check(name: str, passed: bool, reason: str, value: float | None = None) -> CheckResult:
    return CheckResult(name, bool(passed), None if passed else reason, value)


def _pose(matrix: object, config: ValidationConfig) -> tuple[float, float, float] | None:
    if matrix is None:
        return None
    try:
        m = np.asarray(matrix, dtype=np.float64)
    except (ValueError, TypeError):
        return None
    if m.shape != (4, 4) or not np.isfinite(m).all():
        return None
    a = m[:3, :3]
    s = np.cbrt(np.linalg.det(a))
    if not np.isfinite(s) or s <= 0:
        return None
    r = a / s
    if not np.isfinite(r).all() or not np.allclose(r.T @ r, np.eye(3), rtol=0, atol=config.orthogonality_tolerance):
        return None
    yaw = math.asin(float(np.clip(-r[2, 0], -1, 1)))
    if abs(math.cos(yaw)) <= config.singularity_tolerance:
        return None
    return (math.degrees(yaw), math.degrees(math.atan2(r[2, 1], r[2, 2])),
            math.degrees(math.atan2(r[1, 0], r[0, 0])))


def _quality(image: DecodedImage, config: ValidationConfig) -> list[CheckResult]:
    rgb = image.rgb
    longest = max(image.width, image.height)
    if longest > config.quality_max_dimension:
        scale = config.quality_max_dimension / longest
        width = max(1, math.floor(image.width * scale + 0.5))
        height = max(1, math.floor(image.height * scale + 0.5))
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var())
    mean = float(gray.mean())
    black = float(np.count_nonzero(gray <= config.black_max) / gray.size)
    white = float(np.count_nonzero(gray >= config.white_min) / gray.size)
    return [
        _check("blur", blur >= config.blur, "blurred", blur),
        _check("dark", mean >= config.dark, "too_dark", mean),
        _check("bright", mean <= config.bright, "too_bright", mean),
        _check("black_clipping", black < config.black_clipping, "shadow_clipping", black),
        _check("white_clipping", white < config.white_clipping, "highlight_clipping", white),
    ]


def validate(image: DecodedImage, landmarks: LandmarkResult, config: ValidationConfig = ValidationConfig()) -> ValidationResult:
    """Return all executable checks; failed prerequisites suppress dependent checks."""
    checks = []
    count = landmarks.face_count
    checks.append(_check("face_count", count == 1, "no_face" if count == 0 else "multiple_faces", float(count)))
    if count == 1:
        points = landmarks.faces[0]
        valid = all(i < len(points) and math.isfinite(points[i].x) and math.isfinite(points[i].y)
                    and 0 <= points[i].x <= 1 and 0 <= points[i].y <= 1 for i in REQUIRED)
        checks.append(_check("required_landmarks", valid, "required_landmark_missing"))
        if valid:
            width = max(points[i].x for i in FACE_OVAL) - min(points[i].x for i in FACE_OVAL)
            checks.append(_check("face_size", width >= config.face_size, "face_too_small", width))
            edge = all(config.edge_margin <= coord <= 1 - config.edge_margin
                       for i in REQUIRED for coord in (points[i].x, points[i].y))
            checks.append(_check("edge", edge, "landmark_near_edge"))

            def distance(a: int, b: int) -> float:
                p, q = points[a], points[b]
                return pixel_distance(p.x, p.y, q.x, q.y, image.width, image.height)

            outer_width = distance(33, 263)
            checks.append(_check("outer_eye_width", math.isfinite(outer_width) and outer_width > 0,
                                 "invalid_measurement", outer_width if math.isfinite(outer_width) else None))
            for name, upper, lower, left, right, threshold, reason, is_max in (
                ("mouth", 13, 14, 61, 291, config.mouth_ratio, "mouth_open", True),
                ("right_eye", 159, 145, 33, 133, config.eye_ratio, "eye_closed", False),
                ("left_eye", 386, 374, 263, 362, config.eye_ratio, "eye_closed", False),
            ):
                numerator, denominator = distance(upper, lower), distance(left, right)
                ratio = numerator / denominator if denominator > 0 else math.nan
                passed = math.isfinite(ratio) and (ratio <= threshold if is_max else ratio >= threshold)
                checks.append(_check(name, passed, reason, ratio if math.isfinite(ratio) else None))
        matrix = landmarks.facial_transformation_matrices[0] if landmarks.facial_transformation_matrices else None
        angles = _pose(matrix, config)
        checks.append(_check("transform_matrix", angles is not None, "invalid_transform_matrix"))
        if angles is not None:
            for name, angle, limit in zip(("yaw", "pitch", "roll"), angles, (config.yaw, config.pitch, config.roll)):
                checks.append(_check(name, abs(angle) <= limit, name + "_out_of_range", angle))
    checks.extend(_quality(image, config))
    return ValidationResult(tuple(checks))
