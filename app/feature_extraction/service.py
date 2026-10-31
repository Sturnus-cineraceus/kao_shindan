"""Pure numeric conversion from application landmarks to statistics features."""

from __future__ import annotations

import math
from collections.abc import Sequence

from app.landmarks import FaceLandmark

from .models import FaceFeatures, FaceMeasurements


class FeatureExtractionError(ValueError):
    """Invalid landmarks, dimensions, or derived measurements/features."""


# MediaPipe Face Landmarker の対応点。各ペアの pixel distance が
# statistics に渡す4計測値となり、外眼角間幅は後段の比率の共通分母になる。
_MEASUREMENT_PAIRS = (
    ("interpupillary_breadth", 468, 473),
    ("interocular_breadth", 133, 362),
    ("biectocanthion_breadth", 33, 263),
    ("mouth_breadth", 61, 291),
)


def _finite_coordinate(value: float, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FeatureExtractionError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise FeatureExtractionError(f"{label} must be finite")
    return number


def extract_measurements(
    landmarks: Sequence[FaceLandmark], image_width: int, image_height: int
) -> FaceMeasurements:
    """Measure the four specified pixel distances from a single face."""
    if image_width <= 0 or image_height <= 0:
        raise FeatureExtractionError("image dimensions must be positive")

    values: dict[str, float] = {}
    for name, first_index, second_index in _MEASUREMENT_PAIRS:
        if len(landmarks) <= max(first_index, second_index):
            raise FeatureExtractionError(f"required landmark index is missing: {max(first_index, second_index)}")
        first = landmarks[first_index]
        second = landmarks[second_index]
        x1 = _finite_coordinate(first.x, f"landmark {first_index}.x")
        y1 = _finite_coordinate(first.y, f"landmark {first_index}.y")
        x2 = _finite_coordinate(second.x, f"landmark {second_index}.x")
        y2 = _finite_coordinate(second.y, f"landmark {second_index}.y")
        distance = pixel_distance(x1, y1, x2, y2, image_width, image_height)
        if not math.isfinite(distance):
            raise FeatureExtractionError(f"{name} is not finite")
        values[name] = distance

    return FaceMeasurements(**values)


def pixel_distance(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> float:
    """Calculate a 2D distance in image pixels from normalized coordinates."""
    return math.hypot((x2 - x1) * width, (y2 - y1) * height)


def build_features(measurements: FaceMeasurements) -> FaceFeatures:
    """Convert pixel measurements to three fixed-order dimensionless ratios."""
    denominator = _finite_coordinate(
        measurements.biectocanthion_breadth, "biectocanthion_breadth"
    )
    if denominator <= 0:
        raise FeatureExtractionError("biectocanthion_breadth must be positive")

    ratios: list[float] = []
    for name in ("interpupillary_breadth", "interocular_breadth", "mouth_breadth"):
        numerator = _finite_coordinate(getattr(measurements, name), name)
        ratio = numerator / denominator
        if not math.isfinite(ratio):
            raise FeatureExtractionError(f"{name} ratio is not finite")
        ratios.append(ratio)
    return FaceFeatures(*ratios)


# AISTの形態学顔高・頬弓幅に対応させる写真上の代理点。
FACE_ASPECT_LANDMARKS = frozenset((168, 152, 234, 454))


def extract_face_aspect_ratio(landmarks: Sequence[FaceLandmark], image_width: int, image_height: int) -> float:
    """写真上の代理点をピクセル距離に変換し、顔の縦横比を近似する。"""
    if image_width <= 0 or image_height <= 0:
        raise FeatureExtractionError("image dimensions must be positive")
    coordinates = {}
    for index in FACE_ASPECT_LANDMARKS:
        if index >= len(landmarks):
            raise FeatureExtractionError(f"required landmark index is missing: {index}")
        coordinates[index] = (_finite_coordinate(landmarks[index].x, f"landmark {index}.x"),
                              _finite_coordinate(landmarks[index].y, f"landmark {index}.y"))

    def distance(a: int, b: int) -> float:
        return pixel_distance(*coordinates[a], *coordinates[b], image_width, image_height)

    height, width = distance(168, 152), distance(234, 454)
    if not math.isfinite(width) or width <= 0:
        raise FeatureExtractionError("face width must be finite and positive")
    ratio = height / width
    if not math.isfinite(ratio) or ratio <= 0:
        raise FeatureExtractionError("face aspect ratio must be finite and positive")
    return ratio
