from dataclasses import dataclass


@dataclass(frozen=True)
class FaceLandmark:
    x: float
    y: float
    z: float


TransformMatrix = tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class LandmarkResult:
    image_width: int
    image_height: int
    face_count: int
    faces: tuple[tuple[FaceLandmark, ...], ...]
    facial_transformation_matrices: tuple[TransformMatrix | None, ...]
