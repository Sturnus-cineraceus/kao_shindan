from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationConfig:
    face_size: float = 0.30
    edge_margin: float = 0.02
    yaw: float = 10.0
    pitch: float = 10.0
    roll: float = 5.0
    mouth_ratio: float = 0.08
    eye_ratio: float = 0.10
    blur: float = 80.0
    dark: float = 40.0
    bright: float = 215.0
    black_clipping: float = 0.15
    white_clipping: float = 0.15
    orthogonality_tolerance: float = 1e-3
    singularity_tolerance: float = 1e-6
    quality_max_dimension: int = 512
    black_max: int = 5
    white_min: int = 250
