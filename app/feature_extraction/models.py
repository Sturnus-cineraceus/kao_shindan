"""Immutable measurement and feature values for one face."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FaceMeasurements:
    """Pixel distances measured from one face's landmarks."""

    interpupillary_breadth: float
    interocular_breadth: float
    biectocanthion_breadth: float
    mouth_breadth: float


@dataclass(frozen=True)
class FaceFeatures:
    """Dimensionless ratios in statistics FEATURE_KEYS order."""

    interpupillary_breadth_ratio: float
    interocular_breadth_ratio: float
    mouth_breadth_ratio: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (
            self.interpupillary_breadth_ratio,
            self.interocular_breadth_ratio,
            self.mouth_breadth_ratio,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "interpupillary_breadth_ratio": self.interpupillary_breadth_ratio,
            "interocular_breadth_ratio": self.interocular_breadth_ratio,
            "mouth_breadth_ratio": self.mouth_breadth_ratio,
        }
