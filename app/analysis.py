from dataclasses import asdict, dataclass

from app.image_processing import decode_image
from app.image_processing.decoder import ImageDecodeError, InvalidImageSizeError
from app.landmarks.service import LandmarkError, LandmarkModelError
from app.validation import validate
from app.feature_extraction import extract_measurements, build_features, FeatureExtractionError
from app.reference_data import load_reference_data, ReferenceDataError
from app.reference_data.loader import EXPECTED_GROUPS
from app.statistics import compare_features, ModelError


class AnalysisError(Exception):
    def __init__(self, code: str, checks: tuple = ()) -> None:
        super().__init__(code)
        self.code = code
        self.checks = checks


@dataclass(frozen=True)
class AnalysisResult:
    reference_group: str
    model_mode: str
    measurements: dict
    features: dict
    z_scores: dict
    distance: float
    score: float
    quality_checks: tuple
    approximation_notice: str
    algorithm_version: str
    reference_model_version: str
    regularization_used: bool


class AnalysisService:
    def __init__(self, landmarker):
        self.landmarker = landmarker

    def analyze(self, data: bytes, reference_group: str) -> AnalysisResult:
        if reference_group not in EXPECTED_GROUPS:
            raise AnalysisError('invalid_reference_group')
        try:
            image = decode_image(data)
            landmarks = self.landmarker.analyze(image)
            quality = validate(image, landmarks)
            if not quality.passed:
                failures = tuple(check for check in quality.checks if not check.passed)
                reason = failures[0].reason
                raise AnalysisError(reason if reason in ('no_face', 'multiple_faces') else 'validation_failed', failures)
            measurements = extract_measurements(landmarks.faces[0], image.width, image.height)
            features = build_features(measurements)
            reference = load_reference_data()
            statistics = compare_features(reference_group, features.as_dict(), reference_data=reference)
            return AnalysisResult(statistics.reference_group, statistics.model_mode, asdict(measurements),
                statistics.features, statistics.z_scores, statistics.distance, statistics.score,
                quality.checks, '公開統計量に基づく近似値であり、実測人口順位ではありません。',
                '1', statistics.reference_model_version, statistics.regularization_used)
        except InvalidImageSizeError as exc:
            raise AnalysisError('image_size_limit') from exc
        except ImageDecodeError as exc:
            raise AnalysisError('invalid_image') from exc
        except LandmarkModelError as exc:
            raise AnalysisError('model_unavailable') from exc
        except (LandmarkError, FeatureExtractionError, ReferenceDataError, ModelError) as exc:
            raise AnalysisError('analysis_error') from exc
