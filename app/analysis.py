import logging
from dataclasses import asdict, dataclass

from app.image_processing import decode_image
from app.image_processing.decoder import ImageDecodeError, InvalidImageSizeError
from app.landmarks.service import LandmarkError, LandmarkModelError
from app.validation import validate
from app.feature_extraction import extract_measurements, build_features, FeatureExtractionError
from app.reference_data import load_reference_data, ReferenceDataError
from app.reference_data.loader import EXPECTED_GROUPS
from app.statistics import compare_features, ModelError

logger = logging.getLogger(__name__)


class AnalysisError(Exception):
    def __init__(self, code: str, checks: tuple = ()) -> None:
        super().__init__(code)
        self.code, self.checks = code, checks


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
    quality_warnings: tuple
    approximation_notice: str
    algorithm_version: str
    reference_model_version: str
    regularization_used: bool


class AnalysisService:
    def __init__(self, landmarker): self.landmarker = landmarker

    def analyze(self, data: bytes, reference_group: str) -> AnalysisResult:
        logger.info("analysis started")
        logger.debug("input image bytes=%d", len(data))
        if reference_group not in EXPECTED_GROUPS:
            logger.warning("analysis rejected: invalid reference group")
            raise AnalysisError('invalid_reference_group')
        try:
            image = decode_image(data)
            logger.debug("image decoded width=%d height=%d", image.width, image.height)
            landmarks = self.landmarker.analyze(image)
            logger.debug("faces detected count=%d", landmarks.face_count)
            quality = validate(image, landmarks)
            for check in quality.checks:
                logger.debug("validation name=%s passed=%s value=%s threshold=%s missing_ids=%s",
                             check.name, check.passed, check.value, check.threshold, check.missing_ids)
            if not quality.passed:
                failures = tuple(c for c in quality.checks if not c.passed)
                if any(c.name == 'face_count' and c.reason == 'no_face' for c in failures):
                    failures = tuple(c for c in failures if c.name == 'face_count')
                logger.warning("analysis validation failed checks=%s", [c.name for c in failures])
                reason = failures[0].reason
                raise AnalysisError(reason if reason in ('no_face', 'multiple_faces') else 'validation_failed', failures)
            measurements = extract_measurements(quality.analysis_landmarks, image.width, image.height)
            logger.debug("measurements=%s", asdict(measurements))
            features = build_features(measurements)
            logger.debug("features=%s", features.as_dict())
            reference = load_reference_data()
            logger.debug("reference data loaded group=%s", reference_group)
            statistics = compare_features(reference_group, features.as_dict(), reference_data=reference)
            logger.debug("z_scores=%s distance=%s score=%s", statistics.z_scores, statistics.distance, statistics.score)
            result = AnalysisResult(statistics.reference_group, statistics.model_mode, asdict(measurements),
                statistics.features, statistics.z_scores, statistics.distance, statistics.score,
                quality.checks, quality.warnings, '公開統計量に基づく近似値であり、実測人口順位ではありません。',
                '1', statistics.reference_model_version, statistics.regularization_used)
            logger.info("analysis completed successfully")
            return result
        except AnalysisError:
            raise
        except InvalidImageSizeError as exc:
            logger.warning("analysis rejected: image size limit")
            raise AnalysisError('image_size_limit') from exc
        except ImageDecodeError as exc:
            logger.warning("analysis rejected: invalid image")
            raise AnalysisError('invalid_image') from exc
        except LandmarkModelError as exc:
            logger.error("landmark model unavailable", exc_info=True)
            raise AnalysisError('model_unavailable') from exc
        except (LandmarkError, FeatureExtractionError, ReferenceDataError, ModelError) as exc:
            logger.error("analysis processing failed", exc_info=True)
            raise AnalysisError('analysis_error') from exc
