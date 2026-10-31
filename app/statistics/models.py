"""Delta-method model and numerical comparison, independent of the web layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi, chi2

from app.reference_data import MEASUREMENT_KEYS, ReferenceData, ReferenceDataError, load_reference_data

FloatArray = NDArray[np.float64]
MODEL_MODE = "summary_delta"
# 共分散行列が正定値でない場合の対角正則化係数（docs/statistics.md の初期値）。
EPSILON = 1e-6
FEATURE_KEYS = (
    "interpupillary_breadth_ratio",
    "interocular_breadth_ratio",
    "mouth_breadth_ratio",
)
NUMERATOR_KEYS = MEASUREMENT_KEYS[:2] + (MEASUREMENT_KEYS[3],)
DENOMINATOR_KEY = "biectocanthion_breadth"
OPTIONAL_NUMERATORS = {
    "face_aspect_ratio": "morphologic_face_height",
    "interpupillary_to_face_width_ratio": "interpupillary_breadth",
    "interocular_to_face_width_ratio": "interocular_breadth",
    "biectocanthion_to_face_width_ratio": "biectocanthion_breadth",
    "mouth_to_face_width_ratio": "mouth_breadth",
}
OPTIONAL_FEATURE_ORDER = tuple(OPTIONAL_NUMERATORS)
REPRESENTATIVE3_FEATURES = ("face_aspect_ratio", "biectocanthion_to_face_width_ratio", "mouth_to_face_width_ratio")


class ModelError(ValueError):
    """Invalid numeric input or a model that cannot be evaluated safely."""


def validate_additional_features(features: Sequence[str]) -> tuple[str, ...]:
    if isinstance(features, (str, bytes)):
        raise ModelError("invalid additional features")
    try:
        selected = tuple(features)
        if any(not isinstance(item, str) or item not in OPTIONAL_FEATURE_ORDER for item in selected) or len(set(selected)) != len(selected):
            raise ModelError("invalid additional features")
    except TypeError as exc:
        raise ModelError("invalid additional features") from exc
    return tuple(key for key in OPTIONAL_FEATURE_ORDER if key in selected)


@dataclass(frozen=True)
class ReferenceModel:
    reference_group: str
    model_mode: str
    features: tuple[str, ...]
    mean: FloatArray
    covariance: FloatArray
    standard_deviations: FloatArray
    reference_model_version: str
    epsilon: float


@dataclass(frozen=True)
class VolumeZonePoint:
    distance: float
    relative_density: float


@dataclass(frozen=True)
class VolumeZoneDistribution:
    mode_distance: float
    points: tuple[VolumeZonePoint, ...]


@dataclass(frozen=True)
class StatisticsResult:
    reference_group: str
    model_mode: str
    features: dict[str, float]
    z_scores: dict[str, float]
    distance: float
    score: float
    regularization_used: bool
    reference_model_version: str
    volume_zone_score: float
    used_features: tuple[str, ...]
    dimension: int
    volume_zone_distribution: VolumeZoneDistribution


def _finite_positive(value: float, label: str) -> float:
    if not np.isfinite(value) or value <= 0:
        raise ModelError(f"{label} must be finite and positive")
    return float(value)


def build_reference_model(
    reference_group: str,
    reference_data: ReferenceData | None = None,
    *,
    epsilon: float = EPSILON,
    additional_features: Sequence[str] = (),
) -> ReferenceModel:
    """Build the fixed-order three-ratio model for exactly one reference group."""
    selected = validate_additional_features(additional_features)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ModelError("epsilon must be finite and positive")
    data = reference_data or load_reference_data()
    try:
        group = data.get_group(reference_group)
    except ReferenceDataError as exc:
        raise ModelError(str(exc)) from exc
    denominator = group.measurements[DENOMINATOR_KEY]
    mu_y = _finite_positive(denominator.mean_mm, "denominator mean")
    sd_y = _finite_positive(denominator.sd_mm, "denominator sd")
    mean_values: list[float] = []
    variances: list[float] = []
    numerator_means: list[float] = []
    for key in NUMERATOR_KEYS:
        summary = group.measurements[key]
        mu_x = _finite_positive(summary.mean_mm, f"{key} mean")
        sd_x = _finite_positive(summary.sd_mm, f"{key} sd")
        mean_values.append(mu_x / mu_y)
        variances.append(sd_x**2 / mu_y**2 + mu_x**2 * sd_y**2 / mu_y**4)
        numerator_means.append(mu_x)
    mean = np.asarray(mean_values, dtype=np.float64)
    covariance = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            if i == j:
                covariance[i, j] = variances[i]
            else:
                covariance[i, j] = numerator_means[i] * numerator_means[j] * sd_y**2 / mu_y**4
    if selected:
        w = group.measurements["bizygomatic_breadth"]
        mu_w = _finite_positive(w.mean_mm, "face width mean")
        sd_w = _finite_positive(w.sd_mm, "face width sd")
        expanded = np.zeros((3 + len(selected), 3 + len(selected)), dtype=np.float64)
        expanded[:3, :3] = covariance
        optional_means = []
        for index, key in enumerate(selected, 3):
            numerator_key = OPTIONAL_NUMERATORS[key]
            x = group.measurements[numerator_key]
            mu_x = _finite_positive(x.mean_mm, f"{key} mean")
            sd_x = _finite_positive(x.sd_mm, f"{key} sd")
            mean = np.append(mean, mu_x / mu_w)
            expanded[index, index] = sd_x**2 / mu_w**2 + mu_x**2 * sd_w**2 / mu_w**4
            for j, core_key in enumerate(NUMERATOR_KEYS):
                if numerator_key == core_key:
                    expanded[j, index] = sd_x**2 / (mu_y * mu_w)
                elif numerator_key == DENOMINATOR_KEY:
                    expanded[j, index] = -numerator_means[j] * sd_y**2 / (mu_y**2 * mu_w)
                expanded[index, j] = expanded[j, index]
            for j, other_mean in enumerate(optional_means):
                expanded[3 + j, index] = other_mean * mu_x * sd_w**2 / mu_w**4
                expanded[index, 3 + j] = expanded[3 + j, index]
            optional_means.append(mu_x)
        covariance = expanded
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ModelError("Reference model contains non-finite values")
    if np.any(np.diag(covariance) <= 0):
        raise ModelError("Ratio variances must be positive")
    # Retain read-only arrays so the model cannot be mutated after construction.
    for array in (mean, covariance):
        array.setflags(write=False)
    return ReferenceModel(
        reference_group, MODEL_MODE, FEATURE_KEYS + selected, mean, covariance,
        np.sqrt(np.diag(covariance)), data.model_version, float(epsilon),
    )



def build_representative3_model(reference_group: str, reference_data: ReferenceData | None = None, *, epsilon: float = EPSILON) -> ReferenceModel:
    """Build the fixed three ratios with their shared face-width denominator."""
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ModelError("epsilon must be finite and positive")
    data = reference_data or load_reference_data()
    try:
        group = data.get_group(reference_group)
    except ReferenceDataError as exc:
        raise ModelError(str(exc)) from exc
    width = group.measurements["bizygomatic_breadth"]
    mu_y = _finite_positive(width.mean_mm, "face width mean")
    sd_y = _finite_positive(width.sd_mm, "face width sd")
    summaries = [group.measurements[OPTIONAL_NUMERATORS[key]] for key in REPRESENTATIVE3_FEATURES]
    means = np.asarray([_finite_positive(x.mean_mm, "numerator mean") for x in summaries])
    sds = np.asarray([_finite_positive(x.sd_mm, "numerator sd") for x in summaries])
    mean = means / mu_y
    covariance = np.outer(means, means) * sd_y**2 / mu_y**4 + np.diag(sds**2 / mu_y**2)
    mean.setflags(write=False)
    covariance.setflags(write=False)
    return ReferenceModel(reference_group, MODEL_MODE, REPRESENTATIVE3_FEATURES, mean, covariance,
                          np.sqrt(np.diag(covariance)), data.model_version, float(epsilon))

def _coerce_features(values: Mapping[str, float] | Sequence[float], keys: tuple[str, ...]) -> FloatArray:
    if isinstance(values, Mapping):
        if set(values) != set(keys):
            raise ModelError(f"features must contain exactly {keys!r}")
        raw = [values[key] for key in keys]
    else:
        if isinstance(values, (str, bytes)):
            raise ModelError("features must contain exactly three numeric values")
        try:
            raw = list(values)
        except TypeError as exc:
            raise ModelError("features must contain exactly three numeric values") from exc
        if len(raw) != len(keys):
            raise ModelError("features must match the model feature count and order")
    try:
        vector = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ModelError("features must be numeric") from exc
    if vector.shape != (len(keys),) or not np.isfinite(vector).all():
        raise ModelError("features must contain finite numeric values for each model feature")
    return vector


def _volume_zone_score(distance: float, dimension: int = 3) -> float:
    if not np.isfinite(distance) or distance < 0:
        raise ModelError("Distance must be finite and non-negative")
    if not isinstance(dimension, int) or dimension < 3:
        raise ModelError("Dimension must be an integer greater than or equal to 3")
    mode = float(np.sqrt(dimension - 1))
    peak = float(chi.pdf(mode, df=dimension))
    if not np.isfinite(peak) or peak <= 0:
        raise ModelError("Volume zone peak must be finite and positive")
    value = float(100.0 * chi.pdf(distance, df=dimension) / peak)
    if not np.isfinite(value):
        raise ModelError("Volume zone calculation produced a non-finite value")
    return max(0.0, min(100.0, value))


def _volume_zone_distribution(distance: float, dimension: int) -> VolumeZoneDistribution:
    _volume_zone_score(distance, dimension)  # Validate before constructing the range.
    mode = float(np.sqrt(dimension - 1))
    maximum = max(float(chi.ppf(0.999, df=dimension)), distance * 1.05, mode * 1.05)
    if not np.isfinite(maximum) or maximum <= 0:
        raise ModelError("Distribution range must be finite and positive")
    points = tuple(VolumeZonePoint(float(x), _volume_zone_score(float(x), dimension))
                   for x in np.linspace(0.0, maximum, 101))
    if any(not np.isfinite(point.distance) for point in points):
        raise ModelError("Distribution contains non-finite distances")
    return VolumeZoneDistribution(mode, points)


def compare_features(
    reference_group: str,
    features: Mapping[str, float] | Sequence[float],
    *,
    reference_data: ReferenceData | None = None,
    epsilon: float = EPSILON,
    additional_features: Sequence[str] = (),
    calculation_profile: str = "legacy",
) -> StatisticsResult:
    """Return approximate z scores, Mahalanobis distance, and chi-square score."""
    if calculation_profile == "representative3":
        if additional_features:
            raise ModelError("additional features are not allowed")
        model = build_representative3_model(reference_group, reference_data, epsilon=epsilon)
    elif calculation_profile == "legacy":
        model = build_reference_model(reference_group, reference_data, epsilon=epsilon, additional_features=additional_features)
    else:
        raise ModelError("invalid calculation profile")
    dimension = len(model.features)
    observed = _coerce_features(features, model.features)
    delta = observed - model.mean
    try:
        # 同じ元寸法から作る比率が多いと、一次近似の行列は特異になりうる。
        if additional_features and np.linalg.eigvalsh(model.covariance)[0] <= np.max(np.diag(model.covariance)) * 1e-12:
            raise np.linalg.LinAlgError("covariance is numerically singular")
        np.linalg.cholesky(model.covariance)
        covariance_for_solve = model.covariance
        regularization_used = False
    except np.linalg.LinAlgError:
        covariance_for_solve = model.covariance + model.epsilon * np.diag(np.diag(model.covariance))
        if not np.isfinite(covariance_for_solve).all():
            raise ModelError("Regularized covariance contains non-finite values")
        try:
            np.linalg.cholesky(covariance_for_solve)
        except np.linalg.LinAlgError as exc:
            raise ModelError("Covariance matrix remains non-positive-definite after regularization") from exc
        regularization_used = True
    try:
        solved = np.linalg.solve(covariance_for_solve, delta)
    except np.linalg.LinAlgError as exc:
        raise ModelError("Covariance matrix is singular") from exc
    d_squared = float(delta @ solved)
    if not np.isfinite(d_squared):
        raise ModelError("Distance calculation produced a non-finite value")
    if d_squared < 0:
        # 浮動小数点誤差で生じた微小な負値だけをゼロへ丸める許容幅。
        tolerance = 1e-12 * max(1.0, float(np.linalg.norm(delta) ** 2))
        if d_squared < -tolerance:
            raise ModelError("Distance squared is materially negative")
        d_squared = 0.0
    distance = float(np.sqrt(d_squared))
    score = float(100.0 * chi2.sf(d_squared, df=dimension))
    if not np.isfinite(score):
        raise ModelError("Score calculation produced a non-finite value")
    return StatisticsResult(
        reference_group=model.reference_group,
        model_mode=model.model_mode,
        features={key: float(observed[i]) for i, key in enumerate(model.features)},
        z_scores={key: float(delta[i] / model.standard_deviations[i]) for i, key in enumerate(model.features)},
        distance=distance,
        score=score,
        regularization_used=regularization_used,
        reference_model_version=model.reference_model_version,
        volume_zone_score=_volume_zone_score(distance, dimension),
        used_features=model.features,
        dimension=dimension,
        volume_zone_distribution=_volume_zone_distribution(distance, dimension),
    )
