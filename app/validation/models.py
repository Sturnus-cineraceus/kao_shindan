from dataclasses import dataclass

from app.landmarks import FaceLandmark


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    reason: str | None = None
    value: float | None = None
    threshold: float | None = None
    missing_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class QualityWarning:
    code: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    checks: tuple[CheckResult, ...]
    warnings: tuple[QualityWarning, ...] = ()
    analysis_landmarks: tuple[FaceLandmark, ...] | None = None

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(check.reason for check in self.checks if check.reason is not None)
