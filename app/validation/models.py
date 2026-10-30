from dataclasses import dataclass


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    reason: str | None = None
    value: float | None = None


@dataclass(frozen=True)
class ValidationResult:
    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(check.reason for check in self.checks if check.reason is not None)
