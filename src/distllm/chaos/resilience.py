"""Resilience scoring for chaos engineering results."""

from dataclasses import dataclass


@dataclass
class ResilienceScore:
    """Overall resilience score from a chaos scenario."""
    overall: float  # 0-100
    recovery_time_score: float  # 0-100
    data_loss_score: float  # 0-100
    error_rate_score: float  # 0-100
    scenario_name: str = ""

    @property
    def grade(self) -> str:
        if self.overall >= 90:
            return "A"
        elif self.overall >= 80:
            return "B"
        elif self.overall >= 70:
            return "C"
        elif self.overall >= 60:
            return "D"
        else:
            return "F"


class ResilienceScorer:
    """Computes resilience scores from chaos scenario results.

    Scoring weights:
    - Recovery time: 40%
    - Data loss: 20%
    - Error rate: 40%
    """

    RECOVERY_WEIGHT = 0.4
    DATA_LOSS_WEIGHT = 0.2
    ERROR_RATE_WEIGHT = 0.4

    @classmethod
    def compute_score(
        cls,
        actual_recovery_time_s: float,
        expected_recovery_time_s: float,
        has_data_loss: bool,
        actual_error_rate: float,
        max_acceptable_error_rate: float,
        scenario_name: str = "",
    ) -> ResilienceScore:
        """Compute resilience scores from scenario metrics.

        Args:
            actual_recovery_time_s: How long it took to recover.
            expected_recovery_time_s: Expected recovery time.
            has_data_loss: Whether any data was lost.
            actual_error_rate: Observed error rate during chaos.
            max_acceptable_error_rate: Maximum acceptable error rate.
            scenario_name: Name of the scenario.

        Returns:
            ResilienceScore with overall and component scores.
        """
        # Recovery time score: 100 if at or under expected, scales down to 0 at 2x expected
        if expected_recovery_time_s <= 0:
            recovery_score = 100.0
        else:
            ratio = actual_recovery_time_s / expected_recovery_time_s
            if ratio <= 1.0:
                recovery_score = 100.0
            else:
                recovery_score = max(0.0, 100.0 * (2.0 - ratio))

        # Data loss score: 100 if no loss, 0 if any loss
        data_loss_score = 0.0 if has_data_loss else 100.0

        # Error rate score: 100 if under threshold, scales down linearly
        if max_acceptable_error_rate <= 0:
            error_score = 0.0 if actual_error_rate > 0 else 100.0
        else:
            ratio = actual_error_rate / max_acceptable_error_rate
            error_score = max(0.0, 100.0 * (1.0 - ratio))

        # Weighted average
        overall = (
            cls.RECOVERY_WEIGHT * recovery_score
            + cls.DATA_LOSS_WEIGHT * data_loss_score
            + cls.ERROR_RATE_WEIGHT * error_score
        )

        return ResilienceScore(
            overall=round(overall, 1),
            recovery_time_score=round(recovery_score, 1),
            data_loss_score=round(data_loss_score, 1),
            error_rate_score=round(error_score, 1),
            scenario_name=scenario_name,
        )
