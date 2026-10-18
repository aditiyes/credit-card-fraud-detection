import pytest

from src.decisions import decision_for_probability


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.0, "APPROVE"), (0.299, "APPROVE"), (0.30, "REVIEW"), (0.699, "REVIEW"), (0.70, "BLOCK"), (1.0, "BLOCK")],
)
def test_decision_thresholds(probability: float, expected: str) -> None:
    assert decision_for_probability(probability) == expected


def test_out_of_range_probability_is_rejected() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        decision_for_probability(1.1)
