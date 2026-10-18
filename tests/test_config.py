import pytest

from src.config import normalize_service_url


@pytest.mark.parametrize(
    ("input_url", "expected"),
    [
        ("http://127.0.0.1:8000", "http://127.0.0.1:8000"),
        ("https://api.example.com/", "https://api.example.com"),
        ("credit-card-fraud-api-au6f", "https://credit-card-fraud-api-au6f.onrender.com"),
    ],
)
def test_service_url_normalization(input_url: str, expected: str) -> None:
    assert normalize_service_url(input_url) == expected
