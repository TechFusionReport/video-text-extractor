import pytest

from video_text_extractor.security import validate_public_url


@pytest.mark.parametrize("url", ["http://127.0.0.1/a", "http://localhost/a", "file:///etc/passwd"])
def test_private_or_non_http_urls_are_blocked(url: str) -> None:
    with pytest.raises(ValueError):
        validate_public_url(url)

