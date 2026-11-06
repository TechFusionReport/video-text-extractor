"""Resolve TikTok share redirects without handing non-video pages to yt-dlp."""

import re
from http.cookiejar import LoadError, MozillaCookieJar
from typing import Any
from urllib.parse import urljoin, urlsplit

from curl_cffi import CurlOpt
from curl_cffi.requests import Session
from yt_dlp.networking._curlcffi import CurlCFFIRH

from .security import validate_public_url


TIKTOK_HOSTS = {"tiktok.com", "www.tiktok.com", "m.tiktok.com", "vm.tiktok.com", "vt.tiktok.com"}
SHORT_HOSTS = {"vm.tiktok.com", "vt.tiktok.com"}
VIDEO_PATH = re.compile(r"^/@([\w.]+)/video/(\d+)/?$", re.ASCII)
REDIRECT_STATUSES = {301, 302, 303, 307, 308}
MAX_RESOLVER_REQUESTS = 6


class TikTokShortLinkError(ValueError):
    """A share link did not identify a canonical video; safe to show to clients."""


def is_tiktok_short_url(url: str) -> bool:
    parsed = urlsplit(url)
    return parsed.hostname in SHORT_HOSTS or (
        parsed.hostname in TIKTOK_HOSTS and parsed.path.startswith("/t/")
    )


def canonical_tiktok_url(url: str) -> str | None:
    parsed = urlsplit(url)
    if parsed.hostname not in TIKTOK_HOSTS or parsed.username is not None or parsed.password is not None:
        return None
    match = VIDEO_PATH.fullmatch(parsed.path)
    if match:
        return f"https://www.tiktok.com/@{match[1]}/video/{match[2]}"
    return None


def _validate_hop(url: str) -> None:
    try:
        parsed = urlsplit(url)
        allowed = (
            parsed.scheme == "https"
            and parsed.hostname in TIKTOK_HOSTS
            and parsed.port in (None, 443)
            and parsed.username is None
            and parsed.password is None
        )
    except ValueError:
        allowed = False
    if not allowed:
        raise TikTokShortLinkError("TikTok short link could not be resolved (unsafe redirect)")
    # Check every redirect before it can cause a request, including canonical targets.
    validate_public_url(url)


def resolve_tiktok_short_url(url: str, request_options: dict[str, Any]) -> str:
    proxy = request_options.get("proxy")
    if not proxy:
        raise TikTokShortLinkError("TikTok short link could not be resolved (a configured proxy is required)")

    cookies = MozillaCookieJar()
    if cookiefile := request_options.get("cookiefile"):
        try:
            cookies.load(cookiefile, ignore_discard=True)
        except (LoadError, OSError):
            raise TikTokShortLinkError(
                "TikTok short link could not be resolved (configured cookies could not be loaded)"
            ) from None

    target = request_options.get("impersonate")
    # Reuse yt-dlp's installed mapping, including version/OS aliases. Building
    # aliases by concatenation is wrong for Safari versions and mobile targets.
    impersonate = None
    if target:
        impersonate = next((
            name for supported, name in CurlCFFIRH._SUPPORTED_IMPERSONATE_TARGET_MAP.items()
            if target in supported
        ), None)
        if impersonate is None:
            raise TikTokShortLinkError(
                "TikTok short link could not be resolved (configured browser identity is unavailable)"
            )
    _validate_hop(url)
    with Session(
        proxy=proxy,
        trust_env=False,
        curl_options={CurlOpt.NOPROXY: ""},
        cookies=cookies,
        impersonate=impersonate,
        allow_redirects=False,
        timeout=20,
    ) as session:
        current = url
        seen: set[str] = set()
        for _ in range(MAX_RESOLVER_REQUESTS):
            if current in seen:
                raise TikTokShortLinkError("TikTok short link could not be resolved (redirect loop)")
            seen.add(current)
            response = session.get(current, allow_redirects=False)
            try:
                status = response.status_code
                location = response.headers.get("Location")
            finally:
                response.close()
            if status in (403, 429):
                raise TikTokShortLinkError(
                    f"TikTok short link could not be resolved (TikTok denied the request: HTTP {status})"
                )
            if status >= 500:
                # Preserve the API's referenced 502 behavior for upstream faults.
                raise RuntimeError(f"TikTok short-link resolution failed (upstream HTTP {status})")
            if status not in REDIRECT_STATUSES or not location:
                raise TikTokShortLinkError(
                    "TikTok short link could not be resolved (no canonical video URL was returned)"
                )
            current = urljoin(current, location)
            _validate_hop(current)
            if canonical := canonical_tiktok_url(current):
                return canonical
            if urlsplit(current).path.rstrip("/") in ("", "/explore", "/login", "/foryou"):
                # A homepage redirect does not prove whether the link expired or
                # access was restricted. Do not guess, or send it to yt-dlp.
                raise TikTokShortLinkError(
                    "TikTok short link could not be resolved (TikTok returned a non-video redirect)"
                )
    raise TikTokShortLinkError("TikTok short link could not be resolved (too many redirects)")
