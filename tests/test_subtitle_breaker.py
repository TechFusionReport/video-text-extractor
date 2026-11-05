import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from yt_dlp.networking.exceptions import HTTPError
from yt_dlp.utils import DownloadError

from video_text_extractor import extractor as module, main
from video_text_extractor.config import Settings
from video_text_extractor.models import TranscriptSegment
from video_text_extractor.subtitle_breaker import SubtitleCircuitBreaker


@pytest.fixture
def harness(monkeypatch):
    now = [100.0]
    breaker = SubtitleCircuitBreaker(clock=lambda: now[0])
    monkeypatch.setattr(module, 'subtitle_breaker', breaker)
    monkeypatch.setattr(main, 'subtitle_breaker', breaker)
    settings = Settings(subtitle_breaker_seconds=1800)
    extractor = module.VideoExtractor(settings)
    monkeypatch.setattr(module, 'validate_public_url', lambda value: None)
    # A sentinel verifies every metadata attempt preserves shared request options.
    identity = object()
    original_options = extractor._yt_dlp_request_options
    monkeypatch.setattr(extractor, '_yt_dlp_request_options',
                        lambda: {**original_options(), 'proxy': identity})
    monkeypatch.setattr(extractor, '_download_audio', lambda *args: Path('unused.wav'))
    transcribe = Mock(return_value=([TranscriptSegment(start=0, end=1, text='spoken words')], None))
    monkeypatch.setattr(extractor, '_transcribe', transcribe)
    state = SimpleNamespace(now=now, breaker=breaker, settings=settings, extractor=extractor,
                            calls=[], outcome='429', transcribe=transcribe, gate=None)

    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options
            state.calls.append(options.copy())
            assert options['proxy'] is identity

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def _write_subtitles(self):
            if state.outcome == '429':
                inner = HTTPError(SimpleNamespace(status=429, reason='throttled'))
                raise DownloadError('subtitle fetch failed', exc_info=(HTTPError, inner, None))
            if state.outcome == 'other':
                raise DownloadError('subtitle fetch failed: HTTP 503')
            if state.outcome == 'unrelated':
                raise RuntimeError('metadata failure')
            if state.outcome == 'captions':
                path = Path(self.options['outtmpl']).parent / 'source.en.json3'
                path.write_text(json.dumps({'events': [{'segs': [{'utf8': 'caption words'}]}]}))

        def extract_info(self, value, download):
            assert download is True
            if 'postprocessors' in self.options:
                (Path(self.options['outtmpl']).parent / 'audio.wav').touch()
                return {}
            assert self.options['writesubtitles'] == self.options['writeautomaticsub']
            if self.options['writesubtitles']:
                if state.gate:
                    entered, release = state.gate
                    entered.set()
                    assert release.wait(timeout=5)
                self._write_subtitles()
            return {'duration': 30, 'description': 'description'}

    monkeypatch.setattr(module, 'YoutubeDL', FakeYoutubeDL)
    # Public URL validation and all I/O are mocked, so no source address is needed.
    state.run = lambda forced=False: extractor.extract('', None, forced)
    return state


def assert_whisper_warning(result):
    assert result.method == 'whisper'
    assert 'subtitles_unavailable' in result.warnings


def test_429_opens_and_next_request_skips_subtitles(harness, caplog):
    h = harness
    with caplog.at_level(logging.INFO):
        assert_whisper_warning(h.run())
        assert len(h.calls) == 2
        assert h.breaker.status(1800).model_dump() == {
            'breaker_open': True, 'retry_after_seconds': 1800,
        }
        h.calls.clear()
        assert_whisper_warning(h.run())
        assert len(h.calls) == 1
        assert h.calls[0]['writesubtitles'] is False
        assert h.calls[0]['writeautomaticsub'] is False
    assert h.transcribe.call_count == 2
    assert [r.getMessage() for r in caplog.records] == ['Subtitle breaker opened window_seconds=1800']


@pytest.mark.parametrize('outcome', ['captions', 'none', '429'])
def test_probe_success_closes_or_429_reopens(harness, outcome, caplog):
    h = harness
    h.run()
    h.now[0] += 1800
    h.calls.clear()
    h.outcome = outcome
    with caplog.at_level(logging.INFO):
        result = h.run()
    assert h.calls[0]['writesubtitles'] is True
    status = h.breaker.status(1800)
    if outcome == '429':
        assert_whisper_warning(result)
        assert len(h.calls) == 2
        assert status.breaker_open and status.retry_after_seconds == 1800
        assert caplog.records[-1].getMessage() == 'Subtitle breaker opened window_seconds=1800'
    else:
        assert len(h.calls) == 1
        assert not status.breaker_open and status.retry_after_seconds == 0
        assert 'subtitles_unavailable' not in result.warnings
        assert result.method == ('captions' if outcome == 'captions' else 'whisper')
        assert caplog.records[-1].getMessage() == 'Subtitle breaker closed'


def test_non_429_failure_retries_without_opening(harness):
    h = harness
    h.outcome = 'other'
    for _ in range(2):
        assert_whisper_warning(h.run())
    assert [c['writesubtitles'] for c in h.calls] == [True, False, True, False]
    assert h.breaker.status(1800).retry_after_seconds == 0


@pytest.mark.parametrize('outcome', ['other', 'unrelated'])
def test_unsuccessful_non_429_probe_leaves_lease_unchanged(harness, outcome):
    h = harness
    h.run()
    h.now[0] += 1800
    h.outcome = outcome
    if outcome == 'other':
        assert_whisper_warning(h.run())
    else:
        with pytest.raises(RuntimeError, match='metadata failure'):
            h.run()
    assert h.breaker.status(1800).retry_after_seconds == 60


def test_forced_transcription_does_not_touch_breaker(harness, monkeypatch):
    h = harness
    h.run()
    before = h.breaker.status(1800)
    h.calls.clear()
    with monkeypatch.context() as patch:
        for method in ('acquire', 'trip', 'succeeded', 'status'):
            patch.setattr(h.breaker, method, Mock(side_effect=AssertionError('breaker touched')))
        result = h.run(forced=True)
    assert h.breaker.status(1800) == before
    assert result.method == 'whisper'
    assert 'subtitles_unavailable' not in result.warnings
    assert len(h.calls) == 1 and h.calls[0]['writesubtitles'] is False


def test_disabled_breaker_keeps_retrying_and_has_no_clock_access(harness):
    h = harness
    h.settings.subtitle_breaker_seconds = 0
    h.breaker._clock = Mock(side_effect=AssertionError('disabled clock touched'))
    for _ in range(2):
        assert_whisper_warning(h.run())
    assert [c['writesubtitles'] for c in h.calls] == [True, False, True, False]
    assert h.breaker.status(0).model_dump() == {'breaker_open': False, 'retry_after_seconds': 0}


def test_expired_breaker_allows_only_one_concurrent_probe(harness):
    h = harness
    h.run()
    h.now[0] += 1800
    h.calls.clear()
    h.outcome = 'captions'
    entered, release = threading.Event(), threading.Event()
    h.gate = entered, release
    start = threading.Barrier(8)

    def run():
        start.wait(timeout=5)
        return h.run()

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(run) for _ in range(8)]
        try:
            assert entered.wait(timeout=5)
            # All seven followers must finish while the probe is still pending.
            done = threading.Event()
            count = [0]
            lock = threading.Lock()

            def finished(future):
                with lock:
                    count[0] += 1
                    if count[0] == 7:
                        done.set()

            for future in futures:
                future.add_done_callback(finished)
            assert done.wait(timeout=5)
            assert h.breaker.status(1800).retry_after_seconds == 60
            assert sum(c['writesubtitles'] for c in h.calls) == 1
        finally:
            release.set()
        results = [future.result(timeout=5) for future in futures]
    assert len(h.calls) == 8
    assert sum(r.method == 'captions' for r in results) == 1
    assert sum('subtitles_unavailable' in r.warnings for r in results) == 7
    assert h.breaker.status(1800).retry_after_seconds == 0


def test_stale_probe_cannot_close_newer_throttle():
    now = [100.0]
    breaker = SubtitleCircuitBreaker(lambda: now[0])
    breaker.trip(1800, None)
    now[0] += 1800
    _, probe = breaker.acquire(1800)
    breaker.trip(1800, None)
    breaker.succeeded(probe)
    assert breaker.status(1800).retry_after_seconds == 1800


@pytest.mark.parametrize('link', ['exc_info', '__cause__', '__context__'])
def test_http_status_detection_walks_exception_chain(link):
    inner = HTTPError(SimpleNamespace(status=429, reason='rate limited'))
    # Avoid relying on HTTPError's formatted message to detect this status.
    inner.args = ('opaque failure',)
    outer = DownloadError('subtitle fetch failed')
    if link == 'exc_info':
        outer.exc_info = (HTTPError, inner, None)
    else:
        setattr(outer, link, inner)
    inner.__context__ = outer  # Also verify cyclic chains terminate.
    assert module._is_http_429(outer)


@pytest.mark.parametrize('message,expected', [
    ('HTTP 429', True), ('TOO MANY REQUESTS', True), ('HTTP 503', False),
])
def test_message_fallback_and_cyclic_chain(message, expected):
    error = DownloadError(message)
    error.__context__ = error
    assert module._is_http_429(error) is expected


def test_non_429_http_error_does_not_trip():
    inner = HTTPError(SimpleNamespace(status=503, reason='unavailable'))
    assert not module._is_http_429(DownloadError('subtitle failed', exc_info=(HTTPError, inner, None)))


@pytest.mark.parametrize('value', [-1, 86401, 1.5, 'not-an-integer'])
def test_config_rejects_invalid_breaker_seconds(value):
    with pytest.raises(ValidationError):
        Settings(subtitle_breaker_seconds=value)


@pytest.mark.parametrize('value', [0, 1800, 86400])
def test_config_accepts_breaker_bounds(monkeypatch, value):
    monkeypatch.setenv('SUBTITLE_BREAKER_SECONDS', str(value))
    assert Settings().subtitle_breaker_seconds == value


def test_info_authentication_state_and_schema(harness, monkeypatch):
    h = harness
    monkeypatch.setattr(main, 'get_settings', lambda: h.settings)
    client = TestClient(main.app)
    auth = {'Authorization': 'Bearer ' + os.environ['EXTRACTOR_API_KEY']}
    assert client.get('/v1/info').status_code == 401

    def status():
        response = client.get('/v1/info', headers=auth)
        assert response.status_code == 200
        body = response.json()
        assert set(body['subtitles']) == {'breaker_open', 'retry_after_seconds'}
        assert '://' not in response.text
        assert os.environ['EXTRACTOR_API_KEY'] not in response.text
        return body['subtitles']

    assert status() == {'breaker_open': False, 'retry_after_seconds': 0}
    h.run()
    h.now[0] += 0.2
    assert status() == {'breaker_open': True, 'retry_after_seconds': 1800}
    h.now[0] += 1799.8
    assert status() == {'breaker_open': False, 'retry_after_seconds': 0}
    # Info polling must not claim a probe lease.
    allowed, probe = h.breaker.acquire(1800)
    assert allowed and probe is not None
    assert status() == {'breaker_open': True, 'retry_after_seconds': 60}
    h.breaker.succeeded(probe)
    assert status() == {'breaker_open': False, 'retry_after_seconds': 0}
    schema = client.get('/openapi.json').json()
    assert 'subtitles' in schema['components']['schemas']['InfoResponse']['properties']


def test_actual_audio_download_keeps_shared_proxy_options(harness, monkeypatch):
    h = harness
    monkeypatch.setattr(h.extractor, '_download_audio',
                        module.VideoExtractor._download_audio.__get__(h.extractor))
    assert_whisper_warning(h.run())
    assert len(h.calls) == 3  # Subtitle attempt, metadata retry, existing audio download.
    h.calls.clear()
    assert_whisper_warning(h.run())
    assert len(h.calls) == 2  # Exactly one metadata attempt, then existing audio download.
    assert h.calls[0]['writesubtitles'] is False
    assert 'postprocessors' in h.calls[1]


def test_unrelated_429_keeps_fatal_behavior_and_does_not_trip(harness, monkeypatch):
    h = harness
    monkeypatch.setattr(h.extractor, '_metadata', Mock(side_effect=DownloadError('HTTP 429')))
    monkeypatch.setattr(main, 'get_extractor', lambda: h.extractor)
    with pytest.raises(DownloadError):
        h.run()
    assert not h.breaker.status(1800).breaker_open
    h.transcribe.assert_not_called()


def test_skipped_subtitle_error_is_not_retried_or_reclassified(harness, monkeypatch):
    h = harness
    h.run()
    before = h.breaker.status(1800)
    metadata = Mock(side_effect=DownloadError('Unable to download video subtitles: 429'))
    monkeypatch.setattr(h.extractor, '_metadata', metadata)
    with pytest.raises(DownloadError):
        h.run()
    metadata.assert_called_once()
    assert metadata.call_args.kwargs['subtitles'] is False
    assert h.breaker.status(1800) == before


def test_probe_lease_expiry_allows_one_new_probe():
    now = [100.0]
    breaker = SubtitleCircuitBreaker(lambda: now[0])
    breaker.trip(1800, None)
    now[0] += 1800
    allowed, first = breaker.acquire(1800)
    assert allowed
    assert breaker.acquire(1800) == (False, None)
    now[0] += 60
    allowed, second = breaker.acquire(1800)
    assert allowed and second != first
    assert breaker.acquire(1800) == (False, None)
    breaker.succeeded(first)
    assert breaker.status(1800).retry_after_seconds == 60
    breaker.succeeded(second)
    assert not breaker.status(1800).breaker_open
