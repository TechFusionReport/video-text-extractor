import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from curl_cffi import CurlOpt
from curl_cffi.requests import RequestsError
from fastapi.testclient import TestClient
from yt_dlp.networking.impersonate import ImpersonateTarget

from video_text_extractor import extractor as extractor_module, main, tiktok
from video_text_extractor.config import Settings
from video_text_extractor.models import TranscriptSegment


SHORT = 'https://www.tiktok.com/t/unit-test/'
CANONICAL = 'https://www.tiktok.com/@test.creator/video/1234567890'
PROXY = 'http://proxy.invalid:8080'


@pytest.fixture
def redirects(monkeypatch):
    state = SimpleNamespace(replies=[], calls=[], options=[], closed=[], validations=[])
    monkeypatch.setattr(tiktok, 'validate_public_url', state.validations.append)

    class FakeSession:
        def __init__(self, **kwargs):
            state.options.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, **kwargs):
            state.calls.append((url, kwargs))
            item = state.replies.pop(0)
            if isinstance(item, Exception):
                raise item
            status, location = item
            return SimpleNamespace(
                status_code=status,
                headers={'Location': location} if location is not None else {},
                close=lambda: state.closed.append(url),
            )

    monkeypatch.setattr(tiktok, 'Session', FakeSession)
    return state


@pytest.mark.parametrize('short', [SHORT, 'https://vm.tiktok.com/unit-test/', 'https://vt.tiktok.com/unit-test/'])
def test_short_link_redirects_to_canonical_through_proxy(redirects, short):
    redirects.replies = [(302, '/t/intermediate/'), (301, CANONICAL + '?tracking=discarded')]
    result = tiktok.resolve_tiktok_short_url(short, {
        'proxy': PROXY, 'impersonate': ImpersonateTarget(client='chrome'),
    })
    assert result == CANONICAL
    assert len(redirects.calls) == 2
    assert redirects.calls[1][0].endswith('/t/intermediate/')
    assert all(kwargs['allow_redirects'] is False for _, kwargs in redirects.calls)
    assert redirects.options[0]['proxy'] == PROXY
    assert redirects.options[0]['trust_env'] is False
    assert redirects.options[0]['curl_options'][CurlOpt.NOPROXY] == ''
    assert redirects.options[0]['impersonate'].startswith('chrome')
    assert len(redirects.closed) == 2
    assert redirects.validations[-1] == CANONICAL + '?tracking=discarded'


def test_environment_proxy_bypass_cannot_override_configured_proxy(monkeypatch, redirects):
    monkeypatch.setenv('NO_PROXY', '*')
    monkeypatch.setenv('HTTPS_PROXY', 'http://wrong.invalid:8080')
    redirects.replies = [(302, CANONICAL)]
    assert tiktok.resolve_tiktok_short_url(SHORT, {'proxy': PROXY}) == CANONICAL
    assert redirects.options[0]['proxy'] == PROXY
    assert redirects.options[0]['trust_env'] is False
    assert redirects.options[0]['curl_options'] == {CurlOpt.NOPROXY: ''}


def test_missing_proxy_never_constructs_a_session(redirects):
    with pytest.raises(tiktok.TikTokShortLinkError, match='configured proxy is required'):
        tiktok.resolve_tiktok_short_url(SHORT, {})
    assert redirects.options == []
    assert redirects.calls == []


def test_proxy_failure_is_not_retried_direct(redirects):
    redirects.replies = [RequestsError('proxy unavailable')]
    with pytest.raises(RequestsError, match='proxy unavailable'):
        tiktok.resolve_tiktok_short_url(SHORT, {'proxy': PROXY})
    assert len(redirects.calls) == 1
    assert len(redirects.options) == 1
    assert redirects.options[0]['proxy'] == PROXY


@pytest.mark.parametrize('landing', ['/?_r=1', '/explore?_r=1', '/login', '/foryou'])
def test_non_video_redirect_is_rejected_without_fetching_landing_page(redirects, landing):
    redirects.replies = [(302, landing)]
    with pytest.raises(tiktok.TikTokShortLinkError, match='non-video redirect'):
        tiktok.resolve_tiktok_short_url(SHORT, {'proxy': PROXY})
    assert len(redirects.calls) == 1


@pytest.mark.parametrize('location', [
    'https://elsewhere.invalid/@test.creator/video/1234567890',
    'https://www.tiktok.com.evil.invalid/@test.creator/video/1234567890',
    'http://www.tiktok.com/@test.creator/video/1234567890',
    'file:///private',
    'https://www.tiktok.com:444/@test.creator/video/1234567890',
    'https://user:dummy-password@www.tiktok.com/@test.creator/video/1234567890',
])
def test_unsafe_redirects_cannot_trigger_another_request(redirects, location):
    redirects.replies = [(302, location)]
    with pytest.raises(tiktok.TikTokShortLinkError, match='unsafe redirect') as error:
        tiktok.resolve_tiktok_short_url(SHORT, {'proxy': PROXY})
    assert len(redirects.calls) == 1
    assert 'dummy-password' not in str(error.value)


def test_redirect_target_public_address_validation_precedes_extraction(monkeypatch, redirects):
    def validate(url):
        if url == CANONICAL:
            raise ValueError('Private, loopback, and link-local sources are blocked')
    monkeypatch.setattr(tiktok, 'validate_public_url', validate)
    redirects.replies = [(302, CANONICAL)]
    with pytest.raises(ValueError, match='sources are blocked'):
        tiktok.resolve_tiktok_short_url(SHORT, {'proxy': PROXY})
    assert len(redirects.calls) == 1


@pytest.mark.parametrize('status', [403, 429])
def test_explicit_denial_has_a_clear_error(redirects, status):
    redirects.replies = [(status, None)]
    with pytest.raises(tiktok.TikTokShortLinkError, match=f'TikTok denied the request: HTTP {status}'):
        tiktok.resolve_tiktok_short_url(SHORT, {'proxy': PROXY})
    assert len(redirects.calls) == 1


@pytest.mark.parametrize('reply', [(200, None), (404, None), (302, None)])
def test_missing_canonical_target_has_a_clear_error(redirects, reply):
    redirects.replies = [reply]
    with pytest.raises(tiktok.TikTokShortLinkError, match='no canonical video URL'):
        tiktok.resolve_tiktok_short_url(SHORT, {'proxy': PROXY})


def test_redirect_loop_is_bounded(redirects):
    redirects.replies = [(302, SHORT)]
    with pytest.raises(tiktok.TikTokShortLinkError, match='redirect loop'):
        tiktok.resolve_tiktok_short_url(SHORT, {'proxy': PROXY})
    assert len(redirects.calls) == 1


def test_redirect_limit_is_bounded(redirects):
    redirects.replies = [(302, f'/t/hop-{i}/') for i in range(20)]
    with pytest.raises(tiktok.TikTokShortLinkError, match='too many redirects'):
        tiktok.resolve_tiktok_short_url(SHORT, {'proxy': PROXY})
    assert len(redirects.calls) == 6


def test_configured_cookies_are_loaded_without_writing_them(redirects, tmp_path):
    cookiefile = tmp_path / 'cookies.txt'
    text = '# Netscape HTTP Cookie File\n.tiktok.com\tTRUE\t/\tTRUE\t2147483647\ttest_cookie\tunit-value\n'
    cookiefile.write_text(text)
    redirects.replies = [(302, CANONICAL)]
    tiktok.resolve_tiktok_short_url(SHORT, {'proxy': PROXY, 'cookiefile': str(cookiefile)})
    assert [cookie.name for cookie in redirects.options[0]['cookies']] == ['test_cookie']
    assert cookiefile.read_text() == text


@pytest.mark.parametrize('malformed', [True, False])
def test_cookie_errors_are_safe_and_do_not_fetch(redirects, tmp_path, malformed):
    path = tmp_path / 'cookies.txt'
    if malformed:
        path.write_text('invalid cookie data')
    with pytest.raises(tiktok.TikTokShortLinkError, match='configured cookies could not be loaded'):
        tiktok.resolve_tiktok_short_url(SHORT, {'proxy': PROXY, 'cookiefile': str(path)})
    assert redirects.calls == []


def test_homepage_redirect_returns_clear_api_error_before_yt_dlp(monkeypatch, redirects):
    extractor = extractor_module.VideoExtractor(Settings(yt_dlp_proxy=PROXY))
    monkeypatch.setattr(extractor_module, 'validate_public_url', lambda value: None)
    monkeypatch.setattr(main, 'get_extractor', lambda: extractor)
    ydl = Mock(side_effect=AssertionError('yt-dlp must not receive a homepage'))
    monkeypatch.setattr(extractor_module, 'YoutubeDL', ydl)
    redirects.replies = [(302, 'https://www.tiktok.com/?_r=1')]
    response = TestClient(main.app).post(
        '/v1/extract', headers={'Authorization': 'Bearer ' + os.environ['EXTRACTOR_API_KEY']},
        json={'url': SHORT},
    )
    assert response.status_code == 422
    assert response.json() == {'detail': 'TikTok short link could not be resolved (TikTok returned a non-video redirect)'}
    ydl.assert_not_called()
    assert len(redirects.calls) == 1


@pytest.mark.parametrize('input_url,is_short', [(SHORT, True), (CANONICAL + '?tracking=discarded', False)])
def test_extraction_uses_canonical_url_and_exposes_it_without_enabling_ocr(monkeypatch, redirects, input_url, is_short):
    extractor = extractor_module.VideoExtractor(Settings(yt_dlp_proxy=PROXY))
    monkeypatch.setattr(extractor_module, 'validate_public_url', lambda value: None)
    monkeypatch.setattr(main, 'get_extractor', lambda: extractor)
    redirects.replies = [(302, CANONICAL + '?tracking=discarded')]
    metadata = Mock(return_value={'id': '1234567890', 'duration': 20, 'description': 'test description'})
    audio = Mock(return_value=Path('unused.wav'))
    monkeypatch.setattr(extractor, '_metadata', metadata)
    monkeypatch.setattr(extractor, '_read_caption', lambda *args: None)
    monkeypatch.setattr(extractor, '_download_audio', audio)
    monkeypatch.setattr(extractor, '_transcribe', lambda *args: ([TranscriptSegment(start=0, end=1, text='test words')], 'en'))
    monkeypatch.setattr(extractor, '_extract_ocr', Mock(side_effect=AssertionError('OCR must stay off')))
    response = TestClient(main.app).post(
        '/v1/extract', headers={'Authorization': 'Bearer ' + os.environ['EXTRACTOR_API_KEY']},
        json={'url': input_url},
    )
    assert response.status_code == 200
    body = response.json()
    assert body['source_url'] == input_url
    assert body['resolved_url'] == CANONICAL
    assert body['ocr_segments'] == []
    assert Settings().ocr_enabled is False
    assert metadata.call_args.args[0] == CANONICAL
    assert audio.call_args.args[0] == CANONICAL
    assert len(redirects.calls) == int(is_short)


def test_resolver_and_all_yt_dlp_calls_keep_proxy_and_canonical_url(monkeypatch, redirects):
    extractor = extractor_module.VideoExtractor(Settings(yt_dlp_proxy=PROXY))
    monkeypatch.setattr(extractor_module, 'validate_public_url', lambda value: None)
    redirects.replies = [(302, CANONICAL)]
    calls = []

    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options
            assert options['proxy'] == PROXY

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download):
            assert url == CANONICAL
            assert download is True
            calls.append(url)
            if 'postprocessors' in self.options:
                (Path(self.options['outtmpl']).parent / 'audio.wav').touch()
            return {'duration': 20, 'description': 'description'}

    monkeypatch.setattr(extractor_module, 'YoutubeDL', FakeYoutubeDL)
    monkeypatch.setattr(extractor, '_transcribe', lambda *args: ([TranscriptSegment(start=0, end=1, text='test words')], 'en'))
    result = extractor.extract(SHORT, None, False)
    assert calls == [CANONICAL, CANONICAL]
    assert result.resolved_url == CANONICAL
    assert redirects.options[0]['proxy'] == PROXY
    assert redirects.options[0]['curl_options'][CurlOpt.NOPROXY] == ''


@pytest.mark.parametrize('target,expected', [
    (ImpersonateTarget(client='chrome', version='110', os='windows', os_version='10'), 'chrome110'),
    (ImpersonateTarget(client='chrome', version='99', os='android', os_version='12'), 'chrome99_android'),
])
def test_configured_browser_version_and_mobile_identity_are_preserved(redirects, target, expected):
    redirects.replies = [(302, CANONICAL)]
    tiktok.resolve_tiktok_short_url(SHORT, {'proxy': PROXY, 'impersonate': target})
    assert redirects.options[0]['impersonate'] == expected


def test_upstream_server_error_keeps_referenced_502(monkeypatch, redirects, caplog):
    extractor = extractor_module.VideoExtractor(Settings(yt_dlp_proxy=PROXY))
    monkeypatch.setattr(extractor_module, 'validate_public_url', lambda value: None)
    monkeypatch.setattr(main, 'get_extractor', lambda: extractor)
    redirects.replies = [(503, None)]
    response = TestClient(main.app).post(
        '/v1/extract', headers={'Authorization': 'Bearer ' + os.environ['EXTRACTOR_API_KEY']},
        json={'url': SHORT},
    )
    assert response.status_code == 502
    assert 'ref ' in response.json()['detail']
    assert '503' not in response.text
    assert 'upstream HTTP 503' in caplog.text
    assert len(redirects.calls) == 1


def test_real_session_configuration_is_compatible_without_network(monkeypatch):
    # Exercise the real curl_cffi constructor in both supported dependency sets.
    calls = []
    monkeypatch.setattr(tiktok, 'validate_public_url', lambda value: None)

    def get(session, url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=302, headers={'Location': CANONICAL}, close=lambda: None)

    monkeypatch.setattr(tiktok.Session, 'get', get)
    assert tiktok.resolve_tiktok_short_url(SHORT, {
        'proxy': PROXY, 'impersonate': ImpersonateTarget(client='chrome'),
    }) == CANONICAL
    assert calls == [(SHORT, {'allow_redirects': False})]
