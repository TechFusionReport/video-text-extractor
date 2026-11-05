from pathlib import Path
from unittest.mock import Mock

import pytest
from yt_dlp.utils import DownloadError

from video_text_extractor import extractor as module
from video_text_extractor.config import Settings
from video_text_extractor.main import app
from video_text_extractor.models import TranscriptSegment
from fastapi.testclient import TestClient


@pytest.mark.parametrize('failure,forced,duration,status,attempts', [
    ('subtitle', False, 30, 200, 3),
    (None, True, 30, 200, 2),
    ('other', False, 30, 502, 1),
    ('subtitle', False, 61, 422, 2),
    ('retry', False, 30, 502, 2),
    ('subtitle', True, 30, 502, 1),
])
def test_subtitle_flow(monkeypatch, caplog, failure, forced, duration, status, attempts):
    monkeypatch.setattr(module, 'validate_public_url', lambda url: None)
    extractor = module.VideoExtractor(Settings(extractor_api_key='test-secret', whisper_max_media_seconds=60))
    identity = object()
    monkeypatch.setattr(extractor, '_yt_dlp_request_options', lambda: {'proxy': identity})
    calls = []

    class FakeYoutubeDL:
        def __init__(self, options):
            calls.append(options)
            self.options = options
            assert options['proxy'] is identity

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def _write_subtitles(self):
            # Leave a complete-looking caption behind before failing.
            output = Path(self.options['outtmpl']).parent / 'source.en.json3'
            output.write_text('{"events":[{"segs":[{"utf8":"partial"}]}]}')
            raise DownloadError('subtitle HTTP 429')

        def extract_info(self, url, download):
            assert download is True
            if len(calls) == 1 and failure in ('subtitle', 'retry'):
                self._write_subtitles()
            if failure == 'other' or (failure == 'retry' and len(calls) == 2):
                raise DownloadError('unrelated extraction failure')
            if 'postprocessors' in self.options:
                (Path(self.options['outtmpl']).parent / 'audio.wav').touch()
            return {'duration': duration, 'description': 'description'}

    monkeypatch.setattr(module, 'YoutubeDL', FakeYoutubeDL)
    transcribe = Mock(return_value=([TranscriptSegment(start=0, end=1, text='spoken words')], 'en'))
    monkeypatch.setattr(extractor, '_transcribe', transcribe)
    # run_job calls get_extractor directly rather than via FastAPI dependency injection.
    monkeypatch.setattr('video_text_extractor.main.get_extractor', lambda: extractor)
    response = TestClient(app).post('/v1/extract', headers={'Authorization': 'Bearer test-secret'},
                                       json={'url': 'https://example.com/v', 'force_transcription': forced})
    assert response.status_code == status
    assert len(calls) == attempts
    if forced:
        assert all(not options.get('writesubtitles') and not options.get('writeautomaticsub') for options in calls)
    if status == 200:
        assert response.json()['method'] == 'whisper'
        transcribe.assert_called_once()
        assert ('subtitles_unavailable' in response.json()['warnings']) == (failure == 'subtitle')
        assert all(not options.get('writesubtitles') and not options.get('writeautomaticsub') for options in calls[1:])
    else:
        transcribe.assert_not_called()
    if status == 502:
        assert 'ref ' in response.json()['detail']
        assert 'extraction failure' not in response.text
        assert 'DownloadError' in caplog.text


def test_error_logs_redact_url_userinfo(monkeypatch, caplog):
    from urllib.parse import urlunsplit
    secret_url = urlunsplit(('https', 'sensitive-user:sensitive-password@' + 'example.com', '/v', '', ''))

    def fail(*args, **kwargs):
        raise DownloadError('connection failed: ' + secret_url)

    monkeypatch.setattr(module.VideoExtractor, 'extract', fail)
    response = TestClient(app).post('/v1/extract', headers={'Authorization': 'Bearer test-secret'},
                                   json={'url': secret_url})
    assert response.status_code == 502
    assert 'ref ' in response.json()['detail']
    assert 'connection failed' in caplog.text
    for value in (caplog.text, response.text):
        assert 'sensitive-user' not in value
        assert 'sensitive-password' not in value


def test_response_redacts_url_userinfo():
    result = module.VideoExtractor._response(
        'https://sensitive-user:sensitive-password@' + 'example.com/v', {}, 'whisper', [], None, []
    )
    assert 'sensitive-user' not in result.source_url
    assert 'sensitive-password' not in result.source_url
