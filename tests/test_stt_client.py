import wave
import pytest

from stt_client import WhisperClient


class DummyResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json = json_data
        self.text = str(json_data)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._json


class DummySession:
    def __init__(self, response):
        self._response = response
        self.kwargs = {}

    def post(self, url, files=None, data=None, timeout=None):
        # Record parameters and return preset response
        self.kwargs = dict(url=url, files=files, data=data, timeout=timeout)
        return self._response


def test_transcribe_success():
    # Prepare fake PCM data
    raw_pcm = b"\x00\x01\x02\x03" * 100
    # Dummy server returns JSON with text
    dummy_resp = DummyResponse(200, {"text": "hello world"})
    session = DummySession(dummy_resp)
    client = WhisperClient("http://fake", session)
    result = client.transcribe(raw_pcm, channels=1, sample_width=2, rate=16000)
    assert result == "hello world"
    # Ensure WAV was constructed with expected audio parameters
    files = session.kwargs.get("files")
    assert isinstance(files, dict)
    fname, wav_buf, mime = files["file"]
    assert fname == "audio.wav"
    assert mime == "audio/wav"
    wav_buf.seek(0)
    with wave.open(wav_buf, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000


def test_transcribe_http_error():
    raw_pcm = b"\x00\x01"
    # Simulate server error
    dummy_resp = DummyResponse(500, {})
    session = DummySession(dummy_resp)
    client = WhisperClient("http://fake", session)
    with pytest.raises(Exception):
        client.transcribe(raw_pcm, channels=1, sample_width=2, rate=8000)

