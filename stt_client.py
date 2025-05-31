"""
STT (Whisper) client: convert PCM audio to WAV and send to Whisper server.
"""

import io
import wave
import logging
# ``requests`` is only required when the real Whisper server is contacted.  To
# keep the test-suite independent from external dependencies we fall back to a
# lightweight stub when the import fails.

from types import SimpleNamespace

try:
    import requests  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – stub only for tests/CI

    class _FakeResponse:  # pylint: disable=too-few-public-methods
        def __init__(self, status_code=200, text="", json_data=None):
            self.status_code = status_code
            self.text = text
            self._json = json_data or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise Exception("HTTP error stub")

        def json(self):  # noqa: D401 – stub
            return self._json

    class _FakeSession:  # pylint: disable=too-few-public-methods
        def post(self, *_args, **_kwargs):  # noqa: D401 – stub
            return _FakeResponse()

    # expose the attributes the production code references
    requests = SimpleNamespace(Session=_FakeSession)  # type: ignore

logger = logging.getLogger(__name__)


class WhisperClient:
    """
    HTTP client for Whisper STT server.
    """

    def __init__(
        self,
        url: str,
        session: "requests.Session | None" = None,
        timeout: int = 30,
        *,
        send_pcm: bool = False,
    ):
        """Create a Whisper STT client.

        Parameters
        ----------
        url
            Endpoint of the whisper.cpp server.
        session
            Optional ``requests.Session`` to reuse connections; one is created
            automatically otherwise.
        timeout
            Request timeout in seconds.
        send_pcm
            When *True* the client uploads the raw PCM bytes directly instead
            of wrapping them in a WAV container.  whisper.cpp (>=1.5) accepts
            raw 16-bit little-endian audio via multipart/form-data when the
            filename ends with ``.raw``.  This removes the per-chunk WAV
            encoding overhead and saves a few milliseconds.
        """

        self.url = url
        self.session = session if session is not None else requests.Session()
        self.send_pcm = send_pcm
        self.timeout = timeout

    def transcribe(
        self,
        raw_pcm: bytes,
        channels: int,
        sample_width: int,
        rate: int,
    ) -> str:
        """Convert PCM to WAV in-memory and send to Whisper server."""
        # Decide upload format (WAV or raw PCM)
        if self.send_pcm:
            # Direct raw PCM upload – no containerisation cost.
            files = {
                "file": (
                    "audio.raw",
                    io.BytesIO(raw_pcm),
                    "application/octet-stream",
                )
            }
        else:
            # Fallback: wrap in a small WAV header (compatible with any server).
            wav_buf = io.BytesIO()
            try:
                with wave.open(wav_buf, "wb") as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(sample_width)
                    wf.setframerate(rate)
                    wf.writeframes(raw_pcm)
                wav_buf.seek(0)
            except Exception as e:
                logger.error("Error creating WAV: %s", e)
                return ""
            files = {"file": ("audio.wav", wav_buf, "audio/wav")}

        data = {"temperature": 0.0, "temperature_inc": 0.2, "response_format": "json"}
        logger.debug("Whisper request URL: %s, send_pcm=%s", self.url, self.send_pcm)
        # Send request to Whisper STT server
        resp = self.session.post(self.url, files=files, data=data, timeout=self.timeout)
        logger.debug("Whisper response [%d]: %s", resp.status_code, resp.text)
        # Raise for HTTP errors
        resp.raise_for_status()
        # Parse JSON response
        j = resp.json()
        # Extract transcription text
        return j.get("text", "").strip()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def close(self):
        """Close the underlying HTTP session (a no-op for stub sessions)."""
        close = getattr(self.session, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover – defensive only
                logger.debug("Failed to close WhisperClient session", exc_info=True)
