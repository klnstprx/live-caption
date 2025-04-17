"""
STT (Whisper) client: convert PCM audio to WAV and send to Whisper server.
"""
import io
import wave
import logging
import requests

logger = logging.getLogger("vad-whisper-llama")

class WhisperClient:
    """
    HTTP client for Whisper STT server.
    """
    def __init__(self, url: str, session: requests.Session, timeout: int = 30):
        self.url = url
        self.session = session
        self.timeout = timeout

    def transcribe(
        self,
        raw_pcm: bytes,
        channels: int,
        sample_width: int,
        rate: int,
    ) -> str:
        """Convert PCM to WAV in-memory and send to Whisper server."""
        wav_buf = io.BytesIO()
        try:
            with wave.open(wav_buf, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(rate)
                wf.writeframes(raw_pcm)
            wav_buf.seek(0)
        except Exception as e:
            logger.error(f"Error creating WAV: {e}")
            return ""
        files = {"file": ("audio.wav", wav_buf, "audio/wav")}  
        data = {"temperature": 0.0, "temperature_inc": 0.2, "response_format": "json"}
        logger.debug("Whisper request URL: %s, data: %s", self.url, data)
        # Send request to Whisper STT server
        resp = self.session.post(
            self.url, files=files, data=data, timeout=self.timeout
        )
        logger.debug("Whisper response [%d]: %s", resp.status_code, resp.text)
        # Raise for HTTP errors
        resp.raise_for_status()
        # Parse JSON response
        j = resp.json()
        # Extract transcription text
        return j.get("text", "").strip()