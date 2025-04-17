import time
import sys
import logging
from typing import Optional, Any, List, Dict

logger = logging.getLogger("vad-whisper-llama")

def print_and_log(message: str, output_file_handle: Optional[Any] = None):
    """Log message to console (and file via logging handlers)."""
    logger.info(message)

def trim_conversation_history(conversation: List[Dict], max_len: int) -> List[Dict]:
    """Keeps the conversation at the max length (always keeps system prompt)."""
    while len(conversation) > max_len and len(conversation) > 2:
        del conversation[1:3]
    return conversation

def health_check_endpoint(name: str, url: str, timeout: int = 5):
    """Perform a quick HEAD request to verify the service is reachable."""
    import requests
    from requests.exceptions import RequestException
    try:
        resp = requests.head(url, timeout=timeout)
        status = resp.status_code
        if status >= 500:
            logger.error(f"{name} endpoint returned HTTP {status}")
            sys.exit(1)
        logger.info(f"{name} endpoint reachable (HTTP {status})")
    except RequestException as e:
        logger.error(f"{name} health-check failed: {e}")
        sys.exit(1)
  
def get_audio_bytes_per_second(rate: int, channels: int, sample_width: int) -> int:
    """Calculate audio byte rate: bytes per second."""
    return rate * channels * sample_width

def pad_audio(audio_data: bytes, min_bytes: int, sample_width: int) -> bytes:
    """Pads the audio data with silence to reach the minimum byte length."""
    current_bytes = len(audio_data)
    bytes_to_add = max(0, min_bytes - current_bytes)
    if bytes_to_add % sample_width:
        bytes_to_add += sample_width - (bytes_to_add % sample_width)
    return audio_data + (b"\x00" * bytes_to_add)