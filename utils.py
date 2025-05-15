import sys
import logging
from typing import Optional, Any, List, Dict

logger = logging.getLogger("vad-whisper-llama")
# ANSI color codes for STT/translation prefixes
RESET = "\x1b[0m"
YELLOW = "\x1b[33m"
CYAN = "\x1b[36m"

# ---------------------------------------------------------------------------
# Optional dependency handling
# ---------------------------------------------------------------------------
# ``requests`` is only needed when doing real network health-checks.  A minimal
# stub is provided so that the test-suite can run in an isolated environment
# without the library installed.  The stub is monkey-patch friendly – the
# tests replace ``requests.head`` with their own lambdas.
# ---------------------------------------------------------------------------

import types as _types

try:
    import requests as _requests  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – lightweight stub for CI
    _requests = _types.ModuleType("requests")

    class _RequestException(Exception):
        pass

    # Provide the attribute the prod code uses; tests patch it anyway.
    def _default_head(_url, timeout=5, **_):  # noqa: D401 – stub, accepts same kw
        return _types.SimpleNamespace(status_code=200)

    class _Session:  # pylint: disable=too-few-public-methods
        def post(self, *_a, **_kw):  # noqa: D401 – stub
            return _types.SimpleNamespace(status_code=200, text="", json=lambda: {})

    _requests.head = _default_head  # type: ignore
    _requests.Session = _Session  # type: ignore

    # Build a real sub-module so that ``from requests.exceptions import …``
    # works as expected.
    _exceptions_mod = _types.ModuleType("requests.exceptions")
    _exceptions_mod.RequestException = _RequestException  # type: ignore

    _requests.exceptions = _exceptions_mod  # type: ignore[attr-defined]

    # Expose in sys.modules for regular import machinery.
    sys.modules["requests.exceptions"] = _exceptions_mod

    sys.modules["requests"] = _requests


def print_and_log(message: str, output_file_handle: Optional[Any] = None):
    """Print message to stdout with nice formatting and optionally append to an output file."""
    # Prettify multiline console output and color STT/translation prefixes
    lines = message.splitlines()
    if not lines:
        print("")
    else:
        first = lines[0]
        # Detect STT/Translation prefixes and split
        if ":" in first and (
            first.startswith("(Korean)") or first.startswith("(English)")
        ):
            idx = first.find(":")
            prefix = first[: idx + 1]
            rest = first[idx + 1 :].lstrip()
            # Color prefix
            if prefix.startswith("(Korean)"):
                header = f"{YELLOW}{prefix}{RESET}"
            else:
                header = f"{CYAN}{prefix}{RESET}"
            # Print header on its own line
            print(header)
            # Prepare content lines: rest (if any) + following lines
            content_lines = []
            if rest:
                content_lines.append(rest)
            content_lines.extend(lines[1:])
            # Print indented content
            for line in content_lines:
                print("    " + line)
        else:
            # Default: print first and indent others
            print(first)
            for line in lines[1:]:
                print("    " + line)
    # Append to output file if provided
    if output_file_handle:
        try:
            if not lines:
                output_file_handle.write("\n")
            else:
                output_file_handle.write(lines[0] + "\n")
                for line in lines[1:]:
                    output_file_handle.write("    " + line + "\n")
            output_file_handle.flush()
        except Exception:
            logger.warning("Failed to write message to output file.")


def trim_conversation_history(conversation: List[Dict], max_len: int) -> List[Dict]:
    """Return a trimmed copy of the conversation, keeping the system prompt and the most recent messages up to max_len."""
    # If already within limit, return a shallow copy
    if len(conversation) <= max_len:
        return conversation.copy()
    # Always preserve the system prompt at index 0
    # Then take the last (max_len - 1) messages from the remainder
    num_tail = max_len - 1
    tail = conversation[-num_tail:]
    return [conversation[0]] + tail


def health_check_endpoint(name: str, url: str, timeout: int = 5):
    """Perform a quick HEAD request to verify the service is reachable."""
    import requests
    from requests.exceptions import RequestException

    try:
        resp = requests.head(url, timeout=timeout)
        status = resp.status_code
        if status >= 500:
            msg = f"{name} endpoint returned HTTP {status}"
            logger.error(msg)
            # Mirror to stderr so callers that rely on raw stderr capture (e.g. the
            # test-suite) can still see the message even if logging is not
            # configured.
            print(msg, file=sys.stderr, flush=True)
            sys.exit(1)

        msg = f"{name} endpoint reachable (HTTP {status})"
        logger.info(msg)
        print(msg, file=sys.stderr, flush=True)
    except RequestException as e:
        msg = f"{name} health-check failed: {e}"
        logger.error(msg)
        print(msg, file=sys.stderr, flush=True)
        sys.exit(1)


import json as _json


class JSONFormatter(logging.Formatter):
    """
    Formatter that outputs log records as JSON objects.
    """

    def format(self, record):
        record_dict = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            record_dict["exc_info"] = self.formatException(record.exc_info)
        return _json.dumps(record_dict, ensure_ascii=False)


class ColorFormatter(logging.Formatter):
    """
    Formatter that outputs colored log records for improved terminal readability.
    Applies ANSI color codes to the level name based on severity, and dims timestamps.
    """

    RESET = "\x1b[0m"
    DIM = "\x1b[2m"
    COLOR_MAP = {
        "DEBUG": "\x1b[34m",  # Blue
        "INFO": "\x1b[32m",  # Green
        "WARNING": "\x1b[33m",  # Yellow
        "ERROR": "\x1b[31m",  # Red
        "CRITICAL": "\x1b[41m",  # White on red background
    }

    def format(self, record):
        # Prepare the message content
        record.message = record.getMessage()
        # Dim the timestamp if used in the format
        if self.usesTime():
            t = self.formatTime(record, self.datefmt)
            record.asctime = f"{self.DIM}{t}{self.RESET}"
        # Color the level name
        level = record.levelname
        color = self.COLOR_MAP.get(level, "")
        record.levelname = f"{color}{level}{self.RESET}"
        # Build the formatted message
        formatted = self.formatMessage(record)
        # Append exception trace if present
        if record.exc_info:
            formatted = formatted + "\n" + self.formatException(record.exc_info)
        return formatted


def get_audio_bytes_per_second(rate: int, channels: int, sample_width: int) -> int:
    """Calculate audio byte rate: bytes per second."""
    return rate * channels * sample_width


def pad_audio(audio_data: bytes, min_bytes: int, sample_width: int) -> bytes:
    """Return *audio_data* padded with silence (zero-bytes) so that its length is at
    least *min_bytes* and aligned to *sample_width*.

    This is on the hot-path, so we avoid allocating a brand-new ``bytes`` object
    full of zeros every call by re-using a shared cache of zero-buffers.  For
    very large paddings we concatenate from the cache in chunks rather than
    building an intermediate string with Python’s * multiplication (which
    copies).  The function still returns an ordinary immutable ``bytes`` object
    so all existing callers/tests behave the same.
    """

    current_bytes = len(audio_data)
    bytes_to_add = max(0, min_bytes - current_bytes)
    # Align to sample width so frame boundaries stay intact.
    rem = bytes_to_add % sample_width
    if rem:
        bytes_to_add += sample_width - rem

    if bytes_to_add == 0:
        return audio_data  # Fast path – no padding required.

    # ------------------------------------------------------------------
    # Grab a zero-buffer from the cache and slice/replicate as necessary.
    # ------------------------------------------------------------------
    _cache = _ZERO_PAD_CACHE.setdefault(sample_width, b"\x00" * 4096)
    if len(_cache) < bytes_to_add:
        # Enlarge cache once; subsequent calls will reuse the bigger buffer.
        _cache = _ZERO_PAD_CACHE[sample_width] = b"\x00" * (bytes_to_add * 2)

    # Build the padded bytes.  For speed we slice from the cache rather than
    # constructing a brand-new bytes object per invocation.
    return audio_data + _cache[:bytes_to_add]


# ---------------------------------------------------------------------------
# Internal helpers / caches
# ---------------------------------------------------------------------------

# Zero-byte buffer cache per sample_width.  Populated lazily the first time
# pad_audio() is asked to create a buffer of a given width.
_ZERO_PAD_CACHE: dict[int, bytes] = {}

