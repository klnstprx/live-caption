import sys
import logging
from typing import Optional, Any, List, Dict

logger = logging.getLogger("vad-whisper-llama")
# ANSI color codes for STT/translation prefixes
RESET = "\x1b[0m"
YELLOW = "\x1b[33m"
CYAN = "\x1b[36m"


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
            logger.error(f"{name} endpoint returned HTTP {status}")
            sys.exit(1)
        logger.info(f"{name} endpoint reachable (HTTP {status})")
    except RequestException as e:
        logger.error(f"{name} health-check failed: {e}")
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
    """Pads the audio data with silence to reach the minimum byte length."""
    current_bytes = len(audio_data)
    bytes_to_add = max(0, min_bytes - current_bytes)
    if bytes_to_add % sample_width:
        bytes_to_add += sample_width - (bytes_to_add % sample_width)
    return audio_data + (b"\x00" * bytes_to_add)

