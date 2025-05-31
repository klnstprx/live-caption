"""
Live caption orchestrator: capture audio via VAD, transcribe with Whisper,
 translate with Llama, and output in real time.
"""

import os
import sys
import argparse
import logging

# Optional: load environment variables from a .env file
try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv():
        pass


def check_required_dependencies():
    """Ensure required third-party libraries are installed."""
    if any(arg in ("-h", "--help") for arg in sys.argv):
        return
    missing = []
    try:
        import webrtcvad  # noqa: F401
    except ImportError:
        missing.append("webrtcvad")
    try:
        import pyaudio  # noqa: F401
    except ImportError:
        missing.append("pyaudio")
    try:
        import requests  # noqa: F401
        from requests.adapters import HTTPAdapter  # noqa: F401
        from urllib3.util.retry import Retry  # noqa: F401
    except ImportError:
        missing.append("requests")
    if missing:
        print(
            f"Error: Missing required libraries: {', '.join(missing)}. "
            "Please install them via pip.",
            file=sys.stderr,
        )
        sys.exit(1)


check_required_dependencies()

import audio_capture
from utils import health_check_endpoint, ColorFormatter

# Default configuration
DEFAULT_WHISPER_SERVER_URL = "http://127.0.0.1:8081/inference"
DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_FRAME_DURATION_MS = 30
DEFAULT_VAD_MODE = 2
DEFAULT_MAX_SILENCE_FRAMES = 25
DEFAULT_MODEL_NAME = "gemma-3-12b-it"
DEFAULT_MIN_CHUNK_DURATION_MS = 1000
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert translator specializing in **physics**, fluent in Korean "
    "and English. Always respond strictly in English.\n\n"
    "Your primary task is to translate Korean input from an ongoing "
    "**physics lecture** into accurate and clear English.\n\n"
    "Pay meticulous attention to the correct translation of **physics "
    "terminology (e.g., force, velocity, quantum, field), scientific "
    "concepts, mathematical equations, units (e.g., m/s, Joules), and "
    "principles.** Accuracy in these technical details is paramount. Where "
    "a choice exists between a common phrasing and precise technical "
    "language, **opt for the technical term** to ensure scientific accuracy.\n\n"
    "Preserve the original meaning and the **formal, academic tone** typical "
    "of a lecture setting. Produce grammatically correct English translations "
    "that are easily understood within a scientific context.\n\n"
    "Keep in mind that each input is a segment of a longer lecture, so "
    "context may build over time.\n\n"
    "Your response **must** be a structured JSON object with a single key, "
    "`translatedText`, containing the English translation."
)
DEFAULT_MAX_CONVERSATION_MESSAGES = 2 * 4 + 1
DEFAULT_OUTPUT_FILE = None

logger = logging.getLogger(__name__)


def create_arg_parser():
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Real-time VAD + Whisper + OpenAI Chat translation"
    )

    # Server options
    server = parser.add_argument_group("Server options")
    server.add_argument(
        "--whisper-url",
        type=str,
        default=os.environ.get("WHISPER_URL", DEFAULT_WHISPER_SERVER_URL),
        help="URL of the Whisper STT server",
    )

    # Audio capture & VAD options
    audio = parser.add_argument_group("Audio capture & VAD options")
    audio.add_argument(
        "--rate",
        type=int,
        choices=[8000, 16000, 32000, 48000],
        default=int(os.environ.get("RATE", str(DEFAULT_RATE))),
        help="Audio sampling rate in Hz",
    )
    audio.add_argument(
        "--channels",
        type=int,
        choices=[1],
        default=int(os.environ.get("CHANNELS", DEFAULT_CHANNELS)),
        help="Number of audio channels",
    )
    audio.add_argument(
        "--frame-duration-ms",
        type=int,
        choices=[10, 20, 30],
        default=int(os.environ.get("FRAME_DURATION_MS", DEFAULT_FRAME_DURATION_MS)),
        help="Frame duration in milliseconds",
    )
    audio.add_argument(
        "--vad-mode",
        type=int,
        choices=[0, 1, 2, 3],
        default=int(os.environ.get("VAD_MODE", DEFAULT_VAD_MODE)),
        help="VAD aggressiveness (0-3)",
    )
    audio.add_argument(
        "--max-silence-frames",
        type=int,
        default=int(os.environ.get("MAX_SILENCE_FRAMES", DEFAULT_MAX_SILENCE_FRAMES)),
        help="Maximum number of consecutive silent frames",
    )
    audio.add_argument(
        "--min-chunk-duration-ms",
        type=int,
        default=int(
            os.environ.get("MIN_CHUNK_DURATION_MS", DEFAULT_MIN_CHUNK_DURATION_MS)
        ),
        help="Minimum duration of audio chunks in milliseconds",
    )
    audio.add_argument(
        "--device-index",
        type=int,
        default=(
            int(os.environ["DEVICE_INDEX"]) if os.environ.get("DEVICE_INDEX") else None
        ),
        help="Index of audio input device (as listed by PyAudio).",
    )

    # Model & prompt customization
    model = parser.add_argument_group("Model & prompt options")
    model.add_argument(
        "--model-name",
        type=str,
        default=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME),
        help="Translation model name",
    )
    model.add_argument(
        "--system-prompt",
        type=str,
        default=os.environ.get("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        help="System prompt for translation model",
    )
    model.add_argument(
        "--max-conversation-messages",
        type=int,
        default=int(os.environ.get("MAX_CONV_MSGS", DEFAULT_MAX_CONVERSATION_MESSAGES)),
        help="Maximum number of messages in conversation history",
    )

    # Translation API settings
    trans = parser.add_argument_group("Translation options")
    trans.add_argument(
        "--llm-base-url",
        type=str,
        default=os.environ.get("LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
        help="LLM base URL",
    )

    # Output & logging
    out = parser.add_argument_group("Output & logging options")
    out.add_argument(
        "--output-file",
        type=str,
        default=os.environ.get("OUTPUT_FILE", DEFAULT_OUTPUT_FILE),
        help="File to write translated output",
    )
    out.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    out.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug logging (sets log level to DEBUG)",
    )

    return parser


def validate_device_index(args, parser):
    """Ensure the --device-index is valid and has input channels."""
    if args.device_index is None:
        return
    import pyaudio

    pa = pyaudio.PyAudio()
    try:
        info = pa.get_device_info_by_index(args.device_index)
    except Exception as e:
        pa.terminate()
        parser.error(f"Invalid device index {args.device_index}: {e}")
    pa.terminate()
    if info.get("maxInputChannels", 0) <= 0:
        parser.error(f"Device {args.device_index} has no input channels")


def configure_logging(debug: bool):
    """Set up colored console logging for all modules."""
    level = logging.DEBUG if debug else logging.ERROR
    root = logging.getLogger()
    root.setLevel(level)
    # Remove existing handlers on the root logger
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    # Create and attach a colored console handler
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(
        ColorFormatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(console)
    if debug:
        root.debug("Debug logging enabled")


def validate_args(args, parser):
    """Check miscellaneous argument constraints."""
    if args.max_conversation_messages < 1 or args.max_conversation_messages % 2 == 0:
        parser.error("--max-conversation-messages must be an odd number >= 1")
    if args.min_chunk_duration_ms < 0:
        parser.error("--min-chunk-duration-ms must be >= 0")


def main(args):
    """
    Orchestrate:
        1) Health-check services
        2) Capture audio via VAD
        3) Transcribe with Whisper
        4) Translate with OpenAI Chat
        5) Output results
    """
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    from stt_client import WhisperClient
    from translation_client import TranslationClient
    from audio_capture import vad_capture_thread  # noqa: F401
    from pipeline import run_pipeline

    logger.debug(f"Whisper URL: {args.whisper_url}")
    health_check_endpoint("Whisper", args.whisper_url)

    import pyaudio

    paudio = pyaudio.PyAudio()
    args.sample_width = paudio.get_sample_size(pyaudio.paInt16)

    # Build a retry-capable HTTP session
    session = requests.Session()
    retry = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
        backoff_factor=1,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    stt_client = WhisperClient(args.whisper_url, session=session)
    translation_client = TranslationClient(
        model_name=args.model_name,
        base_url=args.llm_base_url,
        timeout=60,
    )

    run_pipeline(args, stt_client, translation_client)

    try:
        paudio.terminate()
    except Exception as e:
        logger.error(f"Error terminating PyAudio: {e}")


if __name__ == "__main__":
    load_dotenv()
    parser = create_arg_parser()
    args = parser.parse_args()

    if args.list_devices:
        audio_capture.list_audio_devices()
        sys.exit(0)

    validate_device_index(args, parser)
    configure_logging(args.debug)
    validate_args(args, parser)
    main(args)
