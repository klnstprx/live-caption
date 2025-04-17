"""Live caption orchestrator: capture audio via VAD, transcribe with Whisper, translate with Llama, and output in real time."""

import os
import sys
import argparse
import queue
import threading
import time
import json
import logging

# Required dependencies: if missing, exit with error (allow --help)
if not any(arg in ('-h','--help') for arg in sys.argv):
    missing = []
    try:
        import webrtcvad
    except ImportError:
        missing.append('webrtcvad')
    try:
        import pyaudio
    except ImportError:
        missing.append('pyaudio')
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
    except ImportError:
        missing.append('requests')
    if missing:
        print(
            f"Error: Missing required libraries: {', '.join(missing)}."
            " Please install them via pip.",
            file=sys.stderr,
        )
        sys.exit(1)

# Optional dependencies
try:
    from dotenv import load_dotenv
except ImportError:
    # If python-dotenv is not installed, define a no-op
    def load_dotenv():
        pass

from utils import (
    print_and_log,
    trim_conversation_history,
    health_check_endpoint,
    get_audio_bytes_per_second,
    pad_audio,
    JSONFormatter,
)
# Heavy external modules are imported within main to allow CLI help without dependencies

DEFAULT_WHISPER_SERVER_URL = "http://127.0.0.1:8081/inference"
DEFAULT_LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_FRAME_DURATION_MS = 30
DEFAULT_VAD_MODE = 2  # 1-3 VAD aggressiveness, higher value = more aggressive chunking = shorter chunks (most likely)
DEFAULT_MAX_SILENCE_FRAMES = 25
DEFAULT_MODEL_NAME = "gemma-3-12b-it"
DEFAULT_MIN_CHUNK_DURATION_MS = 1000

DEFAULT_SYSTEM_PROMPT = """You are an expert translator specializing in **physics**, fluent in Korean and English. Always respond strictly in English.

Your primary task is to translate Korean input from an ongoing **physics lecture** into accurate and clear English.

Pay meticulous attention to the correct translation of **physics terminology (e.g., force, velocity, quantum, field), scientific concepts, mathematical equations, units (e.g., m/s, Joules), and principles.** Accuracy in these technical details is paramount. Where a choice exists between a common phrasing and precise technical language, **opt for the technical term** to ensure scientific accuracy.

Preserve the original meaning and the **formal, academic tone** typical of a lecture setting. Produce grammatically correct English translations that are easily understood within a scientific context.

Keep in mind that each input is a segment of a longer lecture, so context may build over time.

Your response **must** be a structured JSON object with a single key, `translatedText`, containing the English translation."""

DEFAULT_MAX_CONVERSATION_MESSAGES = 2 * 4 + 1
DEFAULT_OUTPUT_FILE = None

logger = logging.getLogger("vad-whisper-llama")


def main(args):
    """
    Main orchestration loop:
    - Health-check services
    - Capture audio via VAD
    - Transcribe with Whisper
    - Translate with Llama
    - Print/log translated output
    """
    logger.info(f"Whisper URL: {args.whisper_url}")
    logger.info(f"Llama URL: {args.llama_url}")
    # Import pipeline components here to allow CLI help without dependencies
    import audio_capture
    from audio_capture import vad_capture_thread, list_audio_devices
    from stt_client import WhisperClient
    from translation_client import LlamaClient
    from concurrent.futures import ThreadPoolExecutor
    health_check_endpoint("Whisper", args.whisper_url)
    health_check_endpoint("Llama", args.llama_url)
    paudio = pyaudio.PyAudio()
    paudio = pyaudio.PyAudio()
    args.sample_width = paudio.get_sample_size(pyaudio.paInt16)
    # Configure HTTP session with retry strategy
    http_session = requests.Session()
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
        backoff_factor=1,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http_session.mount("http://", adapter)
    http_session.mount("https://", adapter)
    http_session = requests.Session()
    stt_client = WhisperClient(args.whisper_url, session=http_session)
    llama_client = LlamaClient(args.llama_url, args.model_name, session=http_session)

    conversation = [{"role": "system", "content": args.system_prompt}]
    audio_queue = queue.Queue(maxsize=10)
    exit_event = threading.Event()

    # For audio padding computation
    bytes_per_sample = args.sample_width
    bytes_per_second = get_audio_bytes_per_second(
        args.rate, args.channels, bytes_per_sample
    )
    min_bytes = int((args.min_chunk_duration_ms / 1000.0) * bytes_per_second)
    min_bytes += (-min_bytes) % bytes_per_sample  # Multiple of sample width

    # Set up optional file logging if args.output_file is provided
    file_handler = None
    output_file_handle = None
    if args.output_file:
        try:
            file_handler = logging.FileHandler(args.output_file, encoding="utf-8")
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)
            logger.info("Logging to file: %s", args.output_file)
            logger.info("--- Log Start: %s ---", time.strftime("%Y-%m-%d %H:%M:%S"))
        except Exception as e:
            logger.error(f"Error creating log file handler: {e}")

    # Start VAD capture in background with exception propagation
    executor = ThreadPoolExecutor(max_workers=1)
    capture_future = executor.submit(vad_capture_thread, audio_queue, exit_event, args)

    try:
        while True:
            # Stop if capture thread finished or exit event set
            if capture_future.done():
                if capture_future.exception():
                    logger.error(
                        "Capture thread error", exc_info=capture_future.exception()
                    )
                break
            try:
                audio_data = audio_queue.get(timeout=0.5)
            except queue.Empty:
                if exit_event.is_set():
                    break
                continue

            # Padding audio chunk
            audio_data = pad_audio(audio_data, min_bytes, bytes_per_sample)
            current_duration_ms = 1000 * len(audio_data) / (bytes_per_second or 1)
            print_and_log(
                f"Padded audio to {current_duration_ms:.1f}ms", output_file_handle
            )

            # Whisper (STT)
            t0 = time.perf_counter()
            recognized_ko = stt_client.transcribe(
                audio_data, args.channels, args.sample_width, args.rate
            )
            t1 = time.perf_counter()
            logger.info("Whisper STT latency: %.1f ms", (t1 - t0) * 1000)
            if not recognized_ko:
                print_and_log("(No transcription result)", output_file_handle)
                continue

            print_and_log(f"(Korean) STT: {recognized_ko}", output_file_handle)
            conversation.append({"role": "user", "content": recognized_ko})

            # Llama translation
            t0 = time.perf_counter()
            llama_reply_raw = llama_client.translate(conversation)
            t1 = time.perf_counter()
            logger.info("Llama translation latency: %.1f ms", (t1 - t0) * 1000)
            # If Llama returned nothing, drop the last user message to avoid context pollution
            if not llama_reply_raw:
                print_and_log("(No translation result from Llama)", output_file_handle)
                # remove the user message appended just before
                if conversation and conversation[-1].get("role") == "user":
                    conversation.pop()
                continue

            # Handle Llama JSON response
            translation = "(No translation)"
            assistant_content_to_save = llama_reply_raw
            try:
                llama_reply_json = json.loads(str(llama_reply_raw))
                if isinstance(llama_reply_json, dict):
                    translation = llama_reply_json.get(
                        "translatedText", "KeyError: 'translatedText'"
                    )
                    assistant_content_to_save = json.dumps(
                        llama_reply_json, ensure_ascii=False, indent=2
                    )
            except (json.JSONDecodeError, TypeError) as e:
                err_msg = f"LLM returned invalid JSON ({e}). Raw: {llama_reply_raw}"
                logger.error(err_msg)
                print_and_log(err_msg, output_file_handle)

            conversation.append(
                {"role": "assistant", "content": assistant_content_to_save}
            )
            print_and_log(f"(English) Translation:\n{translation}", output_file_handle)
            print_and_log("-" * 60, output_file_handle)

            trim_conversation_history(conversation, args.max_conversation_messages)

    except KeyboardInterrupt:
        logger.info("Stopping...")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
    finally:
        exit_event.set()
        executor.shutdown(wait=False)

        # Finalize file logging
        if file_handler:
            logger.info("--- Log End: %s ---", time.strftime("%Y-%m-%d %H:%M:%S"))
            logger.removeHandler(file_handler)
            try:
                file_handler.close()
            except Exception:
                pass
        # Terminate the PyAudio instance used for sample width
        try:
            paudio.terminate()
        except Exception as e:
            logger.error(f"Error terminating PyAudio: {e}")


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Real-time VAD + Whisper + Llama translation"
    )
    # Argument groups for better CLI UX
    server_group = parser.add_argument_group("Server options")
    server_group.add_argument(
        "--whisper-url",
        type=str,
        default=os.environ.get("WHISPER_URL", DEFAULT_WHISPER_SERVER_URL),
    )
    server_group.add_argument(
        "--llama-url",
        type=str,
        default=os.environ.get("LLAMA_URL", DEFAULT_LLAMA_SERVER_URL),
    )
    # Audio capture & VAD options
    audio_group = parser.add_argument_group("Audio capture & VAD options")
    audio_group.add_argument(
        "--rate",
        type=int,
        choices=[8000, 16000, 32000, 48000],
        default=int(os.environ.get("RATE", DEFAULT_RATE)),
        help="Audio sampling rate in Hz (webrtcvad supports only 8000, 16000, 32000, 48000)",
    )
    audio_group.add_argument(
        "--channels",
        type=int,
        default=int(os.environ.get("CHANNELS", DEFAULT_CHANNELS)),
        choices=[1],
    )
    audio_group.add_argument(
        "--frame-duration-ms",
        type=int,
        default=int(os.environ.get("FRAME_DURATION_MS", DEFAULT_FRAME_DURATION_MS)),
        choices=[10, 20, 30],
    )
    audio_group.add_argument(
        "--vad-mode",
        type=int,
        default=int(os.environ.get("VAD_MODE", DEFAULT_VAD_MODE)),
        choices=[0, 1, 2, 3],
    )
    audio_group.add_argument(
        "--max-silence-frames",
        type=int,
        default=int(os.environ.get("MAX_SILENCE_FRAMES", DEFAULT_MAX_SILENCE_FRAMES)),
    )
    # Model & prompt options
    model_group = parser.add_argument_group("Model & prompt options")
    model_group.add_argument(
        "--model-name",
        type=str,
        default=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME),
    )
    model_group.add_argument(
        "--system-prompt",
        type=str,
        default=os.environ.get("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
    )
    model_group.add_argument(
        "--max-conversation-messages",
        type=int,
        default=int(os.environ.get("MAX_CONV_MSGS", DEFAULT_MAX_CONVERSATION_MESSAGES)),
    )
    audio_group.add_argument(
        "--min-chunk-duration-ms",
        type=int,
        default=int(
            os.environ.get("MIN_CHUNK_DURATION_MS", DEFAULT_MIN_CHUNK_DURATION_MS)
        ),
    )
    # Output & logging options
    out_group = parser.add_argument_group("Output & logging options")
    out_group.add_argument(
        "--output-file",
        type=str,
        default=os.environ.get("OUTPUT_FILE", DEFAULT_OUTPUT_FILE),
    )
    audio_group.add_argument(
        "--device-index",
        type=int,
        default=(
            int(os.environ["DEVICE_INDEX"])
            if os.environ.get("DEVICE_INDEX") is not None
            else None
        ),
        help="Index of audio input device (as listed by PyAudio).",
    )
    out_group.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug logging (sets log level to DEBUG)",
    )
    out_group.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices (with index) and exit",
    )
    args = parser.parse_args()
    # Handle listing audio input devices
    if getattr(args, "list_devices", False):
        pa = pyaudio.PyAudio()
        print("Available audio input devices:")
        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
            except Exception:
                continue
            if int(info.get("maxInputChannels", 0)) > 0:
                channels = info.get("maxInputChannels")
                print(
                    f"{i}: {info.get('name')} ({channels} input channel{'s' if channels != 1 else ''})"
                )
        pa.terminate()
        sys.exit(0)
    # Validate device-index if specified
    if args.device_index is not None:
        pa = pyaudio.PyAudio()
        try:
            info = pa.get_device_info_by_index(args.device_index)
        except Exception as e:
            parser.error(f"Invalid device index {args.device_index}: {e}")
        finally:
            pa.terminate()
        if int(info.get("maxInputChannels", 0)) <= 0:
            parser.error(f"Device {args.device_index} has no input channels")
    # Configure debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    # Configure JSON structured logging
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(JSONFormatter())
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    # Validate max_conversation_messages
    if args.max_conversation_messages < 1 or args.max_conversation_messages % 2 == 0:
        parser.error("--max-conversation-messages must be an odd number >= 1")
    # Validate min_chunk_duration_ms
    if args.min_chunk_duration_ms < 0:
        parser.error("--min-chunk-duration-ms must be >= 0")
    main(args)
