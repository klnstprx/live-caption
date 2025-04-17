# Standard and third-party imports
from dotenv import load_dotenv
import os
import sys
import argparse
import queue
import threading
import time
import json
import logging

import pyaudio
import requests

# Local module imports
from utils import (
    print_and_log,
    trim_conversation_history,
    health_check_endpoint,
    get_audio_bytes_per_second,
    pad_audio,
)
from audio_capture import vad_capture_thread
from stt_client import WhisperClient
from translation_client import LlamaClient

# --- Constants ---
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

# --- Logging Setup ---
logger = logging.getLogger("vad-whisper-llama")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def main(args):
    logger.info(f"Whisper URL: {args.whisper_url}")
    logger.info(f"Llama URL: {args.llama_url}")
    # Health-check the Whisper and Llama endpoints
    health_check_endpoint("Whisper", args.whisper_url)
    health_check_endpoint("Llama", args.llama_url)
    # Instantiate PyAudio once to determine sample width (bytes per sample for paInt16)
    paudio = pyaudio.PyAudio()
    args.sample_width = paudio.get_sample_size(pyaudio.paInt16)
    # Create HTTP session and clients for STT and translation
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

    # Prepare file handler for logging to file if requested
    file_handler = None
    # Dummy output_file_handle for legacy print_and_log signature
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

    capture_thread = threading.Thread(
        target=vad_capture_thread, args=(audio_queue, exit_event, args), daemon=True
    )
    capture_thread.start()

    try:
        while True:
            if not capture_thread.is_alive() and exit_event.is_set():
                logger.error("Capture thread exited. Stopping.")
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
        if capture_thread.is_alive():
            logger.info("Waiting for capture thread to exit...")
            capture_thread.join(timeout=2.0)

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


# --- CLI ---
if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Real-time VAD + Whisper + Llama translation"
    )
    parser.add_argument(
        "--whisper-url",
        type=str,
        default=os.environ.get("WHISPER_URL", DEFAULT_WHISPER_SERVER_URL),
    )
    parser.add_argument(
        "--llama-url",
        type=str,
        default=os.environ.get("LLAMA_URL", DEFAULT_LLAMA_SERVER_URL),
    )
    parser.add_argument(
        "--rate",
        type=int,
        choices=[8000, 16000, 32000, 48000],
        default=int(os.environ.get("RATE", DEFAULT_RATE)),
        help="Audio sampling rate in Hz (webrtcvad supports only 8000, 16000, 32000, 48000)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=int(os.environ.get("CHANNELS", DEFAULT_CHANNELS)),
        choices=[1],
    )
    parser.add_argument(
        "--frame-duration-ms",
        type=int,
        default=int(os.environ.get("FRAME_DURATION_MS", DEFAULT_FRAME_DURATION_MS)),
        choices=[10, 20, 30],
    )
    parser.add_argument(
        "--vad-mode",
        type=int,
        default=int(os.environ.get("VAD_MODE", DEFAULT_VAD_MODE)),
        choices=[0, 1, 2, 3],
    )
    parser.add_argument(
        "--max-silence-frames",
        type=int,
        default=int(os.environ.get("MAX_SILENCE_FRAMES", DEFAULT_MAX_SILENCE_FRAMES)),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME),
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=os.environ.get("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
    )
    parser.add_argument(
        "--max-conversation-messages",
        type=int,
        default=int(os.environ.get("MAX_CONV_MSGS", DEFAULT_MAX_CONVERSATION_MESSAGES)),
    )
    parser.add_argument(
        "--min-chunk-duration-ms",
        type=int,
        default=int(
            os.environ.get("MIN_CHUNK_DURATION_MS", DEFAULT_MIN_CHUNK_DURATION_MS)
        ),
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=os.environ.get("OUTPUT_FILE", DEFAULT_OUTPUT_FILE),
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=(
            int(os.environ["DEVICE_INDEX"])
            if os.environ.get("DEVICE_INDEX") is not None
            else None
        ),
        help="Index of audio input device (as listed by PyAudio).",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug logging (sets log level to DEBUG)",
    )
    parser.add_argument(
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
            if info.get("maxInputChannels", 0) > 0:
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
        if info.get("maxInputChannels", 0) <= 0:
            parser.error(f"Device {args.device_index} has no input channels")
    # Configure debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    # Validate max_conversation_messages
    if args.max_conversation_messages < 1 or args.max_conversation_messages % 2 == 0:
        parser.error("--max-conversation-messages must be an odd number >= 1")
    # Validate min_chunk_duration_ms
    if args.min_chunk_duration_ms < 0:
        parser.error("--min-chunk-duration-ms must be >= 0")
    main(args)
