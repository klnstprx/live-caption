import webrtcvad
from dotenv import load_dotenv
import os
import argparse
import queue
import threading
import time
import json
import requests
import pyaudio
import wave
import io
import logging
from typing import Optional, List, Dict, Any

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


# --- Helper Functions ---
def print_and_log(message: str, output_file_handle: Optional[Any] = None):
    # Log to console (with timestamp and level via logger configuration)
    logger.info(message)
    # File logging is handled by configured logging handlers


def get_audio_bytes_per_second(rate: int, channels: int, sample_width: int) -> int:
    return rate * channels * sample_width


def pad_audio(audio_data: bytes, min_bytes: int, sample_width: int) -> bytes:
    """Pads the audio data with silence to reach the minimum byte length."""
    current_bytes = len(audio_data)
    bytes_to_add = max(0, min_bytes - current_bytes)
    # Ensure even padding for sample width
    if bytes_to_add % sample_width:
        bytes_to_add += sample_width - (bytes_to_add % sample_width)
    padded_audio = audio_data + (b"\x00" * bytes_to_add)
    return padded_audio


def trim_conversation_history(conversation: List[Dict], max_len: int):
    """Keeps the conversation at the max length (always keeps system prompt at index 0)."""
    while len(conversation) > max_len and len(conversation) > 2:
        del conversation[1:3]  # Remove oldest user/assistant pair


# --- Audio Capture Thread ---
def vad_capture_thread(audio_queue, exit_event, args):
    vad = webrtcvad.Vad(args.vad_mode)
    chunk_size = int(args.rate * args.frame_duration_ms / 1000)
    p = pyaudio.PyAudio()
    stream = None
    try:
        # Open audio stream; if device_index is None, PyAudio will use default
        stream = p.open(
            rate=args.rate,
            channels=args.channels,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=chunk_size * 4,
            input_device_index=args.device_index,
        )
        logger.info("Microphone capture thread started. Listening...")
    except Exception as e:
        logger.error(f"Error opening audio stream: {e}")
        exit_event.set()
        return

    speech_frames, num_silent, in_speech = [], 0, False

    try:
        while not exit_event.is_set():
            try:
                frame = stream.read(chunk_size, exception_on_overflow=False)
            except OSError as e:
                logger.warning(f"Audio buffer overflow or read error: {e}")
                time.sleep(0.05)
                continue
            if len(frame) < chunk_size * 2:
                time.sleep(0.01)
                continue

            try:
                is_speech = vad.is_speech(frame, args.rate)
            except Exception as e:
                logger.warning(f"VAD processing error: {e}")
                continue

            if is_speech:
                speech_frames.append(frame)
                num_silent = 0
                in_speech = True
            elif in_speech:
                num_silent += 1
                speech_frames.append(frame)
                if num_silent > args.max_silence_frames:
                    utterance = b"".join(speech_frames)
                    try:
                        audio_queue.put_nowait(utterance)
                    except queue.Full:
                        logger.warning("Audio queue full, dropping utterance.")
                    speech_frames, in_speech, num_silent = [], False, 0
    finally:
        if in_speech and speech_frames:
            try:
                audio_queue.put(b"".join(speech_frames), timeout=0.5)
            except queue.Full:
                logger.warning("Audio queue full on thread exit.")
        # Always clean up
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()
        logger.info("Microphone capture thread exited.")


def whisper_transcribe_chunk(raw_pcm: bytes, args) -> str:
    """Convert raw PCM to WAV in-memory and POST to Whisper server."""
    wav_buf = io.BytesIO()
    try:
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(args.channels)
            # use sample width determined in main via args.sample_width
            wf.setsampwidth(args.sample_width)
            wf.setframerate(args.rate)
            wf.writeframes(raw_pcm)
        wav_buf.seek(0)
    except Exception as e:
        logger.error(f"Error creating WAV: {e}")
        return ""
    files = {"file": ("audio.wav", wav_buf, "audio/wav")}  
    # Ensure numeric types for temperature parameters  
    data = {"temperature": 0.0, "temperature_inc": 0.2, "response_format": "json"}  
    # Debug: log request details  
    logger.debug("Whisper request URL: %s, data: %s", args.whisper_url, data)  
    try:  
        resp = requests.post(args.whisper_url, files=files, data=data, timeout=30)  
        # Debug: log response status and body  
        logger.debug("Whisper response [%d]: %s", resp.status_code, resp.text)  
        resp.raise_for_status()  
        return resp.json().get("text", "").strip()  
    except Exception as e:  
        logger.error(f"Whisper error: {e}")  
        return ""  


def llama_translate(conversation: List[Dict], args) -> str:
    """Send conversation to llama-server, expect a JSON with 'translatedText'."""
    payload = {
        "model": args.model_name,
        "messages": conversation,
        "temperature": 0.7,
        "max_tokens": 1024,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "translation",
                "schema": {
                    "type": "object",
                    "properties": {
                        "translatedText": {
                            "type": "string",
                            "description": "The translated text in the target language.",
                        }
                    },
                    "required": ["translatedText"],
                },
            },
        },
    }
    headers = {"Content-Type": "application/json"}
    try:
        # Debug: log Llama request
        logger.debug("Llama request URL: %s, payload: %s", args.llama_url, json.dumps(payload, ensure_ascii=False))
        resp = requests.post(args.llama_url, json=payload, headers=headers, timeout=60)
        # Debug: log Llama response
        logger.debug("Llama response [%d]: %s", resp.status_code, resp.text)
        resp.raise_for_status()
        j = resp.json()
        choices = j.get("choices", [])
        if choices and "message" in choices[0] and "content" in choices[0]["message"]:
            return choices[0]["message"]["content"]
        logger.error(f"Unexpected Llama resp structure: {j}")
        return ""
    except Exception as e:
        logger.error(f"Llama error: {e}")
        return ""


def main(args):
    logger.info(f"Whisper URL: {args.whisper_url}")
    logger.info(f"Llama URL: {args.llama_url}")
    # Instantiate PyAudio once to determine sample width (bytes per sample for paInt16)
    paudio = pyaudio.PyAudio()
    args.sample_width = paudio.get_sample_size(pyaudio.paInt16)

    conversation = [{"role": "system", "content": args.system_prompt}]
    audio_queue = queue.Queue(maxsize=10)
    exit_event = threading.Event()

    # For audio padding computation
    # use sample width obtained above
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

            # Whisper
            recognized_ko = whisper_transcribe_chunk(audio_data, args)
            if not recognized_ko:
                print_and_log("(No transcription result)", output_file_handle)
                continue

            print_and_log(f"(Korean) STT: {recognized_ko}", output_file_handle)
            conversation.append({"role": "user", "content": recognized_ko})

            # Llama
            llama_reply_raw = llama_translate(conversation, args)
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
            except Exception:
                print_and_log(
                    f"LLM did not return valid JSON. Raw: {llama_reply_raw}",
                    output_file_handle,
                )

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
    # Server and audio arguments as before...
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
    # Optional: specify audio input device index (PyAudio)
    parser.add_argument(
        "--device-index",
        type=int,
        default=(int(os.environ["DEVICE_INDEX"]) if os.environ.get("DEVICE_INDEX") is not None else None),
        help="Index of audio input device (as listed by PyAudio).",
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug logging (sets log level to DEBUG)",
    )
    args = parser.parse_args()
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
