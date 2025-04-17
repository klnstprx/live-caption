"""
Live translation pipeline: decouples audio capture, STT, translation, and output formatting.
"""

import time
import json
import queue
import threading
import logging
from concurrent.futures import ThreadPoolExecutor

from openai.types.chat import ChatCompletionMessage

from utils import (
    print_and_log,
    trim_conversation_history,
    pad_audio,
    get_audio_bytes_per_second,
)


def run_pipeline(args, stt_client, translation_client):
    logger = logging.getLogger("vad-whisper-llama")
    # Audio capture VAD thread
    from audio_capture import vad_capture_thread

    # Initialize conversation with system prompt
    conversation = [{"role": "system", "content": args.system_prompt}]
    # Set up audio queue and exit event
    audio_queue = queue.Queue(maxsize=10)
    exit_event = threading.Event()

    # Compute padding parameters
    bytes_per_sample = args.sample_width
    bytes_per_second = get_audio_bytes_per_second(
        args.rate, args.channels, bytes_per_sample
    )
    min_bytes = int((args.min_chunk_duration_ms / 1000.0) * bytes_per_second)
    min_bytes += (-min_bytes) % bytes_per_sample

    # Optional output file for STT and translation logs
    output_file_handle = None
    if args.output_file:
        try:
            output_file_handle = open(args.output_file, mode="a", encoding="utf-8")
            output_file_handle.write(
                f"--- Log Start: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
            )
            output_file_handle.flush()
        except Exception as e:
            logger.error(f"Error opening output file: {e}")

    # Start VAD capture thread
    executor = ThreadPoolExecutor(max_workers=1)
    capture_future = executor.submit(vad_capture_thread, audio_queue, exit_event, args)

    try:
        while True:
            # Check if capture thread has terminated
            if capture_future.done():
                if capture_future.exception():
                    logger.error(
                        "Capture thread error", exc_info=capture_future.exception()
                    )
                break
            # Get next audio chunk
            try:
                audio_data = audio_queue.get(timeout=0.5)
            except queue.Empty:
                if exit_event.is_set():
                    break
                continue

            # Pad audio chunk
            audio_data = pad_audio(audio_data, min_bytes, bytes_per_sample)
            current_duration_ms = 1000 * len(audio_data) / (bytes_per_second or 1)
            logger.debug(f"Padded audio to {current_duration_ms:.1f}ms")

            # STT transcription
            try:
                t0 = time.perf_counter()
                text_in = stt_client.transcribe(
                    audio_data, args.channels, args.sample_width, args.rate
                )
                t1 = time.perf_counter()
                logger.debug("STT latency: %.1f ms", (t1 - t0) * 1000)
            except Exception:
                logger.error("STT client error", exc_info=True)
                print_and_log("(STT error)", output_file_handle)
                continue
            if not text_in:
                logger.debug("(No transcription result)")
                continue

            print_and_log(f"(Korean) STT: {text_in}", output_file_handle)
            conversation.append({"role": "user", "content": text_in})

            # Translation
            try:
                t0 = time.perf_counter()
                reply_raw = translation_client.translate(conversation)
                t1 = time.perf_counter()
                logger.debug("Translation latency: %.1f ms", (t1 - t0) * 1000)
            except Exception:
                logger.error("Translation client error", exc_info=True)
                print_and_log("(Translation error)", output_file_handle)
                # Remove last user message to avoid context pollution
                if conversation and conversation[-1].get("role") == "user":
                    conversation.pop()
                continue
            if not reply_raw:
                print_and_log("(No translation result)", output_file_handle)
                if conversation and conversation[-1].get("role") == "user":
                    conversation.pop()
                continue

            # Parse and display translation result
            translation = "(No translation)"
            assistant_content_to_save = reply_raw
            try:
                # Parse the raw JSON response directly
                parsed = json.loads(reply_raw)
                if isinstance(parsed, dict):
                    translation = parsed.get("translatedText", translation)
                    assistant_content_to_save = json.dumps(
                        parsed, ensure_ascii=False, indent=2
                    )
            except (json.JSONDecodeError, TypeError) as e:
                err_msg = f"LLM returned invalid JSON ({e}). Raw: {reply_raw}"
                logger.error(err_msg)
                print_and_log(err_msg, output_file_handle)

            conversation.append(
                {"role": "assistant", "content": assistant_content_to_save}
            )
            print_and_log(f"(English) Translation:\n{translation}", output_file_handle)

            # Trim conversation history
            conversation = trim_conversation_history(
                conversation, args.max_conversation_messages
            )
    except KeyboardInterrupt:
        logger.info("Stopping...")
    except Exception as e:
        logger.error(f"Unexpected error in pipeline: {e}", exc_info=True)
    finally:
        exit_event.set()
        executor.shutdown(wait=False)
        if output_file_handle:
            try:
                output_file_handle.write(
                    f"--- Log End: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
                )
                output_file_handle.close()
            except Exception:
                logger.warning("Failed to finalize output file.")

