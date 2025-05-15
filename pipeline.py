"""
Live translation pipeline: decouples audio capture, STT, translation, and output formatting.
"""

import time
import json
import queue
import threading
import logging
#
# Major performance optimisation: decouple the heavy STT and translation stages so they
# can operate concurrently with audio capture.  We introduce two additional work queues
# and a couple of lightweight worker threads – one (configurable) pool for STT and one
# single thread for translation.  This lets the microphone thread keep delivering audio
# while the previous chunk is still being transcribed or translated, hiding most of the
# network latency of those calls.
#
# The public API (``run_pipeline``) remains unchanged, so existing callers/tests are not
# affected.  All new objects live only inside this module.
#

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
    # -------------------------------
    # 1.  Setup shared state & queues
    # -------------------------------

    # Audio capture VAD thread (imports kept local to avoid heavy deps during unit-tests)
    from audio_capture import vad_capture_thread

    # Queues
    audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=50)  # raw PCM from mic
    stt_queue: queue.Queue[str] = queue.Queue(maxsize=100)  # Korean text

    exit_event = threading.Event()

    # Pre-compute values used by the STT workers
    bytes_per_sample = args.sample_width
    bytes_per_second = get_audio_bytes_per_second(
        args.rate, args.channels, bytes_per_sample
    )
    min_bytes = int((args.min_chunk_duration_ms / 1000.0) * bytes_per_second)
    # ensure alignment on sample width
    min_bytes += (-min_bytes) % bytes_per_sample

    # Optional output log file
    output_file_handle = None
    if args.output_file:
        try:
            output_file_handle = open(args.output_file, "a", encoding="utf-8")
            output_file_handle.write(
                f"--- Log Start: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
            )
            output_file_handle.flush()
        except Exception as e:
            logger.error("Error opening output file: %s", e)

    # ----------------------
    # 2.   Launch executors
    # ----------------------

    # Microphone capture – keep separate tiny executor so we can still join() on it.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="capture")
    capture_future = executor.submit(vad_capture_thread, audio_queue, exit_event, args)

    # STT workers – configurable in args, default 2.
    stt_workers = max(1, getattr(args, "stt_workers", 2))
    stt_executor = ThreadPoolExecutor(max_workers=stt_workers, thread_name_prefix="stt")

    # Helper so all STT workers share the same logic without re-computing constants.
    def _stt_worker():
        while not exit_event.is_set():
            try:
                audio_data = audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Pad so Whisper always receives at least min_bytes of audio
            audio_data = pad_audio(audio_data, min_bytes, bytes_per_sample)
            logger.debug(
                "STT worker got %d bytes (%.1f ms)",
                len(audio_data),
                1000 * len(audio_data) / (bytes_per_second or 1),
            )

            try:
                t0 = time.perf_counter()
                text_in = stt_client.transcribe(
                    audio_data, args.channels, args.sample_width, args.rate
                )
                dt_ms = (time.perf_counter() - t0) * 1000
                logger.debug("STT latency: %.1f ms", dt_ms)
            except Exception:
                logger.error("STT client error", exc_info=True)
                print_and_log("(STT error)", output_file_handle)
                continue

            if not text_in:
                logger.debug("(No transcription result)")
                continue

            # Forward to translation stage – drop if queue is full.
            try:
                stt_queue.put_nowait(text_in)
            except queue.Full:
                logger.warning("STT queue full, dropping transcription result.")

    # Kick off STT worker pool
    for _ in range(stt_workers):
        stt_executor.submit(_stt_worker)

    # ---------------------------
    # 3.   Translation processing
    # ---------------------------

    def _translation_loop():
        # Keep its own conversation context so ordering is preserved.
        conversation = [{"role": "system", "content": args.system_prompt}]

        while not exit_event.is_set() or not stt_queue.empty():
            try:
                text_in = stt_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            print_and_log(f"(Korean) STT: {text_in}", output_file_handle)
            conversation.append({"role": "user", "content": text_in})

            try:
                t0 = time.perf_counter()
                reply_raw = translation_client.translate(conversation)
                dt_ms = (time.perf_counter() - t0) * 1000
                logger.debug("Translation latency: %.1f ms", dt_ms)
            except Exception:
                logger.error("Translation client error", exc_info=True)
                print_and_log("(Translation error)", output_file_handle)
                # Remove the user msg to keep context clean
                if conversation and conversation[-1].get("role") == "user":
                    conversation.pop()
                continue

            if not reply_raw:
                print_and_log("(No translation result)", output_file_handle)
                if conversation and conversation[-1].get("role") == "user":
                    conversation.pop()
                continue

            # Parse JSON structured result
            translation = "(No translation)"
            assistant_content_to_save = None  # what we'll store in conversation
            try:
                parsed = json.loads(reply_raw)
                if isinstance(parsed, dict):
                    translation = parsed.get("translatedText", translation)
                    assistant_content_to_save = translation  # only store plain text
            except (json.JSONDecodeError, TypeError) as e:
                err_msg = f"LLM returned invalid JSON ({e}). Raw: {reply_raw}"
                logger.error(err_msg)
                print_and_log(err_msg, output_file_handle)

            # Fallback if JSON parse failed
            if assistant_content_to_save is None:
                assistant_content_to_save = reply_raw

            conversation.append(
                {"role": "assistant", "content": assistant_content_to_save}
            )
            print_and_log(f"(English) Translation:\n{translation}", output_file_handle)

            # Trim to keep runtime cheap
            conversation = trim_conversation_history(
                conversation, args.max_conversation_messages
            )

    # Translation thread – single so ordering is preserved
    translation_thread = threading.Thread(
        target=_translation_loop, name="translation", daemon=True
    )
    translation_thread.start()

    # ----------------------------------
    # 4.   Wait for termination signals
    # ----------------------------------
    try:
        while not exit_event.is_set():
            # If microphone thread crashed, propagate error
            if capture_future.done():
                if capture_future.exception():
                    logger.error(
                        "Capture thread error", exc_info=capture_future.exception()
                    )
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        logger.info("Stopping...")
    except Exception as e:
        logger.error("Unexpected error in pipeline: %s", e, exc_info=True)
    finally:
        # Signal all threads to exit then wait a bit
        exit_event.set()
        capture_future.cancel()
        stt_executor.shutdown(wait=False)
        executor.shutdown(wait=False)
        translation_thread.join(timeout=2.0)

        if output_file_handle:
            try:
                output_file_handle.write(
                    f"--- Log End: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
                )
                output_file_handle.close()
            except Exception:
                logger.warning("Failed to finalize output file.")
