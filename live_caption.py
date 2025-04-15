import webrtcvad
import math
import argparse
import sys
import queue
import threading
import time
import json
import requests
import pyaudio
import wave
import io

# --- Default Constants ---
# Server endpoints
DEFAULT_WHISPER_SERVER_URL = "http://127.0.0.1:8081/inference"
DEFAULT_LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"

# Microphone settings
DEFAULT_RATE = 16000  # sample rate
DEFAULT_CHANNELS = 1
DEFAULT_FRAME_DURATION_MS = 30  # webrtcvad supports 10, 20, or 30 ms frames
DEFAULT_VAD_MODE = 2  # 0-3, with 3 being the most aggressive (favors detecting speech)
DEFAULT_MAX_SILENCE_FRAMES = 25  # how many silent frames before we say "utterance ended"? 25 frames * 20 ms = 500 ms

# Model name
DEFAULT_MODEL_NAME = "gemma-3-12b-it"
DEFAULT_SYSTEM_PROMPT = """
You are an expert translator specializing in **physics**, fluent in Korean and English. Always respond strictly in English.

Your primary task is to translate Korean input from an ongoing **physics lecture** into accurate and clear English.

Pay meticulous attention to the correct translation of **physics terminology (e.g., force, velocity, quantum, field), scientific concepts, mathematical equations, units (e.g., m/s, Joules), and principles.** Accuracy in these technical details is paramount. Where a choice exists between a common phrasing and precise technical language, **opt for the technical term** to ensure scientific accuracy.

Preserve the original meaning and the **formal, academic tone** typical of a lecture setting. Produce grammatically correct English translations that are easily understood within a scientific context.

Keep in mind that each input is a segment of a longer lecture, so context may build over time.

Your response **must** be a structured JSON object with a single key, `translatedText`, containing the English translation.
"""
DEFAULT_MAX_CONVERSATION_MESSAGES = 2 * 4 + 1  # System + 3 pairs
DEFAULT_OUTPUT_FILE = None  # Default: don't save to file


# --- Helper Function for Printing and Logging ---
def print_and_log(message, output_file_handle):
    """Prints to console and optionally writes to a file."""
    print(message)
    if output_file_handle:
        try:
            output_file_handle.write(message + "\n")
            output_file_handle.flush()  # Ensure it's written immediately
        except Exception as e:
            print(f"Error writing to output file: {e}", file=sys.stderr)
            # Optionally disable further writing? For now, just report.


# --- Audio Capture Thread ---
def vad_capture_thread(audio_queue, exit_event, args):
    """
    This thread continuously reads from the microphone using settings from args
    and uses WebRTC VAD to form utterances. Each utterance is placed into
    audio_queue as raw PCM data.
    """
    vad = webrtcvad.Vad(args.vad_mode)
    chunk_size = int(args.rate * args.frame_duration_ms / 1000)
    max_silence_frames = args.max_silence_frames

    p = pyaudio.PyAudio()
    stream = None
    try:
        stream = p.open(
            rate=args.rate,
            channels=args.channels,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=chunk_size * 4,  # Buffer multiple VAD chunks
        )
        print("Microphone capture thread started. Listening...")
    except Exception as e:
        print(f"Error opening audio stream: {e}", file=sys.stderr)
        exit_event.set()  # Signal other threads to exit
        return  # Exit this thread

    speech_frames = []
    num_silent = 0
    in_speech = False

    try:
        while not exit_event.is_set():
            try:
                # read next frame, ignoring overflow
                frame = stream.read(chunk_size, exception_on_overflow=False)
            except OSError as e:
                # if overflow or other error, skip this frame
                print(
                    f"Warning: audio buffer overflow or read error: {e}",
                    file=sys.stderr,
                )
                # Add a small delay to prevent tight loop on continuous errors
                time.sleep(0.05)
                continue

            if len(frame) < chunk_size * 2:
                time.sleep(0.01)  # Small delay
                continue

            try:
                is_speech = vad.is_speech(frame, args.rate)
            except Exception as e:
                print(f"Error during VAD processing: {e}", file=sys.stderr)
                continue  # Skip this frame

            if is_speech:
                speech_frames.append(frame)
                num_silent = 0
                if not in_speech:
                    in_speech = True
            else:
                if in_speech:
                    num_silent += 1
                    speech_frames.append(
                        frame
                    )  # Keep appending silence shortly after speech
                    if num_silent > max_silence_frames:
                        # end of utterance
                        utterance = b"".join(speech_frames)
                        try:
                            audio_queue.put_nowait(utterance)
                        except queue.Full:
                            print(
                                "Warning: audio queue is full, dropping utterance.",
                                file=sys.stderr,
                            )
                        speech_frames = []
                        in_speech = False
                        num_silent = 0
                else:
                    # still silent, do nothing
                    pass
    finally:
        # on exit or error, close the stream
        if in_speech and speech_frames:
            # flush last partial utterance
            utterance = b"".join(speech_frames)
            # Use put with timeout to avoid blocking indefinitely if queue is full on exit
            try:
                audio_queue.put(utterance, timeout=0.5)
            except queue.Full:
                print(
                    "Warning: audio queue full when flushing last utterance.",
                    file=sys.stderr,
                )

        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()
        print("Microphone capture thread exited.")


# --- Whisper Transcription Function ---
def whisper_transcribe_chunk(raw_pcm, args):
    """
    Convert raw PCM data to WAV in memory and POST to whisper-server.
    Uses settings from args.
    """
    wav_buf = io.BytesIO()
    try:
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(args.channels)
            wf.setsampwidth(
                pyaudio.get_sample_size(pyaudio.paInt16)
            )  # Use PyAudio to get sample width
            wf.setframerate(args.rate)
            wf.writeframes(raw_pcm)
        wav_buf.seek(0)
    except Exception as e:
        print(f"Error creating WAV in memory: {e}", file=sys.stderr)
        return ""

    files = {"file": ("audio.wav", wav_buf, "audio/wav")}
    # These whisper params could also be CLI args if needed
    data = {
        "temperature": "0.0",
        "temperature_inc": "0.2",
        "response_format": "json",
    }

    try:
        resp = requests.post(args.whisper_url, files=files, data=data, timeout=30)
        resp.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        j = resp.json()
        return j.get("text", "").strip()
    except requests.exceptions.RequestException as e:
        print(
            f"Error contacting whisper-server at {args.whisper_url}: {e}",
            file=sys.stderr,
        )
        return ""
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from whisper-server: {e}", file=sys.stderr)
        print(
            f"Whisper Response Text: {resp.text[:500]}..."
        )  # Log part of the response
        return ""
    except Exception as e:
        print(
            f"An unexpected error occurred during whisper transcription: {e}",
            file=sys.stderr,
        )
        return ""


# --- Llama Translation Function ---
def llama_translate(full_conversation, args):
    """
    Sends the entire conversation to llama-server.
    Uses settings from args.
    Expects model to reply in JSON with 'translatedText' field.
    """
    payload = {
        "model": args.model_name,
        "messages": full_conversation,
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
        resp = requests.post(
            args.llama_url,
            json=payload,
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        j = resp.json()

        # Handle potential variations in response structure
        if (
            "choices" in j
            and len(j["choices"]) > 0
            and "message" in j["choices"][0]
            and "content" in j["choices"][0]["message"]
        ):
            reply_content = j["choices"][0]["message"]["content"]
            return reply_content
        else:
            print(f"Unexpected JSON structure from llama-server: {j}", file=sys.stderr)
            return ""

    except requests.exceptions.RequestException as e:
        print(
            f"Error contacting llama-server at {args.llama_url}: {e}", file=sys.stderr
        )
        return ""
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from llama-server: {e}", file=sys.stderr)
        print(f"Llama Response Text: {resp.text[:500]}...")  # Log part of the response
        return ""
    except KeyError as e:
        print(f"Missing expected key in llama-server response: {e}", file=sys.stderr)
        print(f"Llama Response JSON: {j}")
        return ""
    except Exception as e:
        print(
            f"An unexpected error occurred during llama translation: {e}",
            file=sys.stderr,
        )
        return ""


# --- Main Function ---
def main(args):
    """Main application logic."""
    print("=== Real-Time VAD + Whisper + Llama Translation (Threaded) ===")
    print(f"Whisper Server URL: {args.whisper_url}")
    print(f"Llama Server URL: {args.llama_url}")
    print(
        f"Audio Settings: Rate={args.rate}, Channels={args.channels}, Frame={args.frame_duration_ms}ms"
    )
    print(
        f"VAD Settings: Mode={args.vad_mode}, Silence Frames={args.max_silence_frames}"
    )
    print(
        f"LLM Settings: Model={args.model_name}, Max History={args.max_conversation_messages}"
    )
    if args.output_file:
        print(f"Saving output to: {args.output_file}")
    print("-" * 60)

    # Initialize conversation with the system prompt from args
    conversation = [{"role": "system", "content": args.system_prompt}]

    audio_queue = queue.Queue(maxsize=10)  # Increased queue size slightly
    exit_event = threading.Event()

    # Pass args to the capture thread
    capture_thread = threading.Thread(
        target=vad_capture_thread, args=(audio_queue, exit_event, args), daemon=True
    )
    capture_thread.start()
    min_duration_ms = 1000  # Define the target minimum duration
    print(f"Target Minimum Audio Chunk Duration: {min_duration_ms}ms")
    output_file_handle = None
    if args.output_file:
        try:
            # Open file in append mode with utf-8 encoding
            output_file_handle = open(args.output_file, "a", encoding="utf-8")
            output_file_handle.write(
                f"\n--- Log Start: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
            )
            output_file_handle.write(f"System Prompt: {args.system_prompt.strip()}\n")
            output_file_handle.write("-" * 60 + "\n")
        except IOError as e:
            print(f"Error opening output file {args.output_file}: {e}", file=sys.stderr)
            output_file_handle = None  # Ensure it's None if opening failed

    # Calculate bytes per millisecond for duration checks
    bytes_per_sample = pyaudio.get_sample_size(
        pyaudio.paInt16
    )  # Should be 2 for paInt16
    bytes_per_second = args.rate * args.channels * bytes_per_sample
    # Use floating point for intermediate calculations to avoid precision loss

    if bytes_per_second > 0:
        target_min_bytes_float = (min_duration_ms / 1000.0) * bytes_per_second
        # Round up to the nearest whole byte
        target_min_bytes = math.ceil(target_min_bytes_float)
        # Ensure target_min_bytes is a multiple of bytes_per_sample (e.g., even for 16-bit)
        remainder = target_min_bytes % bytes_per_sample
        if remainder != 0:
            target_min_bytes += bytes_per_sample - remainder
    else:
        target_min_bytes = 0  # Avoid issues if rate/channels are somehow zero
    print(
        f"Target Minimum Audio Chunk Duration: {min_duration_ms}ms (Calculated Target Bytes: {target_min_bytes})"
    )

    try:
        while True:
            # Check if capture thread is still alive
            if not capture_thread.is_alive() and exit_event.is_set():
                print(
                    "Capture thread exited unexpectedly. Stopping main loop.",
                    file=sys.stderr,
                )
                break

            # 1) Get an utterance from the queue
            try:
                # Use a slightly longer timeout to reduce busy-waiting
                audio_data = audio_queue.get(timeout=0.5)
            except queue.Empty:
                if exit_event.is_set():  # Check if exit was signaled while waiting
                    break
                continue  # No audio yet, loop back

            # --- Start Padding Logic ---
            current_bytes = len(audio_data)
            current_duration_ms = (
                (current_bytes / bytes_per_second) * 1000.0
                if bytes_per_second > 0
                else 0
            )  # Avoid division by zero

            if (
                bytes_per_second > 0 and current_bytes < target_min_bytes
            ):  # Check bytes_per_second > 0
                bytes_to_add = target_min_bytes - current_bytes
                # Ensure we add an even number of bytes since samples are 2 bytes each
                if bytes_to_add % bytes_per_sample != 0:
                    # Add padding to align to the next sample boundary
                    bytes_to_add += bytes_per_sample - (bytes_to_add % bytes_per_sample)

                # Calculate number of silent samples to add (each sample is bytes_per_sample bytes)
                samples_to_add = bytes_to_add // bytes_per_sample

                # Generate silence: null bytes ('\x00'). Since it's 16-bit, each sample is b'\x00\x00'
                # Handle potential different sample widths if needed, but paInt16 is standard here.
                silence_padding = (b"\x00" * bytes_per_sample) * samples_to_add

                original_len = len(audio_data)
                audio_data += silence_padding
                padded_len = len(audio_data)
                padded_duration_ms = (padded_len / bytes_per_second) * 1000.0

                padding_log_msg = (
                    f"Padding audio chunk: Original={current_duration_ms:.2f}ms ({original_len} bytes), "
                    f"Target={min_duration_ms}ms ({target_min_bytes} bytes), "
                    f"Added={len(silence_padding)} bytes, "
                    f"New Duration={padded_duration_ms:.2f}ms ({padded_len} bytes)"
                )
                print_and_log(padding_log_msg, output_file_handle)
            # --- End Padding Logic ---
            #
            # 2) Transcribe using Whisper
            # Pass args to the transcription function
            recognized_ko = whisper_transcribe_chunk(audio_data, args)
            if not recognized_ko:
                print_and_log("(No transcription result)", output_file_handle)
                continue  # Skip if transcription failed or returned empty

            log_msg_ko = f"(Korean) STT: {recognized_ko}"
            print_and_log(log_msg_ko, output_file_handle)

            # 3) Append user message to conversation history
            conversation.append({"role": "user", "content": recognized_ko})

            # 4) Translate using Llama
            # Pass args to the translation function
            llama_reply_raw = llama_translate(
                conversation, args
            )  # Get the raw reply (string)

            translation = (
                "Error: LLM call failed or returned empty."  # Default error message
            )
            assistant_content_to_save = llama_reply_raw  # Save the raw reply by default

            if llama_reply_raw:
                # Attempt to parse the JSON response from Llama
                try:
                    # Ensure the raw reply is treated as a string before loading
                    llama_reply_json = json.loads(str(llama_reply_raw))

                    # Check if the parsed result is a dictionary (as expected for JSON object)
                    if isinstance(llama_reply_json, dict):
                        translation = llama_reply_json.get(
                            "translatedText",
                            "Error: 'translatedText' key missing in JSON",
                        )
                        # If parsing succeeded, save the structured JSON string representation
                        assistant_content_to_save = json.dumps(
                            llama_reply_json, ensure_ascii=False, indent=2
                        )
                    else:
                        # Handle cases where Llama returns a plain string that happens to be valid JSON (e.g., just "some text")
                        # Or if it returned JSON but not an object {}
                        print_and_log(
                            f"LLM returned valid JSON, but not the expected object format. Raw Reply: {llama_reply_raw}",
                            output_file_handle,
                        )
                        translation = (
                            f"Info: LLM returned non-object JSON: {llama_reply_raw}"
                        )
                        # Keep assistant_content_to_save as the raw string

                except json.JSONDecodeError:
                    # If the reply wasn't valid JSON, log it and use a default error message
                    print_and_log(
                        f"LLM did not return valid JSON. Full reply:\n{llama_reply_raw}",
                        output_file_handle,
                    )
                    translation = "Error: LLM response was not valid JSON."
                    # Keep assistant_content_to_save as the raw string (which is not JSON)
            else:
                # If llama_translate returned empty string (due to network error etc.)
                assistant_content_to_save = '{"error": "LLM call failed or returned empty"}'  # Save an error JSON

            # Append the content that Llama *actually* returned to the conversation history
            conversation.append(
                {"role": "assistant", "content": assistant_content_to_save}
            )

            # Print the extracted/error translation
            log_msg_en = f"(English) Translation:\n{translation}"
            print_and_log(log_msg_en, output_file_handle)
            print_and_log("-" * 60, output_file_handle)

            # 5) Trim conversation history if it exceeds the maximum length
            # Keep system prompt (index 0), remove oldest user/assistant pairs (indices 1 and 2)
            while len(conversation) > args.max_conversation_messages:
                if len(conversation) > 2:  # Ensure we don't remove system prompt
                    conversation.pop(1)  # Remove the oldest user message
                    conversation.pop(
                        1
                    )  # Remove the oldest assistant message (now at index 1)
                else:
                    break  # Should not happen with max_conv > 1, but safety check

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nAn unexpected error occurred in the main loop: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()  # Print stack trace for debugging
    finally:
        print("Exiting...")
        exit_event.set()  # Signal threads to exit

        # Wait for the capture thread to finish
        if capture_thread.is_alive():
            print("Waiting for capture thread to exit...")
            capture_thread.join(timeout=2.0)  # Add a timeout
            if capture_thread.is_alive():
                print("Warning: Capture thread did not exit cleanly.", file=sys.stderr)

        # Close the output file if it was opened
        if output_file_handle:
            try:
                output_file_handle.write(
                    f"--- Log End: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
                )
                output_file_handle.close()
                print(f"Translation log saved to {args.output_file}")
            except Exception as e:
                print(f"Error closing output file: {e}", file=sys.stderr)

        print("All done.")


# --- Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time VAD + Whisper + Llama translation"
    )

    # Server Endpoints
    parser.add_argument(
        "--whisper-url",
        type=str,
        default=DEFAULT_WHISPER_SERVER_URL,
        help="URL of the whisper-server inference endpoint",
    )
    parser.add_argument(
        "--llama-url",
        type=str,
        default=DEFAULT_LLAMA_SERVER_URL,
        help="URL of the llama-server chat completions endpoint",
    )

    # Audio Settings
    parser.add_argument(
        "--rate",
        type=int,
        default=DEFAULT_RATE,
        help="Sample rate for microphone audio (Hz)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=DEFAULT_CHANNELS,
        choices=[1],  # Currently only supports mono
        help="Number of audio channels (only 1 supported)",
    )
    parser.add_argument(
        "--frame-duration-ms",
        type=int,
        default=DEFAULT_FRAME_DURATION_MS,
        choices=[10, 20, 30],
        help="Duration of each audio frame for VAD (ms)",
    )

    # VAD Settings
    parser.add_argument(
        "--vad-mode",
        type=int,
        default=DEFAULT_VAD_MODE,
        choices=[0, 1, 2, 3],
        help="WebRTC VAD agressiveness mode (0-3, 3 is most aggressive)",
    )
    parser.add_argument(
        "--max-silence-frames",
        type=int,
        default=DEFAULT_MAX_SILENCE_FRAMES,
        help="Number of consecutive silent frames to detect end of utterance",
    )

    # LLM Settings
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Name of the model to use in llama-server",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for the translation model",
    )
    parser.add_argument(
        "--max-conversation-messages",
        type=int,
        default=DEFAULT_MAX_CONVERSATION_MESSAGES,
        help="Max number of messages (system + user/assistant pairs) in history (must be odd)",
    )

    # Output File
    parser.add_argument(
        "--output-file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="Path to a file to save the Korean STT and English translation log",
    )

    # Validate max_conversation_messages is odd and >= 1
    parsed_args = parser.parse_args()
    if (
        parsed_args.max_conversation_messages < 1
        or parsed_args.max_conversation_messages % 2 == 0
    ):
        parser.error("--max-conversation-messages must be an odd number >= 1")

    main(parsed_args)
