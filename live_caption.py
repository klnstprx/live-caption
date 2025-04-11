import webrtcvad
import queue
import threading
import time
import json
import requests
import pyaudio
import wave
import io

# The endpoints for your servers:
WHISPER_SERVER_URL = "http://127.0.0.1:9000/inference"
LLAMA_SERVER_URL = "http://127.0.0.1:9998/v1/chat/completions"

# Microphone settings
RATE = 16000  # sample rate
CHANNELS = 1
FRAME_DURATION_MS = 20  # webrtcvad supports 10, 20, or 30 ms frames
CHUNK_SIZE = int(RATE * FRAME_DURATION_MS / 1000)  # e.g. 16000 * 20/1000 = 320
VAD_MODE = 3  # 0-3, with 3 being the most aggressive (favors detecting speech)
MAX_SILENCE_FRAMES = 25  # how many silent frames before we say "utterance ended"?
# 25 frames * 20 ms = 500 ms

# Model name or parameters for llama-server
MODEL_NAME = "gemma-3-12b-it"
SYSTEM_PROMPT = """
    You are an expert translator fluent in Korean and English. Always respond strictly in English. 
    Your task is to translate the given Korean sentences accurately and naturally into English. 
    Preserve the original meaning, tone, and nuance of the Korean input while producing clear, readable, and grammatically correct English translations.
    The input you receive is a part of a longer lecture, please keep that in mind.
    Your response should be a structured json.
    """
MAX_CONVERSATION_MESSAGES = 2 * 3 + 1


def vad_capture_thread(audio_queue, exit_event):
    """
    This thread continuously reads from the microphone and uses WebRTC VAD to form
    utterances. Each utterance is placed into audio_queue as raw PCM data.
    """
    vad = webrtcvad.Vad(VAD_MODE)

    p = pyaudio.PyAudio()
    stream = p.open(
        rate=RATE,
        channels=CHANNELS,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=CHUNK_SIZE * 4,  # bigger buffer
    )

    print("Microphone capture thread started. Listening...")

    speech_frames = []
    num_silent = 0
    in_speech = False

    try:
        while not exit_event.is_set():
            try:
                # read next frame, ignoring overflow
                frame = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            except OSError as e:
                # if overflow or other error, skip this frame
                print("Warning: audio buffer overflow or read error:", e)
                continue

            if len(frame) < 2:
                time.sleep(0.01)
                continue

            is_speech = vad.is_speech(frame, RATE)
            if is_speech:
                speech_frames.append(frame)
                num_silent = 0
                if not in_speech:
                    in_speech = True
            else:
                if in_speech:
                    num_silent += 1
                    speech_frames.append(frame)
                    if num_silent > MAX_SILENCE_FRAMES:
                        # end of utterance
                        utterance = b"".join(speech_frames)
                        try:
                            audio_queue.put_nowait(utterance)
                        except queue.Full:
                            print("Warning: audio queue is full, dropping utterance.")
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
            audio_queue.put(utterance)
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Microphone capture thread exited.")


def whisper_transcribe_chunk(raw_pcm):
    """
    Convert raw PCM data to WAV in memory and POST to whisper-server.
    """
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(RATE)
        wf.writeframes(raw_pcm)
    wav_buf.seek(0)

    files = {"file": ("audio.wav", wav_buf, "audio/wav")}
    data = {
        "temperature": "0.0",
        "temperature_inc": "0.2",
        "response_format": "json",
    }

    try:
        resp = requests.post(WHISPER_SERVER_URL, files=files, data=data, timeout=30)
        resp.raise_for_status()
        j = resp.json()
        return j.get("text", "").strip()
    except Exception as e:
        print("Error contacting whisper-server:", e)
        return ""


def llama_translate(full_conversation):
    """
    Sends the entire conversation (system + user + assistant messages) to llama-server.
    Expects model to reply in JSON with 'translatedText' field.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": full_conversation,
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(
            LLAMA_SERVER_URL, json=payload, headers=headers, timeout=60
        )
        resp.raise_for_status()
        j = resp.json()
        reply_content = j["choices"][0]["message"]["content"]
        return reply_content
    except Exception as e:
        print("Error contacting llama-server:", e)
        return ""


def main():
    print("=== Real-Time VAD + Whisper + Llama Translation (Threaded) ===")

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    audio_queue = queue.Queue(maxsize=5)
    exit_event = threading.Event()
    capture_thread = threading.Thread(
        target=vad_capture_thread, args=(audio_queue, exit_event), daemon=True
    )
    capture_thread.start()

    try:
        while True:
            # 1) Get an utterance from the queue
            try:
                audio_data = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # 2) Transcribe
            recognized_ko = whisper_transcribe_chunk(audio_data)
            if not recognized_ko:
                continue
            print(f"(Korean) STT: {recognized_ko}")

            # 3) Append user message
            conversation.append({"role": "user", "content": recognized_ko})

            # 4) Translate
            llama_reply = llama_translate(conversation).strip()
            if not llama_reply:
                # If llama call failed or empty
                error_json = '{"translatedText": "Error calling LLM"}'
                conversation.append({"role": "assistant", "content": error_json})
                print("Empty or error from llama.")
            else:
                # Try to parse JSON
                try:
                    j = json.loads(llama_reply)
                    translation = j.get("translatedText", "")
                except json.JSONDecodeError:
                    print("LLM did not return valid JSON. Full reply:\n", llama_reply)
                    translation = "Error: invalid JSON"

                conversation.append({"role": "assistant", "content": llama_reply})
                print(f"(English) Translation:\n{translation}")

            print("-" * 60)

            # 5) Trim conversation if it is too large (remove pairs)
            while len(conversation) > MAX_CONVERSATION_MESSAGES:
                # remove user message at index 1
                conversation.pop(1)
                # now remove assistant (which was at index 2, but after pop(1), it's index 1)
                conversation.pop(1)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        exit_event.set()
        capture_thread.join()
        print("All done.")


if __name__ == "__main__":
    main()
