import webrtcvad
import pyaudio
import time
import queue
import logging

logger = logging.getLogger("vad-whisper-llama")


def vad_capture_thread(audio_queue, exit_event, args):
    """
    Voice activity detection capture loop. Listens on microphone, chunks speech, and enqueues audio frames.
    """
    vad = webrtcvad.Vad(args.vad_mode)
    chunk_size = int(args.rate * args.frame_duration_ms / 1000)
    pa = pyaudio.PyAudio()
    stream = None
    try:
        stream = pa.open(
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

    speech_frames = []
    num_silent = 0
    in_speech = False

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
                    speech_frames = []
                    in_speech = False
                    num_silent = 0
    finally:
        if in_speech and speech_frames:
            try:
                audio_queue.put(b"".join(speech_frames), timeout=0.5)
            except queue.Full:
                logger.warning("Audio queue full on thread exit.")
        if stream:
            stream.stop_stream()
            stream.close()
        pa.terminate()
        logger.info("Microphone capture thread exited.")

