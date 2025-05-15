"""Audio capture with optional VAD support.

This module attempts to import the heavyweight ``webrtcvad`` and ``pyaudio``
libraries at runtime.  When they are not available (e.g. in a CI environment
that only runs unit–tests) we fall back to very small stubs so the import does
not crash — the parts of this file used by the test-suite (``list_audio_devices``)
do not actually depend on those libraries working.
"""

from types import SimpleNamespace

try:
    import webrtcvad  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – executed only in test env
    # Create a minimal stub that mimics the real interface well enough for the
    # unit-tests.  We default to always returning "speech" so that any call site
    # relying on the result does not break during tests.
    class _FakeVad:  # pylint: disable=too-few-public-methods
        def __init__(self, *_, **__):
            pass

        def is_speech(self, *_):  # noqa: D401 – simple stub
            return True

    webrtcvad = SimpleNamespace(Vad=_FakeVad)  # type: ignore

try:
    import pyaudio  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    # A stub of the bits we access inside this module.  Only what the tests need
    # is provided.
    class _FakeStream:  # pylint: disable=too-few-public-methods
        def read(self, *_args, **_kwargs):
            return b""

        def stop_stream(self):
            pass

        def close(self):
            pass

    class _FakePyAudio:  # pylint: disable=too-few-public-methods
        def open(self, *_, **__):  # noqa: D401 – stub
            return _FakeStream()

        def get_device_count(self):  # noqa: D401 – stub
            return 0

        def get_device_info_by_index(self, _):  # noqa: D401 – stub
            return {}

        def terminate(self):  # noqa: D401 – stub
            pass

    pyaudio = SimpleNamespace(  # type: ignore
        PyAudio=_FakePyAudio,
        paInt16=8,  # constant value is irrelevant for tests
    )
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
    
def list_audio_devices():
    """List available audio input devices."""
    pa = pyaudio.PyAudio()
    print("Available audio input devices:")
    for i in range(pa.get_device_count()):
        try:
            info = pa.get_device_info_by_index(i)
        except Exception:
            continue
        if info.get('maxInputChannels', 0) > 0:
            channels = info.get('maxInputChannels')
            name = info.get('name')
            print(f"{i}: {name} ({channels} input channel{'s' if channels != 1 else ''})")
    pa.terminate()

