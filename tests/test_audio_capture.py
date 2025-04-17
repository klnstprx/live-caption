import pytest
import audio_capture


class DummyInfo:
    def __init__(self, name, max_input):
        self._name = name
        self._max = max_input

    def get(self, key, default=None):
        if key == "name":
            return self._name
        if key == "maxInputChannels":
            return self._max
        return default


class DummyPyAudio:
    def __init__(self):
        pass

    def get_device_count(self):
        return 3

    def get_device_info_by_index(self, i):
        # Only index 1 has input channels
        if i == 1:
            return {"name": "Mic1", "maxInputChannels": 2}
        return {"name": f"Dev{i}", "maxInputChannels": 0}

    def terminate(self):
        pass


@pytest.mark.usefixtures("monkeypatch")
def test_list_audio_devices(monkeypatch, capsys):
    # Monkey-patch PyAudio to our dummy
    monkeypatch.setattr(audio_capture.pyaudio, "PyAudio", lambda: DummyPyAudio())
    # Capture stdout
    audio_capture.list_audio_devices()
    captured = capsys.readouterr().out
    # Should list only device 1
    assert "1: Mic1 (2 input channels)" in captured
    # Devices 0 and 2 should be skipped (0 input channels)
    assert "0:" not in captured
    assert "2:" not in captured

