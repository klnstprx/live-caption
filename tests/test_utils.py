import pytest
import io
from utils import pad_audio, get_audio_bytes_per_second, trim_conversation_history

def test_get_audio_bytes_per_second():
    # 16kHz, mono, 2-byte samples => 16000 * 1 * 2 = 32000 bytes/s
    assert get_audio_bytes_per_second(16000, 1, 2) == 32000
    # 8kHz, stereo, 2-byte samples => 8000 * 2 * 2 = 32000 bytes/s
    assert get_audio_bytes_per_second(8000, 2, 2) == 32000

def test_pad_audio_no_padding():
    data = b"\x01\x02\x03\x04"
    # min_bytes equal to current length, no padding
    result = pad_audio(data, min_bytes=4, sample_width=2)
    assert result == data

def test_pad_audio_with_padding():
    data = b"\x01\x02\x03\x04"  # 4 bytes
    # require at least 6 bytes; padding to nearest multiple of sample_width (2)
    padded = pad_audio(data, min_bytes=6, sample_width=2)
    # padded length should be >=6 and multiple of sample_width (2)
    assert len(padded) >= 6 and len(padded) % 2 == 0
    assert padded.startswith(data)
    # zeros appended
    assert padded[4:] == b"\x00" * (len(padded) - 4)

def test_trim_conversation_history():
    # Prepare conversation with system + 2 user/assistant pairs
    conv = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    # Trim to max length 3 (system + one user/assistant)
    trimmed = trim_conversation_history(conv.copy(), max_len=3)
    assert len(trimmed) == 3
    # Check system prompt remains
    assert trimmed[0]["role"] == "system"
    # Last user/assistant preserved
    assert trimmed[1]["role"] == "user" and trimmed[1]["content"] == "u2"
    assert trimmed[2]["role"] == "assistant" and trimmed[2]["content"] == "a2"