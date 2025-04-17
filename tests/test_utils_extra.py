import json
import logging

from utils import (
    print_and_log,
    JSONFormatter,
    ColorFormatter,
    trim_conversation_history,
)


def test_print_and_log_basic(tmp_path, capsys):
    fpath = tmp_path / "out.txt"
    fh = open(fpath, "w", encoding="utf-8")
    msg = "Hello\nWorld"
    print_and_log(msg, fh)
    # Verify stdout
    captured = capsys.readouterr().out
    assert "Hello" in captured
    assert "    World" in captured
    # Verify file contents
    fh.close()
    content = open(fpath, encoding="utf-8").read()
    assert "Hello" in content
    assert "    World" in content


def test_print_and_log_colored_prefix(tmp_path, capsys):
    fpath = tmp_path / "out2.txt"
    fh = open(fpath, "w", encoding="utf-8")
    # Use a Korean prefix
    msg = "(Korean): 안녕\n테스트"
    print_and_log(msg, fh)
    out = capsys.readouterr().out
    # Should color the prefix and indent the rest
    assert "(Korean):" in out
    assert "    테스트" in out
    fh.close()


def test_json_formatter_outputs_valid_json():
    formatter = JSONFormatter()
    record = logging.LogRecord("name", logging.INFO, __file__, 10, "msg", None, None)
    out = formatter.format(record)
    data = json.loads(out)
    assert data["message"] == "msg"
    assert data["level"] == "INFO"


def test_color_formatter_includes_ansi_codes():
    # Simple format showing level and message
    fmt = ColorFormatter("[%(levelname)s] %(message)s")
    record = logging.LogRecord("name", logging.ERROR, __file__, 20, "oops", None, None)
    out = fmt.format(record)
    # Expect ANSI escape codes for color
    assert "\x1b" in out
    assert "oops" in out


def test_trim_conversation_history():
    # system + 4 messages = 5 total
    conv = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    trimmed = trim_conversation_history(conv, max_len=3)
    # Should keep system and last two
    assert len(trimmed) == 3
    assert trimmed[0]["role"] == "system"
    assert trimmed[1]["content"] == "u2"
    assert trimmed[2]["content"] == "a2"

