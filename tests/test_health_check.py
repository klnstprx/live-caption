import pytest

from utils import health_check_endpoint


class DummyResp:
    def __init__(self, status_code):
        self.status_code = status_code


def test_health_check_success(monkeypatch, capsys):
    # Monkey-patch requests.head to return 200
    import requests

    monkeypatch.setattr(requests, "head", lambda url, timeout: DummyResp(200))
    # Should not exit
    health_check_endpoint("TestService", "http://fake", timeout=1)
    # Check that info was logged to stderr
    err = capsys.readouterr().err
    assert "reachable" in err


def test_health_check_server_error(monkeypatch):
    import requests

    monkeypatch.setattr(requests, "head", lambda url, timeout: DummyResp(500))
    with pytest.raises(SystemExit):
        health_check_endpoint("TestService", "http://fake", timeout=1)


def test_health_check_exception(monkeypatch):
    import requests
    from requests.exceptions import RequestException

    # Simulate network error
    def raise_exc(url, timeout):
        raise RequestException("fail")

    monkeypatch.setattr(requests, "head", raise_exc)
    with pytest.raises(SystemExit):
        health_check_endpoint("TestService", "http://fake", timeout=1)

