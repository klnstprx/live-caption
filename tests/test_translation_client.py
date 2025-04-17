import pytest

from translation_client import TranslationClient


class DummyChoice:
    def __init__(self, content):
        # Emulate openai.ChatCompletionChoice.message.content
        self.message = self
        self.content = content


class DummyCompletions:
    def __init__(self, choices):
        self.choices = choices


class DummyChatClient:
    def __init__(self, response_content):
        self.response_content = response_content
        self.chat = self
        self.completions = None

    def create(self, model, messages, temperature, response_format):
        # Return object with choices list
        choice = DummyChoice(self.response_content)
        resp = type('Resp', (), {})()
        resp.choices = [choice]
        return resp


@pytest.fixture(autouse=True)
def patch_client(monkeypatch):
    # Intercept the OpenAI() construction inside TranslationClient
    monkeypatch.setattr(TranslationClient, '__init__', lambda self, model_name, base_url='', timeout=60: None)
    return


def test_translate_returns_content(monkeypatch):
    # Instantiate and monkeypatch client.chat.completions
    tc = TranslationClient('model-x', 'http://base', timeout=30)
    dummy = DummyChatClient('{"translatedText": "foo bar"}')
    # Attach fake client with chat.completions.create
    tc.client = dummy
    result = TranslationClient.translate(tc, [{'role': 'system', 'content': 'hi'}])
    assert result == '{"translatedText": "foo bar"}'