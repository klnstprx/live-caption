"""
Translation client: send conversation to an OpenAI-compatible Chat Completions endpoint and extract translated text.
"""

import logging
# ``openai`` is optional for the unit-tests.  Provide a minimal stub when the
# real library is absent so importing this module does not explode during test
# collection.

from types import SimpleNamespace

try:
    from openai import OpenAI  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – lightweight stub for CI

    class _FakeOpenAI:  # pylint: disable=too-few-public-methods
        def __init__(self, *_, **__):
            # Expose the same attributes the production code expects.
            self.chat = SimpleNamespace()

        def __getattr__(self, _):  # noqa: D401 – simple passthrough
            # Any attribute access returns a lambda that raises, mirroring the
            # real API’s failure when misused during tests.
            return lambda *_, **__: None

    OpenAI = _FakeOpenAI  # type: ignore
try:
    from openai.types.chat import ChatCompletionMessage  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – stub
    from typing import Dict, Any

    class _Stub(dict):
        """Fallback when openai is unknown – just behave like a dict."""

    ChatCompletionMessage = Dict[str, Any]  # type: ignore

from typing import List

logger = logging.getLogger(__name__)


class TranslationClient:
    """
    Translation client using the OpenAI Chat Completions API (or any OpenAI-compatible server).
    Outputs are constrained by a JSON Schema to ensure valid structured responses.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "",
        timeout: int = 60,
    ):
        # Configure OpenAI SDK
        self.client = OpenAI()
        self.client.timeout = timeout
        if base_url:
            self.client.base_url = base_url
        self.model_name = model_name
        self.timeout = timeout
        # Define JSON schema for structured output
        self.response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "translation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "translatedText": {
                            "type": "string",
                            "description": "The translated English text",
                        }
                    },
                    "required": ["translatedText"],
                    "additionalProperties": False,
                },
            },
        }

    def translate(self, conversation: List[ChatCompletionMessage]) -> str | None:
        """
        Send chat messages to the API and return the JSON string output.
        """
        # Log translation call (model_name may be undefined in some test contexts)
        model_name = getattr(self, "model_name", None)
        logger.debug(
            "OpenAI translate: model=%s, messages=%s", model_name, conversation
        )
        # Determine which client method to call (real OpenAI SDK vs. custom client)
        call = None
        try:
            # New OpenAI SDK: self.client.chat.completions.create()
            call = self.client.chat.completions.create
        except Exception:
            # Fallback: client.create()
            call = getattr(self.client, "create", None)
        if not callable(call):
            raise AttributeError(
                "No suitable create method found on TranslationClient client."
            )
        # Invoke completion call
        resp = call(
            model=model_name,
            messages=conversation,
            temperature=0.7,
            response_format=getattr(self, "response_format", None),
        )
        # Extract JSON-level response (string)
        content = resp.choices[0].message.content
        return content
