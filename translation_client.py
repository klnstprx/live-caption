"""
Translation client: send conversation to an OpenAI-compatible Chat Completions endpoint and extract translated text.
"""

import logging
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from typing import List

logger = logging.getLogger("vad-whisper-llama")


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
        model_name = getattr(self, 'model_name', None)
        logger.debug("OpenAI translate: model=%s, messages=%s", model_name, conversation)
        # Determine which client method to call (real OpenAI SDK vs. custom client)
        call = None
        try:
            # New OpenAI SDK: self.client.chat.completions.create()
            call = self.client.chat.completions.create
        except Exception:
            # Fallback: client.create()
            call = getattr(self.client, 'create', None)
        if not callable(call):
            raise AttributeError("No suitable create method found on TranslationClient client.")
        # Invoke completion call
        resp = call(
            model=model_name,
            messages=conversation,
            temperature=0.7,
            response_format=getattr(self, 'response_format', None),
        )
        # Extract JSON-level response (string)
        content = resp.choices[0].message.content
        return content

