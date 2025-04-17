"""
Translation client: send conversation to Llama server and extract translated text.
"""
import json
import logging
import requests
from requests.exceptions import RequestException
from typing import List, Dict

logger = logging.getLogger("vad-whisper-llama")

class LlamaClient:
    """
    HTTP client for Llama translation server.
    """
    def __init__(
        self,
        url: str,
        model_name: str,
        session: requests.Session,
        timeout: int = 60,
    ):
        self.url = url
        self.model_name = model_name
        self.session = session
        self.timeout = timeout

    def translate(self, conversation: List[Dict]) -> str:
        """Send conversation to Llama server and extract 'translatedText'."""
        payload = {
            "model": self.model_name,
            "messages": conversation,
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "translation",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "translatedText": {"type": "string"}
                        },
                        "required": ["translatedText"],
                    },
                },
            },
        }
        headers = {"Content-Type": "application/json"}
        try:
            logger.debug(
                "Llama request URL: %s, payload: %s",
                self.url,
                json.dumps(payload, ensure_ascii=False),
            )
            resp = self.session.post(
                self.url, json=payload, headers=headers, timeout=self.timeout
            )
            logger.debug("Llama response [%d]: %s", resp.status_code, resp.text)
            resp.raise_for_status()
        except RequestException as e:
            logger.error(f"Llama request failed: {e}")
            return ""
        try:
            j = resp.json()
        except ValueError as e:
            logger.error(f"Invalid JSON from Llama: {e}. Response text: {resp.text}")
            return ""
        choices = j.get("choices", [])
        if choices and "message" in choices[0] and "content" in choices[0]["message"]:
            return choices[0]["message"]["content"]
        logger.error(f"Unexpected Llama resp structure: {j}")
        return ""