"""
AuditClient — thin wrapper around the Anthropic Messages API.

Loads ANTHROPIC_API_KEY from the project root .env file automatically.
"""

import json
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# Project root is two levels up from this file (processing_scripts/llm_client/)
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_FILE)


class AuditClient:
    """
    Sends filled accessibility-audit prompts to the Claude API and returns
    parsed JSON responses.

    Parameters
    ----------
    model : str
        Claude model ID to use (default: claude-sonnet-4-6).
    temperature : float
        Sampling temperature (default: 0.1 for consistent structured output).
    max_tokens : int
        Maximum tokens to generate per response (default: 8192).
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.1,
        max_tokens: int = 8192,
    ):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                f"ANTHROPIC_API_KEY not found. Expected in {_ENV_FILE}"
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def call(self, prompt_text: str, payload: dict) -> tuple[dict, dict]:
        """
        Fill `{payload}` in *prompt_text* with JSON-serialised *payload*,
        send to the API, and return ``(parsed_response, usage)``.

        The response is expected to be a JSON object (possibly wrapped in a
        markdown code fence, which is stripped automatically).

        Returns
        -------
        response_json : dict
            Parsed JSON from the model's reply.
        usage : dict
            ``{"input_tokens": int, "output_tokens": int}``
        """
        filled = prompt_text.replace("{payload}", json.dumps(payload, separators=(",", ":")))

        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": filled}],
        )

        usage = {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        }

        raw = message.content[0].text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0].strip()

        return json.loads(raw), usage
