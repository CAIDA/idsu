"""OpenAI-compatible chat client.

One convention for the endpoint: OPENAI_API_KEY and OPENAI_BASE_URL, from the
environment or a .env file at the repository root (see .env.example).
"""

from __future__ import annotations

import os
import random
import sys
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from idsu.config import REPO_ROOT


def make_client(timeout: float = 600) -> OpenAI:
    load_dotenv(REPO_ROOT / ".env")
    missing = [k for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL") if not os.environ.get(k)]
    if missing:
        sys.exit(f"[ERROR] Not set: {', '.join(missing)}. Copy .env.example to .env and fill it in.")
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"], timeout=timeout)


def thinking_extra_body(model: str) -> dict[str, Any] | None:
    """vLLM-served Gemma: turn on its thinking trace, as the reported runs did."""
    if "gemma" in (model or "").lower():
        return {"chat_template_kwargs": {"enable_thinking": True}}
    return None


def chat(client: OpenAI, model: str, messages: list[dict[str, str]], *, temperature: float | None = 0,
         json_mode: bool = True, max_retries: int = 5, log=print):
    """One chat completion, retried with exponential backoff. Returns None if every attempt fails."""
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    extra = thinking_extra_body(model)
    if extra is not None:
        kwargs["extra_body"] = extra
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            log(f"Model call failed ({attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep((2 ** attempt) + random.random())
    return None
