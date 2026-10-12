"""Model slot.

Nothing in the repository stores a key. A local .env that is not committed
supplies LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL. DEEPSEEK_API_KEY is
accepted as another name for the key. While any of the three is empty,
planning stays on the deterministic search. Dispatch sentences stay with
the in-repo interpreter either way.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
from pydantic import BaseModel


class ModelCallError(Exception):
    """The configured model endpoint failed. The message never includes the key."""


class LlmSettings(BaseModel):
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    reasoning_effort: str = ""
    thinking: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    @classmethod
    def from_env(cls) -> "LlmSettings":
        _load_dotenv_file()
        return cls(
            api_key=_env("LLM_API_KEY") or _env("DEEPSEEK_API_KEY"),
            base_url=_env("LLM_BASE_URL"),
            model=_env("LLM_MODEL"),
            reasoning_effort=_env("LLM_REASONING_EFFORT"),
            thinking=_env("LLM_THINKING"),
        )


def chat_completions_url(base_url: str) -> str:
    """OpenAI-SDK style: host or host/v1, then /chat/completions."""
    root = base_url.strip().rstrip("/")
    if root.endswith("/chat/completions"):
        return root
    return root + "/chat/completions"


def complete_json(settings: LlmSettings, system: str, user: str, timeout_s: float | None = None) -> dict:
    """One JSON object from an OpenAI-compatible chat endpoint."""
    body = _body(settings, system, user, json_mode=True)
    response = _post(settings, body, timeout_s)
    if response.status_code == 400 and "response_format" in body:
        body.pop("response_format", None)
        response = _post(settings, body, timeout_s)
    if response.status_code == 400 and (settings.thinking or settings.reasoning_effort):
        plain = LlmSettings(api_key=settings.api_key, base_url=settings.base_url, model=settings.model)
        body = _body(plain, system, user, json_mode=True)
        response = _post(plain, body, timeout_s)
    if response.status_code >= 400:
        detail = response.text[:180].replace(settings.api_key, "")
        raise ModelCallError(f"模型接口返回 {response.status_code} {detail}")
    try:
        message = response.json()["choices"][0]["message"]
        content = message.get("content") or ""
        return _json_object(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError) as exc:
        raise ModelCallError("模型没有返回可用的 JSON") from exc


def _body(settings: LlmSettings, system: str, user: str, *, json_mode: bool) -> dict:
    body: dict = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    if settings.reasoning_effort:
        body["reasoning_effort"] = settings.reasoning_effort
    else:
        body["temperature"] = 0
    if settings.thinking:
        body["thinking"] = {"type": settings.thinking}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    return body


def _post(settings: LlmSettings, body: dict, timeout_s: float | None = None) -> httpx.Response:
    if timeout_s is None:
        timeout = 120.0 if settings.thinking or settings.reasoning_effort else 30.0
    else:
        timeout = timeout_s
    try:
        return httpx.post(
            chat_completions_url(settings.base_url),
            headers={"Authorization": f"Bearer {settings.api_key}"},
            json=body,
            timeout=timeout,
        )
    except httpx.TimeoutException as exc:
        raise ModelCallError("模型这一轮超时") from exc
    except httpx.HTTPError as exc:
        raise ModelCallError("模型接口连不上") from exc


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _json_object(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("expected an object")
    return parsed


def _load_dotenv_file() -> None:
    """Fill missing LLM_* variables from a local .env. Never overrides the environment."""
    path = Path(os.environ.get("LLM_DOTENV_PATH", ".env"))
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
