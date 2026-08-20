"""
Provider-agnostic chat client.

Not `src/vlms`: that flattens every turn into one user message and re-sends it
each iteration. Real multi-turn history costs far fewer tokens, since the
provider caches the stable prefix.

Internal message format:

    {"role": "user" | "assistant",
     "content": [{"type": "text", "text": ...}, {"type": "image", "path": ...}]}

Add a provider with a `_Client` subclass registered in `FAMILIES`.
"""

import base64
import json
import os
import os.path as osp
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


# ----------------------------------------------------------------------------
# message construction helpers
# ----------------------------------------------------------------------------

def text_part(value: str) -> dict:
    return {"type": "text", "text": value}


def image_part(path: str) -> dict:
    return {"type": "image", "path": path}


def user(*parts) -> dict:
    """Build a user turn from strings (text) and existing part dicts."""
    content = [text_part(p) if isinstance(p, str) else p for p in parts if p]
    return {"role": "user", "content": content}


def assistant(text_value: str) -> dict:
    return {"role": "assistant", "content": [text_part(text_value)]}


def _encode(path: str):
    ext = osp.splitext(path)[1].lower()
    media = MEDIA_TYPES.get(ext, "image/png")
    with open(path, "rb") as fh:
        return media, base64.b64encode(fh.read()).decode("ascii")


@dataclass
class LLMResponse:
    text: str = ""
    thinking: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Any = None
    error: Optional[str] = None


# ----------------------------------------------------------------------------
# base
# ----------------------------------------------------------------------------

class _Client:
    """One provider. `config` is a model entry from the JSON config."""

    def __init__(self, config: dict):
        self.config = config
        self.model = config["model"]

    def chat(self, system: str, messages: List[dict]) -> LLMResponse:
        raise NotImplementedError

    # --- shared retry wrapper ------------------------------------------------
    def chat_with_retry(self, system: str, messages: List[dict],
                        attempts: int = 3, backoff: float = 4.0) -> LLMResponse:
        last = None
        for attempt in range(attempts):
            try:
                return self.chat(system, messages)
            except Exception as exc:  # noqa: BLE001 - rate limits, 5xx, transport
                last = exc
                message = str(exc)
                retriable = any(
                    token in message.lower()
                    for token in ("rate", "overload", "timeout", "429", "500",
                                  "502", "503", "504", "connection")
                )
                if attempt == attempts - 1 or not retriable:
                    break
                sleep_for = backoff * (2 ** attempt)
                print(f"    [llm] {type(exc).__name__}: {message[:160]} "
                      f"- retrying in {sleep_for:.0f}s")
                time.sleep(sleep_for)
        return LLMResponse(error=f"{type(last).__name__}: {last}")


# ----------------------------------------------------------------------------
# Anthropic
# ----------------------------------------------------------------------------

class AnthropicClient(_Client):
    def __init__(self, config):
        super().__init__(config)
        import anthropic

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        self.client = anthropic.Anthropic(api_key=key)

    def _content(self, part):
        if part["type"] == "text":
            return {"type": "text", "text": part["text"]}
        media, data = _encode(part["path"])
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media, "data": data},
        }

    def chat(self, system, messages):
        payload = [
            {"role": m["role"], "content": [self._content(p) for p in m["content"]]}
            for m in messages
        ]

        # Cache the system prompt and the (stable) opening turn so every later
        # iteration only pays for the new tail.
        system_blocks = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
        if payload and payload[0]["content"]:
            payload[0]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        kwargs = {
            "model": self.model,
            "max_tokens": self.config.get("max_tokens", 16000),
            "system": system_blocks,
            "messages": payload,
        }
        thinking = self.config.get("thinking")
        if thinking and thinking.get("type") == "enabled":
            kwargs["thinking"] = thinking
            # Anthropic requires max_tokens > thinking budget.
            kwargs["max_tokens"] = max(
                kwargs["max_tokens"], thinking.get("budget_tokens", 0) + 4000
            )
        else:
            kwargs["temperature"] = self.config.get("temperature", 0.0)

        response = self.client.messages.create(**kwargs)

        text = "\n".join(b.text for b in response.content if b.type == "text")
        thought = "\n".join(b.thinking for b in response.content if b.type == "thinking")
        usage = {
            "input_tokens": getattr(response.usage, "input_tokens", 0),
            "output_tokens": getattr(response.usage, "output_tokens", 0),
            "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            "cache_write_tokens": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        }
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        return LLMResponse(text=text, thinking=thought, usage=usage, raw=response)


# ----------------------------------------------------------------------------
# OpenAI (Responses API)
# ----------------------------------------------------------------------------

class OpenAIClient(_Client):
    def __init__(self, config):
        super().__init__(config)
        import openai

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.client = openai.OpenAI(api_key=key)

    def _content(self, part, role):
        if part["type"] == "text":
            kind = "output_text" if role == "assistant" else "input_text"
            return {"type": kind, "text": part["text"]}
        media, data = _encode(part["path"])
        return {"type": "input_image", "image_url": f"data:{media};base64,{data}"}

    def chat(self, system, messages):
        payload = [{"role": "system", "content": [{"type": "input_text", "text": system}]}]
        for m in messages:
            payload.append({
                "role": m["role"],
                "content": [self._content(p, m["role"]) for p in m["content"]],
            })

        kwargs = {"model": self.model, "input": payload}
        if "reasoning_level" in self.config:
            kwargs["reasoning"] = {
                "effort": self.config["reasoning_level"],
                "summary": "auto",
            }
        if "max_output_tokens" in self.config:
            kwargs["max_output_tokens"] = self.config["max_output_tokens"]

        response = self.client.responses.create(**kwargs)

        summaries = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "reasoning":
                for s in (getattr(item, "summary", None) or []):
                    summaries.append(getattr(s, "text", ""))

        usage_obj = getattr(response, "usage", None)
        usage = {
            "input_tokens": getattr(usage_obj, "input_tokens", 0) if usage_obj else 0,
            "output_tokens": getattr(usage_obj, "output_tokens", 0) if usage_obj else 0,
            "total_tokens": getattr(usage_obj, "total_tokens", 0) if usage_obj else 0,
        }
        return LLMResponse(
            text=(response.output_text or "").strip(),
            thinking="\n".join(summaries),
            usage=usage,
            raw=response,
        )


# ----------------------------------------------------------------------------
# Gemini
# ----------------------------------------------------------------------------

class GeminiClient(_Client):
    def __init__(self, config):
        super().__init__(config)
        from google import genai

        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
        self.client = genai.Client(api_key=key)
        self._genai = genai

    def chat(self, system, messages):
        from google.genai import types

        contents = []
        for m in messages:
            parts = []
            for part in m["content"]:
                if part["type"] == "text":
                    parts.append(types.Part.from_text(text=part["text"]))
                else:
                    media, _ = _encode(part["path"])
                    with open(part["path"], "rb") as fh:
                        parts.append(types.Part.from_bytes(data=fh.read(), mime_type=media))
            contents.append(types.Content(
                role="model" if m["role"] == "assistant" else "user", parts=parts
            ))

        cfg: Dict[str, Any] = {"system_instruction": system}
        if "thinking_level" in self.config:
            budget = {"low": 2048, "medium": 8192, "high": 24576}.get(
                self.config["thinking_level"], 8192
            )
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)
        if "temperature" in self.config:
            cfg["temperature"] = self.config["temperature"]

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**cfg),
        )

        meta = getattr(response, "usage_metadata", None)
        usage = {
            "input_tokens": getattr(meta, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(meta, "candidates_token_count", 0) or 0,
            "thinking_tokens": getattr(meta, "thoughts_token_count", 0) or 0,
            "total_tokens": getattr(meta, "total_token_count", 0) or 0,
        }
        return LLMResponse(text=(response.text or "").strip(), usage=usage, raw=response)


# ----------------------------------------------------------------------------
# OpenRouter (OpenAI-compatible chat completions)
# ----------------------------------------------------------------------------

class OpenRouterClient(_Client):
    def __init__(self, config):
        super().__init__(config)
        import openai

        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY is not set.")
        self.client = openai.OpenAI(
            api_key=key, base_url="https://openrouter.ai/api/v1"
        )

    def _content(self, part):
        if part["type"] == "text":
            return {"type": "text", "text": part["text"]}
        media, data = _encode(part["path"])
        return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}}

    def chat(self, system, messages):
        payload = [{"role": "system", "content": system}]
        for m in messages:
            payload.append({
                "role": m["role"],
                "content": [self._content(p) for p in m["content"]],
            })

        kwargs = {"model": self.model, "messages": payload}
        if "reasoning" in self.config:
            kwargs["extra_body"] = {"reasoning": self.config["reasoning"]}

        response = self.client.chat.completions.create(**kwargs)
        usage_obj = getattr(response, "usage", None)
        usage = {
            "input_tokens": getattr(usage_obj, "prompt_tokens", 0) if usage_obj else 0,
            "output_tokens": getattr(usage_obj, "completion_tokens", 0) if usage_obj else 0,
            "total_tokens": getattr(usage_obj, "total_tokens", 0) if usage_obj else 0,
        }
        return LLMResponse(
            text=(response.choices[0].message.content or "").strip(),
            usage=usage,
            raw=response,
        )


FAMILIES = {
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "gemini": GeminiClient,
    "openrouter": OpenRouterClient,
}


def build_client(model_config: dict) -> _Client:
    family = model_config.get("family")
    if family not in FAMILIES:
        raise ValueError(
            f"Unknown model family {family!r}. Known: {', '.join(FAMILIES)}"
        )
    return FAMILIES[family](model_config)


# ----------------------------------------------------------------------------
# JSON extraction - models wrap, fence, and chatter around their JSON
# ----------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Optional[dict]:
    """
    Best-effort parse of a JSON object out of a model response.

    Tries, in order: the whole string, any fenced block, the outermost balanced
    braces, and finally `json_repair` if it is installed.
    """
    if not text:
        return None

    candidates: List[str] = [text.strip()]
    candidates.extend(match.strip() for match in _FENCE.findall(text))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue

    try:
        from json_repair import repair_json

        for candidate in candidates:
            try:
                parsed = json.loads(repair_json(candidate))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:  # noqa: BLE001
                continue
    except ImportError:
        pass

    return None
