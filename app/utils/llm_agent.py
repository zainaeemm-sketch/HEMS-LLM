# app/utils/llm_agent.py
"""
LLM client for the Assistant page. Works with any OpenAI-compatible provider
(OpenAI, VectorEngine, OpenRouter, Gemini's OpenAI endpoint, resellers ...).

Configuration (environment variables or Streamlit Secrets, never in code):
    LLM_API_KEY  or OPENAI_API_KEY  or VECTORENGINE_API_KEY   API key
    LLM_BASE_URL or OPENAI_BASE_URL                           provider URL, e.g. https://lanyiapi.com/v1
    LLM_MODEL                                                 model name
    LLM_API_STYLE                                             auto (default) | responses | chat

With LLM_API_STYLE=auto the Responses API is tried first and, if the provider
doesn't support it, Chat Completions is used and remembered for this session.
"""
from __future__ import annotations

import base64
import os
from typing import List, Dict, Optional, Tuple

import httpx
import openai
from openai import OpenAI

DEFAULT_MODEL = "gpt-5-mini-2025-08-07"
VECTORENGINE_URL = "https://api.vectorengine.ai/v1"

# (base_url, model) -> "responses" | "chat", learned at runtime in auto mode.
_STYLE_CACHE: Dict[Tuple[str, str], str] = {}


class LLMError(RuntimeError):
    pass


class LLMImageError(LLMError):
    """The request carried images and the model or provider rejected it."""


class LLMAuthError(LLMError):
    """The provider rejected the API key."""


def _cfg(*names: str) -> Optional[str]:
    """First non-empty value among env vars, then Streamlit Secrets."""
    for n in names:
        v = os.environ.get(n)
        if v and v.strip():
            return v.strip()
    try:
        import streamlit as st  # optional: only when running in Streamlit
        for n in names:
            v = st.secrets.get(n)
            if v and str(v).strip():
                return str(v).strip()
    except Exception:
        pass
    return None


def llm_settings() -> Dict[str, Optional[str]]:
    """Resolved provider settings (the key itself is never returned)."""
    key = _cfg("LLM_API_KEY", "OPENAI_API_KEY")
    base_url = _cfg("LLM_BASE_URL", "OPENAI_BASE_URL")
    if not key:
        key = _cfg("VECTORENGINE_API_KEY")
        if key and not base_url:
            base_url = VECTORENGINE_URL  # old setup: VectorEngine key only
    return {
        "has_key": bool(key),
        "base_url": base_url or "https://api.openai.com/v1",
        "model": _cfg("LLM_MODEL") or DEFAULT_MODEL,
        "api_style": (_cfg("LLM_API_STYLE") or "auto").lower(),
        "_key": key,
    }


def _make_client(api_key: str, base_url: str) -> OpenAI:
    timeout = httpx.Timeout(connect=10.0, read=90.0, write=20.0, pool=10.0)
    return OpenAI(api_key=api_key, base_url=base_url, http_client=httpx.Client(timeout=timeout))


def _messages_to_input(messages: List[Dict[str, str]]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts)


def _data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _build_input(messages: List[Dict[str, str]], images: Optional[List[Tuple[bytes, str]]]):
    """Responses API input: plain text, or one user turn with text + images."""
    text = _messages_to_input(messages)
    if not images:
        return text
    content = [{"type": "input_text", "text": text}]
    for data, mime in images:
        content.append({"type": "input_image", "image_url": _data_url(data, mime)})
    return [{"role": "user", "content": content}]


def _build_chat_messages(messages: List[Dict[str, str]], images: Optional[List[Tuple[bytes, str]]]):
    """Chat Completions messages; images are attached to the last user turn."""
    out = []
    for m in messages:
        content = (m.get("content") or "").strip()
        role = m.get("role", "user")
        if content and role in ("system", "user", "assistant"):
            out.append({"role": role, "content": content})
    if images:
        last = next((i for i in range(len(out) - 1, -1, -1) if out[i]["role"] == "user"), None)
        if last is None:
            out.append({"role": "user", "content": ""})
            last = len(out) - 1
        parts = [{"type": "text", "text": out[last]["content"]}]
        parts += [{"type": "image_url", "image_url": {"url": _data_url(d, m)}} for d, m in images]
        out[last] = {"role": "user", "content": parts}
    return out


def _get_output_text(resp) -> str:
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    try:
        out = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) == "output_text" and getattr(c, "text", ""):
                    out.append(c.text)
        if out:
            return "\n".join(out).strip()
    except Exception:
        pass
    return ""


def _chat_text(resp) -> str:
    try:
        content = resp.choices[0].message.content
    except Exception:
        return ""
    if isinstance(content, list):  # some providers return content parts
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return (content or "").strip()


def _call_responses(client, model, messages, images, max_tokens, reasoning_effort) -> str:
    inp = _build_input(messages, images)
    try:
        resp = client.responses.create(
            model=model, input=inp, max_output_tokens=int(max_tokens),
            reasoning={"effort": reasoning_effort},
        )
    except (TypeError, openai.BadRequestError):
        # Model or provider doesn't accept the reasoning option: retry without it.
        resp = client.responses.create(model=model, input=inp, max_output_tokens=int(max_tokens))
    return _get_output_text(resp)


def _call_chat(client, model, messages, images, max_tokens) -> str:
    msgs = _build_chat_messages(messages, images)
    try:
        # Newer OpenAI models take max_completion_tokens ...
        resp = client.chat.completions.create(
            model=model, messages=msgs, max_completion_tokens=int(max_tokens))
    except openai.BadRequestError as e:
        if "max_completion_tokens" not in str(e) and "unsupported" not in str(e).lower():
            raise
        # ... many other providers only know max_tokens.
        resp = client.chat.completions.create(model=model, messages=msgs, max_tokens=int(max_tokens))
    return _chat_text(resp)


def chat_with_llm(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    max_output_tokens: int = 600,
    reasoning_effort: str = "minimal",
    images: Optional[List[Tuple[bytes, str]]] = None,
) -> str:
    """Send the conversation (and optional images for the latest question) and
    return the reply text. Raises LLMAuthError, LLMImageError or LLMError."""
    cfg = llm_settings()
    key = api_key or cfg["_key"]
    if not key:
        raise LLMError("Missing API key: set LLM_API_KEY (or OPENAI_API_KEY) in your secrets.")
    base_url, model = cfg["base_url"], model or cfg["model"]
    client = _make_client(key, base_url)

    style = cfg["api_style"]
    if style == "auto":
        order = [_STYLE_CACHE[(base_url, model)]] if (base_url, model) in _STYLE_CACHE else ["responses", "chat"]
    else:
        order = ["chat" if style == "chat" else "responses"]

    last_err: Optional[Exception] = None
    for attempt in order:
        try:
            if attempt == "responses":
                text = _call_responses(client, model, messages, images, max_output_tokens, reasoning_effort)
            else:
                text = _call_chat(client, model, messages, images, max_output_tokens)
            if not text:
                raise LLMError(f"The provider returned an empty reply ({attempt} API).")
            _STYLE_CACHE[(base_url, model)] = attempt
            return text
        except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
            raise LLMAuthError(
                f"The AI provider at {base_url} rejected the API key "
                f"({type(e).__name__}): {e}"
            ) from e
        except openai.RateLimitError as e:
            raise LLMError(f"The AI provider is rate-limiting or out of credit: {e}") from e
        except openai.APIConnectionError as e:
            raise LLMError(f"Couldn't reach the AI provider at {base_url}: {e}") from e
        except openai.NotFoundError as e:
            # Endpoint or model not found: try the other API style if any.
            last_err = e
        except openai.BadRequestError as e:
            last_err = e
        except LLMError as e:
            last_err = e

    detail = f"{type(last_err).__name__}: {last_err}" if last_err else "unknown error"
    if images and isinstance(last_err, openai.BadRequestError):
        raise LLMImageError(f"The AI provider rejected the request with the image: {detail}")
    raise LLMError(f"AI provider call failed ({base_url}, model {model}): {detail}")


# Backward-compatible name used by main.py.
chat_with_vectorengine = chat_with_llm
