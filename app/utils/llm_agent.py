# app/utils/llm_agent.py
from __future__ import annotations

import base64
import os
from typing import List, Dict, Optional, Tuple

import httpx
from openai import OpenAI


class LLMError(RuntimeError):
    pass


class LLMImageError(LLMError):
    """The request carried images and the model or gateway rejected it."""


def _make_client(api_key: str) -> OpenAI:
    timeout = httpx.Timeout(connect=10.0, read=60.0, write=20.0, pool=10.0)
    http_client = httpx.Client(timeout=timeout)
    return OpenAI(
        api_key=api_key,
        base_url="https://api.vectorengine.ai/v1",
        http_client=http_client,
    )


def _messages_to_input(messages: List[Dict[str, str]]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts)


def _get_output_text(resp) -> str:
    # Best path (new SDKs)
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    # Fallback scan (older/newer variants)
    try:
        out = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) == "output_text":
                    t = getattr(c, "text", "")
                    if t:
                        out.append(t)
        if out:
            return "\n".join(out).strip()
    except Exception:
        pass

    return ""


def _build_input(messages: List[Dict[str, str]], images: Optional[List[Tuple[bytes, str]]]):
    """Plain text input as before; with images, one user turn holding the
    conversation text followed by the images (Responses API content parts)."""
    text = _messages_to_input(messages)
    if not images:
        return text
    content = [{"type": "input_text", "text": text}]
    for data, mime in images:
        b64 = base64.b64encode(data).decode("ascii")
        content.append({"type": "input_image", "image_url": f"data:{mime};base64,{b64}"})
    return [{"role": "user", "content": content}]


def chat_with_vectorengine(
    messages: List[Dict[str, str]],
    model: str = "gpt-5-mini-2025-08-07",
    api_key: Optional[str] = None,
    max_output_tokens: int = 600,
    reasoning_effort: str = "minimal",
    images: Optional[List[Tuple[bytes, str]]] = None,
) -> str:
    """
    VectorEngine + OpenAI SDK compatible Responses API call.

    IMPORTANT:
    - Some SDK versions do NOT support `verbosity=...`
    - Some gateways may or may not support `reasoning=...`
    This function tries with reasoning first, then retries without it.
    """
    key = api_key or os.environ.get("VECTORENGINE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise LLMError("Missing VECTORENGINE_API_KEY (or OPENAI_API_KEY) in .env")

    client = _make_client(key)
    inp = _build_input(messages, images)
    # Images attached to the latest question are sent with this call only.
    err_cls = LLMImageError if images else LLMError

    # Attempt 1: with reasoning (if supported by your SDK/gateway)
    try:
        resp = client.responses.create(
            model=model,
            input=inp,
            max_output_tokens=int(max_output_tokens),
            reasoning={"effort": reasoning_effort},
        )
        text = _get_output_text(resp)
        if text:
            return text
    except TypeError:
        # reasoning not supported OR signature mismatch -> retry without reasoning
        pass
    except Exception as e:
        raise err_cls(f"VectorEngine Responses call failed: {type(e).__name__}: {e}")

    # Attempt 2: without reasoning
    try:
        resp = client.responses.create(
            model=model,
            input=inp,
            max_output_tokens=int(max_output_tokens),
        )
    except Exception as e:
        raise err_cls(f"VectorEngine Responses call failed (no reasoning): {type(e).__name__}: {e}")

    text = _get_output_text(resp)
    if not text:
        # If still empty, surface debugging info
        try:
            dump = resp.model_dump()
        except Exception:
            dump = str(resp)
        raise LLMError(f"Empty response text from VectorEngine Responses API. Dump:\n{dump}")

    return text
