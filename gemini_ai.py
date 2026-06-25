"""
gemini_ai.py
------------
Gemini 2.5 Flash AI module for Drug Discovery Pipeline v5.0
B.Tech Biotechnology · KL University

Provides:
    get_gemini_api_key()  → str | None
    call_gemini(system, messages) → str

Drop-in replacement for the old Claude (_call_claude) interface used in app.py.
The `messages` list follows the OpenAI-style format:
    [{"role": "user"|"assistant", "content": "..."}]

Requirements:
    pip install google-generativeai

Secrets (Streamlit Cloud or .streamlit/secrets.toml):
    GEMINI_API_KEY = "AIza..."   # https://aistudio.google.com/app/apikey
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════
# PART 1 — Imports · Configuration · API key handling
# ═══════════════════════════════════════════════════════════════════

import os
import re
import time
import logging
from typing import Optional

# Streamlit is imported lazily so this module can be used outside Streamlit too
try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False

# Google Generative AI SDK
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False

# ── Constants ────────────────────────────────────────────────────────
_MODEL_NAME   = "gemini-2.5-flash"          # model string for the API
_MAX_TOKENS   = 2048                         # max output tokens per call
_TEMPERATURE  = 0.7                          # 0 = deterministic, 1 = creative
_RETRY_LIMIT  = 3                            # max retry attempts on transient errors
_RETRY_DELAY  = 2.0                          # seconds to wait between retries

# Safety settings — relaxed for scientific / pharmacology content
_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT:       HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH:      HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
} if _HAS_GENAI else {}

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
_log = logging.getLogger(__name__)

# ── Cached model instance (module-level singleton) ───────────────────
_model: "genai.GenerativeModel | None" = None  # type: ignore[name-defined]


# ─────────────────────────────────────────────────────────────────────
def get_gemini_api_key() -> str | None:
    """
    Resolve the Gemini API key using the following priority order:
      1. st.secrets["GEMINI_API_KEY"]   (Streamlit Cloud / secrets.toml)
      2. os.environ["GEMINI_API_KEY"]   (environment variable / CI)

    Returns the key string if found, otherwise None.
    Called by app.py to decide whether to show the AI tab or an error banner.
    """
    # 1 — Streamlit secrets (preferred on Streamlit Cloud)
    if _HAS_STREAMLIT:
        try:
            key = st.secrets.get("GEMINI_API_KEY", None)
            if key:
                return str(key).strip()
        except Exception:
            pass  # secrets not available in this environment

    # 2 — Environment variable (local dev / Docker / GitHub Actions)
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key

    return None


# ─────────────────────────────────────────────────────────────────────
def _get_model() -> "genai.GenerativeModel":  # type: ignore[name-defined]
    """
    Initialise (or return cached) the GenerativeModel singleton.
    Raises RuntimeError if the SDK is not installed or no API key is found.
    """
    global _model

    if not _HAS_GENAI:
        raise RuntimeError(
            "google-generativeai is not installed. "
            "Run: pip install google-generativeai"
        )

    if _model is None:
        api_key = get_gemini_api_key()
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not found. "
                "Add it to .streamlit/secrets.toml or set the environment variable."
            )

        genai.configure(api_key=api_key)

        generation_cfg = genai.types.GenerationConfig(
            temperature=_TEMPERATURE,
            max_output_tokens=_MAX_TOKENS,
        )

        _model = genai.GenerativeModel(
            model_name=_MODEL_NAME,
            generation_config=generation_cfg,
            safety_settings=_SAFETY_SETTINGS,
        )
        _log.info("Gemini model initialised: %s", _MODEL_NAME)

    return _model


# ═══════════════════════════════════════════════════════════════════
# PART 2 — call_gemini() · Multi-turn support · Error handling · Retries
# ═══════════════════════════════════════════════════════════════════

def _build_gemini_history(
    messages: list[dict],
) -> tuple[list[dict], str]:
    """
    Convert OpenAI-style message list to Gemini's history + last_user_message format.

    Gemini's GenerativeModel.start_chat() accepts a `history` list of
    {"role": "user"|"model", "parts": ["..."]} dicts.  The final user
    message must be sent via chat.send_message(), not included in history.

    Returns:
        history       — all prior turns (role translated: "assistant" → "model")
        last_user_msg — the most recent user message text to send
    """
    if not messages:
        return [], ""

    # Work on a copy; normalise roles
    msgs = [
        {"role": "model" if m["role"] == "assistant" else "user",
         "content": str(m.get("content", ""))}
        for m in messages
    ]

    # The last message must be from the user
    last_user_msg = ""
    while msgs and msgs[-1]["role"] != "user":
        msgs.pop()          # discard any trailing assistant turns
    if msgs:
        last_user_msg = msgs.pop()["content"]

    # Build Gemini history list
    history = [
        {"role": m["role"], "parts": [m["content"]]}
        for m in msgs
    ]

    return history, last_user_msg


def call_gemini(
    system: str,
    messages: list[dict],
) -> str:
    """
    Call Gemini 2.5 Flash with a system prompt and multi-turn message history.

    Parameters
    ----------
    system   : str
        System / persona prompt (injected as the first user turn followed by a
        model acknowledgement, since Gemini doesn't have a native system role in
        all SDK versions).
    messages : list[dict]
        Conversation history in OpenAI format:
            [{"role": "user"|"assistant", "content": "..."}]
        The last entry must be the user's latest message.

    Returns
    -------
    str
        The assistant's reply, cleaned of Markdown artefacts where appropriate.
        On unrecoverable error, returns a user-friendly error string instead of
        raising an exception (keeps the Streamlit UI alive).
    """
    # ── Pre-flight checks ─────────────────────────────────────────────
    if not _HAS_GENAI:
        return (
            "⚠️ **google-generativeai not installed.**  "
            "Run `pip install google-generativeai` and restart the app."
        )

    api_key = get_gemini_api_key()
    if not api_key:
        return (
            "⚠️ **GEMINI_API_KEY not set.**  "
            "Add it to `.streamlit/secrets.toml`:\n```\nGEMINI_API_KEY = 'AIza...'\n```"
        )

    if not messages:
        return "⚠️ No messages provided."

    # ── Build prompt ──────────────────────────────────────────────────
    history, last_user_msg = _build_gemini_history(messages)

    if not last_user_msg:
        return "⚠️ Could not extract a user message to send."

    # Prepend system context as a synthetic first exchange if provided
    if system and system.strip():
        system_exchange = [
            {"role": "user",  "parts": [
                f"[System instruction — follow these guidelines for all replies]\n{system.strip()}"
            ]},
            {"role": "model", "parts": [
                "Understood. I will follow those guidelines throughout our conversation."
            ]},
        ]
        history = system_exchange + history

    # ── Retry loop ────────────────────────────────────────────────────
    last_error: Exception | None = None

    for attempt in range(1, _RETRY_LIMIT + 1):
        try:
            model = _get_model()
            chat  = model.start_chat(history=history)
            resp  = chat.send_message(last_user_msg)

            # Extract text from response
            reply = _extract_text(resp)

            # Clean up and return
            return _clean_markdown(reply)

        except Exception as exc:
            last_error = exc
            err_str = str(exc).lower()

            # Non-retryable errors — return immediately
            if any(kw in err_str for kw in ("api_key", "invalid", "permission", "quota exceeded")):
                _log.error("Non-retryable Gemini error: %s", exc)
                return _format_api_error(exc)

            # Retryable (rate limit, timeout, server error)
            if attempt < _RETRY_LIMIT:
                wait = _RETRY_DELAY * attempt
                _log.warning(
                    "Gemini call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt, _RETRY_LIMIT, exc, wait,
                )
                time.sleep(wait)
                _model = None  # force model re-initialisation on next attempt
            else:
                _log.error("Gemini call failed after %d attempts: %s", _RETRY_LIMIT, exc)

    return _format_api_error(last_error)


# ═══════════════════════════════════════════════════════════════════
# PART 3 — Helper functions · Markdown cleanup · Streamlit compatibility
# ═══════════════════════════════════════════════════════════════════

def _extract_text(response) -> str:
    """
    Safely extract the text string from a Gemini GenerateContentResponse.
    Handles blocked responses, empty candidates, and unexpected shapes.
    """
    try:
        # Preferred path: response.text property
        text = response.text
        if text:
            return text
    except (AttributeError, ValueError):
        pass

    # Fallback: walk candidates → content → parts
    try:
        for candidate in response.candidates or []:
            parts = getattr(getattr(candidate, "content", None), "parts", None) or []
            for part in parts:
                t = getattr(part, "text", None)
                if t:
                    return t
    except Exception as exc:
        _log.warning("Could not extract text from Gemini response: %s", exc)

    # Check for safety block
    try:
        finish = response.candidates[0].finish_reason
        if str(finish) in ("SAFETY", "2"):
            return (
                "⚠️ Response blocked by Gemini safety filters. "
                "Try rephrasing your question."
            )
    except Exception:
        pass

    return "⚠️ Gemini returned an empty response. Please try again."


def _clean_markdown(text: str) -> str:
    """
    Light cleanup of the AI response for Streamlit's st.markdown renderer.

    - Strips leading/trailing whitespace
    - Normalises Windows line endings
    - Removes stray zero-width characters
    - Leaves all valid Markdown intact (bold, code blocks, lists, etc.)
    """
    if not text:
        return text

    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove zero-width characters that occasionally appear in API output
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _format_api_error(exc: Exception | None) -> str:
    """
    Return a user-friendly Markdown error string for display in Streamlit.
    Includes just enough detail to help the user self-diagnose without
    exposing raw stack traces.
    """
    if exc is None:
        return "⚠️ An unknown error occurred while contacting the Gemini API."

    err_str = str(exc)

    # Rate limit / quota
    if any(kw in err_str.lower() for kw in ("429", "quota", "rate limit", "resource_exhausted")):
        return (
            "⚠️ **Gemini rate limit reached.**  \n"
            "You've hit the free-tier quota. Wait a minute and try again, "
            "or check your usage at https://aistudio.google.com."
        )

    # Invalid API key
    if any(kw in err_str.lower() for kw in ("api_key", "invalid", "api key")):
        return (
            "⚠️ **Invalid Gemini API key.**  \n"
            "Check the value in `.streamlit/secrets.toml` or your environment variable.  \n"
            "Get a free key at https://aistudio.google.com/app/apikey"
        )

    # Network / timeout
    if any(kw in err_str.lower() for kw in ("timeout", "connect", "network", "connection")):
        return (
            "⚠️ **Network error connecting to Gemini.**  \n"
            "Check your internet connection and try again.  \n"
            f"Details: `{err_str[:120]}`"
        )

    # Generic fallback
    return (
        f"⚠️ **Gemini API error.**  \n"
        f"```\n{err_str[:300]}\n```\n"
        "If this persists, restart the app or check the Gemini API status page."
    )


# ─────────────────────────────────────────────────────────────────────
# Utility: quick connectivity / key test (optional, not called by app.py)
# ─────────────────────────────────────────────────────────────────────

def test_connection() -> dict:
    """
    Send a minimal test message to verify the API key and network connectivity.
    Returns a dict: {"ok": bool, "model": str, "message": str}

    Usage (e.g. from a debug tab or CLI):
        import gemini_ai
        print(gemini_ai.test_connection())
    """
    try:
        reply = call_gemini(
            system="You are a test assistant. Reply with exactly one sentence.",
            messages=[{"role": "user", "content": "Respond with: 'Gemini connection OK.'"}],
        )
        ok = "⚠️" not in reply
        return {"ok": ok, "model": _MODEL_NAME, "message": reply}
    except Exception as exc:
        return {"ok": False, "model": _MODEL_NAME, "message": str(exc)}
