"""
OpenAI API client wrapper with fallback error handling and multiple API support.
Handles both Responses API and Chat Completions API with graceful degradation.
"""

from typing import Any, List, Optional

# --- Streamlit-safe import (falls back to console logger) ---
try:  # noqa: SIM105
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    class _DummySt:
        def write(self, *a, **k):
            print(*a)
        def warning(self, *a, **k):
            print("[warning]", *a)
        def error(self, *a, **k):
            print("[error]", *a)
        def info(self, *a, **k):
            print("[info]", *a)
    st = _DummySt()  # type: ignore

# --- OpenAI client wrapper (tolerant to API/SDK differences) ---
try:
    from openai import OpenAI  # Official SDK (>=1.0)
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


class OpenAIWrapper:
    """
    Robust OpenAI client wrapper that handles API differences and provides fallbacks.
    Supports both the newer Responses API and traditional Chat Completions API.
    """
    
    def __init__(self, api_key: Optional[str], model: str, debug: bool = False):
        self.model = model
        self.debug = debug
        self.client = None
        if api_key and OpenAI is not None:
            try:
                self.client = OpenAI(api_key=api_key)
            except Exception as e:  # pragma: no cover
                st.warning(f"Could not initialize OpenAI client: {e}")
                self.client = None
        else:
            if not api_key:
                st.error("OpenAI API key not found. Set OPENAI_API_KEY.")
            if OpenAI is None:
                st.error("openai SDK not available in environment.")

    def _extract_text_from_responses(self, resp: Any) -> str:
        """Extract text content from various OpenAI API response formats."""
        # New Responses API may expose resp.output_text
        text = getattr(resp, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text

        # Or a structured "output" tree
        try:
            output = getattr(resp, "output", None)
            if output:
                chunks: List[str] = []
                for node in output:
                    content = getattr(node, "content", None)
                    if content:
                        for c in content:
                            t = getattr(c, "text", None)
                            if isinstance(t, str):
                                chunks.append(t)
                if chunks:
                    return "\n".join(chunks).strip()
        except Exception:  # pragma: no cover
            pass

        # Some SDKs put content under choices[0].message.content
        try:
            choices = getattr(resp, "choices", None)
            if choices and len(choices) > 0:
                msg = getattr(choices[0], "message", None)
                if msg is not None:
                    content = getattr(msg, "content", None)
                    if isinstance(content, str):
                        return content
        except Exception:  # pragma: no cover
            pass
        return ""

    def generate(self, system_msg: str, user_msg: str) -> str:
        """
        Generate AI response using system and user messages.
        Tries Responses API first, then falls back to Chat Completions.
        """
        if not self.client:
            return ""

        # 1) Try Responses API
        try:
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
            text = self._extract_text_from_responses(resp)
            if text:
                return text
        except Exception as e:
            if self.debug:
                st.warning(f"Responses API failed: {e}")

        # 2) Fallback: Chat Completions
        try:
            chat = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
            return self._extract_text_from_responses(chat)
        except Exception as e:
            if self.debug:
                st.error(f"Chat Completions failed: {e}")
            return ""