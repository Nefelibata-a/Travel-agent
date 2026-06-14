"""
Memory manager for travel preferences and conversation history.

Two-layer: short-term conversation window + long-term preference summary.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Deque

import tiktoken
from langchain_core.messages import BaseMessage
from loguru import logger

_TOKENIZER = tiktoken.get_encoding("cl100k_base")
SHORT_TERM_WINDOW = 10
SUMMARY_TRIGGER_TOKENS = 4096


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


class MemoryManager:
    def __init__(self):
        self._short_term: dict[str, Deque[BaseMessage]] = {}
        self._long_term: dict[str, str] = {}

    def add_message(self, session_id: str, message: BaseMessage) -> None:
        if session_id not in self._short_term:
            self._short_term[session_id] = deque(maxlen=SHORT_TERM_WINDOW)
        self._short_term[session_id].append(message)

        total = sum(
            _count_tokens(m.content if isinstance(m.content, str) else str(m.content))
            for m in self._short_term[session_id]
        )
        if total > SUMMARY_TRIGGER_TOKENS:
            self._compress(session_id)

    def _compress(self, session_id: str) -> None:
        messages = self.get_short_term_messages(session_id)
        summary = " | ".join(
            f"[{m.__class__.__name__}] {m.content[:200]}" for m in messages
        )
        prev = self._long_term.get(session_id, "")
        self._long_term[session_id] = (prev + "\n" + summary) if prev else summary
        self._short_term[session_id] = deque(maxlen=SHORT_TERM_WINDOW)
        logger.info(f"[memory] Compressed session '{session_id}'")

    def get_short_term_messages(self, session_id: str) -> list[BaseMessage]:
        return list(self._short_term.get(session_id, []))

    def get_long_term_summary(self, session_id: str) -> str:
        return self._long_term.get(session_id, "")

    def clear_session(self, session_id: str) -> None:
        self._short_term.pop(session_id, None)
        self._long_term.pop(session_id, None)

    # --- Travel-specific helpers ---

    def remember_preference(self, session_id: str, key: str, value: str) -> None:
        existing = self._long_term.get(session_id, "")
        entry = f"[pref:{key}={value}]"
        self._long_term[session_id] = (existing + " " + entry).strip()

    def get_preferences(self, session_id: str) -> dict[str, str]:
        raw = self.get_long_term_summary(session_id)
        prefs = {}
        import re
        for m in re.finditer(r"\[pref:(\w+)=([^\]]+)\]", raw):
            prefs[m.group(1)] = m.group(2)
        return prefs
