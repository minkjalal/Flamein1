"""
Conversation memory with JSON persistence for the Flame in Freefall application.
Stores conversation turns with timestamps and cited sources.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from config import MEMORY_FILE, MEMORY_MAX_TURNS

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Persistent conversation history."""

    def __init__(self, memory_file: str | None = None):
        self._file = memory_file or MEMORY_FILE
        self._turns: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self.load()

    # ── Public API ────────────────────────────────────────────────────────────

    def add_turn(
        self,
        user_msg: str,
        ai_response: str,
        sources: Optional[list[Any]] = None,
    ) -> None:
        """Append a conversation turn.

        Args:
            user_msg: The user's message / question.
            ai_response: The AI-generated answer.
            sources: Optional list of source metadata dicts.
        """
        turn = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_message": user_msg,
            "ai_response": ai_response,
            "sources": sources or [],
        }
        with self._lock:
            self._turns.append(turn)
            if MEMORY_MAX_TURNS > 0 and len(self._turns) > MEMORY_MAX_TURNS:
                del self._turns[:-MEMORY_MAX_TURNS]
            self.save()

    def get_recent_context(self, n: int = 5) -> str:
        """Format the last *n* turns into a string suitable for prompt context."""
        with self._lock:
            recent = list(self._turns[-n:]) if n > 0 else []
        if not recent:
            return ""

        parts: list[str] = ["## Recent Conversation"]
        for turn in recent:
            parts.append(f"User: {turn.get('user_message', '')}")
            # Truncate long AI responses to avoid blowing up the context window
            response = str(turn.get("ai_response", ""))
            response_preview = response[:500]
            if len(response) > 500:
                response_preview += "…"
            parts.append(f"Assistant: {response_preview}")
            parts.append("")  # blank line between turns

        return "\n".join(parts)

    def get_full_history(self) -> list[dict[str, Any]]:
        """Return a copy of all stored turns."""
        with self._lock:
            return [dict(turn) for turn in self._turns]

    def clear(self) -> None:
        """Reset conversation memory and delete the backing file."""
        with self._lock:
            self._turns.clear()
            if os.path.exists(self._file):
                try:
                    os.remove(self._file)
                    logger.info("Memory file deleted: %s", self._file)
                except OSError as e:
                    logger.warning("Could not delete memory file: %s", e)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Write the conversation history atomically to disk as JSON."""
        with self._lock:
            payload = list(self._turns)
            try:
                directory = os.path.dirname(self._file) or "."
                os.makedirs(directory, exist_ok=True)
                tmp_path = f"{self._file}.tmp"
                with open(tmp_path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2, ensure_ascii=False)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_path, self._file)
            except OSError as e:
                logger.error("Failed to save memory: %s", e)

    def load(self) -> None:
        """Load conversation history from disk if the file exists."""
        with self._lock:
            if not os.path.isfile(self._file):
                self._turns = []
                return

            try:
                with open(self._file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    self._turns = [
                        turn for turn in data if isinstance(turn, dict)
                    ]
                    if MEMORY_MAX_TURNS > 0 and len(self._turns) > MEMORY_MAX_TURNS:
                        self._turns = self._turns[-MEMORY_MAX_TURNS:]
                    logger.info("Loaded %d conversation turns from %s", len(self._turns), self._file)
                else:
                    logger.warning("Unexpected memory format — starting fresh.")
                    self._turns = []
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not load memory (%s) — starting fresh.", e)
                self._turns = []

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._turns)

    def __bool__(self) -> bool:
        return bool(self._turns)
