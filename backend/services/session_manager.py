"""Thread-safe, TTL-bound assistant sessions."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Set

from backend.services.orchestrator import Orchestrator


@dataclass
class AssistantSession:
    assistant: Orchestrator
    last_access: float


class SessionManager:
    def __init__(self, factory: Callable[[], Orchestrator] = Orchestrator, ttl_seconds: int = 3600):
        self._factory = factory
        self._ttl_seconds = ttl_seconds
        self._sessions: Dict[str, AssistantSession] = {}
        self._documents: Dict[str, Set[str]] = {}
        self._lock = threading.RLock()

    def get(self, session_id: str) -> Optional[Orchestrator]:
        now = time.monotonic()
        with self._lock:
            self._evict_expired(now)
            session = self._sessions.get(session_id)
            if session:
                session.last_access = now
                return session.assistant
        return None

    def create_or_replace(self, session_id: str, factory: Optional[Callable[[], Orchestrator]] = None) -> Orchestrator:
        assistant = (factory or self._factory)()
        with self._lock:
            self._sessions[session_id] = AssistantSession(assistant, time.monotonic())
            self._documents.setdefault(session_id, set())
        return assistant

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._documents.pop(session_id, None)

    def register_document(self, session_id: str, document_id: str) -> None:
        with self._lock:
            self._documents.setdefault(session_id, set()).add(document_id)

    def unregister_document(self, session_id: str, document_id: str) -> None:
        with self._lock:
            self._documents.get(session_id, set()).discard(document_id)

    def owns_document(self, session_id: str, document_id: str) -> bool:
        with self._lock:
            return document_id in self._documents.get(session_id, set())

    def _evict_expired(self, now: float) -> None:
        expired = [sid for sid, session in self._sessions.items() if now - session.last_access > self._ttl_seconds]
        for session_id in expired:
            self._sessions.pop(session_id, None)
            self._documents.pop(session_id, None)

    @property
    def active_count(self) -> int:
        with self._lock:
            self._evict_expired(time.monotonic())
            return len(self._sessions)