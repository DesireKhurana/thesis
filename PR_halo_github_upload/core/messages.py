from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional
import time
import uuid

Role = Literal["system", "agent", "tool"]


def _uuid() -> str:
    return str(uuid.uuid4())


@dataclass(frozen=True)
class Message:
    role: Role
    sender: str
    content: str
    ts: float = field(default_factory=lambda: time.time())
    id: str = field(default_factory=_uuid)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    call_id: str = field(default_factory=_uuid)


@dataclass(frozen=True)
class Observation:
    source: str                  # "system", "agent:<name>", "tool:<name>"
    content: str                 # text payload
    ok: bool = True
    related_call_id: Optional[str] = None
