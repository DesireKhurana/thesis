from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

ToolFn = Callable[..., Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    fn: ToolFn
    description: str = ""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, name: str, fn: ToolFn, description: str = "") -> None:
        if not name or not isinstance(name, str):
            raise ValueError("Tool name must be a non-empty string")
        self._tools[name] = ToolSpec(name=name, fn=fn, description=description)

    def call(self, name: str, **kwargs) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name].fn(**kwargs)

    def names(self) -> List[str]:
        return sorted(self._tools.keys())

    def describe(self) -> str:
        # Used in prompts: keeps tool list readable.
        lines = []
        for name in self.names():
            desc = self._tools[name].description.strip()
            lines.append(f"- {name}: {desc}" if desc else f"- {name}")
        return "\n".join(lines)
