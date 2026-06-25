from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import ast
import json
import re

from core.messages import Observation, ToolCall
from core.tools import ToolRegistry


@dataclass
class FcCallerAgent:
    name: str
    tools: ToolRegistry

    def execute(self, call: ToolCall) -> Observation:
        try:
            result = self.tools.call(call.name, **call.arguments)
            return Observation(
                source=f"tool:{call.name}",
                content=str(result),
                ok=True,
                related_call_id=call.call_id,
            )
        except Exception as e:
            return Observation(
                source=f"tool:{call.name}",
                content=f"ERROR: {type(e).__name__}: {e}",
                ok=False,
                related_call_id=call.call_id,
            )

    def _extract_action_and_input(self, text: str) -> tuple[str, str]:
        text = text.strip()

        action_match = re.search(r"(?m)^\s*ACTION:\s*(.+?)\s*$", text)
        if not action_match:
            raise ValueError("Missing ACTION: line")
        tool_name = action_match.group(1).strip()

        # Capture everything after ACTION_INPUT: (often models add fences/newlines)
        ain_match = re.search(r"(?s)^\s*ACTION_INPUT:\s*(.+?)\s*$", text, re.MULTILINE)
        if not ain_match:
            raise ValueError("Missing ACTION_INPUT:")
        raw = ain_match.group(1).strip()

        # Remove markdown code fences if present
        raw = raw.strip("`").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        # Extract the JSON-ish object between the first { and last }
        l = raw.find("{")
        r = raw.rfind("}")
        if l == -1 or r == -1 or r <= l:
            raise ValueError("ACTION_INPUT does not contain a JSON object {...}")
        obj_str = raw[l:r + 1].strip()

        return tool_name, obj_str

    def _parse_args(self, obj_str: str) -> Dict[str, Any]:
        # 1) strict JSON
        try:
            data = json.loads(obj_str)
            if not isinstance(data, dict):
                raise ValueError("ACTION_INPUT must be a JSON object")
            return data
        except json.JSONDecodeError:
            pass

        # 2) fallback: Python literal dict (handles single quotes safely)
        # (This is a common defensive parsing technique when LLM output isn't strict JSON.) [web:426]
        data = ast.literal_eval(obj_str)
        if not isinstance(data, dict):
            raise ValueError("ACTION_INPUT must be an object/dict")
        return data

    def execute_from_text(self, react_text: str) -> Observation:
        try:
            tool_name, obj_str = self._extract_action_and_input(react_text)
            args = self._parse_args(obj_str)
            return self.execute(ToolCall(name=tool_name, arguments=args))
        except Exception as e:
            # Never crash the run: return an Observation so the agent can correct itself. [web:432]
            return Observation(source="fc_caller", content=f"ERROR parsing tool call: {type(e).__name__}: {e}", ok=False)
