from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import re

from core.messages import Observation


@dataclass
class StepResult:
    raw: str
    final: Optional[str] = None
    action_text: Optional[str] = None   # raw text containing ACTION/ACTION_INPUT


class ReActAgent:
    def __init__(self, name: str, llm_client, tool_descriptions: str, identity_mode: str = "neutral") -> None:
        self.name = name
        self.llm = llm_client
        self.tool_descriptions = tool_descriptions
        self.identity_mode = (identity_mode or "neutral").strip().lower()

    def build_prompt(self, task: str, observations: List[Observation], allow_tools: bool) -> str:
        obs_block = "\n".join(
            [f"- {o.source} ({'ok' if o.ok else 'error'}): {o.content}" for o in observations[-10:]]
        )

        if allow_tools:
            tools_block = f"You can use tools (name + description):\n{self.tool_descriptions}\n"
            format_block = """Use EXACTLY this output format:
THOUGHT: (1 short sentence)
ACTION: <tool_name>
ACTION_INPUT: <JSON object>

ACTION_INPUT must be valid JSON:
- Use double quotes for all keys and string values.
- Output ONLY the JSON object (no extra text, no comments).
- No trailing commas.
- Do NOT wrap JSON in ``` fences.

OR, if you are done:
THOUGHT: (1 short sentence)
FINAL: <answer>
"""
        else:
            tools_block = "TOOLS DISABLED.\n"
            format_block = "Reply with exactly one line starting with 'FINAL: '.\nFINAL: "
        if self.identity_mode in {"demographic", "demographics", "profile", "profiles"}:
            neutral_rules = (
                "Identity rules (demographic ablation):\n"
                "- People may have short profile cards in the history (age/gender/location/etc.).\n"
                "- You may reference ONLY what is explicitly provided in the history. Do not invent extra details.\n"
                "- Refer to people using the identifiers shown (e.g., \"person N\").\n"
                "- Keep language respectful.\n"
            )
        else:
            neutral_rules = (
                "Neutrality rules:\n"
                "- Refer to agents only as \"person N\" identifiers shown in observations.\n"
                "- Do not infer or mention demographics, names, ethnicity, nationality, religion, age, gender, health, or class.\n"
                "- Use neutral, descriptive language.\n"
                "- Avoid goal-oriented or competitive framing (e.g., \"win\", \"beat\", \"dominate\", \"optimize\").\n"
            )

        return f"""You are an agent in a controlled simulation.

        {neutral_rules}

        {tools_block}

        {format_block}

        Task: {task}

        Observations so far:
        {obs_block if obs_block else "- (none)"}
        """

    def step(self, task: str, observations: List[Observation], allow_tools: bool = True) -> StepResult:
        prompt = self.build_prompt(task, observations, allow_tools=allow_tools)
        raw = self.llm.complete(prompt, allow_tools=allow_tools).strip()

        # FINAL (always allowed)
        final_match = re.search(r"(?s)\bFINAL:\s*(.*)\s*$", raw)
        if final_match:
            return StepResult(raw=raw, final=final_match.group(1).strip())

        # Tools disabled: accept raw text as the final answer
        if not allow_tools:
            return StepResult(raw=raw, final=raw.strip())

        # ACTION only if tools are allowed
        if allow_tools:
            has_action = re.search(r"(?m)^\s*ACTION:\s*.+$", raw) is not None
            has_input = "ACTION_INPUT:" in raw
            if has_action and has_input:
                return StepResult(raw=raw, action_text=raw)

        # Fallback
        if allow_tools:
            return StepResult(raw=raw, final="Invalid format (expected ACTION/ACTION_INPUT or FINAL).")
        return StepResult(raw=raw, final="Invalid format (expected FINAL).")
