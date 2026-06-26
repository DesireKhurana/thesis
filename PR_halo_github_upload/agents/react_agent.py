from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import re
from core.messages import Observation


@dataclass
class StepResult: #store result of agent step
    raw: str #raw text return by llm
    final: Optional[str] = None #final answer if agent is done
    action_text: Optional[str] = None   # raw text with action and parameters if agent uses tools


class ReActAgent:
    def __init__(self, name: str, llm_client, tool_descriptions: str, identity_mode: str = "neutral") -> None:
        self.name = name #name of the agent
        self.llm = llm_client # llm client used
        self.tool_descriptions = tool_descriptions
        self.identity_mode = (identity_mode or "neutral").strip().lower()  # check if agents have neutral identy or not

    # build the prompt that will be sent to the llm
    def build_prompt(self, task: str, observations: List[Observation], allow_tools: bool) -> str:
        obs_block = "\n".join( #convert last 10 observ to bullet points
            [f"- {o.source} ({'ok' if o.ok else 'error'}): {o.content}" for o in observations[-10:]]
        )
    # if tools are allowed , tell the agent which tool they can use
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
        else: #if tools are not allowed
            tools_block = "TOOLS DISABLED.\n"
            format_block = "Reply with exactly one line starting with 'FINAL: '.\nFINAL: "
        if self.identity_mode in {"demographic", "demographics", "profile", "profiles"}: #if it's demographic
            neutral_rules = (
                "Identity rules (demographic ablation):\n"
                "- People may have short profile cards in the history (age/gender/location/etc.).\n"
                "- You may reference ONLY what is explicitly provided in the history. Do not invent extra details.\n"
                "- Refer to people using the identifiers shown (e.g., \"person N\").\n"
                "- Keep language respectful.\n"
            )
        else:
            neutral_rules = ( #if neutral
                "Neutrality rules:\n"
                "- Refer to agents only as \"person N\" identifiers shown in observations.\n"
                "- Do not infer or mention demographics, names, ethnicity, nationality, religion, age, gender, health, or class.\n"
                "- Use neutral, descriptive language.\n"
                "- Avoid goal-oriented or competitive framing (e.g., \"win\", \"beat\", \"dominate\", \"optimize\").\n"
            )
        #comnine all
        return f"""You are an agent in a controlled simulation.

        {neutral_rules}

        {tools_block}

        {format_block}

        Task: {task}

        Observations so far:
        {obs_block if obs_block else "- (none)"}
        """

    def step(self, task: str, observations: List[Observation], allow_tools: bool = True) -> StepResult:
        #to run a reasoning step
        prompt = self.build_prompt(task, observations, allow_tools=allow_tools)   #build the prompt using the task, prev obs and tools
        raw = self.llm.complete(prompt, allow_tools=allow_tools).strip()  #send the prompt to the llm without spaces or newlines
        # see if llm produced a final answer
        final_match = re.search(r"(?s)\bFINAL:\s*(.*)\s*$", raw)
        if final_match:    # if there is one return it
            return StepResult(raw=raw, final=final_match.group(1).strip())
              #if tools are not allowed and FINAL was not found just return the raw text
        if not allow_tools:
            return StepResult(raw=raw, final=raw.strip())

        #if tool are allowed check if the llm requested one
        if allow_tools:
            has_action = re.search(r"(?m)^\s*ACTION:\s*.+$", raw) is not None
            has_input = "ACTION_INPUT:" in raw
            if has_action and has_input:
                return StepResult(raw=raw, action_text=raw)   #if there is both action and action input then return the raw text as tool request
        if allow_tools:    #fallback
            return StepResult(raw=raw, final="Invalid format (expected ACTION/ACTION_INPUT or FINAL).")
        return StepResult(raw=raw, final="Invalid format (expected FINAL).")
