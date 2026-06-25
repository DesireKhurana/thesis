from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Iterable

from core.messages import Observation
from agents.react_agent import ReActAgent
from agents.fc_caller import FcCallerAgent


@dataclass
class Runtime:
    fc: FcCallerAgent
    agents: Dict[str, ReActAgent]

    inbox: Dict[str, List[Observation]] = field(default_factory=dict)
    queued: Dict[str, List[Observation]] = field(default_factory=dict)

    # internal_key -> neutral label shown to LLM
    display_names: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in self.agents:
            self.inbox.setdefault(name, [])
            self.queued.setdefault(name, [])

        if not self.display_names:
            worker_keys = sorted([k for k in self.agents.keys() if k != "BOSS"])
            self.display_names["BOSS"] = "person 0"
            for i, k in enumerate(worker_keys, start=1):
                self.display_names[k] = f"person {i}"

    def _shown(self, key: str) -> str:
        return self.display_names.get(key, key)

    def _obs_source(self, sender: str) -> str:
        if sender in ("runtime", "system"):
            return "system"
        return f"agent:{self._shown(sender)}"

    def flush_queued(self) -> int:
        delivered = 0
        for agent_name in self.agents.keys():
            buf = self.queued.get(agent_name, [])
            if buf:
                self.inbox[agent_name].extend(buf)
                delivered += len(buf)
                buf.clear()
        return delivered

    def send_direct(self, sender: str, to: str, content: str, defer: bool = False) -> None:
        if to not in self.agents:
            raise KeyError(f"Unknown agent: {to}")

        obs = Observation(source=self._obs_source(sender), content=content, ok=True)

        if defer:
            self.queued.setdefault(to, [])
            self.queued[to].append(obs)
        else:
            self.inbox[to].append(obs)

        # --- Audit copy to supervisor so BOSS has full history (paper) ---
        if "BOSS" in self.agents and to != "BOSS" and sender != "BOSS":
            audit = Observation(
                source=obs.source,
                content=f"[PRIVATE -> {self._shown(to)}] {content}",
                ok=True,
            )
            if defer:
                self.queued["BOSS"].append(audit)
            else:
                self.inbox["BOSS"].append(audit)

    def send_group(self, sender: str, group: Iterable[str], content: str, defer: bool = False) -> None:
        for agent_name in group:
            if agent_name != sender:
                self.send_direct(sender, agent_name, content, defer=defer)

    def broadcast(self, sender: str, content: str, defer: bool = False) -> None:
        for agent_name in self.agents:
            if agent_name != sender:
                self.send_direct(sender, agent_name, content, defer=defer)

    def run_chat(self, agent_name: str, prompt: str) -> str:
        if agent_name not in self.agents:
            raise KeyError(f"Unknown agent '{agent_name}'")

        obs = self.inbox[agent_name]
        step = self.agents[agent_name].step(prompt, obs, allow_tools=False)

        spoken = (step.final or "").strip()
        if not spoken.upper().startswith("FINAL:"):
            spoken = "FINAL: " + spoken

        # chat is an interaction -> defer
        self.broadcast(sender="runtime", content=f"[CHAT] {self._shown(agent_name)}: {spoken}", defer=True)
        return spoken

    def run_task(self, agent_name: str, task: str, max_steps: int = 8) -> str:
        if agent_name not in self.agents:
            raise KeyError(f"Unknown agent '{agent_name}'")

        obs = self.inbox[agent_name]

        for _ in range(max_steps):
            step = self.agents[agent_name].step(task, obs, allow_tools=False)

            if step.final is not None:
                # task results are system notifications -> immediate
                self.broadcast(
                    sender="runtime",
                    content=f"[RESULT] {self._shown(agent_name)} FINAL: {step.final}",
                    defer=False,
                )
                return step.final

            tool_obs = self.fc.execute_from_text(step.action_text or "")
            obs.append(tool_obs)

            self.broadcast(
                sender="runtime",
                content=f"[RESULT] {self._shown(agent_name)} {tool_obs.source}: {tool_obs.content}",
                defer=False,
            )

        self.broadcast(sender="runtime", content=f"[RESULT] {self._shown(agent_name)} stopped (max_steps).", defer=False)
        return "Stopped: max_steps reached."

    def discussion_round(self, participants: List[str], topic: str) -> Dict[str, str]:
        discussion_task = (
            "DISCUSSION MODE (no tools). Reply with exactly one line starting with 'FINAL: '.\n"
            f"Topic: {topic}"
        )

        spoken_map: Dict[str, str] = {}
        for a in participants:
            step = self.agents[a].step(discussion_task, self.inbox[a], allow_tools=False)
            raw = (step.final or "").strip()
            first = raw.splitlines()[0].strip() if raw.splitlines() else ""
            if not first.upper().startswith("FINAL:"):
                first = "FINAL: " + first

            spoken_map[a] = first
            # discussion is interaction -> defer
            self.broadcast(sender="runtime", content=f"[DISCUSSION] {self._shown(a)}: {first}", defer=True)

        return spoken_map
