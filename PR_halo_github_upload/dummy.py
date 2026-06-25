# dummy.py
import random

from core.tools import ToolRegistry
from agents.fc_caller import FcCallerAgent
from agents.react_agent import ReActAgent
from core.runtime import Runtime
import episode_runner  # so we can monkeypatch reset_runtime for debug printing


import ast
import json
import re

class DummyLLMClient:
    def complete(self, prompt: str, allow_tools: bool = True) -> str:
        prompt = prompt or ""

        # -------------------------
        # Supervisor: person -> task
        # -------------------------
        if "You are the supervisor" in prompt and "Available task_ids" in prompt:
            task_ids = ["t1", "t2", "t3", "t4"]
            agents = ["person 1", "person 2", "person 3", "person 4", "person 5"]

            m_tasks = re.search(r"Available task_ids:\s*(\[[^\n]+\])", prompt)
            if m_tasks:
                try:
                    task_ids = ast.literal_eval(m_tasks.group(1))
                except Exception:
                    pass

            m_agents = re.search(r"Agents:\s*(\[[^\n]+\])", prompt)
            if m_agents:
                try:
                    agents = ast.literal_eval(m_agents.group(1))
                except Exception:
                    pass

            # deterministic mapping, same as before
            mapping = {a: task_ids[i % len(task_ids)] for i, a in enumerate(agents)}
            return "FINAL: " + json.dumps(mapping)

        # ----------------------------------------
        # Peer evaluation: ratings depend on history
        # ----------------------------------------
        if "PEER EVALUATION" in prompt and "Targets:" in prompt and "Task IDs:" in prompt:
            # parse task IDs and targets from the prompt
            task_ids = ["t1", "t2", "t3", "t4"]
            targets = ["person 2", "person 3", "person 4", "person 5"]

            m_tasks = re.search(r"Task IDs:\s*(\[[^\n]+\])", prompt)
            if m_tasks:
                try:
                    task_ids = ast.literal_eval(m_tasks.group(1))
                except Exception:
                    pass

            m_targets = re.search(r"Targets:\s*(\[[^\n]+\])", prompt)
            if m_targets:
                try:
                    targets = ast.literal_eval(m_targets.group(1))
                except Exception:
                    pass

            # extract evaluator label ("person N") so ratings can vary per evaluator
            evaluator = "person 0"
            m_eval = re.search(r"Visible history for you\s*\((person \d+)\)\s*:", prompt)
            if m_eval:
                evaluator = m_eval.group(1)

            # pull just the visible history block (between the "Visible history..." line and "Targets:")
            history_block = ""
            m_hist = re.search(r"Visible history for you\s*\(person \d+\)\s*:\s*(.*?)\n\nTargets:", prompt, re.S)
            if m_hist:
                history_block = m_hist.group(1)

            # count outcomes per person from history: success=1/0
            succ = {}  # person -> successes
            fail = {}  # person -> failures
            for person_label, s in re.findall(r"agent=(person \d+).*?success=(\d)", history_block):
                if s == "1":
                    succ[person_label] = succ.get(person_label, 0) + 1
                else:
                    fail[person_label] = fail.get(person_label, 0) + 1

            # count assignments per person per task from history lines
            assigned = {}  # person -> {tid: count}
            for person_label, tid in re.findall(r"assigned\s+(person \d+)\s+->\s+(t\d+)", history_block):
                assigned.setdefault(person_label, {})
                assigned[person_label][tid] = assigned[person_label].get(tid, 0) + 1

            def clamp(v: int) -> int:
                return max(1, min(10, v))

            def tiny_tiebreak(key: str) -> int:
                # deterministic small variation in {-1,0,+1}
                return (abs(hash(key)) % 3) - 1

            out = {}
            for tgt in targets:
                s_count = succ.get(tgt, 0)
                f_count = fail.get(tgt, 0)
                net = s_count - f_count  # can be negative

                out[tgt] = {}
                for tid in task_ids:
                    # baseline + outcome signal + "experience" signal + tiny stable jitter
                    base = 5
                    outcome_signal = 2 * net                 # success streak pushes ratings up
                    exp_signal = assigned.get(tgt, {}).get(tid, 0)  # more assignments -> perceived fit
                    jitter = tiny_tiebreak(f"{evaluator}|{tgt}|{tid}")

                    score = clamp(base + outcome_signal + exp_signal + jitter)
                    out[tgt][tid] = score

            return "FINAL: " + json.dumps(out)

        # -------------------------
        # Normal agent behavior
        # -------------------------
        if allow_tools:
            return "THOUGHT: ok\nFINAL: ok"
        return "FINAL: ok"


def add(a: int, b: int) -> int:
    return a + b


def echo(text: str) -> str:
    return text


def main() -> None:
    random.seed(0)

    # Tools (won't really be used, but keeps your agent prompt/tool registry consistent)
    tools = ToolRegistry()
    tools.register("add", add, "Add two integers. Args: {a:int, b:int}")
    tools.register("echo", echo, "Echo text. Args: {text:str}")

    fc = FcCallerAgent(name="fc_caller", tools=tools)

    llm = DummyLLMClient()

    agents = {
        "BOSS": ReActAgent(name="BOSS", llm_client=llm, tool_descriptions=tools.describe()),
        "W1": ReActAgent(name="W1", llm_client=llm, tool_descriptions=tools.describe()),
        "W2": ReActAgent(name="W2", llm_client=llm, tool_descriptions=tools.describe()),
        "W3": ReActAgent(name="W3", llm_client=llm, tool_descriptions=tools.describe()),
        "W4": ReActAgent(name="W4", llm_client=llm, tool_descriptions=tools.describe()),
        "W5": ReActAgent(name="W5", llm_client=llm, tool_descriptions=tools.describe()),
    }

    rt = Runtime(fc=fc, agents=agents)

    # --- Debug: print inbox length each episode start WITHOUT editing episode_runner.py ---
    _orig_reset = episode_runner.reset_runtime

    def reset_runtime_with_debug(rt_: Runtime, episode_id: int) -> None:
        print(
            f"[debug] pre-reset ep {episode_id}: "
            f"W1 inbox={len(rt_.inbox.get('W1', []))} "
            f"W1 queued={len(rt_.queued.get('W1', []))}"
        )
        _orig_reset(rt_, episode_id)
        print(
            f"[debug] post-reset ep {episode_id}: "
            f"W1 inbox={len(rt_.inbox.get('W1', []))} "
            f"W1 queued={len(rt_.queued.get('W1', []))}"
        )

    episode_runner.reset_runtime = reset_runtime_with_debug

    # -------------------------------------------------------------------------------

    print("Running dummy (no real LLM). Expect inbox length to grow on episode 2.")
    episode_runner.run_experiment(
        rt,
        n_episodes=4,
        random_phase_episodes=2,
        p0=0.8,
        log_path="episodes_dummy.jsonl",
    )
    print("Done. Check episodes_dummy.jsonl and episodes_anon.jsonl")


if __name__ == "__main__":
    main()
