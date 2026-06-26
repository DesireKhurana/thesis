from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random


@dataclass
class HistoryItem:
    episode: int
    step: int
    task_id: str
    agent_id: str
    success: int  # either 0 or 1


def sample_success(p_success: float = 0.8) -> int:
    return 1 if random.random() < p_success else 0 #randomly choose


def phi_epsilon_greedy(
    history: List[HistoryItem],
    task_ids: List[str],
    agent_ids: List[str],
    epsilon: float = 0.1,
    prior: float = 0.5,
) -> Tuple[str, str]:
    #counts attempt and successes for each task,agent pair
    n: Dict[Tuple[str, str], int] = {}
    wins: Dict[Tuple[str, str], int] = {}

    for h in history:
        key = (h.task_id, h.agent_id)
        n[key] = n.get(key, 0) + 1
        wins[key] = wins.get(key, 0) + int(h.success)

    # exploartion, choose random task
    if random.random() < epsilon:
        return random.choice(task_ids), random.choice(agent_ids)

    # exploit, pick best pair with highest success rate
    best_pair = (task_ids[0], agent_ids[0])
    best_rate = -1.0
    for t in task_ids:
        for a in agent_ids:
            key = (t, a)
            if n.get(key, 0) == 0:
                rate = prior
            else:
                rate = wins[key] / n[key]
            if rate > best_rate:
                best_rate = rate
                best_pair = (t, a)

    return best_pair
