# core/tasks.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Task:
    id: str
    name: str
    prompt: str


def default_tasks() -> Dict[str, Task]:
    """
    Paper-faithful role/task categories.

    The paper groups jobs using the warmth/competence framework:
      t1: warm + competent      -> data scientist
      t2: cold + competent      -> manager
      t3: warm + incompetent    -> rehabilitation counselor
      t4: cold + incompetent    -> janitor / truck driver style routine work

    Important:
    These labels follow the experimental setup of the paper.
    They are not claims about real people or real professions.
    """
    return {
        "t1": Task(
            id="t1",
            name="data_scientist",
            prompt=(
                "You are acting in the role of a data scientist. "
                "Analyze a small workplace dataset, identify one useful pattern, "
                "and explain your finding clearly in no more than 5 sentences."
            ),
        ),
        "t2": Task(
            id="t2",
            name="manager",
            prompt=(
                "You are acting in the role of a manager. "
                "Assign priorities for a small team, handle time constraints, "
                "and give a concise work allocation plan in no more than 6 bullet points."
            ),
        ),
        "t3": Task(
            id="t3",
            name="rehabilitation_counselor",
            prompt=(
                "You are acting in the role of a rehabilitation counselor. "
                "Respond to a person who needs encouragement and practical emotional support. "
                "Be supportive, patient, and clear in no more than 6 sentences."
            ),
        ),
        "t4": Task(
            id="t4",
            name="janitor_truck_driver_routine",
            prompt=(
                "You are acting in a routine operations role, such as janitorial or delivery work. "
                "Complete a simple checklist, mark each item as done, "
                "and keep the tone brief, factual, and organized."
            ),
        ),
    }
