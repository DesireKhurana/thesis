from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class JobType:
    id: str
    label: str
    warmth: int          # 0 is low and 1 is high
    competence: int      


def default_job_types() -> Dict[str, JobType]:
    # four categories
    return {
        "warm_competent": JobType("warm_competent", "Warm & Competent", 1, 1),
        "cold_competent": JobType("cold_competent", "Cold & Competent", 0, 1),
        "warm_incompetent": JobType("warm_incompetent", "Warm & Incompetent", 1, 0),
        "cold_incompetent": JobType("cold_incompetent", "Cold & Incompetent", 0, 0),
    }


def default_worker_job_map() -> Dict[str, str]:
    # 5 workers
    return {
        "W1": "warm_competent",
        "W2": "warm_competent",   # replicate
        "W3": "cold_competent",
        "W4": "warm_incompetent",
        "W5": "cold_incompetent",
    }
