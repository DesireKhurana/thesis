# agents/halo_profiles.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class HaloProfile:
    """
    Presentation-only cues.
    IMPORTANT: must not introduce demographics, identity claims, credentials, or backstory.
    """
    name: str
    style_prompt: str                 # injected into the agent's prompt (private)
    visible_artifact: Optional[str] = None  # optionally shown to others (public)


def default_halo_profiles() -> Dict[str, HaloProfile]:
    return {
        "polite": HaloProfile(
            name="polite",
            style_prompt=(
                "Presentation cue: be consistently polite and considerate. "
                "Use brief appreciation and soft language. "
                "Do not invent personal facts."
            ),
        ),
        "verbose": HaloProfile(
            name="verbose",
            style_prompt=(
                "Presentation cue: be more verbose than necessary (within limits). "
                "Add extra explanation and detail. "
                "Do not invent personal facts."
            ),
        ),
        "structured": HaloProfile(
            name="structured",
            style_prompt=(
                "Presentation cue: write in a structured way when possible: "
                "short headings and numbered points. "
                "Do not invent personal facts."
            ),
        ),
        "authoritative": HaloProfile(
            name="authoritative",
            style_prompt=(
                "Presentation cue: sound confident and decisive (fewer hedges). "
                "Do not claim credentials or expertise you were not given. "
                "Do not invent personal facts."
            ),
        ),
        "avatar": HaloProfile(
            name="avatar",
            style_prompt=(
                "Presentation cue: you are perceived as having a visually appealing, professional avatar. "
                "Do not mention demographics or identity. "
                "Do not invent personal facts."
            ),
            # This is the part that makes the avatar cue visible to others.
            # We'll decide in Step 3 where/how to show it.
            visible_artifact="Profile shows a very polished, professional avatar.",
        ),
        "combo": HaloProfile(
            name="combo",
            style_prompt=(
                "Presentation cue: combine polite + structured + confident writing. "
                "Keep it natural. Do not invent personal facts."
            ),
        ),
    }
