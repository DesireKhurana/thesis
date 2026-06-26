from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class HaloProfile:
    name: str
    style_prompt: str     # agemnt rprompt
    visible_artifact: Optional[str] = None  # if it should show to others


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
            # part that make it visible if told
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
