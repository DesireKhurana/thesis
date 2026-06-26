from __future__ import annotations   #to handle type hints more safely since classes refer to each other
from dataclasses import dataclass, field   # to create simple classes to store data
from typing import Any, Dict, Literal, Optional
import time
import uuid # to create uniques ids (universally unique identifier)
# role can only be one of these
Role = Literal["system", "agent", "tool"]


#to generate a random string that is a unique id
def _uuid() -> str:
    return str(uuid.uuid4())


@dataclass(frozen=True) #frozen = true means that once the message is stated it cannot be changed
class Message:
    role: Role #role of the person who sends the message
    sender: str #name of the person
    content: str #content of the message
    ts: float = field(default_factory=lambda: time.time()) #timestamp of the moment that the messahe was created
    id: str = field(default_factory=_uuid) # id of the message and if there is none create a new one


@dataclass(frozen=True)  #after tool is asked it cannot be taken back
class ToolCall:
    name: str #name tool that should be called
    arguments: Dict[str, Any] #parameters  given to tool
    call_id: str = field(default_factory=_uuid) #id of the request


@dataclass(frozen=True)  # the rsult after happend cannot be changed
class Observation:
    source: str                  # tell from where so the role
    content: str                 #the text that was yield
    ok: bool = True    #tells if the observation was succesfull or not
    related_call_id: Optional[str] = None #if it is related to a tool it also stores the tool id, othershise none
