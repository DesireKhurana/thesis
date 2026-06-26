from __future__ import annotations   #to handle type hints more safely since classes refer to each other
from dataclasses import dataclass # to create simple classes to store data
from typing import Any, Dict
import ast #to parse dictionaries
import json #to parse json files
import re #regular expression
from core.messages import Observation, ToolCall
from core.tools import ToolRegistry


@dataclass
class FcCallerAgent: #agent responsible for tool calls
    name: str #name agent
    tools: ToolRegistry
    #execute the tool call and return an observation
    def execute(self, call: ToolCall) -> Observation:
        try:
            result = self.tools.call(call.name, **call.arguments) #call toll with arguments
            return Observation(
                source=f"tool is {call.name}",
                content=str(result),
                ok=True, #if it was succesfull
                related_call_id=call.call_id, # connect result to tool
            )
        except Exception as e: #if it fails return an error
            return Observation(
                source=f"tool si {call.name}",
                content=f"the error-> {type(e).__name__}: {e}",
                ok=False, #not successfull
                related_call_id=call.call_id,
            )
    #extract tool name and parameter from the agents text
    def _extract_action_and_input(self, text: str) -> tuple[str, str]:
        text = text.strip() #remove spaces and new lines
        action_match = re.search(r"(?m)^\s*ACTION:\s*(.+?)\s*$", text) # search for a line that starts with action
        if not action_match:
            raise ValueError("No line")
        tool_name = action_match.group(1).strip() #get the name
    #sercah for the input
        ain_match = re.search(r"(?s)^\s*ACTION_INPUT:\s*(.+?)\s*$", text, re.MULTILINE)
        if not ain_match:
            raise ValueError("No input")
        raw = ain_match.group(1).strip() #extract the raw parameter text
        #replace everything with the quotes
        raw = raw.strip("`").strip()
        raw = raw.replace("```json", "").replace("```", "").strip() #
        #find the opening and closing to get the json
        l = raw.find("{")
        r = raw.rfind("}")
        if l == -1 or r == -1 or r <= l:
            raise ValueError("No json object")
        obj_str = raw[l:r + 1].strip() # extract json as text
        return tool_name, obj_str

    #convert the para to a dict
    def _parse_args(self, obj_str: str) -> Dict[str, Any]:
        try: #json parsing
            data = json.loads(obj_str)
            if not isinstance(data, dict):
                raise ValueError("is not json object")
            return data
        except json.JSONDecodeError:
            pass
        data = ast.literal_eval(obj_str)  #safer parsing as security
        if not isinstance(data, dict):
            raise ValueError("is not a dictionary")
        return data

    #if text output
    def execute_from_text(self, react_text: str) -> Observation:
        try:
            tool_name, obj_str = self._extract_action_and_input(react_text) #extract tool name and para from text
            args = self._parse_args(obj_str) # convert string to dict
            return self.execute(ToolCall(name=tool_name, arguments=args)) #create an instance and execute it
        except Exception as e: #if fails give observation with eror
            return Observation(source="fc_caller", content=f"there is an error in parsing: {type(e).__name__}: {e}", ok=False)
