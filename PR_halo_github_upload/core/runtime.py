from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Iterable

from core.messages import Observation
from agents.react_agent import ReActAgent
from agents.fc_caller import FcCallerAgent

# ro manage communication between agents and tools
@dataclass
class Runtime:
    fc: FcCallerAgent # tool calling agent to get the tool that was asked
    agents: Dict[str, ReActAgent] # all agents name
#to store messages that each agent can currently see
    inbox: Dict[str, List[Observation]] = field(default_factory=dict)
#queue to store messages that will be visible later
    queued: Dict[str, List[Observation]] = field(default_factory=dict)
#maps agents to nautral name that are passed to llm
    display_names: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None: #
        for name in self.agents: #every agent needs to have an inbox and queue
            self.inbox.setdefault(name, [])
            self.queued.setdefault(name, [])
        if not self.display_names: #if it is not provided create nautral name
            worker_keys = sorted([k for k in self.agents.keys() if k != "BOSS"]) #sort all agents execept boss
            self.display_names["BOSS"] = "person 0" #boss will get neutral label 0
            for i, k in enumerate(worker_keys, start=1): #give every person neutral name ex person 1...
                self.display_names[k] = f"person {i}"

    def _shown(self, key: str) -> str:
        return self.display_names.get(key, key) #return the neutral name for agents and if there is no label return the original key

    def _obs_source(self, sender: str) -> str:
        if sender in ("runtime", "system"): # who sent the message if it's runtime or system
            return "system" #yields it is from system
        return f"agent:{self._shown(sender)}" # if not one of those is treated like an agent

    def flush_queued(self) -> int: #deliver all the queued messages into agents inbox
        delivered = 0
        for agent_name in self.agents.keys(): #go through each agent
            buf = self.queued.get(agent_name, []) # get the queue message for this agent
            if buf: #if there are any queues message move to inbox
                self.inbox[agent_name].extend(buf)
                delivered += len(buf)
                buf.clear() #clear the queue
        return delivered

    # send one message from someone to another specific person
    def send_direct(self, sender: str, to: str, content: str, defer: bool = False) -> None:
        if to not in self.agents: #check if the agent exists
            raise KeyError("the agent doesnt exist")
        #create the observation  for the person who gets the message
        obs = Observation(source=self._obs_source(sender), content=content, ok=True)
        # if it has to be delivered later than add to the queue
        if defer:
            self.queued.setdefault(to, [])
            self.queued[to].append(obs)
        else: #otherwhise append to inbox
            self.inbox[to].append(obs)
        #if boss exists then they receive a copy of the message that rae between other agents
        if "BOSS" in self.agents and to != "BOSS" and sender != "BOSS":
            audit = Observation(
                source=obs.source,
                content=f"[PRIVATE -> {self._shown(to)}] {content}",
                ok=True,
            )
            if defer: #if the og message is deferred than also the copy will be delivered later
                self.queued["BOSS"].append(audit)
            else:
                self.inbox["BOSS"].append(audit) #otherwhise go to inbox


#send same message to specific people
    def send_group(self, sender: str, group: Iterable[str], content: str, defer: bool = False) -> None:
        for agent_name in group:
            if agent_name != sender: # do not send the message to the person who sent
                self.send_direct(sender, agent_name, content, defer=defer)

#send same message to everyone
    def broadcast(self, sender: str, content: str, defer: bool = False) -> None:
        for agent_name in self.agents:
            if agent_name != sender: #if sender do not
                self.send_direct(sender, agent_name, content, defer=defer)


#one to one chat
    def run_chat(self, agent_name: str, prompt: str) -> str:
        if agent_name not in self.agents: #check if the agent exists
            raise KeyError(f"the agent does not exist")
        obs = self.inbox[agent_name] #get the inbox of the agent
        step = self.agents[agent_name].step(prompt, obs, allow_tools=False) #tell the agent to respond without any tool
        #get the response
        spoken = (step.final or "").strip()
        if not spoken.upper().startswith("FINAL:"): #make sure that it start with FINAL:
            spoken = "FINAL: " + spoken
        # defer the message so the agents receive it after the end of the episode
        self.broadcast(sender="runtime", content=f"[CHAT] {self._shown(agent_name)}: {spoken}", defer=True)
        return spoken

    def run_task(self, agent_name: str, task: str, max_steps: int = 8) -> str:
        if agent_name not in self.agents: #check if agent exists
            raise KeyError(f"this agent does not exist")
        obs = self.inbox[agent_name] #get the inbox of the agnet
        for _ in range(max_steps): #allow the gaent to use tool ans and to reason in a limited amount of time
            step = self.agents[agent_name].step(task, obs, allow_tools=False)

            if step.final is not None: #if it gives an outcome, broadcast it
                self.broadcast(
                    sender="runtime",
                    content=f"[RESULT] {self._shown(agent_name)} FINAL: {step.final}",
                    defer=False,
                )
                return step.final
            #if there is no final answer check if agent asked a tool and execute it
            tool_obs = self.fc.execute_from_text(step.action_text or "")
            obs.append(tool_obs) #add the tool to the agent observ
            #broadcast the tool result to other agents
            self.broadcast(
                sender="runtime",
                content=f"[RESULT] {self._shown(agent_name)} {tool_obs.source}: {tool_obs.content}",
                defer=False,
            )

        #if the max step is reached stop the task
        self.broadcast(sender="runtime", content=f"[RESULT] {self._shown(agent_name)} stopped (max_steps).", defer=False)
        return "stopped since the max steps is reached."


#run discussion where each person gives one answer, no tools
    def discussion_round(self, participants: List[str], topic: str) -> Dict[str, str]:
        discussion_task = (
            "DISCUSSION MODE (no tools). Reply with exactly one line starting with 'FINAL: '.\n"
            f"Topic: {topic}"
        )
        spoken_map: Dict[str, str] = {} #stores answer of each person
        for a in participants: #ask each paticipant to asnwer
            step = self.agents[a].step(discussion_task, self.inbox[a], allow_tools=False)
            raw = (step.final or "").strip() #get answer
            first = raw.splitlines()[0].strip() if raw.splitlines() else ""   #keep only first line
            if not first.upper().startswith("FINAL:"): #make sure the answer starts the same for evryone
                first = "FINAL: " + first
            spoken_map[a] = first #save the participant ans
            #convo is deffered to everuone gets all the answer at the end of the episode
            self.broadcast(sender="runtime", content=f"[DISCUSSION] {self._shown(a)}: {first}", defer=True)
        return spoken_map #return all participant response

