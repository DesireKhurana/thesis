```markdown
# Halo Induced Stereotype Formation in LLM Based Multi Agent Systems

This repository contains the code for my bachelor thesis experiment.

The experiment is based on a workplace simulation with one supervisor agent, called BOSS, and five worker agents. All workers have the same task success probability. In the halo condition two workers use a polite and structured writing style. The goal is to test whether this writing style changes how agents assign tasks and rate each other.

The final experiment compares neutral and halo conditions with three model setups: DeepSeek, Groq and Qwen2.5 served through Ollama.

## Repository structure

The repository contains the main experiment code, the simulation code, the agent code, run scripts and requirement files.

## Main files

`main.py` starts the experiment. It selects the model setup and creates the agents.

`episode_runner.py` runs the episodes. It handles task assignment, halo instructions, task answers, sampled outcomes, discussion and saved outputs.

`smoke_test_backends.py` checks whether the selected model setup is working before a full run starts.

## Core code

The folder named core contains the main simulation logic.

`core/runtime.py` controls the simulation runtime, messages, discussion rounds and visible names such as person 1.

`core/tasks.py` defines the four task types and their prompts.

`core/evaluation.py` runs peer evaluation and calculates RSI, GBC, CAI and SII.

`core/boss_policy.py` contains helper code for task history and sampled task outcomes.

`core/messages.py` defines message and observation objects.

`core/tools.py` contains the tool interface from the earlier practical work setup. Tools were not used for final worker task answers.

`core/jobs.py` contains helper definitions for jobs and roles.

## Agent code

The folder named agents contains the agent code.

`agents/react_agent.py` defines the agent wrapper used for BOSS and the worker agents.

`agents/halo_profiles.py` contains halo style profile definitions used during development.

`agents/fc_caller.py` contains helper code for function calling support.

## Requirement files

The requirement files list the Python packages used in different environments. The main file is requirements.txt. Other files were used for the Linux server and Python 3.9 setups.

```
