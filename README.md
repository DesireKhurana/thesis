```markdown
# Halo Induced Stereotype Formation in LLM Based Multi Agent Systems

This repository contains the code for my bachelor thesis experiment. The experiment studies whether polite and structured communication can change how LLM agents assign tasks and rate other agents.

The simulation uses one supervisor agent, called BOSS, and five worker agents. All workers have the same task success probability. In the halo condition, two workers use a polite and structured writing style. The final experiment compares neutral and halo conditions across DeepSeek, Groq and Qwen2.5 through Ollama.

## Repository structure

The project is organized into the main experiment files, the core simulation code, the agent code, run scripts and requirement files.

## Main experiment files

| File | Purpose |
|---|---|
| `main.py` | Starts the experiment, selects the model setup, creates the agents and loads the run settings. |
| `episode_runner.py` | Runs the episode loop, including task assignment, halo instructions, task answers, sampled outcomes, discussion and saved outputs. |
| `smoke_test_backends.py` | Checks whether the selected model setup is reachable before a full run is started. |
| `summarize_all_thesis_results.py` | Summarizes the saved results across model setups, seeds and conditions. |
| `dummy.py` | Small test file from early development. It is not part of the final experiment. |

## Core code

| File | Purpose |
|---|---|
| `core/runtime.py` | Controls the simulation runtime, including messages, observations, discussion rounds and visible names such as `person 1`. |
| `core/tasks.py` | Defines the four task types and their prompts. |
| `core/evaluation.py` | Runs peer evaluation and calculates RSI, GBC, CAI and SII. |
| `core/boss_policy.py` | Contains helper code for task history and sampled task outcomes. |
| `core/messages.py` | Defines the message and observation objects used during the simulation. |
| `core/tools.py` | Contains the tool interface from the earlier practical work setup. Tools were not used for final worker task answers. |
| `core/jobs.py` | Contains helper definitions for jobs and roles. |
| `core/__init__.py` | Marks the `core` folder as a Python package. |

## Agent code

| File | Purpose |
|---|---|
| `agents/react_agent.py` | Defines the agent wrapper used for BOSS and the worker agents. |
| `agents/halo_profiles.py` | Stores halo style profile definitions used during development. |
| `agents/fc_caller.py` | Contains helper code for function calling support. |
| `agents/__init__.py` | Marks the `agents` folder as a Python package. |

## Tools folder

| File | Purpose |
|---|---|
| `tools/__init__.py` | Marks the `tools` folder as a Python package. |

## Run scripts

The shell scripts were used to run different parts of the final experiment on the server.

| File | Purpose |
|---|---|
| `run_deepseek_seed2_remaining_seed3.sh` | Runs selected DeepSeek conditions. |
| `run_missing_deepseek_groq_seed4.sh` | Runs missing DeepSeek and Groq seed 4 conditions. |
| `run_groq_seed0.sh` | Runs Groq seed 0 conditions. |
| `run_groq_seed1.sh` | Runs Groq seed 1 conditions. |
| `run_groq_seed1_remaining.sh` | Runs remaining Groq seed 1 conditions. |
| `run_groq_seed2.sh` | Runs Groq seed 2 conditions. |
| `run_groq_seed3.sh` | Runs Groq seed 3 conditions. |
| `run_ollama_5seeds_safe.sh` | Runs the Qwen2.5 setup through Ollama. |
| `run_ollama_llama31_5seeds.sh` | Runs an additional local model test setup. |
| `run_llama31_5seeds_overnight.sh` | Runs overnight local model tests. |

## Requirements files

| File | Purpose |
|---|---|
| `requirements.txt` | Main dependency file. |
| `requirements_linux.txt` | Dependency file for the Linux server. |
| `requirements_py39.txt` | Dependency file for Python 3.9. |
| `requirements_py39_clean.txt` | Shorter cleaned dependency file for Python 3.9. |
| `requirements_server.txt` | Minimal dependency file for the server. |

## Files not included

The repository does not include API keys, `.env` files, virtual environments, run logs or generated result folders. These files were excluded because they are either private or generated during experiment runs.
```
