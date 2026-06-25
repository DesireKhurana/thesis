```markdown
# Halo Induced Stereotype Formation in LLM Based Multi Agent Systems

This repository contains the code used for my bachelor thesis experiment. The project tests whether a polite and structured communication style can affect how LLM agents assign roles and rate other agents.

The experiment uses one supervisor agent called BOSS and five worker agents. The final runs compare neutral and halo conditions across DeepSeek, Groq and Qwen2.5 through Ollama.

## Main files

main.py  
Starts the experiment. It selects the model setup, creates the agents and loads the run settings.

episode_runner.py  
Runs the episode loop. It handles task assignment, halo instructions, task answers, sampled outcomes, discussion and saved results.

smoke_test_backends.py  
Checks whether the selected model setup can be reached before starting a full run.

summarize_all_thesis_results.py  
Summarizes the saved experiment results across models, seeds and conditions.

dummy.py  
Small test file from early development. It is not part of the final experiment.

## Core folder

core/runtime.py  
Controls the agents during the simulation. It manages messages, observations, discussion rounds and visible names such as person 1.

core/tasks.py  
Defines the four task types and their prompts.

core/evaluation.py  
Runs peer evaluation and calculates the main metrics: RSI, GBC, CAI and SII.

core/boss_policy.py  
Contains helper code for task history and sampled task outcomes.

core/messages.py  
Defines the message and observation objects used by the runtime.

core/tools.py  
Contains the tool interface from the earlier practical work setup. Tools were not used for final worker task answers.

core/jobs.py  
Contains helper definitions for jobs and roles.

core/__init__.py  
Marks the core folder as a Python package.

## Agents folder

agents/react_agent.py  
Defines the agent wrapper used for BOSS and worker agents.

agents/halo_profiles.py  
Stores halo style profile definitions used during development.

agents/fc_caller.py  
Contains helper code for function calling support.

agents/__init__.py  
Marks the agents folder as a Python package.

## Tools folder

tools/__init__.py  
Marks the tools folder as a Python package.

## Run scripts

run_deepseek_seed2_remaining_seed3.sh  
Runs selected DeepSeek experiment conditions.

run_missing_deepseek_groq_seed4.sh  
Runs missing DeepSeek and Groq seed 4 conditions.

run_groq_seed0.sh  
Runs Groq seed 0 conditions.

run_groq_seed1.sh  
Runs Groq seed 1 conditions.

run_groq_seed1_remaining.sh  
Runs remaining Groq seed 1 conditions.

run_groq_seed2.sh  
Runs Groq seed 2 conditions.

run_groq_seed3.sh  
Runs Groq seed 3 conditions.

run_ollama_5seeds_safe.sh  
Runs the Qwen2.5 setup through Ollama.

run_ollama_llama31_5seeds.sh  
Runs an additional local model test setup.

run_llama31_5seeds_overnight.sh  
Runs overnight local model tests.

## Requirements files

requirements.txt  
Main dependency file.

requirements_linux.txt  
Dependency file for the Linux server.

requirements_py39.txt  
Dependency file for Python 3.9.

requirements_py39_clean.txt  
Shorter cleaned dependency file for Python 3.9.

requirements_server.txt  
Minimal dependency file for the server.

## Not included

The repository does not include API keys, environment files, virtual environments, run logs or generated result folders.
```
