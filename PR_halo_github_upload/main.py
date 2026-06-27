import os  # read setting from einvironment
import json  # read json text from the result file
import time
import re
from pathlib import Path  # to work with file paths
from dotenv import load_dotenv  # to load the .env file
from openai import RateLimitError, APIStatusError

from core.tools import ToolRegistry
from agents.fc_caller import FcCallerAgent
from agents.react_agent import ReActAgent
from core.runtime import Runtime
from episode_runner import run_experiment

load_dotenv()

def clean_llm_output(text: str) -> str:
    t = (text or "").strip() #convert none to empty string and remove spaces
    while "<think>" in t.lower(): #keep removing hidden reasoning block if they get </think>
        low = t.lower() 
        start = low.find("<think>")
        end = low.find("</think>", start)
        if end != -1: #if there is remove
            t = (t[:start] + t[end + len("</think>"):]).strip()
            continue
        final_pos = low.find("final:") #if there is no closing tag find final: 
        if final_pos != -1: #if there is keep everything final onwards
            t = t[final_pos:].strip()
        else: #otherwhise remove everything
            t = t[:start].strip()
        break
    return t.strip()



load_dotenv()   # load the .env file

# The function is used to send a prompt to the selected llm
def llm_reply(prompt: str, allow_tools: bool = True) -> str:
    # see which backend should be used and make it lower case, ollama is the default model
    backend = os.getenv("LLM_BACKEND", "ollama").lower()

    if backend == "ollama": #if it is ollama
        from ollama import Client
        model = os.getenv("OLLAMA_MODEL", "phi3:mini") #if ollama is not set use phi3:mini
        client = Client(host="http://localhost:11434", timeout=float(os.getenv("OLLAMA_TIMEOUT", "900")))# tells where ollama runs and if it does not answer after tot, it gives up
        # low temperature -> low randomness and num_predictions-> number of tokens that can generate
        base_options = {"temperature": 0.2, "num_predict": 512}
        options = dict(base_options) #make copy if the base_options
        # send prompt to ollama
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}], #message sent to ollama with role and prompt
            options=options,
        )
        return resp["message"]["content"] #return the ollama answer


    if backend == "gemini":
        # import Gemini libraries
        from google import genai
        from google.genai import types
        #read the model from env, default is gemini 2.5 flash
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        client = genai.Client()
        # send the prompt to the llm
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0, #make 0 randomness
                max_output_tokens=int(os.getenv("API_MAX_TOKENS", "512")),
            ),
        )
        return resp.text

    if backend == "openai":
        # import opneai client, for groq is compatible this API
        from openai import OpenAI
        # get the API
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or None  #get api adress from env
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not existend, check env")
        client = OpenAI(api_key=api_key, base_url=base_url)  #the llm to chat
        messages = [{"role": "user", "content": prompt}]
        # Qwen3 models can yield visible <think>...</think> text, Tell them not 
        if "qwen" in model.lower():
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Do not output hidden reasoning. Do not output <think> tags. "
                        "Return only the requested final answer, JSON, or FINAL line."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
        max_retries = int(os.getenv("API_RATE_RETRIES", "8")) #max retried possible
        completion = None
        for attempt in range(max_retries + 1): 
            try:
                completion = client.chat.completions.create( # send prompt to model
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=int(os.getenv("API_MAX_TOKENS", "512")),
                )
                break
            except (RateLimitError, APIStatusError) as e:
                msg = str(e)  #convert error to sring to inspect
                # retry if it is a time limit or size limit
                if ("429" not in msg) and ("rate_limit" not in msg.lower()) and ("rate limit" not in msg.lower()):
                    raise
                if attempt >= max_retries: #if attempts are done then raise
                    raise
                wait_s = float(os.getenv("API_RATE_WAIT_SECONDS", "2.0")) #waiting time before retry
                m = re.search(r"try again in ([0-9.]+)\s*(ms|s)", msg, re.IGNORECASE) #see if error gives a suggestion for waiting time
                if m: #if ir contains then use that
                    val = float(m.group(1))
                    unit = m.group(2).lower()
                    wait_s = (val / 1000.0) if unit == "ms" else val #convert millisecond tos ec if needed
                wait_s = max(wait_s, 0.5) #make sure at least waiting is 0.5
                print(f"[api-retry] rate limit, sleeping {wait_s:.2f}s then retrying attempt {attempt + 1}/{max_retries}")
                time.sleep(wait_s)
        return clean_llm_output(completion.choices[0].message.content or "")
    raise ValueError(f"Unknown LLM_BACKEND={backend}")


# class to help the ReActAgents that has a method named .complete (they call this method and do not directly call llm_reply())
class EnvLLMClient:
    def complete(self, prompt: str, allow_tools: bool = True) -> str:
        return llm_reply(prompt, allow_tools=allow_tools) #send prompt to the backend and return the answer


def add(a: int, b: int) -> int:
    return a + b


def echo(text: str) -> str:
    return text


# builds the multi agent einv
def build_runtime(identity_mode: str = "neutral") -> Runtime:
    tools = ToolRegistry() # creates an empty tool registary where we can add the tools that the agents can use
    tools.register("add", add, "Add two integers. Args: {a:int, b:int}")
    tools.register("echo", echo, "Echo text. Args: {text:str}")

    fc = FcCallerAgent(name="fc_caller", tools=tools) #helps to correctly call tools
    llm = EnvLLMClient()
    #create the agents and the boss
    #during random phase the boss is not assigning task
    #during the phase of the boss, this agent reads history and gives the task
    agents = {
        "BOSS": ReActAgent(name="BOSS", llm_client=llm, tool_descriptions=tools.describe(),
                           identity_mode=identity_mode),
        "W1": ReActAgent(name="W1", llm_client=llm, tool_descriptions=tools.describe(), identity_mode=identity_mode),
        "W2": ReActAgent(name="W2", llm_client=llm, tool_descriptions=tools.describe(), identity_mode=identity_mode),
        "W3": ReActAgent(name="W3", llm_client=llm, tool_descriptions=tools.describe(), identity_mode=identity_mode),
        "W4": ReActAgent(name="W4", llm_client=llm, tool_descriptions=tools.describe(), identity_mode=identity_mode),
        "W5": ReActAgent(name="W5", llm_client=llm, tool_descriptions=tools.describe(), identity_mode=identity_mode),
    }
    #returns runtime that puts together the whole episode
    return Runtime(fc=fc, agents=agents)


def _safe(s: str) -> str: #makes a safe string taht can be used in the folder name
    #go through every character and replace character that are not letters, number, dots, underscores or hyphens with -
    return "".join(c if c.isalnum() or c in "._-" else "-" for c in (s or ""))


#  help to prepare the einvaronment according to the llm choosen
def configure_backend(name: str) -> tuple[str, str]:
    name = name.strip().lower() #make smaller case letters
    if name == "ollama":
        os.environ["LLM_BACKEND"] = "ollama" #tell to use ollama to llm.reply
        return "ollama", os.getenv("OLLAMA_MODEL", "phi3:mini")

    if name == "gemini":
        os.environ["LLM_BACKEND"] = "gemini" #tell to use gemini block
        return "gemini", os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if name in {"openai", "groq", "deepseek", "openrouter", "cloudflare"}:
        os.environ["LLM_BACKEND"] = "openai"
        #check which one we are using
        if name == "openai":
            os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            return "openai", os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if name == "cloudflare":
            acct = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
            if not acct:
                raise ValueError("CLOUDFLARE_ACCOUNT_ id missing in .env")
            os.environ["OPENAI_API_KEY"] = os.getenv("CLOUDFLARE_API_KEY", "")
            os.environ["OPENAI_MODEL"] = os.getenv("CLOUDFLARE_MODEL", "@cf/meta/llama-3.1-8b-instruct")
            os.environ["OPENAI_BASE_URL"] = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/v1"
            return "cloudflare", os.environ["OPENAI_MODEL"]
        # get it from env (groq uses the same as deepseek so they convert it to the same one
        prefix = name.upper()
        os.environ["OPENAI_API_KEY"] = os.getenv(f"{prefix}_API_KEY", "")
        os.environ["OPENAI_MODEL"] = os.getenv(f"{prefix}_MODEL", "")
        os.environ["OPENAI_BASE_URL"] = os.getenv(f"{prefix}_BASE_URL", "")
        #check if all values exists
        if not os.environ["OPENAI_API_KEY"] or not os.environ["OPENAI_MODEL"] or not os.environ["OPENAI_BASE_URL"]:
            raise ValueError(
                f"Missing one"
            )

        return name, os.environ["OPENAI_MODEL"] #return name and model for the name of the folder
    raise ValueError(f"Check provider") # if none of them check



# to read the evaluation file and keep the main values, used to give a summary at the end
def read_stage_eval(path: Path) -> dict:
    latest = {} #to store the lates result for each stage
    if not path.exists(): #if the stage_eval.json does not exist return an empty dictionary
        return latest
    #read the file as text and separets in lines
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): # continue if the line is empty
            continue
        rec = json.loads(line) #convert json text line to python dictionary
        latest[rec["stage"]] = { # stores the metrics and episonde number and stage that is either random or boss
            "episode_end": rec["episode_end"],
            "RSI": rec["rsi"],
            "GBC": rec["gbc"],
            "CAI": rec["cai"],
            "SII": rec["sii"],
        }
    return latest


def main():
    #find folder where is main and turns to absolute path
    root = Path(__file__).resolve().parent
    runs_root = root / "runs" #path for the results
    runs_root.mkdir(exist_ok=True) #if it doesnt exist it creates the path
    # read the backend that is used and if it is not set uses ollama and gemini
    providers = [p.strip() for p in os.getenv("RUN_BACKENDS", "ollama,gemini").split(",") if p.strip()]
    n_episodes = int(os.getenv("N_EPISODES", "10")) #total number episodes
    random_phase = int(os.getenv("RANDOM_PHASE_EPISODES", "5")) #how many are random episodes
    p0 = float(os.getenv("P0", "0.8")) #probability for the worker to succeed in the task
    identity_mode = os.getenv("IDENTITY_MODE", "neutral") #for the replica of the paper the gants are shown as neutral numbered of people
    summary = [] #store summary
    for prov in providers: #run for each llm
        backend_label, model = configure_backend(prov)
        # runs/<provider>/<model>_<condition>_<timestamp>/ history grouped by provider
        halo_mode = os.getenv("HALO_MODE", "none").strip().lower()
        halo_style = os.getenv("HALO_STYLE", "polite_structured").strip().lower()
        seed = os.getenv("SEED", "0").strip()
        control_label = "control" if os.getenv("ALL_RANDOM_CONTROL", "0").strip() == "1" else "boss"
        if halo_mode == "none":
            condition_label = f"neutral_{control_label}_seed{seed}"
        else:
            condition_label = f"halo-{halo_mode}_{halo_style}_{control_label}_seed{seed}"

        run_dir = runs_root / backend_label / f"{_safe(model)}_{condition_label}_{int(time.time())}" #crete a folder for the results of this run
        run_dir.mkdir(parents=True, exist_ok=True)
        run_config = {
            "provider": backend_label,
            "model": model,
            "n_episodes": n_episodes,
            "random_phase_episodes": random_phase,
            "p0": p0,
            "identity_mode": identity_mode,
            "seed": seed,
            "all_random_control": os.getenv("ALL_RANDOM_CONTROL", "0").strip(),
            "halo_mode": halo_mode,
            "halo_workers": os.getenv("HALO_WORKERS", "").strip(),
            "halo_style": halo_style,
            "condition_label": condition_label,
        }

        with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2, ensure_ascii=False)
        print("\n==============================")
        print("RUN:", backend_label)
        print("MODEL:", model)
        print("DIR:", run_dir)
        print("CONFIG:", run_dir / "run_config.json")
        print("==============================\n")
        #prepare the episode
        rt = build_runtime(identity_mode=identity_mode)
        run_experiment(
            rt,
            n_episodes=n_episodes,
            random_phase_episodes=random_phase,
            p0=p0,
            identity_mode=identity_mode,
            log_path="episodes.jsonl",
            out_dir=run_dir,
        )
        #read the final values
        stages = read_stage_eval(run_dir / "stage_eval.jsonl")
                #add the run to the summary
        summary.append({"provider": backend_label, "model": model, "stages": stages})
        #stop before going to the next provider for gemini or groq to avoid rate limit
        time.sleep(1)  

    print("\n=== SUMMARY (latest per stage) ===")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
