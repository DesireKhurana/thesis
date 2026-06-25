import os
import json
import time
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import RateLimitError, APIStatusError

from core.tools import ToolRegistry
from agents.fc_caller import FcCallerAgent
from agents.react_agent import ReActAgent
from core.runtime import Runtime
from episode_runner import run_experiment

load_dotenv()

def clean_llm_output(text: str) -> str:
    """Remove visible reasoning tags that some models emit, such as <think>...</think>."""
    t = (text or "").strip()

    while "<think>" in t.lower():
        low = t.lower()
        start = low.find("<think>")
        end = low.find("</think>", start)

        if end != -1:
            t = (t[:start] + t[end + len("</think>"):]).strip()
            continue

        # If the model started a think block but never closed it, keep only any FINAL part if present.
        final_pos = low.find("final:")
        if final_pos != -1:
            t = t[final_pos:].strip()
        else:
            t = t[:start].strip()
        break

    return t.strip()



def llm_reply(prompt: str, allow_tools: bool = True) -> str:
    backend = os.getenv("LLM_BACKEND", "ollama").lower()

    if backend == "ollama":
        from ollama import Client

        model = os.getenv("OLLAMA_MODEL", "phi3:mini")
        client = Client(host="http://localhost:11434", timeout=float(os.getenv("OLLAMA_TIMEOUT", "900")))

        base_options = {"temperature": 0.2, "num_predict": 512}
        options = dict(base_options)
       # if not allow_tools:
            #options["stop"] = ["\n"]

        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options=options,
        )
        return resp["message"]["content"]

    if backend == "gemini":
        # Make sure GOOGLE_API_KEY is set in your .env
        from google import genai
        from google.genai import types

        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        client = genai.Client()

        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=int(os.getenv("API_MAX_TOKENS", "512")),
            ),
        )
        return resp.text

    if backend == "openai":
        # OpenAI-compatible: OpenAI, Groq, DeepSeek, OpenRouter, Cloudflare, etc.
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or None
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing (check your .env)")

        client = OpenAI(api_key=api_key, base_url=base_url)

        messages = [{"role": "user", "content": prompt}]

        # Qwen3 models can emit visible <think>...</think> text.
        # Tell them not to, and still clean the output defensively.
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

        max_retries = int(os.getenv("API_RATE_RETRIES", "8"))
        completion = None

        for attempt in range(max_retries + 1):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=int(os.getenv("API_MAX_TOKENS", "512")),
                )
                break
            except (RateLimitError, APIStatusError) as e:
                msg = str(e)

                # Retry only rate/request-size style transient limits.
                if ("429" not in msg) and ("rate_limit" not in msg.lower()) and ("rate limit" not in msg.lower()):
                    raise

                if attempt >= max_retries:
                    raise

                wait_s = float(os.getenv("API_RATE_WAIT_SECONDS", "2.0"))
                m = re.search(r"try again in ([0-9.]+)\s*(ms|s)", msg, re.IGNORECASE)
                if m:
                    val = float(m.group(1))
                    unit = m.group(2).lower()
                    wait_s = (val / 1000.0) if unit == "ms" else val

                wait_s = max(wait_s, 0.5)
                print(f"[api-retry] rate limit, sleeping {wait_s:.2f}s then retrying attempt {attempt + 1}/{max_retries}")
                time.sleep(wait_s)

        return clean_llm_output(completion.choices[0].message.content or "")

    raise ValueError(f"Unknown LLM_BACKEND={backend}")


class EnvLLMClient:
    def complete(self, prompt: str, allow_tools: bool = True) -> str:
        return llm_reply(prompt, allow_tools=allow_tools)


def add(a: int, b: int) -> int:
    return a + b


def echo(text: str) -> str:
    return text


def build_runtime(identity_mode: str = "neutral") -> Runtime:
    tools = ToolRegistry()
    tools.register("add", add, "Add two integers. Args: {a:int, b:int}")
    tools.register("echo", echo, "Echo text. Args: {text:str}")

    fc = FcCallerAgent(name="fc_caller", tools=tools)
    llm = EnvLLMClient()

    agents = {
        "BOSS": ReActAgent(name="BOSS", llm_client=llm, tool_descriptions=tools.describe(),
                           identity_mode=identity_mode),
        "W1": ReActAgent(name="W1", llm_client=llm, tool_descriptions=tools.describe(), identity_mode=identity_mode),
        "W2": ReActAgent(name="W2", llm_client=llm, tool_descriptions=tools.describe(), identity_mode=identity_mode),
        "W3": ReActAgent(name="W3", llm_client=llm, tool_descriptions=tools.describe(), identity_mode=identity_mode),
        "W4": ReActAgent(name="W4", llm_client=llm, tool_descriptions=tools.describe(), identity_mode=identity_mode),
        "W5": ReActAgent(name="W5", llm_client=llm, tool_descriptions=tools.describe(), identity_mode=identity_mode),
    }

    return Runtime(fc=fc, agents=agents)


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "-" for c in (s or ""))


def configure_backend(name: str) -> tuple[str, str]:
    """
    Sets env vars that llm_reply uses.
    Returns (backend_label, model_label)
    """
    name = name.strip().lower()

    if name == "ollama":
        os.environ["LLM_BACKEND"] = "ollama"
        return "ollama", os.getenv("OLLAMA_MODEL", "phi3:mini")

    if name == "gemini":
        os.environ["LLM_BACKEND"] = "gemini"
        return "gemini", os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # OpenAI-compatible providers (and OpenAI itself)
    if name in {"openai", "groq", "deepseek", "openrouter", "cloudflare"}:
        os.environ["LLM_BACKEND"] = "openai"

        if name == "openai":
            os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            # expects OPENAI_API_KEY and optionally OPENAI_MODEL already set in .env
            return "openai", os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        if name == "cloudflare":
            acct = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
            if not acct:
                raise ValueError("CLOUDFLARE_ACCOUNT_ID missing in .env")

            os.environ["OPENAI_API_KEY"] = os.getenv("CLOUDFLARE_API_KEY", "")
            os.environ["OPENAI_MODEL"] = os.getenv("CLOUDFLARE_MODEL", "@cf/meta/llama-3.1-8b-instruct")
            os.environ["OPENAI_BASE_URL"] = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/v1"

            return "cloudflare", os.environ["OPENAI_MODEL"]

        # groq/deepseek/openrouter: expects GROQ_API_KEY/GROQ_MODEL/GROQ_BASE_URL etc.
        prefix = name.upper()
        os.environ["OPENAI_API_KEY"] = os.getenv(f"{prefix}_API_KEY", "")
        os.environ["OPENAI_MODEL"] = os.getenv(f"{prefix}_MODEL", "")
        os.environ["OPENAI_BASE_URL"] = os.getenv(f"{prefix}_BASE_URL", "")

        if not os.environ["OPENAI_API_KEY"] or not os.environ["OPENAI_MODEL"] or not os.environ["OPENAI_BASE_URL"]:
            raise ValueError(
                f"Missing one of {prefix}_API_KEY / {prefix}_MODEL / {prefix}_BASE_URL in .env"
            )

        return name, os.environ["OPENAI_MODEL"]

    raise ValueError(f"Unknown provider '{name}'")


def read_stage_eval(path: Path) -> dict:
    """
    stage_eval.jsonl has multiple lines. We keep the latest line per stage.
    """
    latest = {}
    if not path.exists():
        return latest

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        latest[rec["stage"]] = {
            "episode_end": rec["episode_end"],
            "RSI": rec["rsi"],
            "GBC": rec["gbc"],
            "CAI": rec["cai"],
            "SII": rec["sii"],
        }
    return latest


def main():
    root = Path(__file__).resolve().parent
    runs_root = root / "runs"
    runs_root.mkdir(exist_ok=True)

    providers = [p.strip() for p in os.getenv("RUN_BACKENDS", "ollama,gemini").split(",") if p.strip()]
    n_episodes = int(os.getenv("N_EPISODES", "10"))
    random_phase = int(os.getenv("RANDOM_PHASE_EPISODES", "5"))
    p0 = float(os.getenv("P0", "0.8"))
    identity_mode = os.getenv("IDENTITY_MODE", "neutral")
    summary = []

    for prov in providers:
        backend_label, model = configure_backend(prov)

        # HISTORY GROUPED BY PROVIDER:
        # runs/<provider>/<model>_<condition>_<timestamp>/
        halo_mode = os.getenv("HALO_MODE", "none").strip().lower()
        halo_style = os.getenv("HALO_STYLE", "polite_structured").strip().lower()
        seed = os.getenv("SEED", "0").strip()
        control_label = "control" if os.getenv("ALL_RANDOM_CONTROL", "0").strip() == "1" else "boss"

        if halo_mode == "none":
            condition_label = f"neutral_{control_label}_seed{seed}"
        else:
            condition_label = f"halo-{halo_mode}_{halo_style}_{control_label}_seed{seed}"

        run_dir = runs_root / backend_label / f"{_safe(model)}_{condition_label}_{int(time.time())}"
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

        rt = build_runtime(identity_mode=identity_mode)
        # Uses your updated episode_runner (out_dir supported)
        run_experiment(
            rt,
            n_episodes=n_episodes,
            random_phase_episodes=random_phase,
            p0=p0,
            identity_mode=identity_mode,
            log_path="episodes.jsonl",
            out_dir=run_dir,
        )

        stages = read_stage_eval(run_dir / "stage_eval.jsonl")
        summary.append({"provider": backend_label, "model": model, "stages": stages})

        time.sleep(1)  # small pause for rate limits

    print("\n=== SUMMARY (latest per stage) ===")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
