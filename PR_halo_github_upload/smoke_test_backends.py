import os

def clean_llm_output(text: str) -> str:
    t = (text or "").strip()
    while "<think>" in t.lower():
        low = t.lower()
        start = low.find("<think>")
        end = low.find("</think>", start)
        if end != -1:
            t = (t[:start] + t[end + len("</think>"):]).strip()
            continue
        final_pos = low.find("final:")
        if final_pos != -1:
            t = t[final_pos:].strip()
        else:
            t = t[:start].strip()
        break
    return t.strip()


def test_ollama():
    from ollama import Client
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    c = Client(host=host, timeout=300)
    r = c.chat(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        options={"temperature": 0.0, "num_predict": 8},
    )
    print("[ollama]", r["message"]["content"].strip())

def test_openai_compatible(label="openai"):
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("GROQ_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")
    model = os.getenv("OPENAI_MODEL") or os.getenv("GROQ_MODEL") or os.getenv("DEEPSEEK_MODEL")
    if not api_key or not model:
        print(f"[{label}] missing API key or model env vars")
        return
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    messages = [{"role":"user","content":"Reply with exactly: OK"}]
    if model and "qwen" in model.lower():
        messages = [
            {"role": "system", "content": "Do not output hidden reasoning. Do not output <think> tags. Reply only with the requested text."},
            {"role":"user","content":"Reply with exactly: OK"},
        ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=64,
        temperature=0.0,
    )
    print(f"[{label}]", clean_llm_output(resp.choices[0].message.content))

def test_anthropic():
    import anthropic
    key = os.getenv("ANTHROPIC_API_KEY")
    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
    if not key:
        print("[anthropic] missing ANTHROPIC_API_KEY")
        return
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model=model,
        max_tokens=8,
        temperature=0.0,
        messages=[{"role":"user","content":"Reply with exactly: OK"}],
    )
    out = "".join([b.text for b in msg.content if hasattr(b, "text")]).strip()
    print("[anthropic]", out)

def test_gemini():
    from google import genai
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    if not key:
        print("[gemini] missing GEMINI_API_KEY (or GOOGLE_API_KEY)")
        return
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(model=model, contents="Reply with exactly: OK")
    print("[gemini]", (resp.text or "").strip())

if __name__ == "__main__":
    backend = os.getenv("BACKEND", "ollama").lower()
    if backend == "ollama":
        test_ollama()
    elif backend in {"openai", "groq", "deepseek"}:
        test_openai_compatible(label=backend)
    elif backend == "anthropic":
        test_anthropic()
    elif backend == "gemini":
        test_gemini()
    else:
        print("Unknown BACKEND. Use one of: ollama|openai|groq|deepseek|anthropic|gemini")
