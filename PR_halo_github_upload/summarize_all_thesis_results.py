import json
import csv
from pathlib import Path
from statistics import mean, pstdev

PROVIDERS = ["deepseek", "groq", "ollama"]
METRICS = ["rsi", "gbc", "cai", "sii"]

def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def latest_stage(stage_rows, stage_name):
    rows = [r for r in stage_rows if r.get("stage") == stage_name]
    return rows[-1] if rows else None

all_runs = []

for provider in PROVIDERS:
    root = Path("runs") / provider
    if not root.exists():
        continue

    for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        cfg_path = run_dir / "run_config.json"
        stage_path = run_dir / "stage_eval.jsonl"
        episodes_path = run_dir / "episodes.jsonl"

        if not cfg_path.exists() or not stage_path.exists() or not episodes_path.exists():
            continue

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        # Keep only real thesis runs, not 2-episode tests.
        if int(cfg.get("n_episodes", 0)) != 30:
            continue
        if int(cfg.get("random_phase_episodes", 0)) != 15:
            continue
        if cfg.get("identity_mode") != "neutral":
            continue

        episodes = [line for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(episodes) != 30:
            print("SKIP incomplete episode count:", run_dir, "episodes=", len(episodes))
            continue

        stage_rows = read_jsonl(stage_path)
        random_stage = latest_stage(stage_rows, "random")

        is_control = str(cfg.get("all_random_control", "0")) == "1"
        final_stage_name = "random_control" if is_control else "boss"
        final_stage = latest_stage(stage_rows, final_stage_name)

        if random_stage is None or final_stage is None:
            print("SKIP missing stages:", run_dir)
            continue

        halo_mode = cfg.get("halo_mode", "none")
        condition_type = "halo" if halo_mode != "none" else "neutral"
        strategy = "control" if is_control else "boss"
        seed = str(cfg.get("seed"))

        row = {
            "provider": provider,
            "model": cfg.get("model"),
            "seed": seed,
            "condition_type": condition_type,
            "strategy": strategy,
            "run_dir": str(run_dir),
        }

        for m in METRICS:
            row[f"random_{m.upper()}"] = float(random_stage[m])
            row[f"final_{m.upper()}"] = float(final_stage[m])
            row[f"delta_{m.upper()}"] = float(final_stage[m]) - float(random_stage[m])

        all_runs.append(row)

# Deduplicate: keep newest folder for same provider/model/seed/condition/strategy.
dedup = {}
for row in all_runs:
    key = (
        row["provider"],
        row["model"],
        row["seed"],
        row["condition_type"],
        row["strategy"],
    )
    dedup[key] = row

all_runs = list(dedup.values())

per_run_path = Path("results/thesis_all_models_30ep_per_run_deltas.csv")
with open(per_run_path, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "provider", "model", "seed", "condition_type", "strategy",
        "random_RSI", "final_RSI", "delta_RSI",
        "random_GBC", "final_GBC", "delta_GBC",
        "random_CAI", "final_CAI", "delta_CAI",
        "random_SII", "final_SII", "delta_SII",
        "run_dir",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in sorted(all_runs, key=lambda r: (r["provider"], int(r["seed"]), r["condition_type"], r["strategy"])):
        writer.writerow(row)

paired = []

for provider in PROVIDERS:
    seeds = sorted({r["seed"] for r in all_runs if r["provider"] == provider}, key=int)

    for seed in seeds:
        for condition_type in ["neutral", "halo"]:
            boss = [
                r for r in all_runs
                if r["provider"] == provider
                and r["seed"] == seed
                and r["condition_type"] == condition_type
                and r["strategy"] == "boss"
            ]
            control = [
                r for r in all_runs
                if r["provider"] == provider
                and r["seed"] == seed
                and r["condition_type"] == condition_type
                and r["strategy"] == "control"
            ]

            if not boss or not control:
                continue

            boss = boss[-1]
            control = control[-1]

            row = {
                "provider": provider,
                "model": boss["model"],
                "seed": seed,
                "condition_type": condition_type,
                "boss_run_dir": boss["run_dir"],
                "control_run_dir": control["run_dir"],
            }

            for m in ["RSI", "GBC", "CAI", "SII"]:
                row[f"boss_delta_{m}"] = boss[f"delta_{m}"]
                row[f"control_delta_{m}"] = control[f"delta_{m}"]
                row[f"boss_advantage_{m}"] = boss[f"delta_{m}"] - control[f"delta_{m}"]

            paired.append(row)

paired_path = Path("results/thesis_all_models_30ep_boss_advantages.csv")
with open(paired_path, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "provider", "model", "seed", "condition_type",
        "boss_delta_RSI", "control_delta_RSI", "boss_advantage_RSI",
        "boss_delta_GBC", "control_delta_GBC", "boss_advantage_GBC",
        "boss_delta_CAI", "control_delta_CAI", "boss_advantage_CAI",
        "boss_delta_SII", "control_delta_SII", "boss_advantage_SII",
        "boss_run_dir", "control_run_dir",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in sorted(paired, key=lambda r: (r["provider"], r["condition_type"], int(r["seed"]))):
        writer.writerow(row)

print("\n=== RUN COUNT CHECK ===")
for provider in PROVIDERS:
    rows = [r for r in all_runs if r["provider"] == provider]
    print(f"{provider}: {len(rows)} valid 30-episode runs found")

    for condition_type in ["neutral", "halo"]:
        for strategy in ["boss", "control"]:
            sub = [
                r for r in rows
                if r["condition_type"] == condition_type and r["strategy"] == strategy
            ]
            seeds = sorted([r["seed"] for r in sub], key=int)
            print(f"  {condition_type:7s} {strategy:7s}: {len(sub)} runs, seeds={seeds}")

print("\n=== BOSS ADVANTAGE BY SEED ===")
for row in sorted(paired, key=lambda r: (r["provider"], r["condition_type"], int(r["seed"]))):
    print(
        f"{row['provider']:8s} seed {row['seed']} {row['condition_type']:7s} | "
        f"RSI={row['boss_advantage_RSI']:+.3f} "
        f"GBC={row['boss_advantage_GBC']:+.3f} "
        f"CAI={row['boss_advantage_CAI']:+.3f} "
        f"SII={row['boss_advantage_SII']:+.3f}"
    )

print("\n=== MEAN BOSS ADVANTAGE ===")
for provider in PROVIDERS:
    for condition_type in ["neutral", "halo"]:
        sub = [r for r in paired if r["provider"] == provider and r["condition_type"] == condition_type]
        if not sub:
            continue

        print(f"\n{provider} {condition_type}: n={len(sub)} seeds")
        for m in ["RSI", "GBC", "CAI", "SII"]:
            vals = [r[f"boss_advantage_{m}"] for r in sub]
            pos = sum(1 for v in vals if v > 0)
            sd = pstdev(vals) if len(vals) > 1 else 0.0
            print(f"  {m}: mean={mean(vals):+.3f}, std={sd:.3f}, positive={pos}/{len(vals)}")

print("\n=== HALO MINUS NEUTRAL MEAN ADVANTAGE ===")
for provider in PROVIDERS:
    print(f"\n{provider}:")
    for m in ["RSI", "GBC", "CAI", "SII"]:
        neutral = [
            r[f"boss_advantage_{m}"] for r in paired
            if r["provider"] == provider and r["condition_type"] == "neutral"
        ]
        halo = [
            r[f"boss_advantage_{m}"] for r in paired
            if r["provider"] == provider and r["condition_type"] == "halo"
        ]
        if neutral and halo:
            print(f"  {m}: halo_mean - neutral_mean = {mean(halo) - mean(neutral):+.3f}")

print("\nSaved:")
print(per_run_path)
print(paired_path)
