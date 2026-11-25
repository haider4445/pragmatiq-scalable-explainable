"""Experiment 03: time each stage of the encoder pipeline independently.

A short nano smoke test, not a throughput benchmark or proof of convergence.
Keep generated data/checkpoints outside Git; export only aggregate metrics.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from pragmatiq import api


def run(root, out, n_users=1000, steps=30, seed=42):
    root, out = Path(root), Path(out)
    if root.exists():
        raise FileExistsError(f"{root} exists; choose a fresh --root to avoid overwriting runs")
    root.mkdir(parents=True)
    out.mkdir(parents=True, exist_ok=True)
    times = {}
    def stage(name, fn):
        print(f"\n--- {name} ---", flush=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times[name] = time.perf_counter() - start
        print(f"{name}: {times[name]:.2f}s", flush=True)
        return result

    # Match quickstart's settings; isolate costs rather than change the model.
    raw, tok = root / "raw", root / "tok"
    stage("synthesize", lambda: api.synthesize(
        {"n_users": n_users, "seed": seed}, out=raw, n_workers=0, write_report=False))
    stage("tokenize", lambda: api.tokenize(raw, tok,
        config={"target_vocab": 28000, "n_buckets": 64}, n_workers=0))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train = stage("pretrain", lambda: api.pretrain(tok, "decomposed",
        model_size="nano", config={"max_steps": steps, "token_budget": 8192,
        "warmup_steps": max(10, steps // 10), "devices": 1}, runs_root=root / "runs"))
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    run_dir = train["run_dir"]
    # Full-history embeddings are for inspecting representations, NOT for risk scoring.
    emb = stage("embed_full_history", lambda: api.embed(tok, run_dir, out=root / "embeddings.parquet"))
    # The probe independently extracts cutoff-truncated embeddings; its runtime includes that work.
    probe = stage("probe_at_eval_cutoff", lambda: api.probe(tok, run_dir,
        raw / "labels/default_12m.parquet"))
    import pyarrow.parquet as pq
    raw_events = pq.ParquetFile(raw / "events.parquet").metadata.num_rows
    results = {
        "source_revision": subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
        "config": {"n_users": n_users, "steps": steps, "data_seed": seed, "token_budget": 8192,
                   "model_size": "nano", "flash_disabled": os.environ.get("PRAGMATIQ_DISABLE_FLASH")},
        "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "raw_events": raw_events, "stage_seconds": times, "total_stage_seconds": sum(times.values()),
        "pretrain_peak_allocated_bytes": peak,
        "embedding_summary": emb, "probe": probe,
        "limitations": ["Single run, no warmup-controlled benchmark",
                        "Probe includes separate cutoff-aware embedding extraction",
                        "Full-history embeddings not used as risk features",
                        "Probe cutoff alone does not establish leakage-free pretraining or tokenizer fitting",
                        "Peak allocated memory excludes reserved memory and other processes"],
    }
    (out / "03_pipeline_metrics.json").write_text(json.dumps(results, indent=2, default=str)+"\n")
    print(json.dumps(results, indent=2, default=str))
    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="runs/research_decomposed")
    p.add_argument("--out", default="research/results")
    p.add_argument("--n-users", type=int, default=1000)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    run(a.root, a.out, a.n_users, a.steps, a.seed)
