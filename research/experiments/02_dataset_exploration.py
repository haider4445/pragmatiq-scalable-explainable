"""Measure event-length padding waste on the synthetic quickstart dataset.

This is a CPU simulation, not a GPU benchmark. Sorted batches are an idealized
reference, not the exact Pragmatiq event-band sampler. Attention waste counts
dense bidirectional score positions, excluding special tokens and other layers.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def batching_metrics(lengths, batch_size=32):
    useful = allocated = useful_pairs = allocated_pairs = 0
    for start in range(0, len(lengths), batch_size):
        batch = np.asarray(lengths[start:start + batch_size], dtype=np.int64)
        longest = int(batch.max())
        useful += int(batch.sum())
        allocated += len(batch) * longest
        useful_pairs += int((batch ** 2).sum())
        allocated_pairs += len(batch) * longest ** 2
    if not allocated:
        raise ValueError("No nonempty histories")
    return {
        "useful_positions": useful,
        "allocated_positions": allocated,
        "padding_waste_pct": 100 * (1 - useful / allocated),
        "useful_attention_pairs": useful_pairs,
        "allocated_attention_pairs": allocated_pairs,
        "attention_waste_pct": 100 * (1 - useful_pairs / allocated_pairs),
    }


def run(root, out, batch_size=32, seed=42):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Read only needed columns to reduce CPU memory pressure.
    events = pd.read_parquet(root / "raw/events.parquet", columns=["user_id", "source"])
    profiles = pd.read_parquet(root / "raw/profiles.parquet", columns=["user_id"])
    lengths = events.groupby("user_id").size().to_numpy()
    if len(lengths) == 0 or batch_size < 1:
        raise ValueError("Need nonempty data and positive batch size")
    rng = np.random.default_rng(seed)
    stats = pd.Series(lengths).describe(percentiles=[.25,.5,.75,.9,.95,.99])
    random_metrics = batching_metrics(rng.permutation(lengths), batch_size)
    sorted_metrics = batching_metrics(np.sort(lengths), batch_size)
    label_schemas = {}
    import pyarrow.parquet as pq
    for p in sorted((root / "raw/labels").glob("*.parquet")):
        label_schemas[p.name] = str(pq.read_schema(p))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result = {
        "experiment": "02_dataset_exploration",
        "source_revision": revision,
        "seed": seed, "batch_size": batch_size,
        "dataset": {"events": len(events), "profiles": len(profiles), "customers_with_events": len(lengths)},
        "sequence_length": stats.to_dict(),
        "event_counts": events["source"].value_counts().to_dict(),
        "label_schemas": label_schemas,
        "random_batching": random_metrics,
        "length_sorted_batching": sorted_metrics,
        "limitations": ["Event positions only; not token positions or measured GPU FLOPs", "One seeded permutation", "Globally sorted idealized baseline; not actual event-count sharding"],
    }
    (out / "02_dataset_stats.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(lengths, bins=50)
    ax.set(xlabel="Events per customer", ylabel="Customers", title="Customer history length distribution")
    fig.tight_layout()
    fig.savefig(out / "02_sequence_distribution.png", dpi=150)
    plt.close(fig)
    print(json.dumps(result, indent=2, default=float))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="runs/research_baseline")
    p.add_argument("--out", default="research/results")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    run(a.root, a.out, a.batch_size, a.seed)
