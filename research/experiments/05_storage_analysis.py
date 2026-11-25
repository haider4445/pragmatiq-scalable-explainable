"""Experiment 05: inspect upstream Parquet shards, LMDB and lazy loading.

Read-only learning experiment. Timing is illustrative: filesystem and process
caches are not controlled. No new storage optimization is claimed.
"""
import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from pragmatiq.data.sharding import UserIndex, band_of
from pragmatiq.data.dataset import ShardDataset


def run(root, out, sample_size=32):
    root, out = Path(root), Path(out)
    shard_dir = root / "tok"
    manifest = json.loads((shard_dir / "shard_manifest.json").read_text())
    idx = UserIndex(shard_dir)
    assert len(idx) == manifest["n_users"]
    assert int(idx.n_tokens.sum()) == manifest["total_tokens"]
    assert int(idx.n_events.sum()) == manifest["total_events"]
    assert all(band_of(int(n), tuple(manifest["bands"])) == int(b)
               for n,b in zip(idx.n_events,idx.bands))
    shards=[]
    for p in sorted((shard_dir/"shards").glob("*.parquet")):
        f=pq.ParquetFile(p)
        shards.append({"name":p.name,"rows":f.metadata.num_rows,
                       "row_groups":f.metadata.num_row_groups,"bytes":p.stat().st_size})
    assert len(shards)==manifest["n_shards"]
    assert sum(s["rows"] for s in shards)==len(idx)
    bands=[]
    for b,upper in enumerate(manifest["bands"]):
        mask=idx.bands==b
        n=int(mask.sum())
        bands.append({"band":b,"upper_events":upper,"users":n,
                      "events":int(idx.n_events[mask].sum()),"tokens":int(idx.n_tokens[mask].sum()),
                      "min_events":int(idx.n_events[mask].min()) if n else None,
                      "max_events":int(idx.n_events[mask].max()) if n else None})
    # Sample deterministic users without replacement; compare identical outputs.
    rng=np.random.default_rng(42)
    uids=[idx.order[int(i)] for i in rng.choice(len(idx),min(sample_size,len(idx)),replace=False)]
    t=time.perf_counter()
    metas=idx.meta_many(uids)
    index_s=time.perf_counter()-t
    # Independent dataset caches; OS cache can be warmed by the first pass.
    single=ShardDataset(shard_dir,cache_bytes=256*1024**2)
    group=ShardDataset(shard_dir,cache_bytes=256*1024**2)
    t=time.perf_counter()
    records=[single.get(u) for u in uids]
    single_s=time.perf_counter()-t
    t=time.perf_counter()
    batched=group.get_many(uids)
    group_s=time.perf_counter()-t
    for a,b,m in zip(records,batched,metas):
        assert a.user_id==b.user_id==m.user_id
        assert a.n_events==m.n_events and a.n_tokens==m.n_tokens
        for name in ("key_ids","value_ids","positions","event_offsets","event_ts",
                     "time_log","hour","dow","dom","source_ids","prof_key_ids",
                     "prof_value_ids","prof_positions","prof_offsets","prof_time_log","prof_ts","is_text"):
            np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
        assert a.text_values==b.text_values
    single.close()
    group.close()
    idx.close()
    result={"source_revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            "manifest":manifest,"bands":bands,"shards":shards,
            "storage":{"parquet_bytes":sum(s["bytes"] for s in shards),
                       "lmdb_file_bytes":sum(p.stat().st_size for p in (shard_dir/"user_index.lmdb").glob("*") if p.is_file())},
            "sample":{"users":len(uids),"distinct_shards":len({(m.band,m.shard) for m in metas}),
                      "index_lookup_seconds":index_s,"individual_get_seconds":single_s,
                      "grouped_get_seconds":group_s,"array_equality_verified":True,
                      "cache_budget_bytes":256*1024**2},
            "limitations":["Single timing pass; OS cache and execution order confound comparison",
                           "Shard cache can retain one shard larger than its budget",
                           "LMDB file size is not virtual map size or resident memory",
                           "Synthetic data, no distributed benchmark"]}
    out.mkdir(parents=True,exist_ok=True)
    (out/"05_storage_metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",default="runs/research_decomposed")
    p.add_argument("--out",default="research/results")
    p.add_argument("--sample-size",type=int,default=32)
    a=p.parse_args()
    run(a.root,a.out,a.sample_size)
