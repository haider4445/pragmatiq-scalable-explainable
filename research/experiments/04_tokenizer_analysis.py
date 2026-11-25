"""Learning experiment 04: inspect the existing key/value/time tokenizer.

Reuse a saved tokenizer; no model training or new tokenizer fitting required.
Synthetic data only. Store aggregate statistics, not raw customer records.
"""
import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from pragmatiq.data.tokenizer import PragmaTokenizer, iter_user_records, UNK


def run(root, out, max_users=25):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tok = PragmaTokenizer.load(root / "tok/tokenizer")
    keys = {v: k for k, v in tok.key_vocab.items()}
    event_lengths, user_lengths = [], []
    field_tokens, unknown_by_field = Counter(), Counter()
    n_events = n_tokens = n_unknown = n_profiles = 0
    shown = set()
    # Walk real synthetic histories using the same upstream encoder as training.
    for raw in iter_user_records(root / "raw", max_users=max_users):
        rec = tok.encode(raw)
        lengths = np.diff(rec.event_offsets)
        assert len(lengths) == rec.n_events
        assert rec.event_offsets[0] == 0 and rec.event_offsets[-1] == rec.n_tokens
        assert len(rec.key_ids) == len(rec.value_ids) == len(rec.positions)
        assert np.all(np.diff(rec.event_ts) >= 0)
        event_lengths.extend(lengths.tolist())
        user_lengths.append(rec.n_tokens)
        n_events += rec.n_events
        n_tokens += rec.n_tokens
        n_profiles += len(rec.prof_key_ids)
        n_unknown += int((rec.value_ids == UNK).sum())
        for kid, vid in zip(rec.key_ids, rec.value_ids):
            key = keys.get(int(kid), "[UNK]")
            field_tokens[key] += 1
            if vid == UNK:
                unknown_by_field[key] += 1
        # Display a few synthetic examples locally; never write raw fields to Git.
        for i, (_, source, fields) in enumerate(raw.events):
            if source in shown or len(shown) >= 4:
                continue
            shown.add(source)
            lo, hi = map(int, rec.event_offsets[i:i+2])
            print("\nSource:", source, "raw fields:", fields)
            print("Encoded (key, key_id, value_id, within-field position):")
            print([(keys.get(int(k), "[UNK]"), int(k), int(v), int(p))
                   for k,v,p in zip(rec.key_ids[lo:hi],rec.value_ids[lo:hi],rec.positions[lo:hi])])
            print("Decoded:", tok.decode_event(rec, i))
            print("Time log / calendar:", float(rec.time_log[i]), int(rec.hour[i]), int(rec.dow[i]), int(rec.dom[i]))
    if not n_events:
        raise ValueError("No events in sample")
    def summary(a):
        return {"min": int(min(a)), "mean": float(np.mean(a)), "p50": float(np.percentile(a,50)),
                "p95": float(np.percentile(a,95)), "max": int(max(a))}
    result = {
        "source_revision": subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "tokenizer_hash": tok.content_hash,
        "vocab_size": tok.vocab_size,
        "field_kind_counts": dict(Counter(tok.field_kind.values())),
        "field_kinds": tok.field_kind,
        "numeric_bucket_counts": {k:b.n_bins for k,b in tok.binners.items()},
        "sample": {"users":len(user_lengths),"events":n_events,"event_tokens":n_tokens,
                   "profile_tokens":n_profiles,"unknown_value_tokens":n_unknown,
                   "unknown_value_pct":100*n_unknown/n_tokens},
        "tokens_per_event": summary(event_lengths),
        "event_tokens_per_user": summary(user_lengths),
        "field_token_counts":dict(field_tokens),
        "unknown_by_field":dict(unknown_by_field),
        "config": {"max_event_tokens":tok.config.max_event_tokens,
                   "max_profile_tokens":tok.config.max_profile_tokens,
                   "max_events_per_user":tok.config.max_events_per_user,
                   "calendar_tz":tok.config.calendar_tz},
        "limitations":["First N user histories, not a random representative sample",
                       "Tokenizer fitted on same synthetic dataset; UNK rate is not out-of-sample",
                       "Event token counts exclude model-added special tokens; history cap applies in collator",
                       "No new architectural contribution; upstream replication"],
    }
    (out/"04_tokenizer_metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    return result

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",default="runs/research_decomposed")
    p.add_argument("--out",default="research/results")
    p.add_argument("--max-users",type=int,default=25)
    a=p.parse_args()
    run(a.root,a.out,a.max_users)
