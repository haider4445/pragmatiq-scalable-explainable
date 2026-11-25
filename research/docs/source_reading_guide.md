# Upstream source reading guide — experiments 01–05

Read the original implementation alongside our experiments. Paths and approximate line anchors refer to the research branch as inspected before experiment 05 finished; use symbol names if lines shift. Upstream code is unchanged by this guide.

| Experiment | Source | Key symbols and questions |
|---|---|---|
| 01 Quickstart | [api.py](../../pragmatiq/api.py#L857) | quickstart, _resolve_pretrain, _pretrain_inner: how are the stages wired? |
| 02 Data | [schema.py](../../pragmatiq/core/schema.py#L130), [synthetic](../../pragmatiq/data/synthetic) | UserRecord, label_schema, WorldConfig, generate, World: what is an event versus a profile? |
| 03 Pipeline | [api.py](../../pragmatiq/api.py#L141), [probe.py](../../pragmatiq/training/probe.py#L55) | synthesize, tokenize, pretrain, embed, probe, embed_users, cutoffs_from_labels, EmbeddingProbe, RawCountBaseline: which steps repeat embedding extraction? |
| 04 Tokenizer | [tokenizer.py](../../pragmatiq/data/tokenizer.py#L127) | PercentileBinner, TokenizedRecord, fit, _finalize, encode, _encode_field, truncate_record, _train_bpe: how are field types inferred, tokens counted and cutoffs enforced? |
| 05 Storage | [sharding.py](../../pragmatiq/data/sharding.py#L140), [dataset.py](../../pragmatiq/data/dataset.py#L39) | band_of, ShardWriter.add, _flush_band, _write_index, UserIndex, ShardDataset._shard_table, get, get_many: how do we retrieve histories without scanning the dataset? |

## Priority reading (30 minutes)

1. api.quickstart (857) and api.tokenize (199).
2. tokenizer.PercentileBinner (127), PragmaTokenizer._finalize (421), encode (507), _encode_field (632).
3. sharding.band_of (140), ShardWriter.add (179), _flush_band (230), _write_index (269).
4. dataset.ShardDataset._shard_table (71), get (88), get_many (95).
5. Preview dataset.DynamicBatchSampler._plan (176), ShardDataLoader._make (295), collate.VarlenCollator.__call__ (195).

## Read in Colab without editing

```python
import inspect
from pragmatiq import api
from pragmatiq.data.tokenizer import PragmaTokenizer
from pragmatiq.data.dataset import ShardDataset
print(inspect.getsource(api.quickstart))
print(inspect.getsource(PragmaTokenizer._encode_field))
print(inspect.getsource(ShardDataset.get_many))
```

Keep notes on inputs, outputs, shapes, complexity, design decisions and remaining limitations. Separate verified upstream capabilities from proposed extensions. The sparse concept-bottleneck FM, IBM adaptation and matched NVIDIA comparison belong to the post-learning phase. Do not commit raw data, checkpoints or credentials.
