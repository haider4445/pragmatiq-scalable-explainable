# Pragmatiq research roadmap

Independent research extension of [Pragmatiq](https://github.com/dynamiq-ai/pragmatiq), inspired by [Revolut PRAGMA](https://arxiv.org/abs/2604.08649). Compare with the [NVIDIA transaction foundation model](https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model) on IBM TabFormer synthetic data.

## Six-hour hands-on checklist

Each task is timeboxed to 15 minutes; complex extensions are smoke tests, not validated production implementations.

### Hour 1 — Baseline

- [ ] 01. Install editable training package in Colab; record GPU, Python, Torch and Git SHA.
- [ ] 02. Run nano quickstart on synthetic data; save checkpoint and probe metrics.
- [ ] 03. Inspect events, profiles, labels and timestamps.
- [ ] 04. Run generation, tokenization, pretraining and evaluation separately.

### Hour 2 — Data and efficiency

- [ ] 05. Trace field-aware tokenizer and inspect key/value IDs.
- [ ] 06. Inspect event-count shards, offsets and LMDB index.
- [ ] 07. Compare fixed-user vs length-bucketed vs token-budget batching.
- [ ] 08. Inspect packed sequences and attention boundaries; record actual attention backend.

### Hour 3 — Architecture

- [ ] 09. Trace profile and event encoders using forward hooks.
- [ ] 10. Trace history encoder, TimeRoPE, residuals and attention.
- [ ] 11. Inspect token/field/event masking and MLM loss.
- [ ] 12. Save small-model baseline metrics and checkpoint.

### Hour 4 — Adaptation and scaling

- [ ] 13. Run native LoRA; measure trainable parameters.
- [ ] 14. Prototype full supervised fine-tuning and verify gradients.
- [ ] 15. Prototype 12-horizon charge-off hazard head using synthetic targets and censoring masks.
- [ ] 16. Prototype compute-aware sampler; compare batch cost variance.

### Hour 5 — Explainability and inference

- [ ] 17. Extract and store versioned offline embeddings.
- [ ] 18. Prototype sparse autoencoder; record reconstruction and sparsity.
- [ ] 19. Prototype integrated gradients on a differentiable risk head.
- [ ] 20. Test independent event embedding cache against full recomputation.

### Hour 6 — IBM and NVIDIA

- [ ] 21. Inspect IBM TabFormer subset and temporal split with positive fraud labels.
- [ ] 22. Convert IBM to Pragmatiq schema; keep labels separate.
- [ ] 23. Fit tokenizer on training histories and smoke-test MLM.
- [ ] 24. Prepare matched encoder/decoder comparison on the same temporal splits; record limitations.

## Follow-up research

- Distributed preprocessing, customer-hash/time partitions, memory-mapped shards and FSDP2.
- Full SFT vs LoRA, multi-task fraud and credit risk, temporal leakage tests.
- SAE concept validation, integrated gradients and residual analysis.
- Event embedding caching, versioned features, Triton and quantization.
- Matched-compute NVIDIA comparison; report transactions/sec, useful tokens/sec, VRAM, AUPRC, AUROC, calibration and latency.

## Commit discipline

Keep upstream source and license intact until a deliberate extension is needed. Place learning notebooks, extension prototypes and small aggregate metrics under `research/`. Commit after each verified milestone; record environment, seed, source revision and actual metrics. Do not commit credentials, proprietary data, raw datasets, model weights or generated customer records. Keep large artifacts outside Git. No benchmark result is claimed until measured.
