# Next six-hour session: learning through research extensions

## Scope
Experiments 01–05 completed. Preserve their baseline and upstream implementation. Tomorrow is a prototype session, not a publication-quality benchmark. Primary candidate: jointly pretrained hybrid sparse financial concept bottleneck; secondary candidate: hierarchy-aware batching. IBM integration and NVIDIA comparison are follow-on work, with protocol design tomorrow. No claim of novelty until literature review and controlled evaluation.

## Schedule (24 × 15 minutes)
| Time | Task | Deliverable |
|---|---|---|
| 0:00–0:15 | Read CBM, CB-SAE, PRAGMA and related work; formulate falsifiable hypothesis | Related-work notes |
| 0:15–0:30 | Trace PragmaModel.forward, PragmaOutput and MLMHead | Tensor shape trace |
| 0:30–0:45 | Inspect sampler plan and packed collator | Actual batch workload metrics |
| 0:45–1:00 | Prototype hierarchical cost estimator and sampler | Coverage/budget unit tests |
| 1:00–1:15 | Compare native vs proposed sampler at equal coverage | Proxy cost distribution |
| 1:15–1:30 | GPU smoke benchmark, or defer if GPU unavailable | Step time and memory, labeled exploratory |
| 1:30–1:45 | Implement Top-K sparse bottleneck in separate module | PyTorch module |
| 1:45–2:00 | Test gradients, sparsity, shapes and zero activations | Tests |
| 2:00–2:15 | Add supervised and discovered concept channels | Configurable hybrid layer |
| 2:15–2:30 | Define point-in-time financial concept targets | Concept dataset builder |
| 2:30–2:45 | Test target cutoffs and perturbations | Leakage tests |
| 2:45–3:00 | Add concept-only linear fraud/default head | No-dense-bypass test |
| 3:00–3:15 | Integrate user bottleneck into encoder wrapper | Dense baseline unchanged |
| 3:15–3:30 | Inspect existing MLM's dense token/event bypass | Explicit scope and alternative head |
| 3:30–3:45 | Implement bottleneck-mediated MLM variant | Forward/backward test |
| 3:45–4:00 | Add reconstruction, concept alignment and task losses | Configurable loss and gradient tests |
| 4:00–4:15 | Configure dense and sparse nano runs | Matched seeds and budgets |
| 4:15–4:30 | Run short dense smoke test | Baseline metrics |
| 4:30–4:45 | Run short sparse smoke test | Loss, activation and memory metrics |
| 4:45–5:00 | Test concept intervention and bypass invariance | Faithfulness checks |
| 5:00–5:15 | Inspect dead features, coverage, concept fidelity | Diagnostic table |
| 5:15–5:30 | Outline IBM TabFormer chronological split and NVIDIA baseline protocol | Benchmark spec |
| 5:30–5:45 | Record results, limitations, failures and Git revisions | README update |
| 5:45–6:00 | Prioritize next experiments, paper outline | Research decision log |

If implementation takes longer, prioritize the bottleneck, point-in-time targets, concept-only head and forward/backward tests. Defer sampler GPU benchmark and joint MLM training rather than skip correctness. If MLM retains dense token/event routes, call only the downstream user prediction concept-mediated; do not claim the whole FM is bottleneck-mediated. Sparse activations alone do not establish semantic interpretability or faster computation.

## Evaluation protocol
Dense Pragmatiq, frozen/post-hoc SAE, downstream CBM, jointly trained sparse variant, later NVIDIA decoder and XGBoost on identical IBM splits. Match train data, information available at prediction time, model size/compute where practical, and repeat seeds. Report AUROC, average precision (define PR metric), calibration, concept fidelity, intervention response, activation/dead-feature rates, and training/inference cost. IBM is synthetic and does not supply charge-off outcomes; use fraud for comparison and separate synthetic risk tasks. No production scale claims.

## References to examine
- Concept Bottleneck Models: https://arxiv.org/abs/2007.04612
- Concept Bottleneck Models Without Predefined Concepts: https://arxiv.org/abs/2407.03921
- Interpretable and Steerable Concept Bottleneck Sparse Autoencoders: https://arxiv.org/abs/2512.10805
- PRAGMA: https://arxiv.org/abs/2604.08649
- NVIDIA blueprint: https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model
- NVIDIA dynamic context parallelism cost model: https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/
