# Auditable financial FM — revised next-session extension plan

This supersedes the implementation priorities in next_session_research_plan.md, not the prior experiment results. Preserve upstream baseline. Six-hour prototype budget, not a publication-ready result.

| Experiment | Minutes | Deliverable |
|---|---:|---|
| 06 Quantile B-spline | 60 | Train-only quantile knots, continuous numeric payload through tokenizer/shards/collator, differentiable spline embedding, boundary and mask tests |
| 07 Explainable multi-expert concept bottleneck | 90 | Four concept experts, 128 concepts each, global Top-8 STE, single-expert control, diversity and utilization tests |
| 08 Joint pretraining | 60 | Optional RMSNorm, sparse encoder wrapper, configurable reconstruction/MLM/concept losses, explicit dense-bypass scope and gradient smoke tests |
| 09 Auditability | 60 | Local spline weight decomposition, end-to-end gradient/perturbation attribution, training-only top activating examples, concept interventions |
| 10 SFT | 45 | Frozen versus full SFT with concept-only head, monitor concept fidelity and drift |
| 11 Feature extraction | 45 | Logistic and XGBoost on sparse versus dense versus engineered features, same temporal split |

## Research hypotheses
H1: spline encoding retains useful within-bin magnitude information without sacrificing local feature-range interpretability.
H2: multi-expert sparse concept learning improves concept diversity and fidelity at matched dictionary capacity versus a single-expert Top-K model.
H3: concept-only prediction permits exact additive logit decomposition with a linear head; end-to-end range attribution remains approximate for nonlinear transformer paths.
H4: sparse features retain downstream predictive utility and reduce dependence on manual feature engineering.

## Implementation guardrails
Fit knots on train-only data, handle repeated quantiles and out-of-range values, preserve raw numeric values through storage and prevent masked raw value leakage. Keep dense original as baseline; use separate extension modules where practical. Global Top-K after evaluating all experts is not compute-saving MoE; compare early routed experts separately. Sparsity does not prove monosemanticity. Validate known concept labels on held-out histories, check discovered features with retrieval, interventions, redundancy and stability. Exact linear concept contribution is w_j*c_j on the logit scale, not an exact raw-feature attribution through attention. Do not label a model fully bottleneck-mediated when MLM still sees dense event/token paths. Use training-only retrieval examples and avoid raw customer data in Git. SFT can drift concept semantics; compare before/after fidelity. Report predictive metrics, calibration, concept coverage, intervention behavior, runtime and memory. Match splits, training budget, and model capacity where practical; repeat seeds later. IBM fraud and NVIDIA comparisons follow prototype phase; synthetic default is not real credit-risk validation.

If behind schedule, finish numeric and bottleneck forward/backward/leakage tests before downstream evaluations. Start small on nano with 64 hidden dimensions; initial expert count and Top-K are hypotheses, not tuned defaults. No first-of-kind claim until literature review and evidence.
