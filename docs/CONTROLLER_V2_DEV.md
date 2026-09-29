# Controller v2 — Nested Development Validation

**Status:** supervised controller frozen for test evaluation on 2026-09-29.  
**Data:** 12,004 leakage-safe supervised-language development queries.  
**Validation:** nested grouped cross-validation by query.

## Aggregate result

Controller v2:

- Macro-F1: **0.68766**
- Weighted-F1: **0.68800**
- Accuracy: **0.68819**
- selected-context retrieval rate: **34.61%**
- average selected k: **1.50**

No-retrieval baseline:

- Macro-F1: **0.67921**
- Accuracy: **0.67944**

Development gain:

- Macro-F1: **+0.00844**
- Accuracy: **+0.00875**

A paired query bootstrap on the nested out-of-fold predictions gives an approximate 95% CI for the Macro-F1 delta of **[+0.0042, +0.0127]**. This interval is a development-stage diagnostic and does not replace final held-out test evaluation.

## Fold consistency

| Fold | Penalty | Adaptive Macro-F1 | No-RAG Macro-F1 | Delta |
|---|---:|---:|---:|---:|
| 1 | 0.075 | 0.6863 | 0.6826 | +0.0038 |
| 2 | 0.055 | 0.6907 | 0.6780 | +0.0127 |
| 3 | 0.065 | 0.6873 | 0.6806 | +0.0066 |
| 4 | 0.080 | 0.6907 | 0.6778 | +0.0128 |
| 5 | 0.075 | 0.6830 | 0.6769 | +0.0062 |

All five outer folds improve over their corresponding no-retrieval baseline.

The final development-only penalty search peaks at **lambda=0.065**. Performance is relatively flat from approximately 0.060–0.080, which reduces concern that the result depends on a single unstable penalty value.

## Rescue/harm

Relative to no retrieval, the nested out-of-fold controller:

- rescues **382** no-retrieval errors;
- harms **277** no-retrieval successes;
- yields a net gain of **105** correct predictions.

## Per-language Macro-F1 delta

| Language | Adaptive | No-RAG | Delta |
|---|---:|---:|---:|
| por | 0.5811 | 0.5619 | +0.0192 |
| pcm | 0.4499 | 0.4412 | +0.0088 |
| ibo | 0.7846 | 0.7765 | +0.0081 |
| ary | 0.7295 | 0.7234 | +0.0061 |
| tso | 0.5667 | 0.5607 | +0.0061 |
| amh | 0.5255 | 0.5198 | +0.0058 |
| yor | 0.7164 | 0.7131 | +0.0033 |
| hau | 0.7597 | 0.7576 | +0.0021 |
| arq | 0.5677 | 0.5683 | -0.0006 |
| kin | 0.6045 | 0.6052 | -0.0007 |
| swa | 0.4323 | 0.4385 | -0.0062 |
| twi | 0.5933 | 0.6024 | -0.0091 |

The mean of per-language Macro-F1 values rises from approximately **0.6057 to 0.6093**.

## Selected actions

Controller v2 selects:

- no_retrieval: 7,850
- hybrid_same_k5: 1,721
- bm25_same_k5: 797
- hybrid_same_k3: 550
- hybrid_multi_k3: 422
- bm25_same_k3: 394
- dense_same_k5: 257
- dense_same_k3: 10
- dense_multi_k3: 3
- cross-lingual actions: **0**

The absence of selected cross-lingual actions is important. Cross-lingual policies have substantial oracle-only complementary coverage, but Controller v2 does not yet learn when to exploit them.

## Efficiency interpretation

Controller v2 uses action-specific retrieval diagnostics (BM25/dense scores, margins, retrieved-label distributions) before action selection. Therefore the reported 34.6% retrieval rate is a **selected-context rate**, not the fraction of queries on which no retrieval computation occurred.

The system should be described as a **probe-and-route adaptive evidence selector** unless/until a pure query-only gate is shown to work.

Efficiency claims should separate:

1. retrieval-probe compute;
2. selected context size/tokens;
3. downstream LLM/classifier cost.

Do not claim that 65.4% of queries incur zero retrieval computation.

## Freeze decision

The supervised Controller v2 is frozen with:

- retrieval-aware action-success model;
- nested grouped validation protocol;
- development-tuned retrieval penalty = **0.065**;
- current feature set and action set.

No further supervised-controller tuning should use supervised test labels.

## Zero-shot decision

Do **not** yet evaluate Oromo/Tigrinya labels.

Before strict zero-shot evaluation, simulate unseen-language routing on the 12 supervised languages using:

- no target-language training examples;
- multilingual no-RAG probe trained on the other 11 languages;
- cross-lingual retrieval only;
- leave-one-language-out controller evaluation.

Freeze the zero-shot router only after that simulation.
