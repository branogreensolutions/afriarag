# AfriARAG Stage-1 Experiments

This is the operational protocol for the initial Sections 1–69 research scope.

## Primary questions

- **RQ1:** Does retrieval improve African-language sentiment classification over no retrieval?
- **RQ2:** Does the best retrieval strategy vary by query and language?
- **RQ3:** Can a learned controller approximate the development-set oracle and outperform the strongest fixed policy?
- **RQ4:** Does adaptive cross-lingual retrieval help strict zero-shot Oromo and Tigrinya?
- **RQ5:** When does retrieval help, harm, or waste computation?

## Stage 1A — dataset freeze, audit, and leakage-safe view

Run:

```bash
python scripts/download_data.py
python scripts/audit_data.py
python scripts/prepare_leakage_safe_data.py
```

The first audit found substantial exact cross-split duplication in the official AfriSenti files. Because retrieval can exploit verbatim train/evaluation duplicates, **the leakage-safe view is the primary experimental dataset**.

The untouched official split is retained only for secondary historical comparability.

Leakage-safe rules:

1. Never modify the downloaded raw files.
2. Remove exact duplicate text within the training retrieval pool.
3. Remove conflicting-label exact training duplicates from the retrieval pool.
4. Remove supervised dev queries found exactly in training.
5. Remove supervised test queries found exactly in training or supervised dev.
6. Within dev/test, collapse identical duplicates only when their labels agree; if identical text has conflicting labels within the same split, exclude the entire conflicting group rather than arbitrarily retaining one label.
7. Oromo and Tigrinya remain strict zero-shot: their train files contain no labelled examples and target-language labels are never used for tuning.
8. Oromo/Tigrinya test examples are **not** removed merely because the same text appears in target-language dev, because target dev labels are not used by the strict zero-shot pipeline. This choice must remain frozen before zero-shot evaluation.

Required outputs include:

- `outputs/audit/dataset_manifest.csv`
- `outputs/audit/cross_split_duplicates.csv`
- `outputs/audit/duplicate_label_conflicts.csv`
- `outputs/audit/leakage_safe_manifest.csv`
- `outputs/audit/leakage_safe_summary.json`
- `outputs/audit/leakage_safe_exclusions.csv`

## Stage 1B — fixed development policies

Run the 12 policies in `configs/stage1.yaml` on the 12 supervised-language development splits. The cheap no-RAG model is a character n-gram logistic-regression probe; retrieval policies use score-weighted label voting. This isolates retrieval behavior before paid LLM experiments.

Primary metric: macro-F1. Secondary metrics: weighted-F1, accuracy, retrieval rate, average k and latency.

## Stage 1C — oracle

For each development query, identify all policies that predict the gold label correctly. Select the lowest-cost correct policy using the frozen cost rule. If no policy is correct, label the query `unresolved`.

**Decision gate:** move to paid LLM experiments only if the oracle shows material headroom over the strongest fixed policy and the oracle distribution is not degenerate.

## Stage 1D — controller

Train only on supervised development-oracle labels. Inputs are query-side surface features and no-RAG confidence/entropy. No test or Oromo/Tigrinya label enters controller training.

## Stage 1E — frozen supervised test

Only after policies, encoder, controller features and hyperparameters are frozen should the supervised test set be evaluated. Compare adaptive vs strongest fixed policy using paired-bootstrap confidence intervals.

## Stage 1F — strict zero-shot

Oromo (`orm`) and Tigrinya (`tir`) may use only no-retrieval, multilingual or cross-lingual evidence from the 12 supervised languages. Their labels are evaluation-only.

## Initial policy matrix

| Policy | Retriever | Scope | k | alpha |
|---|---|---|---:|---:|
| no_retrieval | none | none | 0 | 0 |
| bm25_same_k3 | BM25 | same | 3 | 0 |
| bm25_same_k5 | BM25 | same | 5 | 0 |
| dense_same_k3 | dense | same | 3 | 1 |
| dense_same_k5 | dense | same | 5 | 1 |
| hybrid_same_k3 | hybrid | same | 3 | .5 |
| hybrid_same_k5 | hybrid | same | 5 | .5 |
| dense_multi_k3 | dense | multilingual | 3 | 1 |
| hybrid_multi_k3 | hybrid | multilingual | 3 | .5 |
| dense_cross_k3 | dense | cross | 3 | 1 |
| hybrid_cross_k3 | hybrid | cross | 3 | .5 |
| hybrid_cross_k5 | hybrid | cross | 5 | .5 |

## Stop/go criteria after the oracle

Proceed when the evidence supports adaptation. In particular, look for at least two of:

1. Oracle macro-F1 materially exceeds the strongest fixed policy.
2. No single action accounts for nearly all successful oracle decisions.
3. Retrieval rescue exceeds retrieval harm for a meaningful subset of languages/query groups.
4. Cross-/multilingual policies are selected non-trivially for difficult queries.

If these are absent, revise the action space or hypothesis before adding LLM cost.
