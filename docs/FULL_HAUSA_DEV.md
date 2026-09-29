# Full Hausa Development Experiment

**Status:** Stage-1 retrieval-isolation result; not a final paper result.  
**Dataset:** leakage-safe Hausa development split, 2,462 queries.  
**Gold distribution:** 862 neutral, 811 positive, 789 negative.  
**Actions:** no retrieval plus same-language BM25, dense and hybrid retrieval at k=3/5.

## Fixed-policy results

| Policy | Macro-F1 | Weighted-F1 | Accuracy |
|---|---:|---:|---:|
| no_retrieval | **0.7576** | **0.7574** | **0.7563** |
| hybrid_same_k5 | 0.6710 | 0.6714 | 0.6779 |
| hybrid_same_k3 | 0.6622 | 0.6624 | 0.6686 |
| bm25_same_k5 | 0.6437 | 0.6435 | 0.6466 |
| bm25_same_k3 | 0.6304 | 0.6301 | 0.6332 |
| dense_same_k5 | 0.5625 | 0.5650 | 0.5914 |
| dense_same_k3 | 0.5538 | 0.5559 | 0.5792 |

The no-retrieval character n-gram probe is the strongest fixed policy for Hausa under the cheap Stage-1 classifier.

## Retrieval rescue and harm

No retrieval is correct on **1,862 / 2,462** queries and wrong on **600**.

At least one retrieval policy rescues **420 / 600 = 70.0%** of no-retrieval errors. Only 180 queries are unresolved by all seven tested actions.

Individual retrieval help/harm relative to no retrieval:

| Policy | Rescue | Harm |
|---|---:|---:|
| bm25_same_k3 | 231 | 534 |
| bm25_same_k5 | 215 | 485 |
| dense_same_k3 | 206 | 642 |
| dense_same_k5 | 211 | 617 |
| hybrid_same_k3 | 230 | 446 |
| hybrid_same_k5 | 223 | 416 |

Thus unconditional retrieval loses overall because harm exceeds rescue, despite a large pool of retrieval-rescuable errors.

## Oracle headroom

A gold-aware set-valued oracle can resolve **2,282 / 2,462** queries:

- oracle accuracy: **0.9269**
- oracle macro-F1: **0.9268**
- no-retrieval accuracy: **0.7563**
- no-retrieval macro-F1: **0.7576**

The oracle gap is approximately **+0.169 Macro-F1**. This is diagnostic headroom, not achieved deployable performance.

## Rescue by gold class

Among no-retrieval errors:

| Gold class | No-RAG errors | Rescued by >=1 retrieval policy | Rescue rate |
|---|---:|---:|---:|
| negative | 244 | 146 | 59.8% |
| neutral | 201 | 149 | 74.1% |
| positive | 155 | 125 | 80.6% |

Retrieval complementarity is therefore not restricted to one class.

## Action complementarity

Some queries are uniquely solved by one tested action. Counts where exactly one action in the full seven-action set is correct include:

- no_retrieval: 134
- bm25_same_k3: 18
- bm25_same_k5: 13
- dense_same_k3: 14
- dense_same_k5: 17
- hybrid_same_k3: 8
- hybrid_same_k5: 7

This supports retaining retriever/depth choice in the action space rather than reducing AfriARAG to a binary gate only.

## Uncertainty signal

Mean no-RAG probe confidence:

- no-RAG correct queries: **0.789**
- no-RAG wrong queries: **0.617**
- retrieval-rescued no-RAG errors: **0.609**
- unresolved no-RAG errors: **0.637**

Uncertainty is informative, but grouped cross-validation shows that a confidence-only gate does not reliably beat no retrieval.

## Preliminary controller validation

A grouped five-fold query-only action-success Random Forest, trained to estimate P(correct | query, action), achieves approximately:

- Macro-F1: **0.745**
- Accuracy: **0.746**
- Retrieval rate: **26%**

This is below the no-retrieval baseline. A nested confidence-threshold retrieval gate is also below no retrieval under cross-validation.

This result should not be treated as a failure of the overall hypothesis. The current experiment has only:

1. one language;
2. same-language retrieval;
3. a cheap weighted-vote retrieval classifier;
4. no multilingual/cross-lingual routing signal.

The next decisive test is therefore the full 12-language development experiment with same-language, multilingual and cross-lingual actions.

## Decision

**GO, with caution.**

The evidence strongly supports *retrieval heterogeneity* and oracle headroom, but does not yet show that the current learned query-only controller can exploit it. The next experiment must test whether cross-language variation and multilingual/cross-lingual actions create learnable routing structure.

Do not evaluate the final test sets yet.
