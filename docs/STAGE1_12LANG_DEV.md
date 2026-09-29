# Stage-1 12-Language Development Results

**Status:** development-stage retrieval-isolation experiment; final test sets remain untouched.

## Scope

- 12 supervised African languages
- 12,004 leakage-safe development queries
- 12 fixed policies per query
- no retrieval, same-language BM25/dense/hybrid, multilingual dense/hybrid, and cross-lingual dense/hybrid

## Strongest fixed policy

Across all 12,004 queries, no retrieval is the strongest global fixed policy:

- Macro-F1: **0.6792**
- Weighted-F1: **0.6795**
- Accuracy: **0.6794**

The strongest fixed retrieval policy is hybrid same-language k=5:

- Macro-F1: **0.6465**
- Accuracy: **0.6475**

One language, Moroccan Arabic/Darija (ary), shows a small fixed-policy gain from hybrid same-language k=5 over no retrieval:

- hybrid_same_k5 Macro-F1: **0.7321**
- no_retrieval Macro-F1: **0.7234**

This language-level exception supports retaining per-language/query adaptation rather than relying only on the global ranking.

## Oracle headroom

Of 12,004 queries:

- no retrieval is correct on **8,156**
- no retrieval is wrong on **3,848**
- at least one retrieval action rescues **3,327 / 3,848 = 86.5%** of no-retrieval errors
- only **521** queries are unresolved by every tested action
- full oracle accuracy: **0.9566**
- full oracle Macro-F1: **0.9564**

Thus the action set has substantial complementarity even though every retrieval action is weaker than no retrieval when applied globally.

## Incremental value of routing scope

Using correctness coverage as an oracle diagnostic:

- no retrieval only: **67.94%**
- no retrieval + same-language actions: **89.88%**
- + multilingual actions: **91.96%**
- + cross-lingual actions: **95.66%**

Among no-retrieval errors:

- same-language actions rescue 2,633 queries
- multilingual actions add **250** queries not rescued by same-language actions
- cross-lingual actions add a further **444** queries not rescued by same-language or multilingual actions

This is the key Stage-1 evidence for retaining language-scope routing. Cross-lingual retrieval is a poor global fixed strategy but has important query-specific complementary value.

## Per-language no-retrieval error rescue

| Language | No-RAG errors | Rescued by >=1 retrieval action | Rescue rate |
|---|---:|---:|---:|
| amh | 683 | 563 | 82.4% |
| arq | 163 | 134 | 82.2% |
| ary | 135 | 116 | 85.9% |
| hau | 600 | 508 | 84.7% |
| ibo | 397 | 351 | 88.4% |
| kin | 312 | 274 | 87.8% |
| pcm | 353 | 294 | 83.3% |
| por | 312 | 282 | 90.4% |
| swa | 209 | 185 | 88.5% |
| tso | 77 | 69 | 89.6% |
| twi | 107 | 98 | 91.6% |
| yor | 500 | 453 | 90.6% |

## Controller v1

The first grouped action-success controller uses query features and action identity but not retrieval diagnostics.

Five-fold grouped development validation:

- Macro-F1: **0.6640**
- Accuracy: **0.6646**
- Retrieval rate: **43.76%**
- Average k: **1.75**

This is below the no-retrieval baseline and therefore does not yet exploit the oracle headroom.

## Controller v2 rationale

The prediction file already contains action-specific retrieval evidence:

- BM25 raw top score and margin
- dense cosine top score and margin
- fused score margin
- retrieved labels
- retrieved scores

Controller v1 ignores these signals. Controller v2 therefore estimates:

`P(action succeeds | query, retrieval diagnostics, action)`

and chooses:

`argmax[P(correct) - lambda * I(retrieval)]`

where the retrieval penalty lambda is tuned only on development data using nested grouped cross-validation.

This controller is intentionally evaluated without touching any final test set.

## Decision

**Continue. Do not evaluate test data yet.**

The fixed-policy results alone do not support the claim that RAG globally improves AfriSenti sentiment classification. They support a more specific and scientifically stronger claim: retrieval usefulness is highly query-dependent, and multilingual/cross-lingual actions contribute complementary rescue coverage despite weak global averages.

The next decision gate is whether retrieval-aware Controller v2 can beat the no-retrieval development baseline under nested grouped validation.
