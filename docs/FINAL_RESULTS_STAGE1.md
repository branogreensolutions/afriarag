# AfriARAG Stage-1 Final Results and RQ Conclusions

**Status:** final held-out Stage-1 analysis after pre-test freeze.  
**Primary metric:** Macro-F1.  
**Confirmatory fixed comparator:** no retrieval, because it was the strongest global fixed development policy.

## 1. Supervised held-out result

Across 28,824 held-out supervised-language test queries:

| System | Macro-F1 | Weighted-F1 | Accuracy | MacroLangF1 |
|---|---:|---:|---:|---:|
| No retrieval | 0.66335 | 0.66304 | 0.66264 | 0.57214 |
| AfriARAG adaptive | **0.66555** | **0.66534** | **0.66528** | **0.57336** |

Point-estimate deltas:

- Macro-F1: **+0.00220**
- Weighted-F1: **+0.00229**
- Accuracy: **+0.00264**
- MacroLangF1: **+0.00121**

Paired language-stratified query bootstrap (5,000 samples):

- Macro-F1 delta 95% CI: **[-0.00073, +0.00512]**
- Accuracy delta 95% CI: **[-0.00024, +0.00555]**

Therefore the supervised aggregate improvement is **not statistically reliable at the 95% two-sided confidence level**.

Paired outcomes relative to no retrieval:

- rescue: 969
- harm: 893
- net correct gain: 76

The adaptive system selects retrieved context for 37.54% of test queries.

### Fixed-policy context

No retrieval remains the strongest fixed policy on the held-out supervised test. The next strongest fixed retrieval policy is hybrid same-language k=5 with Macro-F1 0.61278. Thus AfriARAG has the best supervised point estimate among tested systems, but its advantage over the predeclared no-retrieval comparator is small and statistically inconclusive.

## 2. Strict zero-shot Oromo/Tigrinya result

Across 4,001 strict zero-shot test queries:

| System | Macro-F1 | Accuracy | MacroLangF1 |
|---|---:|---:|---:|
| No retrieval | 0.31434 | 0.32167 | 0.31340 |
| AfriARAG adaptive router | **0.33687** | **0.34016** | **0.33939** |

Adaptive vs no-retrieval deltas:

- Macro-F1: **+0.02253**
- Accuracy: **+0.01850**
- MacroLangF1: **+0.02599**

Paired bootstrap:

- Macro-F1 delta 95% CI: **[+0.01368, +0.03179]**
- Accuracy delta 95% CI: **[+0.00975, +0.02749]**

Thus the frozen adaptive zero-shot router **reliably outperforms the no-retrieval zero-shot baseline**.

Paired outcomes:

- rescue: 202
- harm: 128
- net correct gain: 74

### Critical fixed-policy comparison

The held-out zero-shot fixed-policy table shows:

| Policy | Macro-F1 | Accuracy |
|---|---:|---:|
| dense_cross_k3 | **0.37700** | **0.37741** |
| hybrid_cross_k5 | 0.36081 | 0.36216 |
| hybrid_cross_k3 | 0.35355 | 0.35716 |
| adaptive router | 0.33687 | 0.34016 |
| no retrieval | 0.31434 | 0.32167 |

Therefore the zero-shot experiment supports the value of **cross-lingual retrieval**, but does **not** support the stronger claim that the learned adaptive router beats the strongest fixed zero-shot retrieval strategy.

The fact that dense cross-lingual k=3 substantially outperforms the router on Oromo/Tigrinya was discovered on held-out test and must be reported as a descriptive/post-hoc comparison rather than used to retune the frozen router.

## 3. Research-question conclusions

### RQ1 — Does retrieval improve over no retrieval?

**Context-dependent.**

- Supervised setting: every fixed retrieval policy is worse than no retrieval globally.
- Strict zero-shot setting: cross-lingual dense retrieval substantially improves over no retrieval.

Hence retrieval is not universally beneficial; its value is strongest when target-language supervision is absent.

### RQ2 — Do queries/languages require different retrieval strategies, depths, or sources?

**Supported diagnostically, but only partially converted into deployed gains.**

Development oracle analysis shows substantial query-level complementarity among no retrieval, same-language, multilingual, and cross-lingual actions. Different languages also show different fixed-policy behavior. However, the learned controller does not fully exploit this complementarity on held-out data.

### RQ3 — Does the learned adaptive controller outperform fixed retrieval?

**Not established as a general claim.**

- Supervised test: adaptive has the best point estimate and beats the strongest fixed comparator (no retrieval) by +0.00220 Macro-F1, but the 95% bootstrap interval crosses zero.
- Strict zero-shot test: adaptive reliably beats no retrieval, but is clearly below fixed dense cross-lingual k=3.

The hypothesis that adaptive routing generally outperforms the strongest fixed policy is therefore **not confirmed by Stage 1**.

### RQ4 — Does cross-lingual retrieval help strict zero-shot Oromo/Tigrinya?

**Yes. Strongly supported.**

The adaptive router improves over no retrieval, and fixed dense cross-lingual k=3 improves even more. The strongest Stage-1 finding is therefore that cross-lingual evidence is useful for low-resource unseen-language sentiment classification.

### RQ5 — Can retrieval benefit/harm be predicted from query/retrieval signals?

**Partially supported.**

Retrieval-aware Controller v2 improves nested development performance and produces a positive supervised held-out point estimate. Retrieval diagnostics therefore contain predictive signal. However, generalization is limited: the supervised gain is small and the zero-shot router fails to choose retrieval aggressively enough for Oromo/Tigrinya relative to the fixed dense-cross policy.

## 4. Main scientific interpretation

Stage 1 does **not** show that adaptive RAG universally improves African sentiment classification.

It shows three more defensible findings:

1. Retrieval behavior is highly heterogeneous: the same retrieval action can rescue some queries while harming others.
2. In supervised African-language sentiment classification, a strong no-retrieval character model is difficult to beat globally; adaptive routing yields only a small held-out aggregate gain.
3. In strict unseen-language evaluation, cross-lingual retrieval produces a substantial improvement over no retrieval, but a simple fixed dense cross-lingual policy currently outperforms the learned zero-shot router.

This points to a refined research contribution: **when target-language supervision is absent, cross-lingual retrieval becomes substantially more valuable; adaptive routing remains promising but requires better generalization to unseen languages.**

## 5. Claims to avoid

Do not claim:

- that supervised AfriARAG significantly beats no retrieval;
- that adaptive routing beats the strongest fixed zero-shot strategy;
- that only selected-context queries incur retrieval computation;
- that the oracle is achieved system performance;
- that test results justify further tuning of the frozen Stage-1 controller.

## 6. Remaining publication analyses

Without changing the frozen models:

1. development-only controller feature ablations;
2. development-only action-space ablations;
3. test confusion matrices and per-class metrics;
4. descriptive per-language fixed-policy comparisons;
5. qualitative rescue/harm examples;
6. selected-context and retrieval-probe efficiency accounting;
7. final figures and tables;
8. manuscript drafting.

Any new controller architecture should be presented as a separate Stage-2/future experiment, not silently tuned against the current held-out test.
