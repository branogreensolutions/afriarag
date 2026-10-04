# AfriARAG Stage-2 Transformer Follow-up — Final Results

**Status:** frozen, pre-specified follow-up evaluation completed after Stage-1.
**Model:** `Davlan/afro-xlmr-base-114L`
**Target-label rule:** Oromo/Tigrinya labels were not used for training or model selection.

Stage 2 was motivated after the Stage-1 held-out results were inspected, so it
must not be described as part of the original Stage-1 confirmatory experiment.
However, the Stage-2 model/configuration was frozen before its own held-out
evaluation, and the paired comparison analysis was specified before the Stage-2
test results were inspected.

## 1. Supervised 12-language test

| System | Macro-F1 | Accuracy | MacroLangF1 |
|---|---:|---:|---:|
| AfroXLM-R | **0.68149** | **0.68061** | **0.60059** |
| AfriARAG adaptive | 0.66555 | 0.66528 | 0.57336 |
| Char no-retrieval | 0.66335 | 0.66264 | 0.57214 |

Paired language-stratified bootstrap (5,000 samples):

AfroXLM-R vs AfriARAG adaptive:
- Macro-F1 delta: approximately **+0.01592**
- 95% CI: **[+0.01004, +0.02190]**

AfroXLM-R vs no retrieval:
- Macro-F1 delta: approximately **+0.01818**
- 95% CI: **[+0.01248, +0.02400]**

Thus the transformer comparator is reliably stronger in aggregate than both
Stage-1 systems.

### Supervised language heterogeneity

AfroXLM-R does not dominate every language. Relative to AfriARAG adaptive it is
higher on 7/12 test languages and lower on 5/12.

Large AfroXLM-R gains occur for languages including Amharic, Kinyarwanda,
Portuguese, Swahili, and Algerian Arabic. AfriARAG/char models remain stronger
for some languages including Hausa, Igbo, Yoruba, Moroccan Arabic, and Twi.

This heterogeneity is important: aggregate pretrained-representation strength
does not remove language-specific differences.

## 2. Strict zero-shot Oromo/Tigrinya

| System | Macro-F1 | Accuracy | MacroLangF1 |
|---|---:|---:|---:|
| AfroXLM-R | **0.49791** | **0.51487** | **0.47645** |
| dense_cross_k3 | 0.37700 | 0.37741 | 0.37193 |
| AfriARAG adaptive | 0.33687 | 0.34016 | 0.33939 |
| no retrieval | 0.31434 | 0.32167 | 0.31340 |

Paired bootstrap:

AfroXLM-R vs AfriARAG adaptive:
- Macro-F1 delta: approximately **+0.16103**
- 95% CI: **[+0.14103, +0.18124]**

AfroXLM-R vs dense_cross_k3:
- Macro-F1 delta: approximately **+0.12101**
- 95% CI: **[+0.10112, +0.14088]**

AfroXLM-R vs no retrieval:
- Macro-F1 delta: approximately **+0.18371**
- 95% CI: **[+0.16301, +0.20392]**

All intervals are well above zero.

### Target-language breakdown

Oromo Macro-F1:
- AfroXLM-R: **0.40418**
- dense_cross_k3: 0.35710
- AfriARAG adaptive: 0.32263
- no retrieval: 0.31859

Tigrinya Macro-F1:
- AfroXLM-R: **0.54872**
- dense_cross_k3: 0.38677
- AfriARAG adaptive: 0.35615
- no retrieval: 0.30820

AfroXLM-R therefore beats the retrieval-based systems on both zero-shot target
languages, with an especially large advantage on Tigrinya.

## 3. What Stage 2 changes

The strongest final empirical conclusion is no longer that retrieval is the best
solution for unseen African-language sentiment classification.

Instead:

1. Stage 1 shows that retrieval behavior is heterogeneous and that cross-lingual
   evidence can rescue many predictions.
2. Stage 1 also shows that a fixed dense cross-lingual policy improves strongly
   over the lightweight zero-shot no-retrieval baseline.
3. Stage 2 shows that a strong African multilingual pretrained encoder is
   substantially stronger than both fixed and adaptive retrieval systems in the
   same strict label-zero-shot evaluation.

Therefore the paper should ask not simply **whether retrieval helps**, but
**when retrieval adds value relative to strong pretrained multilingual
representations**.

## 4. Recommended paper framing

A defensible framing is:

> Retrieval is not universally beneficial in low-resource African sentiment
> classification. Its apparent value depends strongly on the strength of the
> underlying representation model. With a lightweight character classifier,
> cross-lingual retrieval yields meaningful zero-shot gains; however, a strong
> African multilingual encoder surpasses both fixed and adaptive retrieval on
> Oromo and Tigrinya. Adaptive routing still exposes useful query-level
> heterogeneity, but does not yet provide state-of-the-art predictive gains.

This is a stronger and more general scientific contribution than claiming that
adaptive RAG universally wins.

## 5. Claims to avoid

Do not claim:
- AfriARAG is the best-performing system;
- adaptive retrieval beats AfroXLM-R;
- retrieval is necessary for zero-shot transfer;
- Stage 2 was part of the original Stage-1 confirmatory design;
- target-language labels were used to train the transformer.

## 6. Next methodological question

A natural future experiment is whether retrieval can add value **on top of**
AfroXLM-R rather than competing against a weaker character classifier.

Because the current Stage-1 and Stage-2 test sets have already been inspected,
such a combined model should be evaluated either:
- by nested cross-validation/development-only analysis; or
- on a new external dataset / newly locked evaluation set.

It must not be tuned and re-evaluated on the already-inspected AfriSenti test
sets as if it were a fresh confirmatory result.
