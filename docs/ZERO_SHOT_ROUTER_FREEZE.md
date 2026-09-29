# Zero-Shot Router Freeze

**Freeze date:** 2026-09-29  
**Validation:** leave-one-language-out (LOLO) over all 12 supervised languages.  
**Target use:** strict zero-shot Oromo (orm) and Tigrinya (tir).

## Validation result

The zero-shot simulation contains 12,004 development queries. For each held-out
language, no target-language training examples are used. The no-retrieval probe is
trained only on the other 11 languages, and retrieval is restricted to
cross-lingual evidence.

Nested LOLO result:

- adaptive Macro-F1: **0.44436**
- no-retrieval Macro-F1: **0.41518**
- delta: **+0.02918**
- adaptive accuracy: **0.45427**
- no-retrieval accuracy: **0.42877**
- selected-context retrieval rate: **34.25%**
- average selected k: **1.14**

The nested result improves Macro-F1 in **11 of 12 held-out languages**. Nigerian
Pidgin is the only negative language-level delta (-0.00487). Swahili is nearly
flat (+0.00281).

## Per-language Macro-F1 delta

| Language | Delta |
|---|---:|
| amh | +0.1350 |
| yor | +0.1008 |
| kin | +0.0705 |
| tso | +0.0561 |
| arq | +0.0387 |
| twi | +0.0368 |
| ibo | +0.0271 |
| por | +0.0261 |
| ary | +0.0199 |
| hau | +0.0197 |
| swa | +0.0028 |
| pcm | -0.0049 |

## Selected policies

Across the nested LOLO selected actions:

- no_retrieval: 7,893
- dense_cross_k3: 2,670
- hybrid_cross_k3: 748
- hybrid_cross_k5: 693

Among retrieved selections, dense cross-lingual k=3 accounts for approximately
65% of selected retrieval actions.

## Frozen deployment penalty

The single development-wide LOLO out-of-fold penalty search peaks at:

`lambda = 0.065`

This penalty yields Macro-F1 0.44734 on the pooled LOLO out-of-fold action
scores. This number is **not** the unbiased nested validation estimate because
the same pooled OOF scores are used to select lambda. The reportable validation
estimate remains the nested value, Macro-F1 **0.44436**.

The frozen zero-shot deployment penalty is therefore:

`lambda_zero_shot = 0.065`

## Interpretation

This experiment supports the claim that cross-lingual routing can improve
sentiment classification for a language treated as unseen during controller
training.

The aggregate improvement is not uniformly distributed across languages, so
final Oromo/Tigrinya results must be reported separately as well as jointly.

The selected-context retrieval rate is not total retrieval-computation rate.
The router uses candidate retrieval diagnostics before selecting an action.

## Frozen strict zero-shot action space

- no_retrieval
- dense_cross_k3
- hybrid_cross_k3
- hybrid_cross_k5

No same-language labelled retrieval is permitted for Oromo or Tigrinya.

## Freeze rule

Oromo/Tigrinya gold labels must not alter:

- the action set;
- encoder;
- feature set;
- retrieval penalty;
- controller architecture;
- controller weights;
- prompts/classifier configuration.

Target gold labels are evaluation-only.
