# AfriARAG Stage-1 Held-Out Test — Preliminary Results

**Status:** confirmatory held-out evaluation completed after pre-test freeze.  
**Significance status:** paired bootstrap/fixed-policy final analysis pending via `scripts/analyze_final_test.py`.

## Pre-test freeze

The pre-test manifest was created before held-out evaluation at git commit:

`b3284594e1be3ce20664dbf81af2a2b9c7260ca9`

Both supervised and zero-shot retrieval penalties were frozen at:

`lambda = 0.065`

The manifest records SHA-256 hashes for Stage-1 config, both controller models/configs, and leakage-safe dataset reports.

## Supervised held-out test

Queries: **28,824**

Frozen adaptive Controller v2:

- Macro-F1: **0.66555**
- Weighted-F1: **0.66534**
- Accuracy: **0.66528**
- selected-context retrieval rate: **37.54%**
- average selected k: **1.60**

Frozen no-retrieval comparator:

- Macro-F1: **0.66335**
- Accuracy: **0.66264**

Delta:

- Macro-F1: **+0.00220**
- Accuracy: **+0.00264**

The held-out gain is positive but substantially smaller than the nested-development gain (+0.00844 Macro-F1). Statistical interpretation is deferred until paired bootstrap is completed.

### Supervised per-language Macro-F1 delta

Positive:
- swa: +0.02524
- arq: +0.01405
- por: +0.00736
- kin: +0.00510
- yor: +0.00384
- ibo: +0.00219
- pcm: +0.00113

Negative:
- twi: -0.00287
- ary: -0.00501
- hau: -0.00758
- amh: -0.01203
- tso: -0.01683

Thus adaptive routing improves 7/12 supervised test languages and regresses on 5/12.

Mean per-language Macro-F1 (MacroLangF1 diagnostic):

- adaptive: **0.57336**
- no retrieval: **0.57214**
- delta: **+0.00121**

## Strict zero-shot Oromo/Tigrinya test

Queries: **4,001**

Frozen zero-shot router:

- Macro-F1: **0.33687**
- Weighted-F1: **0.32895**
- Accuracy: **0.34016**
- selected-context retrieval rate: **23.89%**
- average selected k: **0.775**

Frozen zero-shot no-retrieval comparator:

- Macro-F1: **0.31434**
- Accuracy: **0.32167**

Delta:

- Macro-F1: **+0.02253**
- Accuracy: **+0.01850**

This held-out improvement is directionally consistent with the pre-test LOLO development simulation (+0.02918 Macro-F1).

### Target-language results

Oromo:
- adaptive Macro-F1: **0.32263**
- no retrieval: **0.31859**
- delta: **+0.00404**
- selected-context retrieval rate: **23.74%**

Tigrinya:
- adaptive Macro-F1: **0.35615**
- no retrieval: **0.30820**
- delta: **+0.04795**
- selected-context retrieval rate: **24.05%**

Both target languages improve, but the aggregate zero-shot gain is driven primarily by Tigrinya. The paper must report Oromo and Tigrinya separately, not only their pooled result.

Mean target-language Macro-F1:

- adaptive: **0.33939**
- no retrieval: **0.31340**
- delta: **+0.02599**

## Interpretation before significance testing

The confirmatory test evidence currently supports two different strengths of conclusion:

1. **Supervised setting:** adaptive routing retains a positive held-out aggregate improvement, but the gain is small and heterogeneous across languages. Do not claim a statistically reliable improvement until paired bootstrap is complete.

2. **Strict zero-shot setting:** the adaptive cross-lingual router shows a larger held-out improvement, with positive Macro-F1 deltas on both Oromo and Tigrinya. Significance still requires paired analysis.

The test outputs must not be used to retune penalties, controller architecture, feature set, or action space.

## Next analysis

Run:

```bash
python scripts/analyze_final_test.py
```

This will produce:

- fixed-policy test metric tables;
- confirmatory adaptive-vs-no-retrieval comparisons;
- rescue/harm counts;
- MacroLangF1;
- paired query bootstrap confidence intervals stratified by language.

The development-selected no-retrieval policy remains the confirmatory fixed comparator. Any policy identified as best only after inspecting test metrics must be labeled descriptive/post-hoc.
