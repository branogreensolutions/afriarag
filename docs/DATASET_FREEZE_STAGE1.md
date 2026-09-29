# AfriARAG Stage-1 Dataset Freeze

**Freeze date:** 2026-09-29  
**Primary experimental view:** `data/afrisenti_clean/`  
**Raw official view:** `data/afrisenti/` (untouched)

## Frozen counts

- Raw rows: **111,718**
- Leakage-safe rows: **108,107**
- Excluded rows: **3,611**

| Language | Train | Dev | Test |
|---|---:|---:|---:|
| amh | 5,982 | 1,497 | 1,999 |
| arq | 1,641 | 411 | 958 |
| ary | 5,563 | 492 | 2,961 |
| hau | 14,172 | 2,462 | 5,303 |
| ibo | 10,192 | 1,803 | 3,682 |
| kin | 3,250 | 788 | 989 |
| pcm | 4,578 | 1,004 | 3,032 |
| por | 2,964 | 731 | 3,662 |
| swa | 1,798 | 450 | 747 |
| tso | 804 | 203 | 254 |
| twi | 3,018 | 310 | 730 |
| yor | 8,522 | 1,853 | 4,507 |
| orm | 0 | 396 | 2,001 |
| tir | 0 | 398 | 2,000 |

## Frozen leakage policy

1. Preserve the official downloaded files unchanged.
2. Remove exact duplicate text within the training retrieval pool.
3. Remove entire conflicting-label exact duplicate groups from training.
4. Remove supervised dev examples seen exactly in training.
5. Remove supervised test examples seen exactly in training or supervised dev.
6. Within dev/test, collapse identical duplicates only when labels agree.
7. Within dev/test, remove entire exact-text groups whose labels conflict.
8. Oromo and Tigrinya remain strict zero-shot: no target-language training labels are used.
9. Oromo/Tigrinya test examples are not removed solely for appearing in target-language dev, because target dev labels are not used for tuning.

## Freeze rule

The data-cleaning policy, action space, encoder, controller feature families, and primary metrics must not be changed after reviewing final test results.

Any later change to the data view requires a new named experimental version and must not silently replace this freeze.
