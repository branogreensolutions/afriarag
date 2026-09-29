# Hausa-200 Pilot — Preliminary Stage-1 Result

**Status:** pipeline-validation pilot; not a final paper result.  
**Sample:** 200 stratified Hausa development queries (70 neutral, 66 positive, 64 negative).  
**Policies:** no retrieval plus same-language BM25, dense and hybrid retrieval at k=3/5.

## Fixed-policy results

| Policy | Macro-F1 | Accuracy |
|---|---:|---:|
| no_retrieval | 0.7654 | 0.765 |
| hybrid_same_k3 | 0.7408 | 0.745 |
| hybrid_same_k5 | 0.7167 | 0.725 |
| bm25_same_k5 | 0.6867 | 0.690 |
| bm25_same_k3 | 0.6841 | 0.685 |
| dense_same_k3 | 0.5606 | 0.595 |
| dense_same_k5 | 0.5574 | 0.595 |

No retrieval is the strongest **fixed** policy in this pilot. Hybrid k=3 is the strongest retrieval policy and substantially outperforms both BM25-only and dense-only retrieval.

## Retrieval help/harm relative to no retrieval

The no-retrieval probe correctly classifies 153/200 queries and misses 47.

Among the 47 no-retrieval errors, at least one of the six retrieval policies correctly classifies **35 (74.5%)**. Only 12/200 queries are unresolved by every tested policy.

For hybrid k=3 specifically:

- rescues 25 no-retrieval errors;
- harms 29 queries that no retrieval had classified correctly;
- therefore fixed hybrid retrieval is slightly worse overall even though it has substantial rescue capability.

This is preliminary evidence for the central AfriARAG hypothesis: retrieval can be useful on a substantial subset of difficult queries while being harmful on others, so query-selective retrieval may outperform unconditional retrieval.

## Oracle upper bound

A gold-aware oracle that uses a correct tested action whenever one exists reaches approximately:

- **Macro-F1: 0.9399**
- **Accuracy: 0.9400**

compared with no retrieval:

- **Macro-F1: 0.7654**
- **Accuracy: 0.7650**

This oracle is not deployable and must not be interpreted as achieved system performance. It measures complementarity/headroom in the tested action set.

## Policy complementarity

Among the 47 no-retrieval errors:

- hybrid k=3 rescues 25;
- hybrid k=5 rescues 23;
- BM25 k=3 rescues 22;
- BM25 k=5 rescues 22;
- dense k=3 rescues 18;
- dense k=5 rescues 15.

Four no-retrieval errors in this pilot are rescued by exactly one retrieval policy, including cases unique to hybrid k=3, BM25 k=5 and dense k=3. This suggests that retriever selection—not only retrieve/no-retrieve gating—may matter.

## Important limitations

1. N=200 is a pilot, not a final inferential sample.
2. Only same-language Hausa retrieval is tested here; multilingual/cross-lingual routing is not yet represented.
3. The current latency values reflect retrieval scoring after embeddings are available; final efficiency claims require end-to-end query-encoding and retrieval timing on frozen hardware.
4. The oracle is gold-aware and is only an upper-bound diagnostic.
5. Retrieval-signal logging has been expanded after this pilot to preserve raw top scores, margins, evidence labels and evidence scores for controller analysis.

## Decision

**GO** to full Hausa development evaluation with the same-language action set. If the full Hausa result preserves oracle headroom and help/harm heterogeneity, proceed to the full 12-language multilingual/cross-lingual development experiment on GPU-capable hardware.
