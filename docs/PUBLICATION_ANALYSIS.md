# Publication Analysis Protocol

This document defines the post-freeze analyses used to turn AfriARAG Stage 1
into publication-ready tables, figures, and qualitative evidence.

## Principle

No script in this publication-analysis phase may alter:

- the frozen leakage-safe dataset;
- supervised Controller v2;
- the supervised retrieval penalty;
- the zero-shot router;
- the zero-shot retrieval penalty;
- the Stage-1 action spaces;
- held-out predictions.

The package may summarize, stratify, visualize, or bootstrap frozen outputs.
Development-only ablations may retrain diagnostic models only on development
data and must not be used to revise the held-out Stage-1 system.

## Commands

Development-only explanatory ablations:

```bash
python scripts/run_publication_ablations.py
```

Frozen held-out publication package:

```bash
python scripts/build_publication_package.py
```

## Required package outputs

### Tables

- `table_main_results.csv`
- `table_fixed_policies.csv`
- `table_per_language.csv`
- `table_per_class.csv`
- `table_rescue_harm.csv`
- `table_ablations.csv`
- `efficiency_summary.csv`
- `rescue_harm_examples.csv`

### Figures

- `confusion_supervised_adaptive.png`
- `confusion_supervised_no_retrieval.png`
- `confusion_zero_shot_adaptive.png`
- `confusion_zero_shot_no_retrieval.png`
- `per_language_delta.png`
- `retrieval_policy_comparison_supervised.png`
- `retrieval_policy_comparison_zero_shot.png`
- `controller_feature_ablation.png`
- `action_space_ablation.png`

### Audit

- `publication_manifest.json`
- `README.md`

## Statistical labeling

The supervised adaptive-vs-no-retrieval comparison is confirmatory because no
retrieval was the strongest global fixed policy on development data.

The zero-shot adaptive-vs-no-retrieval comparison is likewise the frozen
confirmatory comparison.

The discovery that `dense_cross_k3` is the strongest fixed zero-shot policy on
the held-out Oromo/Tigrinya test is descriptive/post-hoc. It must not be used to
retune Stage 1.

## Efficiency labeling

The controller uses retrieval diagnostics before selecting an action.
Accordingly:

- report **selected-context retrieval rate**;
- do not call it total retrieval-computation rate;
- separate retrieval scoring latency from query-embedding/probe cost;
- separate both from downstream classifier/LLM token cost.

## Qualitative examples

Rescue/harm examples are selected deterministically from frozen outputs. They
are explanatory examples, not an additional evaluation set.

## Stage 2 boundary

A stronger encoder classifier, a new router, or an LLM-based RAG layer is a
separately specified Stage-2 experiment. Stage-2 results must not be presented
as if they were part of the original pre-test Stage-1 freeze.
