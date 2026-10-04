# Post-review robustness analyses

These analyses were added after independent manuscript review. They are
deliberately separated from the original Stage-1 and Stage-2 confirmatory/frozen
work.

## 1. Stage-2 transformer multi-seed robustness

Purpose: quantify training-seed variability for the fixed AfroXLM-R
configuration.

Fixed seeds:

- 42
- 43
- 44

No hyperparameter may change between seeds. The existing frozen seed-42 run is
reused by default; only missing seeds are trained.

Run:

```bash
python scripts/run_transformer_multiseed_robustness.py
```

Outputs:

- `outputs/transformer_multiseed/seed_metrics.csv`
- `outputs/transformer_multiseed/aggregate_metrics.csv`
- `outputs/transformer_multiseed/robustness_summary.json`

Interpretation: post-specified robustness evidence. Do not relabel it as part of
the original Stage-2 confirmatory design.

## 2. True action-space retraining ablation

Purpose: replace the earlier selection-space restriction diagnostic with a true
development-only ablation in which Controller v2 is retrained separately for
each available action subset.

Action subsets:

1. no retrieval only;
2. no retrieval + same-language actions;
3. no retrieval + same-language + multilingual actions;
4. full action space.

Each trainable subset uses exactly the same five-outer/three-inner grouped CV and
retrieval-penalty search used by Controller v2.

Run:

```bash
python scripts/run_true_action_space_ablation.py
```

Outputs:

- `outputs/publication/table_action_space_retrained_ablation.csv`
- `outputs/publication/action_space_retrained_fold_metrics.csv`
- `outputs/publication/action_space_retrained_selected_actions.csv`
- `outputs/publication/action_space_retrained_ablation.json`

This analysis reads development predictions only and must not be used to retune
the frozen held-out Stage-1 system.

## 3. MacroLangF1 paired bootstrap

Purpose: quantify uncertainty for the equal-weight language aggregate rather
than relying only on pooled Macro-F1.

Run:

```bash
python scripts/analyze_macrolang_bootstrap.py
```

Outputs:

- `outputs/publication/table_macrolang_bootstrap.csv`
- `outputs/publication/macrolang_bootstrap.json`

The bootstrap resamples paired queries independently within each language and
computes per-language Macro-F1 before averaging languages equally.

## Reporting rule

The manuscript should label:

- Stage-1 frozen test comparisons as confirmatory;
- Stage-2 as a separately frozen supporting follow-up;
- these new analyses as post-specified robustness/explanatory analyses.
