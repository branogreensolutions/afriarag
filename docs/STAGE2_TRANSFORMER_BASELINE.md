# Stage 2 — Strong Transformer Comparator

Stage 2 is a separately specified supporting experiment created after the
Stage-1 held-out results were inspected. It must not be described as part of the
original Stage-1 confirmatory evaluation.

## Purpose

Stage 1 intentionally used a lightweight character n-gram classifier to isolate
retrieval behavior. For publication, a reviewer may reasonably ask whether the
retrieval findings remain meaningful relative to a modern pretrained
African/multilingual encoder.

Stage 2 therefore adds one strong classifier comparator:

`Davlan/afro-xlmr-base-114L`

The model is fine-tuned only on the 12 leakage-safe supervised AfriSenti
training languages.

## Fixed configuration

Configuration: `configs/transformer_baseline.yaml`

Primary choices are specified before Stage-2 held-out evaluation:

- maximum sequence length: 128
- epochs: 3 maximum
- learning rate: 2e-5
- weight decay: 0.01
- warmup ratio: 0.10
- effective batch size: 32 (8 x 4 gradient accumulation)
- development selection metric: Macro-F1
- early-stopping patience: 1 epoch
- seed: 42

The best epoch is selected only using the leakage-safe 12-language development
set.

## Strict zero-shot rule

Oromo and Tigrinya labels are not used for:

- training;
- best-epoch selection;
- hyperparameter selection;
- tokenizer/model adaptation.

They are evaluated only after the Stage-2 model and configuration have been
frozen.

The pretrained encoder may have encountered unlabeled Oromo/Tigrinya text during
language-model pretraining. This is compatible with label-zero-shot transfer
and must be stated explicitly.

## Commands

Install optional dependency:

```bash
python -m pip install -r requirements-transformer.txt
```

Train and create a separate pre-test freeze:

```bash
python scripts/train_transformer_baseline.py
```

Review:

- `outputs/transformer_baseline/dev_summary.json`
- `outputs/transformer_baseline/dev_per_language.csv`
- `outputs/transformer_baseline/pretest_freeze_manifest.json`

Then evaluate exactly once:

```bash
python scripts/evaluate_transformer_baseline.py
```

## Interpretation boundary

Stage-2 results may be used as a strong supporting baseline and robustness
comparison. They cannot retroactively convert any Stage-1 exploratory or
non-significant claim into a preregistered confirmatory result.

No Stage-1 model, controller, penalty, or prediction is changed by this
experiment.
