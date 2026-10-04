"""Post-specified 3-seed robustness analysis for the frozen Stage-2 AfroXLM-R setup.

The hyperparameters are copied from configs/transformer_baseline.yaml and are
NOT retuned. The original seed-42 result may be reused; by default this script
trains only missing seeds 43 and 44, then aggregates seeds 42/43/44.

This analysis was specified after the original Stage-2 held-out result and must
be reported as robustness evidence, not as a new confirmatory experiment.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    XLMRobertaTokenizerFast,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afriarag.core import load_split, read_config

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {x: i for i, x in enumerate(LABELS)}
ID2LABEL = {i: x for x, i in LABEL2ID.items()}
FIXED_SEEDS = [42, 43, 44]


class TweetDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length, with_labels=True):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_labels = with_labels

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        enc = self.tokenizer(
            row.text,
            truncation=True,
            max_length=self.max_length,
        )
        if self.with_labels:
            enc["labels"] = LABEL2ID[str(row.label)]
        return enc


def concat_split(root, languages, split):
    return pd.concat(
        [load_split(root, lang, split) for lang in languages],
        ignore_index=True,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(labels, pred),
    }


def per_language(frame):
    rows = []
    for lang, g in frame.groupby("language"):
        rows.append({
            "language": lang,
            "n": int(len(g)),
            "macro_f1": float(
                f1_score(g.label, g.prediction, average="macro", zero_division=0)
            ),
            "accuracy": float(accuracy_score(g.label, g.prediction)),
        })
    return pd.DataFrame(rows)


def summarize(frame):
    pl = per_language(frame)
    return {
        "n": int(len(frame)),
        "languages": int(frame.language.nunique()),
        "macro_f1": float(
            f1_score(frame.label, frame.prediction, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(frame.label, frame.prediction)),
        "macro_lang_f1": float(pl.macro_f1.mean()),
    }


def predict_frame(frame, tokenizer, model, max_length, batch_size, device):
    ds = TweetDataset(frame, tokenizer, max_length, with_labels=False)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    preds = []

    model.eval()
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.append(torch.argmax(logits, dim=-1).detach().cpu().numpy())

    ids = np.concatenate(preds)
    out = frame.copy()
    out["prediction"] = [ID2LABEL[int(i)] for i in ids]
    out["correct"] = out.label.eq(out.prediction)
    return out


def train_and_evaluate_seed(cfg, seed, seed_dir):
    set_seed(seed)
    root = cfg["data"]["root"]
    supervised = cfg["data"]["supervised_languages"]
    zero = cfg["data"]["zero_shot_languages"]
    max_length = int(cfg["model"]["max_length"])
    model_name = cfg["model"]["name"]
    tcfg = cfg["training"]

    train_df = concat_split(root, supervised, "train")
    dev_df = concat_split(root, supervised, "dev")
    test_df = concat_split(root, supervised, "test")
    zero_df = concat_split(root, zero, "test")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = TweetDataset(train_df, tokenizer, max_length, with_labels=True)
    dev_ds = TweetDataset(dev_df, tokenizer, max_length, with_labels=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    checkpoint_dir = seed_dir / "checkpoints"
    model_dir = seed_dir / "model"
    seed_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = bool(tcfg.get("fp16", True)) and torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        learning_rate=float(tcfg["learning_rate"]),
        per_device_train_batch_size=int(tcfg["train_batch_size"]),
        per_device_eval_batch_size=int(tcfg["eval_batch_size"]),
        gradient_accumulation_steps=int(tcfg["gradient_accumulation_steps"]),
        num_train_epochs=float(tcfg["epochs"]),
        weight_decay=float(tcfg["weight_decay"]),
        warmup_ratio=float(tcfg["warmup_ratio"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model=str(tcfg["metric_for_best_model"]),
        greater_is_better=True,
        save_total_limit=2,
        seed=seed,
        data_seed=seed,
        fp16=use_fp16,
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", True)),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=int(tcfg["early_stopping_patience"])
            )
        ],
    )
    trainer.train()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    # Reload explicitly as XLM-R to avoid the unrelated Mistral-regex warning.
    tok_eval = XLMRobertaTokenizerFast.from_pretrained(
        model_dir,
        fix_mistral_regex=False,
    )
    eval_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_model.to(device)

    sup_pred = predict_frame(
        test_df, tok_eval, eval_model, max_length, int(tcfg["eval_batch_size"]), device
    )
    zs_pred = predict_frame(
        zero_df, tok_eval, eval_model, max_length, int(tcfg["eval_batch_size"]), device
    )
    dev_pred = predict_frame(
        dev_df, tok_eval, eval_model, max_length, int(tcfg["eval_batch_size"]), device
    )

    sup_pred.to_csv(seed_dir / "supervised_test_predictions.csv", index=False)
    zs_pred.to_csv(seed_dir / "zero_shot_test_predictions.csv", index=False)
    dev_pred.to_csv(seed_dir / "dev_predictions.csv", index=False)

    per_language(sup_pred).to_csv(
        seed_dir / "supervised_test_per_language.csv", index=False
    )
    per_language(zs_pred).to_csv(
        seed_dir / "zero_shot_test_per_language.csv", index=False
    )

    summary = {
        "seed": seed,
        "analysis_status": "post-specified robustness; fixed hyperparameters",
        "model": model_name,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_dev_metric": trainer.state.best_metric,
        "development": summarize(dev_pred),
        "supervised_test": summarize(sup_pred),
        "strict_zero_shot_test": summarize(zs_pred),
        "target_language_labels_used_for_training_or_selection": False,
    }
    (seed_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def reuse_original_seed42(cfg, root):
    original = Path(cfg["output"]["root"])
    required = [
        original / "dev_summary.json",
        original / "test_summary.json",
        original / "dev_predictions.csv",
        original / "supervised_test_predictions.csv",
        original / "zero_shot_test_predictions.csv",
    ]
    if not all(p.exists() for p in required):
        return None

    target = root / "seed_42"
    target.mkdir(parents=True, exist_ok=True)

    test_summary = json.loads(
        (original / "test_summary.json").read_text(encoding="utf-8")
    )
    dev_summary = json.loads(
        (original / "dev_summary.json").read_text(encoding="utf-8")
    )
    dev = pd.read_csv(original / "dev_predictions.csv")
    sup = pd.read_csv(original / "supervised_test_predictions.csv")
    zs = pd.read_csv(original / "zero_shot_test_predictions.csv")

    if any("label" not in df.columns for df in [dev, sup, zs]):
        raise RuntimeError("Original seed-42 prediction files are missing labels.")

    # Recompute seed-42 metrics through the same functions used for new seeds.
    summary = {
        "seed": 42,
        "analysis_status": "post-specified robustness; reused original frozen seed-42 run",
        "model": cfg["model"]["name"],
        "best_model_checkpoint": dev_summary.get("best_model_checkpoint"),
        "best_dev_metric": dev_summary.get("best_dev_metric"),
        "development": summarize(dev),
        "supervised_test": summarize(sup),
        "strict_zero_shot_test": summarize(zs),
        "target_language_labels_used_for_training_or_selection": False,
    }

    shutil.copy2(
        original / "dev_predictions.csv",
        target / "dev_predictions.csv",
    )
    shutil.copy2(
        original / "supervised_test_predictions.csv",
        target / "supervised_test_predictions.csv",
    )
    shutil.copy2(
        original / "zero_shot_test_predictions.csv",
        target / "zero_shot_test_predictions.csv",
    )
    (target / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def aggregate(root, summaries):
    rows = []
    for s in summaries:
        for setting in ["supervised_test", "strict_zero_shot_test"]:
            block = s[setting]
            rows.append({
                "seed": s["seed"],
                "setting": setting,
                "macro_f1": block["macro_f1"],
                "accuracy": block["accuracy"],
                "macro_lang_f1": block["macro_lang_f1"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(root / "seed_metrics.csv", index=False)

    agg_rows = []
    for setting, g in df.groupby("setting"):
        agg_rows.append({
            "setting": setting,
            "seeds": ",".join(map(str, sorted(g.seed.tolist()))),
            "n_seeds": int(len(g)),
            "macro_f1_mean": float(g.macro_f1.mean()),
            "macro_f1_std": float(g.macro_f1.std(ddof=1)),
            "accuracy_mean": float(g.accuracy.mean()),
            "accuracy_std": float(g.accuracy.std(ddof=1)),
            "macro_lang_f1_mean": float(g.macro_lang_f1.mean()),
            "macro_lang_f1_std": float(g.macro_lang_f1.std(ddof=1)),
        })

    agg = pd.DataFrame(agg_rows)
    agg.to_csv(root / "aggregate_metrics.csv", index=False)
    payload = {
        "analysis_status": "post-specified 3-seed robustness",
        "fixed_seeds": FIXED_SEEDS,
        "hyperparameter_source": "configs/transformer_baseline.yaml",
        "target_labels_used_for_training_or_selection": False,
        "aggregate": agg_rows,
    }
    (root / "robustness_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(agg.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/transformer_baseline.yaml")
    ap.add_argument(
        "--force-seed42-retrain",
        action="store_true",
        help="Retrain seed 42 instead of reusing the already frozen original run.",
    )
    args = ap.parse_args()

    cfg = read_config(args.config)
    root = Path("outputs/transformer_multiseed")
    root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for seed in FIXED_SEEDS:
        seed_dir = root / f"seed_{seed}"
        summary_path = seed_dir / "summary.json"

        if summary_path.exists():
            print(f"[multiseed] reuse completed seed {seed}: {summary_path}")
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue

        if seed == 42 and not args.force_seed42_retrain:
            reused = reuse_original_seed42(cfg, root)
            if reused is not None:
                print("[multiseed] reused original frozen seed-42 Stage-2 outputs")
                summaries.append(reused)
                continue

        print(f"[multiseed] training fixed configuration with seed={seed}")
        summaries.append(train_and_evaluate_seed(cfg, seed, seed_dir))

    aggregate(root, summaries)


if __name__ == "__main__":
    main()
