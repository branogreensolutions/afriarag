"""Train the separately specified Stage-2 AfroXLM-R classifier baseline.

This is NOT a modification of frozen Stage 1. It is a modern encoder comparator
trained only on the leakage-safe supervised training labels and selected by the
supervised development split. Oromo/Tigrinya labels are never used for training
or model selection.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afriarag.core import load_split, read_config

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {x: i for i, x in enumerate(LABELS)}
ID2LABEL = {i: x for x, i in LABEL2ID.items()}


class TweetDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        enc = self.tokenizer(
            row.text,
            truncation=True,
            max_length=self.max_length,
        )
        enc["labels"] = LABEL2ID[str(row.label)]
        return enc


def concat_split(root, languages, split):
    return pd.concat(
        [load_split(root, lang, split) for lang in languages],
        ignore_index=True,
    )


def compute_metrics(eval_pred):
    logits, label_ids = eval_pred
    pred = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(label_ids, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(label_ids, pred, average="weighted", zero_division=0),
        "accuracy": accuracy_score(label_ids, pred),
        "macro_precision": precision_score(label_ids, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(label_ids, pred, average="macro", zero_division=0),
    }


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    cfg_path = Path("configs/transformer_baseline.yaml")
    cfg = read_config(cfg_path)
    seed = int(cfg["seed"])
    set_seed(seed)

    root = cfg["data"]["root"]
    languages = cfg["data"]["supervised_languages"]
    train_df = concat_split(root, languages, "train")
    dev_df = concat_split(root, languages, "dev")

    unknown = sorted((set(train_df.label) | set(dev_df.label)) - set(LABELS))
    if unknown:
        raise SystemExit(f"Unexpected labels: {unknown}")

    model_name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = TweetDataset(train_df, tokenizer, int(cfg["model"]["max_length"]))
    dev_ds = TweetDataset(dev_df, tokenizer, int(cfg["model"]["max_length"]))
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    outroot = Path(cfg["output"]["root"])
    checkpoint_dir = outroot / "checkpoints"
    model_dir = outroot / "model"
    outroot.mkdir(parents=True, exist_ok=True)

    tcfg = cfg["training"]
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

    dev_out = trainer.predict(dev_ds)
    pred_ids = np.argmax(dev_out.predictions, axis=-1)
    pred_labels = [ID2LABEL[int(x)] for x in pred_ids]

    pred_df = dev_df.copy()
    pred_df["prediction"] = pred_labels
    pred_df["correct"] = pred_df.label.eq(pred_df.prediction)
    pred_df.to_csv(outroot / "dev_predictions.csv", index=False)

    metrics = []
    for lang, g in pred_df.groupby("language"):
        metrics.append({
            "language": lang,
            "n": int(len(g)),
            "macro_f1": float(f1_score(g.label, g.prediction, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(g.label, g.prediction, average="weighted", zero_division=0)),
            "accuracy": float(accuracy_score(g.label, g.prediction)),
        })
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(outroot / "dev_per_language.csv", index=False)

    overall = {
        "stage": "Stage-2 separately specified transformer comparator",
        "model": model_name,
        "train_rows": int(len(train_df)),
        "dev_rows": int(len(dev_df)),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_dev_metric": trainer.state.best_metric,
        "dev_macro_f1": float(
            f1_score(pred_df.label, pred_df.prediction, average="macro", zero_division=0)
        ),
        "dev_accuracy": float(accuracy_score(pred_df.label, pred_df.prediction)),
        "zero_shot_target_labels_used": False,
    }
    (outroot / "dev_summary.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )

    freeze_files = [cfg_path]
    for name in [
        "config.json",
        "model.safetensors",
        "tokenizer_config.json",
        "tokenizer.json",
        "sentencepiece.bpe.model",
    ]:
        p = model_dir / name
        if p.exists():
            freeze_files.append(p)

    freeze = {
        "freeze_type": "Stage-2 transformer comparator pre-test freeze",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "label_map": LABEL2ID,
        "supervised_languages": languages,
        "zero_shot_languages": cfg["data"]["zero_shot_languages"],
        "target_labels_used": False,
        "files": [
            {
                "path": str(p),
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
            }
            for p in freeze_files
        ],
    }
    (outroot / "pretest_freeze_manifest.json").write_text(
        json.dumps(freeze, indent=2), encoding="utf-8"
    )

    print(json.dumps(overall, indent=2))
    print(f"saved model to {model_dir}")
    print(f"saved freeze manifest to {outroot / 'pretest_freeze_manifest.json'}")


if __name__ == "__main__":
    main()
