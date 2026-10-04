"""Evaluate the frozen Stage-2 AfroXLM-R classifier comparator.

The model must already be frozen by train_transformer_baseline.py. This script
never trains or tunes. It evaluates:
1) the 12 supervised held-out test languages;
2) strict zero-shot Oromo/Tigrinya, with no target-language labels used in training.

Because Stage 2 was specified after Stage-1 test inspection, these results are
supporting/exploratory comparisons, not replacements for the Stage-1
confirmatory analysis.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, XLMRobertaTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afriarag.core import load_split, read_config

LABELS = ["negative", "neutral", "positive"]


class TextOnlyDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.tokenizer(
            self.frame.iloc[idx].text,
            truncation=True,
            max_length=self.max_length,
        )


def concat_split(root, languages, split):
    return pd.concat(
        [load_split(root, lang, split) for lang in languages],
        ignore_index=True,
    )


def predict(frame, tokenizer, model, max_length, batch_size, device):
    ds = TextOnlyDataset(frame, tokenizer, max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    preds = []
    probs = []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            p = torch.softmax(logits, dim=-1)
            probs.append(p.detach().cpu().numpy())
            preds.append(torch.argmax(p, dim=-1).detach().cpu().numpy())

    pred_ids = np.concatenate(preds)
    prob = np.concatenate(probs)
    out = frame.copy()
    out["prediction"] = [
        model.config.id2label.get(int(i), LABELS[int(i)]).lower()
        for i in pred_ids
    ]
    out["correct"] = out.label.eq(out.prediction)
    for i, label in enumerate(LABELS):
        out[f"prob_{label}"] = prob[:, i]
    return out


def summarize(frame):
    rows = []
    for lang, g in frame.groupby("language"):
        rows.append({
            "language": lang,
            "n": int(len(g)),
            "macro_f1": float(f1_score(g.label, g.prediction, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(g.label, g.prediction, average="weighted", zero_division=0)),
            "accuracy": float(accuracy_score(g.label, g.prediction)),
        })
    return pd.DataFrame(rows)


def overall(frame):
    return {
        "n": int(len(frame)),
        "languages": int(frame.language.nunique()),
        "macro_f1": float(f1_score(frame.label, frame.prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(frame.label, frame.prediction, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(frame.label, frame.prediction)),
        "macro_lang_f1": float(summarize(frame).macro_f1.mean()),
    }


def main():
    cfg = read_config("configs/transformer_baseline.yaml")
    outroot = Path(cfg["output"]["root"])
    model_dir = outroot / "model"
    freeze_manifest = outroot / "pretest_freeze_manifest.json"

    if not freeze_manifest.exists():
        raise SystemExit(
            "Missing transformer pre-test freeze manifest. Train/freeze the comparator before test evaluation."
        )

    # AfroXLM-R is an XLM-R model. Loading the locally saved tokenizer through
    # AutoTokenizer can trigger Transformers' known false-positive
    # fix_mistral_regex warning for non-Mistral tokenizers. Use the explicit
    # XLM-R fast tokenizer class to preserve the saved tokenizer without
    # applying an unrelated Mistral regex rewrite.
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    root = cfg["data"]["root"]
    supervised = cfg["data"]["supervised_languages"]
    zero = cfg["data"]["zero_shot_languages"]
    max_length = int(cfg["model"]["max_length"])
    batch_size = int(cfg["training"]["eval_batch_size"])

    supervised_df = concat_split(root, supervised, "test")
    zero_df = concat_split(root, zero, "test")

    sup_pred = predict(
        supervised_df, tokenizer, model, max_length, batch_size, device
    )
    zs_pred = predict(
        zero_df, tokenizer, model, max_length, batch_size, device
    )

    sup_pred.to_csv(outroot / "supervised_test_predictions.csv", index=False)
    zs_pred.to_csv(outroot / "zero_shot_test_predictions.csv", index=False)

    sup_lang = summarize(sup_pred)
    zs_lang = summarize(zs_pred)
    sup_lang.to_csv(outroot / "supervised_test_per_language.csv", index=False)
    zs_lang.to_csv(outroot / "zero_shot_test_per_language.csv", index=False)

    summary = {
        "stage": "Stage-2 supporting transformer comparator",
        "model": cfg["model"]["name"],
        "supervised_test": overall(sup_pred),
        "strict_zero_shot_test": overall(zs_pred),
        "target_language_labels_used_for_training_or_selection": False,
        "interpretation": (
            "Supporting/exploratory comparator specified after Stage-1 test inspection; "
            "do not relabel as a Stage-1 confirmatory result."
        ),
    }
    (outroot / "test_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
