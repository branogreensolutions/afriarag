"""Apply the frozen retrieval-aware Controller v2 to a fixed-policy prediction file.

Selection uses only query/action/retrieval diagnostics and the frozen development
penalty. Gold labels are used only after selection to compute evaluation metrics.

Important: because retrieval diagnostics are computed for every candidate action
before routing, "retrieval_rate" below means selected-context rate, not the rate
of retrieval-computation calls.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

QUERY_NUM = [
    "char_len","token_len","avg_token_len","digit_ratio","upper_ratio",
    "punct_count","exclaim_count","question_count","mention_count",
    "hashtag_count","url_count","emoji_count","script_mix",
    "probe_confidence","probe_entropy",
]
RETRIEVAL_NUM = [
    "score_margin",
    "bm25_top_raw","bm25_margin_raw",
    "dense_top_raw","dense_margin_raw",
    "retr_neg_share","retr_neu_share","retr_pos_share",
    "retr_label_entropy","retr_majority_share",
    "retr_score_mean","retr_score_std","retr_score_last",
]
ACTION_NUM = ["k","alpha"]
CAT = ["language","policy","retriever","scope"]


def parse_labels(value):
    if pd.isna(value) or str(value).strip() == "":
        return {
            "retr_neg_share": 0.0,
            "retr_neu_share": 0.0,
            "retr_pos_share": 0.0,
            "retr_label_entropy": 0.0,
            "retr_majority_share": 0.0,
        }
    labels = [x.strip() for x in str(value).split(",") if x.strip()]
    n = max(1, len(labels))
    shares = {
        "negative": labels.count("negative") / n,
        "neutral": labels.count("neutral") / n,
        "positive": labels.count("positive") / n,
    }
    vals = np.asarray(list(shares.values()), dtype=float)
    nz = vals[vals > 0]
    entropy = float(-(nz * np.log(nz)).sum()) if len(nz) else 0.0
    return {
        "retr_neg_share": shares["negative"],
        "retr_neu_share": shares["neutral"],
        "retr_pos_share": shares["positive"],
        "retr_label_entropy": entropy,
        "retr_majority_share": float(vals.max()) if len(vals) else 0.0,
    }


def parse_scores(value):
    if pd.isna(value) or str(value).strip() == "":
        return {
            "retr_score_mean": 0.0,
            "retr_score_std": 0.0,
            "retr_score_last": 0.0,
        }
    vals = np.asarray(
        [float(x) for x in str(value).split(",") if str(x).strip() != ""],
        dtype=float,
    )
    if len(vals) == 0:
        return {
            "retr_score_mean": 0.0,
            "retr_score_std": 0.0,
            "retr_score_last": 0.0,
        }
    return {
        "retr_score_mean": float(vals.mean()),
        "retr_score_std": float(vals.std()),
        "retr_score_last": float(vals[-1]),
    }


def enrich(df):
    x = df.copy()
    lf = pd.DataFrame([parse_labels(v) for v in x["retrieved_labels"]], index=x.index)
    sf = pd.DataFrame([parse_scores(v) for v in x["retrieved_scores"]], index=x.index)
    for col in lf:
        x[col] = lf[col]
    for col in sf:
        x[col] = sf[col]
    for col in QUERY_NUM + RETRIEVAL_NUM + ACTION_NUM:
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="outputs/fixed/test_predictions.csv")
    ap.add_argument("--controller", default="outputs/controller_v2/controller.joblib")
    ap.add_argument("--controller-config", default="outputs/controller_v2/controller_config.json")
    ap.add_argument("--tag", default="test")
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    model = joblib.load(args.controller)
    controller_cfg = json.loads(Path(args.controller_config).read_text(encoding="utf-8"))
    penalty = float(controller_cfg["retrieval_penalty"])

    df = enrich(df)
    Xcols = QUERY_NUM + RETRIEVAL_NUM + ACTION_NUM + CAT

    prob = model.predict_proba(df[Xcols])[:, 1]
    df["success_probability"] = prob
    df["selection_utility"] = (
        df["success_probability"]
        - penalty * (df["k"].astype(float) > 0).astype(float)
    )

    idx = (
        df.groupby(["language","query_id"], sort=False)["selection_utility"]
        .idxmax()
    )
    picked = df.loc[idx].copy()

    overall = {
        "queries": int(len(picked)),
        "macro_f1": float(f1_score(picked.gold, picked.prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(picked.gold, picked.prediction, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(picked.gold, picked.prediction)),
        "selected_context_retrieval_rate": float((picked.k > 0).mean()),
        "avg_selected_k": float(picked.k.mean()),
        "retrieval_penalty": penalty,
        "selected_policy_counts": {
            str(k): int(v) for k, v in picked["policy"].value_counts().items()
        },
    }

    no = df[df.policy == "no_retrieval"].copy()
    if len(no):
        overall["no_retrieval_macro_f1"] = float(
            f1_score(no.gold, no.prediction, average="macro", zero_division=0)
        )
        overall["no_retrieval_accuracy"] = float(
            accuracy_score(no.gold, no.prediction)
        )
        overall["macro_f1_delta_vs_no_retrieval"] = (
            overall["macro_f1"] - overall["no_retrieval_macro_f1"]
        )

    per_lang = []
    for lang, g in picked.groupby("language"):
        row = {
            "language": lang,
            "queries": int(len(g)),
            "macro_f1": float(f1_score(g.gold, g.prediction, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(g.gold, g.prediction)),
            "selected_context_retrieval_rate": float((g.k > 0).mean()),
            "avg_selected_k": float(g.k.mean()),
        }
        nlang = no[no.language == lang]
        if len(nlang):
            row["no_retrieval_macro_f1"] = float(
                f1_score(nlang.gold, nlang.prediction, average="macro", zero_division=0)
            )
            row["macro_f1_delta_vs_no_retrieval"] = (
                row["macro_f1"] - row["no_retrieval_macro_f1"]
            )
        per_lang.append(row)

    outdir = Path("outputs/adaptive_v2")
    outdir.mkdir(parents=True, exist_ok=True)
    picked.to_csv(outdir / f"{args.tag}_selected_actions.csv", index=False)
    pd.DataFrame(per_lang).to_csv(outdir / f"{args.tag}_per_language.csv", index=False)
    (outdir / f"{args.tag}_summary.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )

    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
