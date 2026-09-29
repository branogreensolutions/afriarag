import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afriarag.core import read_config

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

    label_features = pd.DataFrame(
        [parse_labels(v) for v in x["retrieved_labels"]],
        index=x.index,
    )
    score_features = pd.DataFrame(
        [parse_scores(v) for v in x["retrieved_scores"]],
        index=x.index,
    )
    for col in label_features:
        x[col] = label_features[col]
    for col in score_features:
        x[col] = score_features[col]

    for col in RETRIEVAL_NUM + ACTION_NUM + QUERY_NUM:
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)

    x["group_id"] = x["language"].astype(str) + "::" + x["query_id"].astype(str)
    return x


def make_pipeline():
    prep = ColumnTransformer([
        ("num", StandardScaler(), QUERY_NUM + RETRIEVAL_NUM + ACTION_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )
    return Pipeline([("prep", prep), ("model", model)])


def select_actions(scored, retrieval_penalty):
    x = scored.copy()
    x["selection_utility"] = (
        x["success_probability"]
        - retrieval_penalty * (x["k"].astype(float) > 0).astype(float)
    )
    idx = (
        x.groupby(["language","query_id"], sort=False)["selection_utility"]
        .idxmax()
    )
    return x.loc[idx].copy()


def selection_macro_f1(scored, penalty):
    chosen = select_actions(scored, penalty)
    return float(
        f1_score(
            chosen["gold"],
            chosen["prediction"],
            average="macro",
            zero_division=0,
        )
    )


def make_oof_scores(df, train_idx, inner_folds):
    train_df = df.iloc[train_idx].copy()
    groups = train_df["group_id"]
    unique_groups = groups.nunique()
    n_splits = min(inner_folds, unique_groups)
    if n_splits < 2:
        raise RuntimeError("Not enough query groups for inner CV.")

    parts = []
    Xcols = QUERY_NUM + RETRIEVAL_NUM + ACTION_NUM + CAT
    gkf = GroupKFold(n_splits=n_splits)

    for itr, iva in gkf.split(train_df[Xcols], train_df["correct"], groups=groups):
        model = make_pipeline()
        model.fit(train_df.iloc[itr][Xcols], train_df.iloc[itr]["correct"].astype(int))
        prob = model.predict_proba(train_df.iloc[iva][Xcols])[:, 1]

        part = train_df.iloc[iva].copy()
        part["success_probability"] = prob
        parts.append(part)

    return pd.concat(parts, ignore_index=True)


def tune_penalty(inner_scored, grid):
    rows = []
    for penalty in grid:
        chosen = select_actions(inner_scored, penalty)
        rows.append({
            "retrieval_penalty": float(penalty),
            "macro_f1": float(
                f1_score(
                    chosen["gold"],
                    chosen["prediction"],
                    average="macro",
                    zero_division=0,
                )
            ),
            "accuracy": float(accuracy_score(chosen["gold"], chosen["prediction"])),
            "retrieval_rate": float((chosen["k"] > 0).mean()),
            "avg_k": float(chosen["k"].mean()),
        })

    table = pd.DataFrame(rows)
    # Deterministic tie-break: best Macro-F1, then lower retrieval rate,
    # then the larger penalty.
    best = table.sort_values(
        ["macro_f1","retrieval_rate","retrieval_penalty"],
        ascending=[False, True, False],
    ).iloc[0]
    return float(best["retrieval_penalty"]), table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="outputs/fixed/dev_predictions.csv")
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--outer-folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=3)
    ap.add_argument("--penalty-max", type=float, default=0.15)
    ap.add_argument("--penalty-step", type=float, default=0.005)
    args = ap.parse_args()

    cfg = read_config(args.config)
    raw = pd.read_csv(args.predictions)

    required = set(
        QUERY_NUM
        + ACTION_NUM
        + CAT
        + [
            "query_id","gold","prediction","correct",
            "score_margin",
            "bm25_top_raw","bm25_margin_raw",
            "dense_top_raw","dense_margin_raw",
            "retrieved_labels","retrieved_scores",
        ]
    )
    missing = sorted(required - set(raw.columns))
    if missing:
        raise SystemExit(
            "The prediction file predates retrieval-aware controller logging. "
            f"Missing columns: {missing}"
        )

    df = enrich(raw)
    Xcols = QUERY_NUM + RETRIEVAL_NUM + ACTION_NUM + CAT
    groups = df["group_id"]

    penalty_grid = np.arange(
        0.0,
        args.penalty_max + args.penalty_step / 2,
        args.penalty_step,
    )

    n_outer = min(args.outer_folds, groups.nunique())
    outer = GroupKFold(n_splits=n_outer)

    selected_parts = []
    action_score_parts = []
    fold_metrics = []
    penalty_tables = []

    for fold, (tr, va) in enumerate(
        outer.split(df[Xcols], df["correct"], groups=groups),
        start=1,
    ):
        print(f"[controller] outer fold {fold}/{n_outer}: tuning retrieval penalty")

        inner_scored = make_oof_scores(df, tr, args.inner_folds)
        best_penalty, penalty_table = tune_penalty(inner_scored, penalty_grid)
        penalty_table["outer_fold"] = fold
        penalty_tables.append(penalty_table)

        model = make_pipeline()
        model.fit(df.iloc[tr][Xcols], df.iloc[tr]["correct"].astype(int))
        prob = model.predict_proba(df.iloc[va][Xcols])[:, 1]

        scored = df.iloc[va].copy()
        scored["fold"] = fold
        scored["success_probability"] = prob
        scored["retrieval_penalty"] = best_penalty
        action_score_parts.append(scored)

        chosen = select_actions(scored, best_penalty)
        selected_parts.append(chosen)

        auc = None
        yva = df.iloc[va]["correct"].astype(int)
        if yva.nunique() > 1:
            auc = float(roc_auc_score(yva, prob))

        fold_metrics.append({
            "fold": fold,
            "queries": int(chosen["group_id"].nunique()),
            "retrieval_penalty": best_penalty,
            "macro_f1": float(
                f1_score(
                    chosen["gold"],
                    chosen["prediction"],
                    average="macro",
                    zero_division=0,
                )
            ),
            "accuracy": float(
                accuracy_score(chosen["gold"], chosen["prediction"])
            ),
            "retrieval_rate": float((chosen["k"] > 0).mean()),
            "avg_k": float(chosen["k"].mean()),
            "action_success_auc": auc,
        })

    selected = pd.concat(selected_parts, ignore_index=True)
    action_scores = pd.concat(action_score_parts, ignore_index=True)
    penalties = pd.concat(penalty_tables, ignore_index=True)

    no_rag = df[df["policy"] == "no_retrieval"].copy()

    overall = {
        "queries": int(selected["group_id"].nunique()),
        "macro_f1": float(
            f1_score(
                selected["gold"],
                selected["prediction"],
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                selected["gold"],
                selected["prediction"],
                average="weighted",
                zero_division=0,
            )
        ),
        "accuracy": float(
            accuracy_score(selected["gold"], selected["prediction"])
        ),
        "retrieval_rate": float((selected["k"] > 0).mean()),
        "avg_k": float(selected["k"].mean()),
        "mean_outer_penalty": float(
            np.mean([r["retrieval_penalty"] for r in fold_metrics])
        ),
        "no_retrieval_macro_f1": float(
            f1_score(
                no_rag["gold"],
                no_rag["prediction"],
                average="macro",
                zero_division=0,
            )
        ),
        "no_retrieval_accuracy": float(
            accuracy_score(no_rag["gold"], no_rag["prediction"])
        ),
        "macro_f1_delta_vs_no_retrieval": float(
            f1_score(
                selected["gold"],
                selected["prediction"],
                average="macro",
                zero_division=0,
            )
            - f1_score(
                no_rag["gold"],
                no_rag["prediction"],
                average="macro",
                zero_division=0,
            )
        ),
        "selected_policy_counts": {
            str(k): int(v)
            for k, v in selected["policy"].value_counts().items()
        },
    }

    # Final dev-tuned penalty for the deployable controller. This uses only
    # out-of-fold development probabilities; the test set remains untouched.
    final_penalty, final_penalty_table = tune_penalty(
        action_scores,
        penalty_grid,
    )
    overall["final_dev_tuned_retrieval_penalty"] = final_penalty

    final_model = make_pipeline()
    final_model.fit(df[Xcols], df["correct"].astype(int))

    outdir = Path("outputs/controller_v2")
    outdir.mkdir(parents=True, exist_ok=True)

    selected.to_csv(outdir / "nested_cv_selected_actions.csv", index=False)
    action_scores.to_csv(outdir / "nested_cv_action_scores.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(
        outdir / "nested_cv_metrics.csv",
        index=False,
    )
    penalties.to_csv(outdir / "nested_penalty_search.csv", index=False)
    final_penalty_table.to_csv(
        outdir / "final_dev_penalty_search.csv",
        index=False,
    )
    (outdir / "nested_cv_summary.json").write_text(
        json.dumps(overall, indent=2),
        encoding="utf-8",
    )

    joblib.dump(final_model, outdir / "controller.joblib")
    (outdir / "controller_config.json").write_text(
        json.dumps({
            "retrieval_penalty": final_penalty,
            "query_numeric": QUERY_NUM,
            "retrieval_numeric": RETRIEVAL_NUM,
            "action_numeric": ACTION_NUM,
            "categorical": CAT,
            "target": "P(action produces correct sentiment | query, retrieval diagnostics, action)",
            "selection": "argmax(P(correct)-retrieval_penalty*I[k>0])",
            "validation": "nested grouped cross-validation by query",
        }, indent=2),
        encoding="utf-8",
    )

    print(pd.DataFrame(fold_metrics).to_string(index=False))
    print(json.dumps(overall, indent=2))
    print(f"saved {outdir / 'controller.joblib'}")


if __name__ == "__main__":
    main()
