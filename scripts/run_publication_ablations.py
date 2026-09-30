"""Development-only publication ablations for AfriARAG Stage 1.

This script never uses held-out test labels. It performs:
1) nested grouped controller feature ablations on dev_predictions.csv;
2) action-space ablations using the already generated outer-fold OOF action scores;
3) writes publication tables/figures under outputs/publication/.

These analyses are explanatory/diagnostic and must not be used to alter the
frozen Stage-1 test systems.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_controller import (
    ACTION_NUM,
    CAT,
    QUERY_NUM,
    RETRIEVAL_NUM,
    enrich,
)

SCORE_NUM = [
    "score_margin",
    "bm25_top_raw", "bm25_margin_raw",
    "dense_top_raw", "dense_margin_raw",
    "retr_score_mean", "retr_score_std", "retr_score_last",
]

LABEL_NUM = [
    "retr_neg_share", "retr_neu_share", "retr_pos_share",
    "retr_label_entropy", "retr_majority_share",
]

FEATURE_SETS = {
    "query_action": QUERY_NUM + ACTION_NUM,
    "query_action_scores": QUERY_NUM + SCORE_NUM + ACTION_NUM,
    "query_action_labels": QUERY_NUM + LABEL_NUM + ACTION_NUM,
    "full": QUERY_NUM + RETRIEVAL_NUM + ACTION_NUM,
}


def make_pipeline(num_features):
    prep = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )
    return Pipeline([("prep", prep), ("model", model)])


def select_actions(scored, penalty):
    x = scored.copy()
    x["selection_utility"] = (
        x["success_probability"]
        - penalty * (pd.to_numeric(x["k"], errors="coerce").fillna(0) > 0).astype(float)
    )
    idx = x.groupby(["language", "query_id"], sort=False)["selection_utility"].idxmax()
    return x.loc[idx].copy()


def tune_penalty(scored, grid):
    rows = []
    for penalty in grid:
        chosen = select_actions(scored, penalty)
        rows.append({
            "retrieval_penalty": float(penalty),
            "macro_f1": float(
                f1_score(chosen.gold, chosen.prediction, average="macro", zero_division=0)
            ),
            "accuracy": float(accuracy_score(chosen.gold, chosen.prediction)),
            "retrieval_rate": float((pd.to_numeric(chosen.k, errors="coerce").fillna(0) > 0).mean()),
            "avg_k": float(pd.to_numeric(chosen.k, errors="coerce").fillna(0).mean()),
        })
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["macro_f1", "retrieval_rate", "retrieval_penalty"],
        ascending=[False, True, False],
    ).iloc[0]
    return float(best.retrieval_penalty)


def inner_oof(df, train_idx, num_features, inner_folds):
    train = df.iloc[train_idx].copy()
    groups = train.group_id
    gkf = GroupKFold(n_splits=min(inner_folds, groups.nunique()))
    cols = num_features + CAT
    parts = []

    for tr, va in gkf.split(train[cols], train.correct, groups=groups):
        model = make_pipeline(num_features)
        model.fit(train.iloc[tr][cols], train.iloc[tr].correct.astype(int))
        prob = model.predict_proba(train.iloc[va][cols])[:, 1]
        part = train.iloc[va].copy()
        part["success_probability"] = prob
        parts.append(part)

    return pd.concat(parts, ignore_index=True)


def feature_ablation(df, outer_folds, inner_folds, penalty_grid):
    groups = df.group_id
    outer = GroupKFold(n_splits=min(outer_folds, groups.nunique()))
    rows = []

    for name, num_features in FEATURE_SETS.items():
        print(f"[feature-ablation] {name}")
        chosen_parts = []
        penalties = []

        for fold, (tr, va) in enumerate(
            outer.split(df[num_features + CAT], df.correct, groups=groups),
            start=1,
        ):
            inner = inner_oof(df, tr, num_features, inner_folds)
            penalty = tune_penalty(inner, penalty_grid)
            penalties.append(penalty)

            model = make_pipeline(num_features)
            model.fit(df.iloc[tr][num_features + CAT], df.iloc[tr].correct.astype(int))
            prob = model.predict_proba(df.iloc[va][num_features + CAT])[:, 1]

            scored = df.iloc[va].copy()
            scored["success_probability"] = prob
            chosen = select_actions(scored, penalty)
            chosen["fold"] = fold
            chosen_parts.append(chosen)

        chosen = pd.concat(chosen_parts, ignore_index=True)
        rows.append({
            "ablation_type": "controller_features",
            "variant": name,
            "macro_f1": float(
                f1_score(chosen.gold, chosen.prediction, average="macro", zero_division=0)
            ),
            "accuracy": float(accuracy_score(chosen.gold, chosen.prediction)),
            "retrieval_rate": float(
                (pd.to_numeric(chosen.k, errors="coerce").fillna(0) > 0).mean()
            ),
            "avg_k": float(pd.to_numeric(chosen.k, errors="coerce").fillna(0).mean()),
            "mean_outer_penalty": float(np.mean(penalties)),
            "notes": "nested grouped CV on development data only",
        })

    return pd.DataFrame(rows)


def action_space_ablation(action_scores, dev_predictions):
    x = action_scores.copy()
    x["k"] = pd.to_numeric(x.k, errors="coerce").fillna(0)
    if "retrieval_penalty" not in x.columns:
        raise SystemExit(
            "nested_cv_action_scores.csv must contain the fold-specific retrieval_penalty column."
        )

    subsets = {
        "no_retrieval_only": lambda d: d.policy.eq("no_retrieval"),
        "no_plus_same": lambda d: d.scope.isin(["none", "same"]),
        "no_plus_same_plus_multi": lambda d: ~d.scope.eq("cross"),
        "full_action_space": lambda d: pd.Series(True, index=d.index),
    }

    rows = []
    for name, mask_fn in subsets.items():
        s = x[mask_fn(x)].copy()
        s["selection_utility_ablation"] = (
            s.success_probability
            - s.retrieval_penalty.astype(float) * (s.k > 0).astype(float)
        )
        idx = s.groupby(["language", "query_id"], sort=False)[
            "selection_utility_ablation"
        ].idxmax()
        chosen = s.loc[idx].copy()

        allowed = set(s.policy.unique())
        dp = dev_predictions[dev_predictions.policy.isin(allowed)].copy()
        oracle = (
            dp.groupby(["language", "query_id"])["correct"]
            .max()
            .astype(bool)
        )

        rows.append({
            "ablation_type": "action_space",
            "variant": name,
            "macro_f1": float(
                f1_score(chosen.gold, chosen.prediction, average="macro", zero_division=0)
            ),
            "accuracy": float(accuracy_score(chosen.gold, chosen.prediction)),
            "retrieval_rate": float((chosen.k > 0).mean()),
            "avg_k": float(chosen.k.mean()),
            "mean_outer_penalty": float(chosen.retrieval_penalty.mean()),
            "oracle_resolvable_rate": float(oracle.mean()),
            "notes": "OOF dev action scores; no retraining; explanatory action-space restriction",
        })

    return pd.DataFrame(rows)


def plot_ablation(df, ablation_type, title, out):
    x = df[df.ablation_type == ablation_type].sort_values("macro_f1").copy()
    fig, ax = plt.subplots(figsize=(8.5, max(4.2, len(x) * 0.55)))
    ax.barh(np.arange(len(x)), x.macro_f1.to_numpy())
    ax.set(
        yticks=np.arange(len(x)),
        yticklabels=x.variant.tolist(),
        xlabel="Development Macro-F1",
        title=title,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-predictions", default="outputs/fixed/dev_predictions.csv")
    ap.add_argument("--oof-action-scores", default="outputs/controller_v2/nested_cv_action_scores.csv")
    ap.add_argument("--outer-folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=3)
    ap.add_argument("--penalty-max", type=float, default=0.15)
    ap.add_argument("--penalty-step", type=float, default=0.005)
    args = ap.parse_args()

    dev_path = Path(args.dev_predictions)
    score_path = Path(args.oof_action_scores)
    if not dev_path.exists():
        raise SystemExit(f"Missing: {dev_path}")
    if not score_path.exists():
        raise SystemExit(f"Missing: {score_path}")

    raw = pd.read_csv(dev_path)
    df = enrich(raw)
    action_scores = pd.read_csv(score_path)

    grid = np.arange(
        0.0,
        args.penalty_max + args.penalty_step / 2,
        args.penalty_step,
    )

    feature = feature_ablation(
        df,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        penalty_grid=grid,
    )
    actions = action_space_ablation(action_scores, raw)
    table = pd.concat([feature, actions], ignore_index=True)

    outdir = Path("outputs/publication")
    outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(outdir / "table_ablations.csv", index=False)

    plot_ablation(
        table,
        "controller_features",
        "Controller feature ablation — development only",
        outdir / "controller_feature_ablation.png",
    )
    plot_ablation(
        table,
        "action_space",
        "Action-space ablation — development only",
        outdir / "action_space_ablation.png",
    )

    print(table.to_string(index=False))
    print(f"saved {outdir / 'table_ablations.csv'}")


if __name__ == "__main__":
    main()
