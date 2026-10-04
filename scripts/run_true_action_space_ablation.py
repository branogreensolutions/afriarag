"""True development-only action-space retraining ablation.

Unlike the earlier selection-space restriction diagnostic, this script retrains
a separate Controller v2 inside each action subset using the same nested grouped
CV and inner penalty tuning protocol. No held-out test labels are read.

This is a post-specified explanatory robustness analysis and must not be used to
change the frozen Stage-1 system.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_controller import (
    ACTION_NUM,
    CAT,
    QUERY_NUM,
    RETRIEVAL_NUM,
    enrich,
    make_oof_scores,
    make_pipeline,
    select_actions,
    tune_penalty,
)

ACTION_SETS = {
    "no_retrieval_only": ["no_retrieval"],
    "no_plus_same": [
        "no_retrieval",
        "bm25_same_k3", "bm25_same_k5",
        "dense_same_k3", "dense_same_k5",
        "hybrid_same_k3", "hybrid_same_k5",
    ],
    "no_plus_same_plus_multi": [
        "no_retrieval",
        "bm25_same_k3", "bm25_same_k5",
        "dense_same_k3", "dense_same_k5",
        "hybrid_same_k3", "hybrid_same_k5",
        "dense_multi_k3", "hybrid_multi_k3",
    ],
    "full_action_space": [
        "no_retrieval",
        "bm25_same_k3", "bm25_same_k5",
        "dense_same_k3", "dense_same_k5",
        "hybrid_same_k3", "hybrid_same_k5",
        "dense_multi_k3", "hybrid_multi_k3",
        "dense_cross_k3", "hybrid_cross_k3", "hybrid_cross_k5",
    ],
}


def oracle_resolvable(raw_subset):
    return float(
        raw_subset.groupby(["language", "query_id"])["correct"].max().astype(bool).mean()
    )


def evaluate_subset(raw, name, policies, outer_folds, inner_folds, penalty_grid):
    sub_raw = raw[raw.policy.isin(policies)].copy()
    if sub_raw.empty:
        raise RuntimeError(f"No rows for action set {name}")

    if name == "no_retrieval_only":
        chosen = sub_raw.copy()
        return {
            "variant": name,
            "actions": len(policies),
            "queries": int(chosen.groupby(["language", "query_id"]).ngroups),
            "macro_f1": float(
                f1_score(chosen.gold, chosen.prediction, average="macro", zero_division=0)
            ),
            "accuracy": float(accuracy_score(chosen.gold, chosen.prediction)),
            "retrieval_rate": 0.0,
            "avg_k": 0.0,
            "mean_outer_penalty": 0.0,
            "oracle_resolvable_rate": oracle_resolvable(sub_raw),
            "notes": "reference; one available action, no controller training required",
        }, pd.DataFrame(), chosen

    df = enrich(sub_raw)
    xcols = QUERY_NUM + RETRIEVAL_NUM + ACTION_NUM + CAT
    groups = df.group_id
    outer = GroupKFold(n_splits=min(outer_folds, groups.nunique()))

    chosen_parts = []
    fold_rows = []

    for fold, (tr, va) in enumerate(
        outer.split(df[xcols], df.correct, groups=groups),
        start=1,
    ):
        print(f"[action-ablation] {name}: outer fold {fold}/{outer.n_splits}")
        inner_scored = make_oof_scores(df, tr, inner_folds)
        penalty, _ = tune_penalty(inner_scored, penalty_grid)

        model = make_pipeline()
        model.fit(df.iloc[tr][xcols], df.iloc[tr].correct.astype(int))
        prob = model.predict_proba(df.iloc[va][xcols])[:, 1]

        scored = df.iloc[va].copy()
        scored["success_probability"] = prob
        scored["retrieval_penalty"] = penalty
        scored["outer_fold"] = fold
        chosen = select_actions(scored, penalty)
        chosen_parts.append(chosen)

        fold_rows.append({
            "variant": name,
            "fold": fold,
            "queries": int(chosen.group_id.nunique()),
            "retrieval_penalty": float(penalty),
            "macro_f1": float(
                f1_score(chosen.gold, chosen.prediction, average="macro", zero_division=0)
            ),
            "accuracy": float(accuracy_score(chosen.gold, chosen.prediction)),
            "retrieval_rate": float((chosen.k > 0).mean()),
            "avg_k": float(chosen.k.mean()),
        })

    selected = pd.concat(chosen_parts, ignore_index=True)
    folds = pd.DataFrame(fold_rows)

    result = {
        "variant": name,
        "actions": len(policies),
        "queries": int(selected.group_id.nunique()),
        "macro_f1": float(
            f1_score(selected.gold, selected.prediction, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(selected.gold, selected.prediction)),
        "retrieval_rate": float((selected.k > 0).mean()),
        "avg_k": float(selected.k.mean()),
        "mean_outer_penalty": float(folds.retrieval_penalty.mean()),
        "oracle_resolvable_rate": oracle_resolvable(sub_raw),
        "notes": "separate Controller v2 retrained with nested grouped CV on this action subset",
    }
    return result, folds, selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="outputs/fixed/dev_predictions.csv")
    ap.add_argument("--outer-folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=3)
    ap.add_argument("--penalty-max", type=float, default=0.15)
    ap.add_argument("--penalty-step", type=float, default=0.005)
    args = ap.parse_args()

    raw = pd.read_csv(args.predictions)
    penalty_grid = np.arange(
        0.0,
        args.penalty_max + args.penalty_step / 2,
        args.penalty_step,
    )

    results = []
    folds_all = []
    selected_all = []

    for name, policies in ACTION_SETS.items():
        result, folds, selected = evaluate_subset(
            raw,
            name,
            policies,
            args.outer_folds,
            args.inner_folds,
            penalty_grid,
        )
        results.append(result)
        if len(folds):
            folds_all.append(folds)
        selected = selected.copy()
        selected["variant"] = name
        selected_all.append(selected)

    outdir = Path("outputs/publication")
    outdir.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame(results)
    table.to_csv(
        outdir / "table_action_space_retrained_ablation.csv",
        index=False,
    )
    if folds_all:
        pd.concat(folds_all, ignore_index=True).to_csv(
            outdir / "action_space_retrained_fold_metrics.csv",
            index=False,
        )
    pd.concat(selected_all, ignore_index=True).to_csv(
        outdir / "action_space_retrained_selected_actions.csv",
        index=False,
    )

    summary = {
        "analysis_status": "post-specified development-only robustness ablation",
        "held_out_test_labels_used": False,
        "protocol": "separate Controller v2 retraining per action subset; 5 outer grouped folds; 3 inner grouped folds; identical penalty grid",
        "action_sets": ACTION_SETS,
        "results": results,
    }
    (outdir / "action_space_retrained_ablation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
