"""Leave-one-language-out validation for the strict zero-shot router.

The outer fold is an entire language, not a query split. This tests whether the
controller can route an unseen African language using only source-language
experience. Oromo/Tigrinya labels are not involved.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_controller import (  # noqa: E402
    ACTION_NUM,
    CAT,
    QUERY_NUM,
    RETRIEVAL_NUM,
    enrich,
    make_pipeline,
    select_actions,
    tune_penalty,
)

ALLOWED = [
    "no_retrieval",
    "dense_cross_k3",
    "hybrid_cross_k3",
    "hybrid_cross_k5",
]


def inner_language_oof(train_df):
    """Score each source language with a model trained on the other source languages."""
    xcols = QUERY_NUM + RETRIEVAL_NUM + ACTION_NUM + CAT
    parts = []
    langs = sorted(train_df.language.unique())

    for held in langs:
        tr = train_df[train_df.language != held]
        va = train_df[train_df.language == held]
        model = make_pipeline()
        model.fit(tr[xcols], tr.correct.astype(int))
        prob = model.predict_proba(va[xcols])[:, 1]
        scored = va.copy()
        scored["success_probability"] = prob
        parts.append(scored)

    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--predictions",
        default="outputs/zero_shot_sim/dev_zero_shot_predictions.csv",
    )
    ap.add_argument("--penalty-max", type=float, default=0.20)
    ap.add_argument("--penalty-step", type=float, default=0.005)
    args = ap.parse_args()

    raw = pd.read_csv(args.predictions)
    raw = raw[raw.policy.isin(ALLOWED)].copy()
    df = enrich(raw)
    xcols = QUERY_NUM + RETRIEVAL_NUM + ACTION_NUM + CAT

    penalty_grid = np.arange(
        0.0,
        args.penalty_max + args.penalty_step / 2,
        args.penalty_step,
    )

    selected_parts = []
    score_parts = []
    metrics = []
    penalty_parts = []

    for target in sorted(df.language.unique()):
        print(f"[zero-shot-router] outer held-out language={target}")
        tr = df[df.language != target].copy()
        va = df[df.language == target].copy()

        inner = inner_language_oof(tr)
        penalty, ptable = tune_penalty(inner, penalty_grid)
        ptable["outer_language"] = target
        penalty_parts.append(ptable)

        model = make_pipeline()
        model.fit(tr[xcols], tr.correct.astype(int))
        prob = model.predict_proba(va[xcols])[:, 1]

        scored = va.copy()
        scored["success_probability"] = prob
        scored["outer_language"] = target
        scored["retrieval_penalty"] = penalty
        score_parts.append(scored)

        chosen = select_actions(scored, penalty)
        selected_parts.append(chosen)

        no = va[va.policy == "no_retrieval"]
        auc = (
            float(roc_auc_score(va.correct.astype(int), prob))
            if va.correct.nunique() > 1
            else None
        )
        metrics.append({
            "language": target,
            "queries": int(chosen.query_id.nunique()),
            "retrieval_penalty": penalty,
            "adaptive_macro_f1": float(
                f1_score(chosen.gold, chosen.prediction, average="macro", zero_division=0)
            ),
            "no_retrieval_macro_f1": float(
                f1_score(no.gold, no.prediction, average="macro", zero_division=0)
            ),
            "macro_f1_delta": float(
                f1_score(chosen.gold, chosen.prediction, average="macro", zero_division=0)
                - f1_score(no.gold, no.prediction, average="macro", zero_division=0)
            ),
            "adaptive_accuracy": float(accuracy_score(chosen.gold, chosen.prediction)),
            "no_retrieval_accuracy": float(accuracy_score(no.gold, no.prediction)),
            "retrieval_rate": float((chosen.k > 0).mean()),
            "avg_k": float(chosen.k.mean()),
            "action_success_auc": auc,
        })

    selected = pd.concat(selected_parts, ignore_index=True)
    scored_all = pd.concat(score_parts, ignore_index=True)
    penalty_search = pd.concat(penalty_parts, ignore_index=True)

    no_all = df[df.policy == "no_retrieval"].copy()
    adaptive_macro = float(
        f1_score(selected.gold, selected.prediction, average="macro", zero_division=0)
    )
    no_macro = float(
        f1_score(no_all.gold, no_all.prediction, average="macro", zero_division=0)
    )

    # Tune final deployment penalty only from leave-one-language-out OOF scores.
    final_penalty, final_table = tune_penalty(scored_all, penalty_grid)

    final_model = make_pipeline()
    final_model.fit(df[xcols], df.correct.astype(int))

    summary = {
        "queries": int(selected.query_id.nunique()),
        "languages": int(selected.language.nunique()),
        "adaptive_macro_f1": adaptive_macro,
        "no_retrieval_macro_f1": no_macro,
        "macro_f1_delta": adaptive_macro - no_macro,
        "adaptive_accuracy": float(accuracy_score(selected.gold, selected.prediction)),
        "no_retrieval_accuracy": float(accuracy_score(no_all.gold, no_all.prediction)),
        "retrieval_rate": float((selected.k > 0).mean()),
        "avg_k": float(selected.k.mean()),
        "final_lolo_tuned_retrieval_penalty": final_penalty,
        "selected_policy_counts": {
            str(k): int(v) for k, v in selected.policy.value_counts().items()
        },
    }

    outdir = Path("outputs/zero_shot_controller")
    outdir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(metrics).to_csv(outdir / "lolo_metrics.csv", index=False)
    selected.to_csv(outdir / "lolo_selected_actions.csv", index=False)
    scored_all.to_csv(outdir / "lolo_action_scores.csv", index=False)
    penalty_search.to_csv(outdir / "nested_penalty_search.csv", index=False)
    final_table.to_csv(outdir / "final_lolo_penalty_search.csv", index=False)
    (outdir / "lolo_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    joblib.dump(final_model, outdir / "controller.joblib")
    (outdir / "controller_config.json").write_text(
        json.dumps({
            "retrieval_penalty": final_penalty,
            "allowed_policies": ALLOWED,
            "validation": "leave-one-language-out; inner leave-one-language-out penalty tuning",
            "target": "strict zero-shot Oromo/Tigrinya after freeze",
        }, indent=2),
        encoding="utf-8",
    )

    print(pd.DataFrame(metrics).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
