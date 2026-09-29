"""Evaluate the frozen strict zero-shot router on unseen target languages.

Expected input is produced by run_fixed.py for zero-shot target languages using:
  no_retrieval dense_cross_k3 hybrid_cross_k3 hybrid_cross_k5

The router/controller and penalty must already be frozen from leave-one-language-out
development simulation over the 12 supervised languages. Target labels are used
only after action selection to compute final metrics.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_controller import (  # noqa: E402
    ACTION_NUM,
    CAT,
    QUERY_NUM,
    RETRIEVAL_NUM,
    enrich,
    select_actions,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="outputs/fixed/test_orm-tir_predictions.csv")
    ap.add_argument("--controller", default="outputs/zero_shot_controller/controller.joblib")
    ap.add_argument("--controller-config", default="outputs/zero_shot_controller/controller_config.json")
    ap.add_argument("--tag", default="zero_shot_test")
    args = ap.parse_args()

    raw = pd.read_csv(args.predictions)
    model = joblib.load(args.controller)
    cfg = json.loads(Path(args.controller_config).read_text(encoding="utf-8"))
    penalty = float(cfg["retrieval_penalty"])
    allowed = cfg.get("allowed_policies", [
        "no_retrieval","dense_cross_k3","hybrid_cross_k3","hybrid_cross_k5"
    ])

    raw = raw[raw.policy.isin(allowed)].copy()
    df = enrich(raw)
    xcols = QUERY_NUM + RETRIEVAL_NUM + ACTION_NUM + CAT

    prob = model.predict_proba(df[xcols])[:, 1]
    df["success_probability"] = prob
    picked = select_actions(df, penalty)

    summary = {
        "queries": int(picked.query_id.nunique()),
        "languages": int(picked.language.nunique()),
        "macro_f1": float(f1_score(picked.gold, picked.prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(picked.gold, picked.prediction, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(picked.gold, picked.prediction)),
        "selected_context_retrieval_rate": float((picked.k > 0).mean()),
        "avg_selected_k": float(picked.k.mean()),
        "retrieval_penalty": penalty,
        "selected_policy_counts": {
            str(k): int(v) for k, v in picked.policy.value_counts().items()
        },
    }

    no = df[df.policy == "no_retrieval"].copy()
    summary["no_retrieval_macro_f1"] = float(
        f1_score(no.gold, no.prediction, average="macro", zero_division=0)
    )
    summary["no_retrieval_accuracy"] = float(
        accuracy_score(no.gold, no.prediction)
    )
    summary["macro_f1_delta_vs_no_retrieval"] = (
        summary["macro_f1"] - summary["no_retrieval_macro_f1"]
    )

    per_lang = []
    for lang, g in picked.groupby("language"):
        nlang = no[no.language == lang]
        per_lang.append({
            "language": lang,
            "queries": int(len(g)),
            "macro_f1": float(f1_score(g.gold, g.prediction, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(g.gold, g.prediction)),
            "no_retrieval_macro_f1": float(f1_score(nlang.gold, nlang.prediction, average="macro", zero_division=0)),
            "no_retrieval_accuracy": float(accuracy_score(nlang.gold, nlang.prediction)),
            "macro_f1_delta_vs_no_retrieval": float(
                f1_score(g.gold, g.prediction, average="macro", zero_division=0)
                - f1_score(nlang.gold, nlang.prediction, average="macro", zero_division=0)
            ),
            "selected_context_retrieval_rate": float((g.k > 0).mean()),
            "avg_selected_k": float(g.k.mean()),
        })

    outdir = Path("outputs/zero_shot_test")
    outdir.mkdir(parents=True, exist_ok=True)
    picked.to_csv(outdir / f"{args.tag}_selected_actions.csv", index=False)
    pd.DataFrame(per_lang).to_csv(outdir / f"{args.tag}_per_language.csv", index=False)
    (outdir / f"{args.tag}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
