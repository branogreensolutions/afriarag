import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
    "probe_confidence","probe_entropy"
]
ACTION_NUM = ["k","alpha"]
CAT = ["language","policy","retriever","scope"]


def make_pipeline(cfg, seed):
    prep = ColumnTransformer([
        ("num", StandardScaler(), QUERY_NUM + ACTION_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    model = RandomForestClassifier(
        n_estimators=cfg["controller"]["n_estimators"],
        min_samples_leaf=cfg["controller"]["min_samples_leaf"],
        class_weight=cfg["controller"]["class_weight"],
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("prep", prep), ("model", model)])


def choose_actions(scored: pd.DataFrame, k_cost: float) -> pd.DataFrame:
    x = scored.copy()
    x["predicted_utility"] = x["success_probability"] - k_cost * x["k"].astype(float)
    idx = x.groupby(["language","query_id"])["predicted_utility"].idxmax()
    return x.loc[idx].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="outputs/fixed/dev_predictions.csv")
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    cfg = read_config(args.config)

    df = pd.read_csv(args.predictions)
    required = set(QUERY_NUM + ACTION_NUM + CAT + ["query_id","gold","prediction","correct"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    # Each query appears once per candidate action. Split by query ID so actions
    # from the same query can never leak between controller train/validation.
    groups = df["query_id"].astype(str)
    n_groups = groups.nunique()
    n_splits = min(args.folds, n_groups)
    if n_splits < 2:
        raise SystemExit("Need at least two distinct queries for grouped validation.")

    X = df[QUERY_NUM + ACTION_NUM + CAT]
    y = df["correct"].astype(int)

    cv_parts = []
    fold_rows = []
    gkf = GroupKFold(n_splits=n_splits)

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups), start=1):
        pipe = make_pipeline(cfg, cfg["seed"] + fold)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        prob = pipe.predict_proba(X.iloc[va])[:, 1]

        scored = df.iloc[va].copy()
        scored["fold"] = fold
        scored["success_probability"] = prob
        cv_parts.append(scored)

        chosen = choose_actions(scored, cfg["oracle"]["k_cost"])
        fold_rows.append({
            "fold": fold,
            "queries": int(chosen["query_id"].nunique()),
            "macro_f1": float(f1_score(chosen["gold"], chosen["prediction"], average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(chosen["gold"], chosen["prediction"])),
            "retrieval_rate": float((chosen["k"] > 0).mean()),
            "avg_k": float(chosen["k"].mean()),
            "action_success_auc": float(roc_auc_score(y.iloc[va], prob)) if y.iloc[va].nunique() > 1 else None,
        })

    cv = pd.concat(cv_parts, ignore_index=True)
    chosen_all = choose_actions(cv, cfg["oracle"]["k_cost"])
    overall = {
        "queries": int(chosen_all["query_id"].nunique()),
        "macro_f1": float(f1_score(chosen_all["gold"], chosen_all["prediction"], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(chosen_all["gold"], chosen_all["prediction"], average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(chosen_all["gold"], chosen_all["prediction"])),
        "retrieval_rate": float((chosen_all["k"] > 0).mean()),
        "avg_k": float(chosen_all["k"].mean()),
        "selected_policy_counts": {str(k): int(v) for k, v in chosen_all["policy"].value_counts().items()},
    }

    # Fit the deployable action-success model on all available development rows.
    final_model = make_pipeline(cfg, cfg["seed"])
    final_model.fit(X, y)

    outdir = Path("outputs/controller")
    outdir.mkdir(parents=True, exist_ok=True)
    cv.to_csv(outdir / "controller_cv_action_scores.csv", index=False)
    chosen_all.to_csv(outdir / "controller_cv_selected_actions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(outdir / "controller_cv_metrics.csv", index=False)
    (outdir / "controller_cv_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    joblib.dump(final_model, outdir / "controller.joblib")
    (outdir / "features.json").write_text(
        json.dumps({
            "query_numeric": QUERY_NUM,
            "action_numeric": ACTION_NUM,
            "categorical": CAT,
            "target": "P(action produces correct sentiment | query, action)",
            "selection": "argmax(P(correct)-k_cost*k)",
        }, indent=2),
        encoding="utf-8",
    )

    print(pd.DataFrame(fold_rows).to_string(index=False))
    print(json.dumps(overall, indent=2))
    print(f"saved {outdir / 'controller.joblib'}")


if __name__ == "__main__":
    main()
