import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afriarag.core import read_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/fixed/dev_predictions.csv")
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--tag", default="dev")
    args = ap.parse_args()

    cfg = read_config(args.config)
    df = pd.read_csv(args.input)
    outdir = Path("outputs/oracle")
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    oracle_preds = []
    oracle_gold = []

    for (lang, qid), g in df.groupby(["language","query_id"], sort=False):
        g = g.copy()
        correct = g[g["correct"] == True].copy()  # noqa: E712
        gold = g["gold"].iloc[0]

        if correct.empty:
            fallback = g[g["policy"] == "no_retrieval"]
            fallback = fallback.iloc[0] if len(fallback) else g.iloc[0]
            rows.append({
                "language": lang,
                "query_id": qid,
                "gold": gold,
                "resolvable": False,
                "correct_actions": "",
                "lowest_cost_correct_actions": "",
                "lowest_cost_k": None,
                "no_retrieval_correct": bool(
                    len(g[g["policy"] == "no_retrieval"]) and
                    g[g["policy"] == "no_retrieval"]["correct"].iloc[0]
                ),
            })
            oracle_preds.append(fallback["prediction"])
            oracle_gold.append(gold)
            continue

        min_k = float(correct["k"].min())
        lowest = correct[correct["k"].astype(float) == min_k]
        correct_actions = sorted(correct["policy"].astype(str).unique())
        lowest_actions = sorted(lowest["policy"].astype(str).unique())

        rows.append({
            "language": lang,
            "query_id": qid,
            "gold": gold,
            "resolvable": True,
            "correct_actions": "|".join(correct_actions),
            "lowest_cost_correct_actions": "|".join(lowest_actions),
            "lowest_cost_k": min_k,
            "no_retrieval_correct": "no_retrieval" in correct_actions,
        })

        # Any correct action yields the same oracle classification outcome.
        oracle_preds.append(correct.iloc[0]["prediction"])
        oracle_gold.append(gold)

    oracle = pd.DataFrame(rows)
    labels_path = outdir / f"{args.tag}_oracle_labels.csv"
    oracle.to_csv(labels_path, index=False)

    # Set-valued action coverage: how often each action belongs to the correct set.
    coverage = []
    for policy, g in df.groupby("policy"):
        coverage.append({
            "policy": policy,
            "queries": int(g["query_id"].nunique()),
            "correct_queries": int(g["correct"].sum()),
            "correct_share": float(g["correct"].mean()),
        })
    coverage_df = pd.DataFrame(coverage).sort_values("correct_share", ascending=False)
    coverage_df.to_csv(outdir / f"{args.tag}_action_coverage.csv", index=False)

    # Lowest-cost compatible-set frequency. Do not collapse ties to one arbitrary label.
    compat_counts = (
        oracle["lowest_cost_correct_actions"]
        .fillna("")
        .replace("", "unresolved")
        .value_counts()
        .rename_axis("compatible_action_set")
        .reset_index(name="n")
    )
    compat_counts["share"] = compat_counts["n"] / len(oracle)
    compat_counts.to_csv(outdir / f"{args.tag}_compatible_action_sets.csv", index=False)

    summary = {
        "queries": int(len(oracle)),
        "resolvable_queries": int(oracle["resolvable"].sum()),
        "unresolved_queries": int((~oracle["resolvable"]).sum()),
        "oracle_accuracy": float(accuracy_score(oracle_gold, oracle_preds)),
        "oracle_macro_f1": float(f1_score(oracle_gold, oracle_preds, average="macro", zero_division=0)),
        "no_retrieval_correct_queries": int(oracle["no_retrieval_correct"].sum()),
        "retrieval_rescuable_queries": int(((~oracle["no_retrieval_correct"]) & oracle["resolvable"]).sum()),
    }
    (outdir / f"{args.tag}_oracle_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print("\nLowest-cost compatible action sets:")
    print(compat_counts.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
