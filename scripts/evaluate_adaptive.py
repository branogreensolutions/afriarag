"""Apply the frozen controller to precomputed fixed-policy predictions.

Stage 1 intentionally separates policy-selection quality from online retrieval cost.
For the final paper, the selected policy will be executed online and true end-to-end
cost will be measured. Never use test labels to alter controller or policy choices.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

FEATURES_NUM = [
    "char_len","token_len","avg_token_len","digit_ratio","upper_ratio",
    "punct_count","exclaim_count","question_count","mention_count",
    "hashtag_count","url_count","emoji_count","script_mix",
    "probe_confidence","probe_entropy"
]
FEATURES_CAT = ["language"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="outputs/fixed/test_predictions.csv")
    ap.add_argument("--controller", default="outputs/controller/controller.joblib")
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    model = joblib.load(args.controller)
    base = df.sort_values("policy").drop_duplicates(["language","query_id"]).copy()
    base["selected_policy"] = model.predict(base[FEATURES_NUM + FEATURES_CAT])

    picked = base[["language","query_id","gold","selected_policy"]].merge(
        df,
        left_on=["language","query_id","selected_policy"],
        right_on=["language","query_id","policy"],
        how="left",
        suffixes=("","_run"),
    )
    picked = picked[picked.prediction.notna()].copy()

    result = {
        "n": len(picked),
        "macro_f1": f1_score(picked.gold, picked.prediction, average="macro", zero_division=0),
        "weighted_f1": f1_score(picked.gold, picked.prediction, average="weighted", zero_division=0),
        "accuracy": accuracy_score(picked.gold, picked.prediction),
        "retrieval_rate": float((picked.k > 0).mean()),
        "avg_k": float(picked.k.mean()),
    }

    outdir = Path("outputs/adaptive")
    outdir.mkdir(parents=True, exist_ok=True)
    picked.to_csv(outdir / "adaptive_predictions.csv", index=False)
    (outdir / "adaptive_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
