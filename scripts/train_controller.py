import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afriarag.core import read_config

FEATURES_NUM = [
    "char_len","token_len","avg_token_len","digit_ratio","upper_ratio",
    "punct_count","exclaim_count","question_count","mention_count",
    "hashtag_count","url_count","emoji_count","script_mix",
    "probe_confidence","probe_entropy"
]
FEATURES_CAT = ["language"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="outputs/oracle/oracle_labels.csv")
    ap.add_argument("--config", default="configs/stage1.yaml")
    args = ap.parse_args()
    cfg = read_config(args.config)

    df = pd.read_csv(args.oracle)
    df = df[df.oracle_policy != "unresolved"].copy()
    X = df[FEATURES_NUM + FEATURES_CAT]
    y = df.oracle_policy

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg["seed"])
    tr, va = next(splitter.split(X, y, groups=df.query_id))

    prep = ColumnTransformer([
        ("num", StandardScaler(), FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
    ])
    model = RandomForestClassifier(
        n_estimators=cfg["controller"]["n_estimators"],
        min_samples_leaf=cfg["controller"]["min_samples_leaf"],
        class_weight=cfg["controller"]["class_weight"],
        random_state=cfg["seed"],
        n_jobs=-1,
    )
    pipe = Pipeline([("prep", prep), ("model", model)])
    pipe.fit(X.iloc[tr], y.iloc[tr])
    pred = pipe.predict(X.iloc[va])
    print(classification_report(y.iloc[va], pred, zero_division=0))

    pipe.fit(X, y)
    outdir = Path("outputs/controller")
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, outdir / "controller.joblib")
    (outdir / "features.json").write_text(
        json.dumps({"numeric": FEATURES_NUM, "categorical": FEATURES_CAT}, indent=2),
        encoding="utf-8",
    )
    print(f"saved {outdir / 'controller.joblib'}")

if __name__ == "__main__":
    main()
