"""Pre-specified Stage-2 transformer comparison analysis.

Create this script before evaluating the Stage-2 transformer on held-out test.
It performs no training or tuning. After transformer test predictions exist, it
compares the frozen Stage-2 encoder with frozen Stage-1 systems using paired,
language-stratified bootstrap.

Stage-2 was specified after Stage-1 test inspection, so all Stage-2 comparisons
are supporting/exploratory rather than replacements for Stage-1 confirmatory
results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

LABELS = ["negative", "neutral", "positive"]


def normalize_transformer(df):
    x = df.copy()
    if "label" in x.columns and "gold" not in x.columns:
        x = x.rename(columns={"label": "gold"})
    return x


def metrics(df):
    return {
        "n": int(len(df)),
        "macro_f1": float(f1_score(df.gold, df.prediction, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(df.gold, df.prediction, labels=LABELS, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(df.gold, df.prediction)),
        "macro_lang_f1": float(np.mean([
            f1_score(g.gold, g.prediction, labels=LABELS, average="macro", zero_division=0)
            for _, g in df.groupby("language")
        ])),
    }


def paired_bootstrap(a, b, samples, seed):
    left = a[["language","query_id","gold","prediction"]].rename(
        columns={"prediction":"pred_a"}
    )
    right = b[["language","query_id","gold","prediction"]].rename(
        columns={"prediction":"pred_b"}
    )
    m = left.merge(
        right,
        on=["language","query_id","gold"],
        how="inner",
        validate="one_to_one",
    )
    if len(m) != len(left) or len(m) != len(right):
        raise RuntimeError(
            f"Paired merge mismatch: A={len(left)}, B={len(right)}, paired={len(m)}"
        )

    rng = np.random.default_rng(seed)
    groups = {
        lang: g.reset_index(drop=True)
        for lang, g in m.groupby("language")
    }

    f1_delta = np.empty(samples)
    acc_delta = np.empty(samples)

    for i in range(samples):
        pieces = []
        for lang, g in groups.items():
            idx = rng.integers(0, len(g), size=len(g))
            pieces.append(g.iloc[idx])
        s = pd.concat(pieces, ignore_index=True)

        f1_delta[i] = (
            f1_score(s.gold, s.pred_a, labels=LABELS, average="macro", zero_division=0)
            - f1_score(s.gold, s.pred_b, labels=LABELS, average="macro", zero_division=0)
        )
        acc_delta[i] = (
            accuracy_score(s.gold, s.pred_a)
            - accuracy_score(s.gold, s.pred_b)
        )

    def summary(arr):
        return {
            "mean": float(arr.mean()),
            "ci95_low": float(np.quantile(arr, 0.025)),
            "ci95_high": float(np.quantile(arr, 0.975)),
            "p_delta_le_0": float(np.mean(arr <= 0)),
        }

    return {
        "macro_f1_delta": summary(f1_delta),
        "accuracy_delta": summary(acc_delta),
    }


def per_language(systems):
    rows = []
    languages = sorted(set().union(*[
        set(df.language.unique()) for df in systems.values()
    ]))
    for lang in languages:
        for name, df in systems.items():
            g = df[df.language == lang]
            if g.empty:
                continue
            rows.append({
                "language": lang,
                "system": name,
                **metrics(g),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transformer-supervised", default="outputs/transformer_baseline/supervised_test_predictions.csv")
    ap.add_argument("--transformer-zero-shot", default="outputs/transformer_baseline/zero_shot_test_predictions.csv")
    ap.add_argument("--stage1-supervised-adaptive", default="outputs/adaptive_v2/supervised_test_selected_actions.csv")
    ap.add_argument("--stage1-supervised-fixed", default="outputs/fixed/test_predictions.csv")
    ap.add_argument("--stage1-zero-shot-adaptive", default="outputs/zero_shot_test/zero_shot_test_selected_actions.csv")
    ap.add_argument("--stage1-zero-shot-fixed", default="outputs/fixed/test_orm-tir_predictions.csv")
    ap.add_argument("--bootstrap-samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=142)
    args = ap.parse_args()

    paths = [Path(getattr(args, key)) for key in [
        "transformer_supervised",
        "transformer_zero_shot",
        "stage1_supervised_adaptive",
        "stage1_supervised_fixed",
        "stage1_zero_shot_adaptive",
        "stage1_zero_shot_fixed",
    ]]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit("Missing required inputs:\n- " + "\n- ".join(missing))

    ts = normalize_transformer(pd.read_csv(args.transformer_supervised))
    tz = normalize_transformer(pd.read_csv(args.transformer_zero_shot))
    sa = pd.read_csv(args.stage1_supervised_adaptive)
    sf = pd.read_csv(args.stage1_supervised_fixed)
    za = pd.read_csv(args.stage1_zero_shot_adaptive)
    zf = pd.read_csv(args.stage1_zero_shot_fixed)

    sn = sf[sf.policy == "no_retrieval"].copy()
    zd = zf[zf.policy == "dense_cross_k3"].copy()
    zn = zf[zf.policy == "no_retrieval"].copy()

    systems_sup = {
        "transformer": ts,
        "afriarag_adaptive": sa,
        "no_retrieval": sn,
    }
    systems_zs = {
        "transformer": tz,
        "afriarag_adaptive": za,
        "dense_cross_k3": zd,
        "no_retrieval": zn,
    }

    summary = {
        "analysis_type": "pre-specified Stage-2 supporting comparison",
        "stage2_status": "supporting/exploratory; specified after Stage-1 test inspection",
        "bootstrap": {
            "method": "paired query bootstrap stratified within language",
            "samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "supervised": {
            "metrics": {name: metrics(df) for name, df in systems_sup.items()},
            "transformer_vs_afriarag": paired_bootstrap(ts, sa, args.bootstrap_samples, args.seed),
            "transformer_vs_no_retrieval": paired_bootstrap(ts, sn, args.bootstrap_samples, args.seed + 1),
        },
        "zero_shot": {
            "metrics": {name: metrics(df) for name, df in systems_zs.items()},
            "transformer_vs_afriarag": paired_bootstrap(tz, za, args.bootstrap_samples, args.seed + 2),
            "transformer_vs_dense_cross_k3": paired_bootstrap(tz, zd, args.bootstrap_samples, args.seed + 3),
            "transformer_vs_no_retrieval": paired_bootstrap(tz, zn, args.bootstrap_samples, args.seed + 4),
        },
    }

    outdir = Path("outputs/transformer_baseline/comparison")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    pl = pd.concat([
        per_language(systems_sup).assign(setting="supervised"),
        per_language(systems_zs).assign(setting="zero_shot"),
    ], ignore_index=True)
    pl.to_csv(outdir / "comparison_per_language.csv", index=False)

    rows = []
    for setting, block in [("supervised", systems_sup), ("zero_shot", systems_zs)]:
        for name, df in block.items():
            rows.append({"setting": setting, "system": name, **metrics(df)})
    pd.DataFrame(rows).to_csv(outdir / "comparison_main.csv", index=False)

    print(json.dumps(summary, indent=2))
    print(f"saved {outdir}")


if __name__ == "__main__":
    main()
