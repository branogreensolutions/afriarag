"""Final held-out test analysis for AfriARAG Stage 1.

This script does not train or tune any model. It compares frozen adaptive outputs
with predeclared fixed policies, computes rescue/harm counts, MacroLangF1, and
paired stratified bootstrap confidence intervals.

Run only after the pre-test freeze manifest has been created.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

LABELS = ["negative", "neutral", "positive"]


def metrics(df):
    return {
        "n": int(len(df)),
        "macro_f1": float(f1_score(df.gold, df.prediction, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(df.gold, df.prediction, labels=LABELS, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(df.gold, df.prediction)),
    }


def macro_lang_f1(df):
    vals = []
    for _, g in df.groupby("language"):
        vals.append(
            f1_score(g.gold, g.prediction, labels=LABELS, average="macro", zero_division=0)
        )
    return float(np.mean(vals)) if vals else float("nan")


def fixed_policy_table(df):
    rows = []
    for policy, g in df.groupby("policy"):
        m = metrics(g)
        m.update({
            "policy": policy,
            "languages": int(g.language.nunique()),
            "macro_lang_f1": macro_lang_f1(g),
            "retrieval_rate": float((g.k.astype(float) > 0).mean()),
            "avg_k": float(g.k.astype(float).mean()),
        })
        rows.append(m)
    return pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)


def paired_frame(adaptive, fixed, fixed_policy="no_retrieval"):
    base = fixed[fixed.policy == fixed_policy][
        ["language","query_id","gold","prediction","correct"]
    ].rename(columns={
        "prediction":"baseline_prediction",
        "correct":"baseline_correct",
    })
    ad = adaptive[
        ["language","query_id","gold","prediction","correct","policy","k"]
    ].rename(columns={
        "prediction":"adaptive_prediction",
        "correct":"adaptive_correct",
        "policy":"adaptive_policy",
        "k":"adaptive_k",
    })
    merged = ad.merge(
        base,
        on=["language","query_id","gold"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(adaptive):
        raise RuntimeError(
            f"Paired merge lost rows: adaptive={len(adaptive)}, paired={len(merged)}"
        )
    return merged


def bootstrap_delta(paired, samples=2000, seed=42):
    rng = np.random.default_rng(seed)
    langs = sorted(paired.language.unique())
    deltas_f1 = np.empty(samples, dtype=float)
    deltas_acc = np.empty(samples, dtype=float)

    groups = {lang: paired[paired.language == lang].reset_index(drop=True) for lang in langs}

    for b in range(samples):
        pieces = []
        for lang in langs:
            g = groups[lang]
            idx = rng.integers(0, len(g), size=len(g))
            pieces.append(g.iloc[idx])
        s = pd.concat(pieces, ignore_index=True)

        af1 = f1_score(
            s.gold, s.adaptive_prediction,
            labels=LABELS, average="macro", zero_division=0
        )
        bf1 = f1_score(
            s.gold, s.baseline_prediction,
            labels=LABELS, average="macro", zero_division=0
        )
        aacc = accuracy_score(s.gold, s.adaptive_prediction)
        bacc = accuracy_score(s.gold, s.baseline_prediction)

        deltas_f1[b] = af1 - bf1
        deltas_acc[b] = aacc - bacc

    def summarize(x):
        return {
            "mean": float(np.mean(x)),
            "ci95_low": float(np.quantile(x, 0.025)),
            "ci95_high": float(np.quantile(x, 0.975)),
            "p_delta_le_0": float(np.mean(x <= 0)),
        }

    return {
        "macro_f1_delta": summarize(deltas_f1),
        "accuracy_delta": summarize(deltas_acc),
    }


def compare(adaptive, fixed, bootstrap_samples, seed):
    paired = paired_frame(adaptive, fixed, "no_retrieval")

    adaptive_m = metrics(
        adaptive.rename(columns={"prediction":"prediction"})
    )
    baseline = fixed[fixed.policy == "no_retrieval"].copy()
    baseline_m = metrics(baseline)

    rescue = int(((~paired.baseline_correct.astype(bool)) & paired.adaptive_correct.astype(bool)).sum())
    harm = int((paired.baseline_correct.astype(bool) & (~paired.adaptive_correct.astype(bool))).sum())
    unchanged_correct = int((paired.baseline_correct.astype(bool) & paired.adaptive_correct.astype(bool)).sum())
    unchanged_wrong = int(((~paired.baseline_correct.astype(bool)) & (~paired.adaptive_correct.astype(bool))).sum())

    return {
        "adaptive": {
            **adaptive_m,
            "macro_lang_f1": macro_lang_f1(adaptive),
            "selected_context_retrieval_rate": float((adaptive.k.astype(float) > 0).mean()),
            "avg_selected_k": float(adaptive.k.astype(float).mean()),
        },
        "no_retrieval": {
            **baseline_m,
            "macro_lang_f1": macro_lang_f1(baseline),
        },
        "delta": {
            "macro_f1": adaptive_m["macro_f1"] - baseline_m["macro_f1"],
            "weighted_f1": adaptive_m["weighted_f1"] - baseline_m["weighted_f1"],
            "accuracy": adaptive_m["accuracy"] - baseline_m["accuracy"],
            "macro_lang_f1": macro_lang_f1(adaptive) - macro_lang_f1(baseline),
        },
        "paired_outcomes": {
            "rescue": rescue,
            "harm": harm,
            "unchanged_correct": unchanged_correct,
            "unchanged_wrong": unchanged_wrong,
            "net_correct_gain": rescue - harm,
        },
        "bootstrap": bootstrap_delta(
            paired,
            samples=bootstrap_samples,
            seed=seed,
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--supervised-fixed", default="outputs/fixed/test_predictions.csv")
    ap.add_argument("--supervised-adaptive", default="outputs/adaptive_v2/supervised_test_selected_actions.csv")
    ap.add_argument("--zero-shot-fixed", default="outputs/fixed/test_orm-tir_predictions.csv")
    ap.add_argument("--zero-shot-adaptive", default="outputs/zero_shot_test/zero_shot_test_selected_actions.csv")
    ap.add_argument("--bootstrap-samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sup_fixed = pd.read_csv(args.supervised_fixed)
    sup_ad = pd.read_csv(args.supervised_adaptive)
    zs_fixed = pd.read_csv(args.zero_shot_fixed)
    zs_ad = pd.read_csv(args.zero_shot_adaptive)

    outdir = Path("outputs/final_analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    sup_table = fixed_policy_table(sup_fixed)
    zs_table = fixed_policy_table(zs_fixed)
    sup_table.to_csv(outdir / "supervised_fixed_policy_metrics.csv", index=False)
    zs_table.to_csv(outdir / "zero_shot_fixed_policy_metrics.csv", index=False)

    sup_comp = compare(sup_ad, sup_fixed, args.bootstrap_samples, args.seed)
    zs_comp = compare(zs_ad, zs_fixed, args.bootstrap_samples, args.seed + 1)

    (outdir / "supervised_comparison.json").write_text(
        json.dumps(sup_comp, indent=2), encoding="utf-8"
    )
    (outdir / "zero_shot_comparison.json").write_text(
        json.dumps(zs_comp, indent=2), encoding="utf-8"
    )

    final = {
        "analysis_type": "post-freeze held-out test analysis",
        "bootstrap": {
            "method": "paired query bootstrap stratified within language",
            "samples": args.bootstrap_samples,
            "seed_supervised": args.seed,
            "seed_zero_shot": args.seed + 1,
        },
        "supervised": sup_comp,
        "zero_shot": zs_comp,
        "note": (
            "The confirmatory fixed comparison is no_retrieval because it was the "
            "strongest global fixed development policy. Any test-best fixed policy "
            "reported from the metric tables is descriptive/post-hoc."
        ),
    }
    (outdir / "final_test_analysis.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )

    print("SUPERVISED FIXED POLICIES")
    print(sup_table.to_string(index=False))
    print("\nSUPERVISED ADAPTIVE VS NO RETRIEVAL")
    print(json.dumps(sup_comp, indent=2))
    print("\nZERO-SHOT FIXED POLICIES")
    print(zs_table.to_string(index=False))
    print("\nZERO-SHOT ADAPTIVE VS NO RETRIEVAL")
    print(json.dumps(zs_comp, indent=2))


if __name__ == "__main__":
    main()
