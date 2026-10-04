"""Paired bootstrap confidence intervals for MacroLangF1.

This reporting-only script uses frozen prediction files. It does not train or
tune any model. Resampling is paired within language; each bootstrap replicate
computes Macro-F1 separately for every language and then averages those
language-level values with equal weight.

Outputs cover Stage-1 confirmatory comparisons and the separately specified
Stage-2 supporting comparisons.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LABELS = ["negative", "neutral", "positive"]


def normalize_transformer(df):
    x = df.copy()
    rename = {}
    if "label" in x.columns and "gold" not in x.columns:
        rename["label"] = "gold"
    if "id" in x.columns and "query_id" not in x.columns:
        rename["id"] = "query_id"
    if rename:
        x = x.rename(columns=rename)
    return x


def macro_lang_f1_from_predictions(frame, pred_col):
    vals = []
    for _, g in frame.groupby("language"):
        vals.append(
            f1_score(
                g.gold,
                g[pred_col],
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        )
    return float(np.mean(vals))


def pair(a, b):
    left = a[["language", "query_id", "gold", "prediction"]].rename(
        columns={"prediction": "pred_a"}
    )
    right = b[["language", "query_id", "gold", "prediction"]].rename(
        columns={"prediction": "pred_b"}
    )
    m = left.merge(
        right,
        on=["language", "query_id", "gold"],
        how="inner",
        validate="one_to_one",
    )
    if len(m) != len(left) or len(m) != len(right):
        raise RuntimeError(
            f"Paired merge mismatch: A={len(left)}, B={len(right)}, paired={len(m)}"
        )
    return m


def bootstrap(a, b, samples, seed):
    m = pair(a, b)
    observed_a = macro_lang_f1_from_predictions(m, "pred_a")
    observed_b = macro_lang_f1_from_predictions(m, "pred_b")
    observed_delta = observed_a - observed_b

    groups = {
        lang: g.reset_index(drop=True)
        for lang, g in m.groupby("language")
    }
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=float)

    for i in range(samples):
        per_lang_delta = []
        for _, g in groups.items():
            idx = rng.integers(0, len(g), size=len(g))
            s = g.iloc[idx]
            a_f1 = f1_score(
                s.gold, s.pred_a, labels=LABELS, average="macro", zero_division=0
            )
            b_f1 = f1_score(
                s.gold, s.pred_b, labels=LABELS, average="macro", zero_division=0
            )
            per_lang_delta.append(a_f1 - b_f1)
        deltas[i] = float(np.mean(per_lang_delta))

    return {
        "macro_lang_f1_a": observed_a,
        "macro_lang_f1_b": observed_b,
        "observed_delta": observed_delta,
        "bootstrap_mean_delta": float(deltas.mean()),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "p_delta_le_0": float(np.mean(deltas <= 0)),
        "samples": int(samples),
        "seed": int(seed),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=242)
    args = ap.parse_args()

    sup_fixed = pd.read_csv("outputs/fixed/test_predictions.csv")
    zs_fixed = pd.read_csv("outputs/fixed/test_orm-tir_predictions.csv")
    sup_ad = pd.read_csv(
        "outputs/adaptive_v2/supervised_test_selected_actions.csv"
    )
    zs_ad = pd.read_csv(
        "outputs/zero_shot_test/zero_shot_test_selected_actions.csv"
    )
    tr_sup = normalize_transformer(
        pd.read_csv("outputs/transformer_baseline/supervised_test_predictions.csv")
    )
    tr_zs = normalize_transformer(
        pd.read_csv("outputs/transformer_baseline/zero_shot_test_predictions.csv")
    )

    sup_no = sup_fixed[sup_fixed.policy == "no_retrieval"].copy()
    zs_no = zs_fixed[zs_fixed.policy == "no_retrieval"].copy()
    zs_dense = zs_fixed[zs_fixed.policy == "dense_cross_k3"].copy()

    comparisons = [
        (
            "stage1_supervised",
            "adaptive_vs_no_retrieval",
            sup_ad, sup_no,
            "confirmatory Stage-1 comparison",
        ),
        (
            "stage1_zero_shot",
            "adaptive_vs_no_retrieval",
            zs_ad, zs_no,
            "confirmatory Stage-1 comparison",
        ),
        (
            "stage2_supervised",
            "transformer_vs_afriarag",
            tr_sup, sup_ad,
            "supporting Stage-2 comparison",
        ),
        (
            "stage2_supervised",
            "transformer_vs_no_retrieval",
            tr_sup, sup_no,
            "supporting Stage-2 comparison",
        ),
        (
            "stage2_zero_shot",
            "transformer_vs_afriarag",
            tr_zs, zs_ad,
            "supporting Stage-2 comparison",
        ),
        (
            "stage2_zero_shot",
            "transformer_vs_dense_cross_k3",
            tr_zs, zs_dense,
            "supporting Stage-2 comparison; dense policy identified post-hoc in Stage 1",
        ),
        (
            "stage2_zero_shot",
            "transformer_vs_no_retrieval",
            tr_zs, zs_no,
            "supporting Stage-2 comparison",
        ),
    ]

    rows = []
    details = {}
    for i, (setting, name, a, b, status) in enumerate(comparisons):
        result = bootstrap(a, b, args.samples, args.seed + i)
        rows.append({
            "setting": setting,
            "comparison": name,
            "status": status,
            **result,
        })
        details[f"{setting}:{name}"] = result

    outdir = Path("outputs/publication")
    outdir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(outdir / "table_macrolang_bootstrap.csv", index=False)

    payload = {
        "analysis_status": "reporting-only paired bootstrap on frozen predictions",
        "metric": "MacroLangF1 = unweighted mean of per-language Macro-F1",
        "method": "paired query bootstrap within each language; equal-weight language averaging per replicate",
        "comparisons": details,
    }
    (outdir / "macrolang_bootstrap.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
