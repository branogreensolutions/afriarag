"""Build the frozen Stage-1 publication-analysis package.

This script is post-freeze analysis only. It does not train models, tune
hyperparameters, or alter any frozen Stage-1 decision. It consumes the final
test predictions/selected actions plus final statistical-analysis outputs and
writes publication-ready CSV/PNG/JSON artifacts under outputs/publication/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

LABELS = ["negative", "neutral", "positive"]


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Required input is missing: {p}")
    return p


def metric_row(df: pd.DataFrame) -> dict:
    return {
        "n": int(len(df)),
        "macro_f1": float(
            f1_score(df.gold, df.prediction, labels=LABELS, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(df.gold, df.prediction, labels=LABELS, average="weighted", zero_division=0)
        ),
        "accuracy": float(accuracy_score(df.gold, df.prediction)),
    }


def macro_lang_f1(df: pd.DataFrame) -> float:
    vals = []
    for _, g in df.groupby("language"):
        vals.append(
            f1_score(g.gold, g.prediction, labels=LABELS, average="macro", zero_division=0)
        )
    return float(np.mean(vals)) if vals else float("nan")


def per_class_rows(df: pd.DataFrame, setting: str, system: str) -> list[dict]:
    p, r, f, s = precision_recall_fscore_support(
        df.gold,
        df.prediction,
        labels=LABELS,
        zero_division=0,
    )
    rows = []
    for label, pp, rr, ff, ss in zip(LABELS, p, r, f, s):
        rows.append({
            "setting": setting,
            "system": system,
            "class": label,
            "precision": float(pp),
            "recall": float(rr),
            "f1": float(ff),
            "support": int(ss),
        })
    return rows


def save_confusion(df: pd.DataFrame, title: str, out: Path) -> None:
    cm = confusion_matrix(df.gold, df.prediction, labels=LABELS)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    image = ax.imshow(cm)
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(LABELS)),
        yticks=np.arange(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Predicted label",
        ylabel="Gold label",
        title=title,
    )
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
            )
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fixed_metrics(df: pd.DataFrame, setting: str) -> pd.DataFrame:
    rows = []
    for policy, g in df.groupby("policy"):
        m = metric_row(g)
        rows.append({
            "setting": setting,
            "policy": policy,
            **m,
            "macro_lang_f1": macro_lang_f1(g),
            "retrieval_rate": float((pd.to_numeric(g.k, errors="coerce").fillna(0) > 0).mean()),
            "avg_k": float(pd.to_numeric(g.k, errors="coerce").fillna(0).mean()),
        })
    return pd.DataFrame(rows).sort_values(
        ["macro_f1", "accuracy"], ascending=False
    ).reset_index(drop=True)


def paired(adaptive: pd.DataFrame, fixed: pd.DataFrame) -> pd.DataFrame:
    base = fixed[fixed.policy == "no_retrieval"][
        ["language", "query_id", "gold", "prediction", "correct"]
    ].rename(columns={
        "prediction": "baseline_prediction",
        "correct": "baseline_correct",
    })

    keep = [
        "language", "query_id", "text", "gold", "prediction", "correct",
        "policy", "k", "retrieved_ids", "retrieved_languages",
        "retrieved_labels", "retrieved_scores", "latency_ms",
    ]
    for optional in ["success_probability", "selection_utility"]:
        if optional in adaptive.columns:
            keep.append(optional)

    ad = adaptive[[c for c in keep if c in adaptive.columns]].rename(columns={
        "prediction": "adaptive_prediction",
        "correct": "adaptive_correct",
        "policy": "adaptive_policy",
        "k": "adaptive_k",
    })

    m = ad.merge(
        base,
        on=["language", "query_id", "gold"],
        how="inner",
        validate="one_to_one",
    )
    if len(m) != len(adaptive):
        raise RuntimeError(
            f"Paired merge mismatch: adaptive={len(adaptive)}, paired={len(m)}"
        )

    m["baseline_correct"] = m["baseline_correct"].astype(bool)
    m["adaptive_correct"] = m["adaptive_correct"].astype(bool)
    m["outcome"] = np.select(
        [
            (~m.baseline_correct) & m.adaptive_correct,
            m.baseline_correct & (~m.adaptive_correct),
            m.baseline_correct & m.adaptive_correct,
        ],
        ["rescue", "harm", "unchanged_correct"],
        default="unchanged_wrong",
    )
    return m


def language_table(
    adaptive: pd.DataFrame,
    fixed: pd.DataFrame,
    setting: str,
) -> pd.DataFrame:
    no = fixed[fixed.policy == "no_retrieval"]
    rows = []
    for lang, g in adaptive.groupby("language"):
        ng = no[no.language == lang]
        af1 = f1_score(g.gold, g.prediction, labels=LABELS, average="macro", zero_division=0)
        nf1 = f1_score(ng.gold, ng.prediction, labels=LABELS, average="macro", zero_division=0)
        rows.append({
            "setting": setting,
            "language": lang,
            "queries": int(len(g)),
            "adaptive_macro_f1": float(af1),
            "no_retrieval_macro_f1": float(nf1),
            "delta_macro_f1": float(af1 - nf1),
            "adaptive_accuracy": float(accuracy_score(g.gold, g.prediction)),
            "no_retrieval_accuracy": float(accuracy_score(ng.gold, ng.prediction)),
            "selected_context_retrieval_rate": float(
                (pd.to_numeric(g.k, errors="coerce").fillna(0) > 0).mean()
            ),
            "avg_selected_k": float(pd.to_numeric(g.k, errors="coerce").fillna(0).mean()),
        })
    return pd.DataFrame(rows)


def plot_language_delta(df: pd.DataFrame, out: Path) -> None:
    x = df.sort_values(["setting", "delta_macro_f1"]).copy()
    labels = [f"{r.setting}:{r.language}" for r in x.itertuples()]
    fig, ax = plt.subplots(figsize=(9.2, max(5.0, len(x) * 0.34)))
    ax.barh(np.arange(len(x)), x.delta_macro_f1.to_numpy())
    ax.axvline(0.0, linewidth=1)
    ax.set(
        yticks=np.arange(len(x)),
        yticklabels=labels,
        xlabel="Adaptive − No-Retrieval Macro-F1",
        title="Held-out Macro-F1 delta by language",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_policy_table(df: pd.DataFrame, setting: str, out: Path) -> None:
    x = df[df.setting == setting].sort_values("macro_f1").copy()
    fig, ax = plt.subplots(figsize=(9.0, max(4.5, len(x) * 0.4)))
    ax.barh(np.arange(len(x)), x.macro_f1.to_numpy())
    ax.set(
        yticks=np.arange(len(x)),
        yticklabels=x.policy.tolist(),
        xlabel="Macro-F1",
        title=f"{setting.replace('_', ' ').title()} fixed-policy comparison",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main_results(
    final_analysis: dict,
    sup_fixed_metrics: pd.DataFrame,
    zs_fixed_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for setting, fixed_table in [
        ("supervised", sup_fixed_metrics),
        ("zero_shot", zs_fixed_metrics),
    ]:
        block = final_analysis[setting]
        rows.append({
            "setting": setting,
            "system": "adaptive",
            **{k: block["adaptive"][k] for k in ["n", "macro_f1", "weighted_f1", "accuracy", "macro_lang_f1"]},
            "delta_macro_f1_vs_no_retrieval": block["delta"]["macro_f1"],
            "bootstrap_ci95_low": block["bootstrap"]["macro_f1_delta"]["ci95_low"],
            "bootstrap_ci95_high": block["bootstrap"]["macro_f1_delta"]["ci95_high"],
            "selected_context_retrieval_rate": block["adaptive"]["selected_context_retrieval_rate"],
            "avg_selected_k": block["adaptive"]["avg_selected_k"],
            "comparison_status": "confirmatory vs no_retrieval",
        })
        rows.append({
            "setting": setting,
            "system": "no_retrieval",
            **{k: block["no_retrieval"][k] for k in ["n", "macro_f1", "weighted_f1", "accuracy", "macro_lang_f1"]},
            "delta_macro_f1_vs_no_retrieval": 0.0,
            "bootstrap_ci95_low": np.nan,
            "bootstrap_ci95_high": np.nan,
            "selected_context_retrieval_rate": 0.0,
            "avg_selected_k": 0.0,
            "comparison_status": "predeclared fixed comparator",
        })

        best = fixed_table.iloc[0]
        if best.policy != "no_retrieval":
            rows.append({
                "setting": setting,
                "system": f"fixed:{best.policy}",
                "n": int(best.n),
                "macro_f1": float(best.macro_f1),
                "weighted_f1": float(best.weighted_f1),
                "accuracy": float(best.accuracy),
                "macro_lang_f1": float(best.macro_lang_f1),
                "delta_macro_f1_vs_no_retrieval": float(
                    best.macro_f1 - block["no_retrieval"]["macro_f1"]
                ),
                "bootstrap_ci95_low": np.nan,
                "bootstrap_ci95_high": np.nan,
                "selected_context_retrieval_rate": float(best.retrieval_rate),
                "avg_selected_k": float(best.avg_k),
                "comparison_status": "descriptive/post-hoc test-best fixed policy",
            })
    return pd.DataFrame(rows)


def efficiency_rows(setting: str, adaptive: pd.DataFrame, fixed: pd.DataFrame) -> list[dict]:
    rows = []
    k = pd.to_numeric(adaptive.k, errors="coerce").fillna(0)
    lat = pd.to_numeric(adaptive.get("latency_ms", 0), errors="coerce").fillna(0)
    retrieved = k > 0
    rows.append({
        "setting": setting,
        "system": "adaptive",
        "selected_context_retrieval_rate": float(retrieved.mean()),
        "avg_selected_k": float(k.mean()),
        "total_selected_documents": int(k.sum()),
        "mean_recorded_selected_action_latency_ms": float(lat.mean()),
        "mean_recorded_latency_when_retrieving_ms": float(lat[retrieved].mean()) if retrieved.any() else 0.0,
        "latency_scope": "recorded retrieval scoring/top-k only; excludes candidate-probe generation and downstream model/token cost",
    })

    for policy, g in fixed.groupby("policy"):
        kk = pd.to_numeric(g.k, errors="coerce").fillna(0)
        ll = pd.to_numeric(g.get("latency_ms", 0), errors="coerce").fillna(0)
        rows.append({
            "setting": setting,
            "system": f"fixed:{policy}",
            "selected_context_retrieval_rate": float((kk > 0).mean()),
            "avg_selected_k": float(kk.mean()),
            "total_selected_documents": int(kk.sum()),
            "mean_recorded_selected_action_latency_ms": float(ll.mean()),
            "mean_recorded_latency_when_retrieving_ms": float(ll[kk > 0].mean()) if (kk > 0).any() else 0.0,
            "latency_scope": "recorded retrieval scoring/top-k only; excludes query-embedding preprocessing and downstream model/token cost",
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--supervised-fixed", default="outputs/fixed/test_predictions.csv")
    ap.add_argument("--supervised-adaptive", default="outputs/adaptive_v2/supervised_test_selected_actions.csv")
    ap.add_argument("--zero-shot-fixed", default="outputs/fixed/test_orm-tir_predictions.csv")
    ap.add_argument("--zero-shot-adaptive", default="outputs/zero_shot_test/zero_shot_test_selected_actions.csv")
    ap.add_argument("--final-analysis", default="outputs/final_analysis/final_test_analysis.json")
    ap.add_argument("--freeze-manifest", default="outputs/freeze/pretest_freeze_manifest.json")
    ap.add_argument("--examples-per-outcome-language", type=int, default=3)
    args = ap.parse_args()

    input_paths = {
        "supervised_fixed": ensure(args.supervised_fixed),
        "supervised_adaptive": ensure(args.supervised_adaptive),
        "zero_shot_fixed": ensure(args.zero_shot_fixed),
        "zero_shot_adaptive": ensure(args.zero_shot_adaptive),
        "final_analysis": ensure(args.final_analysis),
        "freeze_manifest": ensure(args.freeze_manifest),
    }

    sup_fixed = pd.read_csv(input_paths["supervised_fixed"])
    sup_ad = pd.read_csv(input_paths["supervised_adaptive"])
    zs_fixed = pd.read_csv(input_paths["zero_shot_fixed"])
    zs_ad = pd.read_csv(input_paths["zero_shot_adaptive"])
    final = read_json(input_paths["final_analysis"])
    freeze = read_json(input_paths["freeze_manifest"])

    outdir = Path("outputs/publication")
    outdir.mkdir(parents=True, exist_ok=True)

    sup_fixed_table = fixed_metrics(sup_fixed, "supervised")
    zs_fixed_table = fixed_metrics(zs_fixed, "zero_shot")
    fixed_all = pd.concat([sup_fixed_table, zs_fixed_table], ignore_index=True)
    fixed_all.to_csv(outdir / "table_fixed_policies.csv", index=False)

    main_table = main_results(final, sup_fixed_table, zs_fixed_table)
    main_table.to_csv(outdir / "table_main_results.csv", index=False)

    lang_table = pd.concat([
        language_table(sup_ad, sup_fixed, "supervised"),
        language_table(zs_ad, zs_fixed, "zero_shot"),
    ], ignore_index=True)
    lang_table.to_csv(outdir / "table_per_language.csv", index=False)

    class_rows = []
    for setting, adaptive, fixed in [
        ("supervised", sup_ad, sup_fixed),
        ("zero_shot", zs_ad, zs_fixed),
    ]:
        class_rows += per_class_rows(adaptive, setting, "adaptive")
        no = fixed[fixed.policy == "no_retrieval"]
        class_rows += per_class_rows(no, setting, "no_retrieval")
        if setting == "zero_shot":
            dense = fixed[fixed.policy == "dense_cross_k3"]
            if len(dense):
                class_rows += per_class_rows(dense, setting, "dense_cross_k3")
    pd.DataFrame(class_rows).to_csv(outdir / "table_per_class.csv", index=False)

    rescue_rows = []
    examples = []
    for setting, adaptive, fixed in [
        ("supervised", sup_ad, sup_fixed),
        ("zero_shot", zs_ad, zs_fixed),
    ]:
        m = paired(adaptive, fixed)
        counts = m.outcome.value_counts()
        rescue_rows.append({
            "setting": setting,
            "queries": int(len(m)),
            "rescue": int(counts.get("rescue", 0)),
            "harm": int(counts.get("harm", 0)),
            "unchanged_correct": int(counts.get("unchanged_correct", 0)),
            "unchanged_wrong": int(counts.get("unchanged_wrong", 0)),
            "net_correct_gain": int(counts.get("rescue", 0) - counts.get("harm", 0)),
        })

        priority = "success_probability" if "success_probability" in m.columns else "query_id"
        for (lang, outcome), g in m[m.outcome.isin(["rescue", "harm"])].groupby(
            ["language", "outcome"]
        ):
            if priority == "success_probability":
                g = g.sort_values(priority, ascending=False)
            else:
                g = g.sort_values(priority)
            take = g.head(args.examples_per_outcome_language).copy()
            take.insert(0, "setting", setting)
            examples.append(take)

    pd.DataFrame(rescue_rows).to_csv(outdir / "table_rescue_harm.csv", index=False)
    if examples:
        ex = pd.concat(examples, ignore_index=True)
        keep = [
            "setting", "language", "outcome", "query_id", "text", "gold",
            "baseline_prediction", "adaptive_prediction", "adaptive_policy",
            "adaptive_k", "retrieved_languages", "retrieved_labels",
            "retrieved_scores", "retrieved_ids", "success_probability",
            "selection_utility",
        ]
        ex[[c for c in keep if c in ex.columns]].to_csv(
            outdir / "rescue_harm_examples.csv", index=False
        )

    efficiency = pd.DataFrame(
        efficiency_rows("supervised", sup_ad, sup_fixed)
        + efficiency_rows("zero_shot", zs_ad, zs_fixed)
    )
    efficiency.to_csv(outdir / "efficiency_summary.csv", index=False)

    save_confusion(
        sup_ad,
        "Supervised test — AfriARAG adaptive",
        outdir / "confusion_supervised_adaptive.png",
    )
    save_confusion(
        sup_fixed[sup_fixed.policy == "no_retrieval"],
        "Supervised test — No retrieval",
        outdir / "confusion_supervised_no_retrieval.png",
    )
    save_confusion(
        zs_ad,
        "Strict zero-shot test — AfriARAG adaptive",
        outdir / "confusion_zero_shot_adaptive.png",
    )
    save_confusion(
        zs_fixed[zs_fixed.policy == "no_retrieval"],
        "Strict zero-shot test — No retrieval",
        outdir / "confusion_zero_shot_no_retrieval.png",
    )

    plot_language_delta(lang_table, outdir / "per_language_delta.png")
    plot_policy_table(fixed_all, "supervised", outdir / "retrieval_policy_comparison_supervised.png")
    plot_policy_table(fixed_all, "zero_shot", outdir / "retrieval_policy_comparison_zero_shot.png")

    manifest = {
        "package": "AfriARAG Stage-1 publication analysis",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "post-freeze; no model tuning",
        "pretest_git_commit": freeze.get("git_commit"),
        "inputs": [
            {
                "name": name,
                "path": str(path),
                "sha256": sha256(path),
            }
            for name, path in input_paths.items()
        ],
        "important_interpretation": {
            "supervised": (
                "Adaptive Macro-F1 point estimate is above no retrieval, but the "
                "paired bootstrap 95% interval crosses zero."
            ),
            "zero_shot": (
                "Adaptive routing reliably beats no retrieval, but fixed dense_cross_k3 "
                "is the strongest held-out zero-shot fixed policy."
            ),
            "efficiency": (
                "Selected-context rate is not total retrieval-computation rate because "
                "retrieval diagnostics are computed before routing."
            ),
        },
    }
    (outdir / "publication_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    readme = f"""# AfriARAG publication-analysis package

Generated from the frozen Stage-1 held-out outputs.

## Reportable confirmatory comparisons

- Supervised: frozen adaptive Controller v2 vs development-selected no retrieval.
- Strict zero-shot: frozen adaptive router vs development-selected no retrieval.

## Descriptive/post-hoc comparisons

The held-out zero-shot fixed-policy table shows dense_cross_k3 as the strongest
fixed policy. Because this ranking was observed on test, it must be reported as
descriptive/post-hoc and must not be used to retune Stage 1.

## Files

- table_main_results.csv
- table_fixed_policies.csv
- table_per_language.csv
- table_per_class.csv
- table_rescue_harm.csv
- rescue_harm_examples.csv
- efficiency_summary.csv
- confusion_supervised_adaptive.png
- confusion_supervised_no_retrieval.png
- confusion_zero_shot_adaptive.png
- confusion_zero_shot_no_retrieval.png
- per_language_delta.png
- retrieval_policy_comparison_supervised.png
- retrieval_policy_comparison_zero_shot.png
- publication_manifest.json

Development-only ablations are generated separately by
`scripts/run_publication_ablations.py` into this same directory.

## Statistical caution

The supervised Macro-F1 improvement is not statistically reliable at the 95%
two-sided paired-bootstrap level. The strict zero-shot adaptive improvement over
no retrieval is statistically reliable, but the frozen adaptive router is below
fixed dense_cross_k3 on the held-out zero-shot test.

## Efficiency caution

Recorded latency is retrieval scoring/top-k latency from Stage 1. It excludes
candidate retrieval-probe generation and downstream LLM/token cost. Do not
present selected-context retrieval rate as zero-retrieval-compute rate.
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Publication package written to {outdir}")
    print(main_table.to_string(index=False))


if __name__ == "__main__":
    main()
