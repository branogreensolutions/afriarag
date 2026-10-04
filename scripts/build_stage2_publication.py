"""Build publication-ready Stage-2 comparison tables and figures.

This is reporting only. It consumes frozen Stage-2 comparison outputs and does
not train or tune any model.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    root = Path("outputs/transformer_baseline/comparison")
    main_path = root / "comparison_main.csv"
    lang_path = root / "comparison_per_language.csv"
    summary_path = root / "comparison_summary.json"

    for p in [main_path, lang_path, summary_path]:
        if not p.exists():
            raise SystemExit(f"Missing required input: {p}")

    main = pd.read_csv(main_path)
    lang = pd.read_csv(lang_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    outdir = Path("outputs/publication")
    outdir.mkdir(parents=True, exist_ok=True)

    main.to_csv(outdir / "table_stage2_transformer_comparison.csv", index=False)
    lang.to_csv(outdir / "table_stage2_per_language.csv", index=False)

    sig_rows = []
    for setting in ["supervised", "zero_shot"]:
        block = summary[setting]
        for key, value in block.items():
            if not key.startswith("transformer_vs_"):
                continue
            opponent = key.replace("transformer_vs_", "")
            sig_rows.append({
                "setting": setting,
                "comparison": f"transformer vs {opponent}",
                "macro_f1_delta_mean": value["macro_f1_delta"]["mean"],
                "macro_f1_ci95_low": value["macro_f1_delta"]["ci95_low"],
                "macro_f1_ci95_high": value["macro_f1_delta"]["ci95_high"],
                "accuracy_delta_mean": value["accuracy_delta"]["mean"],
                "accuracy_ci95_low": value["accuracy_delta"]["ci95_low"],
                "accuracy_ci95_high": value["accuracy_delta"]["ci95_high"],
            })
    pd.DataFrame(sig_rows).to_csv(
        outdir / "table_stage2_bootstrap_comparisons.csv",
        index=False,
    )

    for setting in ["supervised", "zero_shot"]:
        x = main[main.setting == setting].sort_values("macro_f1").copy()
        fig, ax = plt.subplots(figsize=(8.2, max(4.0, 0.65 * len(x) + 2)))
        ax.barh(np.arange(len(x)), x.macro_f1.to_numpy())
        ax.set(
            yticks=np.arange(len(x)),
            yticklabels=x.system.tolist(),
            xlabel="Macro-F1",
            title=f"{setting.replace('_', ' ').title()} — Stage 2 comparison",
        )
        fig.tight_layout()
        fig.savefig(
            outdir / f"stage2_{setting}_comparison.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)

    z = lang[lang.setting == "zero_shot"].copy()
    systems = ["transformer", "dense_cross_k3", "afriarag_adaptive", "no_retrieval"]
    langs = ["orm", "tir"]
    z = z[z.system.isin(systems) & z.language.isin(langs)]

    width = 0.2
    xpos = np.arange(len(langs))
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for i, system in enumerate(systems):
        vals = []
        for lang_code in langs:
            row = z[(z.language == lang_code) & (z.system == system)]
            vals.append(float(row.macro_f1.iloc[0]) if len(row) else np.nan)
        ax.bar(xpos + (i - 1.5) * width, vals, width=width, label=system)

    ax.set(
        xticks=xpos,
        xticklabels=["Oromo", "Tigrinya"],
        ylabel="Macro-F1",
        title="Strict zero-shot performance by target language",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        outdir / "stage2_zero_shot_by_language.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Stage-2 publication outputs written to {outdir}")


if __name__ == "__main__":
    main()
