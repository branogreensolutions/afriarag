"""Build a strict zero-shot-style development simulation over the 12 supervised languages.

For each target language:
- train the no-retrieval char-ngram probe on the other 11 languages only;
- reuse cross-lingual retrieval predictions that already exclude target-language
  training examples;
- overwrite probe confidence/entropy with the zero-shot probe values;
- retain only actions available when a target language has no labelled training
  corpus.

This produces controller-training data for leave-one-language-out zero-shot
routing without using Oromo or Tigrinya labels.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afriarag.core import (
    CharNgramProbe,
    load_pool,
    load_split,
    query_features,
    read_config,
)

ZERO_SHOT_ACTIONS = [
    "no_retrieval",
    "dense_cross_k3",
    "hybrid_cross_k3",
    "hybrid_cross_k5",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--predictions", default="outputs/fixed/dev_predictions.csv")
    args = ap.parse_args()

    cfg = read_config(args.config)
    root = cfg["data"]["root"]
    languages = cfg["data"]["supervised_languages"]
    full = pd.read_csv(args.predictions)

    parts = []

    for target in languages:
        print(f"[zero-shot-sim] target={target}: fitting no-RAG probe on other 11 languages")
        source_langs = [x for x in languages if x != target]
        train = load_pool(root, source_langs)
        qdf = load_split(root, target, "dev")

        probe = CharNgramProbe(
            cfg["probe"]["max_features"],
            cfg["probe"]["C"],
            cfg["seed"],
        ).fit(train.text, train.label)

        pred = probe.predict(qdf.text)
        prob = probe.predict_proba(qdf.text)

        # Construct strict zero-shot no-retrieval rows.
        no_rows = []
        for i, (_, row) in enumerate(qdf.iterrows()):
            feats = query_features(row.text, target, prob[i])
            gold = row.get("label")
            no_rows.append({
                "query_id": row.id,
                "text": row.text,
                "language": target,
                "gold": gold,
                "policy": "no_retrieval",
                "retriever": "none",
                "scope": "none",
                "k": 0,
                "alpha": 0.0,
                "prediction": pred[i],
                "correct": bool(pred[i] == gold),
                "latency_ms": 0.0,
                "top_score": 0.0,
                "score_margin": 0.0,
                "bm25_top_raw": 0.0,
                "bm25_margin_raw": 0.0,
                "dense_top_raw": 0.0,
                "dense_margin_raw": 0.0,
                "retrieved_ids": "",
                "retrieved_languages": "",
                "retrieved_labels": "",
                "retrieved_scores": "",
                **feats,
            })

        parts.append(pd.DataFrame(no_rows))

        # Existing cross-lingual rows already exclude target-language training.
        cross = full[
            (full.language == target)
            & (full.policy.isin(ZERO_SHOT_ACTIONS[1:]))
        ].copy()

        # Replace the same-language probe confidence used in the original run
        # with the strict zero-shot probe confidence/entropy.
        qfeat = pd.DataFrame([
            {
                "query_id": row.id,
                **query_features(row.text, target, prob[i]),
            }
            for i, (_, row) in enumerate(qdf.iterrows())
        ])
        replace_cols = [
            "char_len","token_len","avg_token_len","digit_ratio","upper_ratio",
            "punct_count","exclaim_count","question_count","mention_count",
            "hashtag_count","url_count","emoji_count","script_mix",
            "probe_confidence","probe_entropy",
        ]
        cross = cross.drop(columns=[c for c in replace_cols if c in cross.columns])
        cross = cross.merge(qfeat[["query_id"] + replace_cols], on="query_id", how="left")
        parts.append(cross)

    out = pd.concat(parts, ignore_index=True)
    outdir = Path("outputs/zero_shot_sim")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "dev_zero_shot_predictions.csv"
    out.to_csv(path, index=False)

    print(f"saved {path}")
    print(out.groupby(["language","policy"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
