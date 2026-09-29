from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afriarag.core import (
    BM25Retriever, CharNgramProbe, DenseRetriever, classification_metrics,
    load_pool, load_split, minmax, query_features, read_config, topk, weighted_vote
)

def scope_pool(train_all, target_lang, scope):
    if scope == "same":
        return train_all[train_all.language == target_lang].reset_index(drop=True)
    if scope == "cross":
        return train_all[train_all.language != target_lang].reset_index(drop=True)
    if scope == "multilingual":
        return train_all.reset_index(drop=True)
    raise ValueError(scope)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--languages", nargs="*")
    ap.add_argument("--limit", type=int, default=None, help="Optional per-language debug limit")
    args = ap.parse_args()

    cfg = read_config(args.config)
    root = cfg["data"]["root"]
    supervised = cfg["data"]["supervised_languages"]
    zero = cfg["data"]["zero_shot_languages"]
    languages = args.languages or supervised
    train_all = load_pool(root, supervised)

    pool_cache = {}
    bm_cache = {}
    dense_cache = {}
    rows, metric_rows = [], []

    for lang in languages:
        qdf = load_split(root, lang, args.split)
        if args.limit:
            qdf = qdf.head(args.limit)

        probe_train = train_all if lang in zero else train_all[train_all.language == lang]
        if probe_train.empty:
            probe_train = train_all
        probe = CharNgramProbe(
            cfg["probe"]["max_features"], cfg["probe"]["C"], cfg["seed"]
        ).fit(probe_train.text, probe_train.label)
        no_pred = probe.predict(qdf.text)
        no_prob = probe.predict_proba(qdf.text)

        for policy in cfg["policies"]:
            if lang in zero and policy["scope"] == "same":
                continue
            pname = policy["name"]

            if policy["retriever"] == "none":
                for i, (_, row) in enumerate(qdf.iterrows()):
                    feats = query_features(row.text, lang, no_prob[i])
                    gold = row.get("label")
                    pred = no_pred[i]
                    rows.append({
                        "query_id": row.id, "text": row.text, "language": lang,
                        "gold": gold, "policy": pname, "retriever": "none",
                        "scope": "none", "k": 0, "alpha": 0.0,
                        "prediction": pred, "correct": bool(pred == gold),
                        "latency_ms": 0.0, "top_score": 0.0,
                        "retrieved_ids": "", "retrieved_languages": "", **feats
                    })
                continue

            scope = policy["scope"]
            pool_key = ("multilingual",) if scope == "multilingual" else (scope, lang)
            if pool_key not in pool_cache:
                pool_cache[pool_key] = scope_pool(train_all, lang, scope)
            pool = pool_cache[pool_key]

            if policy["retriever"] in {"bm25","hybrid"} and pool_key not in bm_cache:
                bm_cache[pool_key] = BM25Retriever(pool.text.tolist())
            if policy["retriever"] in {"dense","hybrid"} and pool_key not in dense_cache:
                dense_cache[pool_key] = DenseRetriever(
                    pool.text.tolist(), cfg["dense"]["model"], cfg["dense"]["batch_size"]
                )

            for i, (_, row) in enumerate(qdf.iterrows()):
                t0 = time.perf_counter()
                if policy["retriever"] == "bm25":
                    scores = minmax(bm_cache[pool_key].score(row.text))
                elif policy["retriever"] == "dense":
                    scores = minmax(dense_cache[pool_key].score(row.text))
                else:
                    bs = minmax(bm_cache[pool_key].score(row.text))
                    ds = minmax(dense_cache[pool_key].score(row.text))
                    scores = policy["alpha"] * ds + (1 - policy["alpha"]) * bs

                idx, vals = topk(scores, policy["k"])
                pred = weighted_vote(pool.iloc[idx].label.tolist(), vals.tolist())
                elapsed = (time.perf_counter() - t0) * 1000
                feats = query_features(row.text, lang, no_prob[i])
                gold = row.get("label")
                rows.append({
                    "query_id": row.id, "text": row.text, "language": lang,
                    "gold": gold, "policy": pname, "retriever": policy["retriever"],
                    "scope": scope, "k": policy["k"], "alpha": policy["alpha"],
                    "prediction": pred, "correct": bool(pred == gold),
                    "latency_ms": elapsed,
                    "top_score": float(vals[0]) if len(vals) else 0.0,
                    "retrieved_ids": ",".join(pool.iloc[idx].id.astype(str)),
                    "retrieved_languages": ",".join(pool.iloc[idx].language.astype(str)),
                    **feats
                })

        lang_frame = pd.DataFrame([r for r in rows if r["language"] == lang])
        for pname, g in lang_frame.groupby("policy"):
            if g.gold.notna().all():
                metric_rows.append({
                    "language": lang, "policy": pname,
                    **classification_metrics(g.gold, g.prediction),
                    "retrieval_rate": float((g.k > 0).mean()),
                    "avg_k": float(g.k.mean()),
                    "latency_ms": float(g.latency_ms.mean())
                })

    outdir = Path("outputs/fixed")
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir / f"{args.split}_predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(outdir / f"{args.split}_metrics.csv", index=False)
    print(json.dumps({
        "predictions": str(outdir / f"{args.split}_predictions.csv"),
        "metrics": str(outdir / f"{args.split}_metrics.csv"),
        "rows": len(rows)
    }, indent=2))

if __name__ == "__main__":
    main()
