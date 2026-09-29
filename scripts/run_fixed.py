from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afriarag.core import (
    BM25Retriever,
    CharNgramProbe,
    DenseEncoder,
    classification_metrics,
    load_split,
    minmax,
    query_features,
    read_config,
    topk,
    weighted_vote,
)


def frame_signature(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for row in df[["id", "text"]].itertuples(index=False):
        h.update(str(row.id).encode("utf-8", errors="ignore"))
        h.update(b"\0")
        h.update(str(row.text).encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)


def cached_embeddings(
    encoder: DenseEncoder,
    df: pd.DataFrame,
    cache_root: str | Path,
    cache_name: str,
    query: bool = False,
) -> np.ndarray:
    cache_dir = Path(cache_root) / model_slug(encoder.model_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sig = frame_signature(df)
    path = cache_dir / f"{cache_name}_{sig}.npy"

    if path.exists():
        arr = np.load(path)
        if arr.shape[0] == len(df):
            print(f"[dense] cache hit: {path} -> {arr.shape}")
            return np.asarray(arr, dtype="float32")
        print(f"[dense] ignoring stale cache with wrong row count: {path}")

    kind = "queries" if query else "passages"
    print(f"[dense] encoding {kind}: {cache_name} ({len(df):,} rows)")
    texts = df["text"].tolist()
    arr = (
        encoder.encode_queries(texts, show_progress_bar=True)
        if query
        else encoder.encode_passages(texts, show_progress_bar=True)
    )
    np.save(path, arr)
    print(f"[dense] saved cache: {path} -> {arr.shape}")
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--languages", nargs="*")
    ap.add_argument("--limit", type=int, default=None, help="Optional per-language debug limit")
    ap.add_argument(
        "--policies",
        nargs="*",
        help="Optional exact policy names to run. Useful for a small smoke test.",
    )
    ap.add_argument("--run-name", default=None, help="Optional output tag")
    args = ap.parse_args()

    cfg = read_config(args.config)
    root = cfg["data"]["root"]
    supervised = cfg["data"]["supervised_languages"]
    zero = cfg["data"]["zero_shot_languages"]
    languages = args.languages or supervised

    configured = cfg["policies"]
    if args.policies:
        known = {p["name"] for p in configured}
        missing = sorted(set(args.policies) - known)
        if missing:
            raise SystemExit(f"Unknown policies: {missing}. Known policies: {sorted(known)}")
        policies = [p for p in configured if p["name"] in set(args.policies)]
    else:
        policies = configured

    print("Policies:", ", ".join(p["name"] for p in policies))

    train_by_lang = {}
    for lang in supervised:
        train_by_lang[lang] = load_split(root, lang, "train")
    train_all = pd.concat([train_by_lang[l] for l in supervised], ignore_index=True)

    applicable_dense = [
        p for p in policies if p["retriever"] in {"dense", "hybrid"}
    ]
    dense_needed = bool(applicable_dense)
    dense_scopes = {p["scope"] for p in applicable_dense}

    # Encode each training language once and cache it. Per-language caches let a
    # small same-language smoke test be reused by later multilingual runs.
    emb_by_lang: dict[str, np.ndarray] = {}
    encoder = None
    if dense_needed:
        if dense_scopes & {"multilingual", "cross"}:
            dense_train_langs = list(supervised)
        else:
            dense_train_langs = [l for l in languages if l in supervised]

        if dense_train_langs:
            encoder = DenseEncoder(cfg["dense"]["model"], cfg["dense"]["batch_size"])
            for lang in dense_train_langs:
                emb_by_lang[lang] = cached_embeddings(
                    encoder,
                    train_by_lang[lang],
                    cfg["dense"]["cache_dir"],
                    f"train_{lang}",
                    query=False,
                )

    pool_cache: dict[tuple, pd.DataFrame] = {}
    emb_pool_cache: dict[tuple, np.ndarray] = {}
    bm_cache: dict[tuple, BM25Retriever] = {}
    rows, metric_rows = [], []

    def get_pool(target_lang: str, scope: str):
        key = ("multilingual",) if scope == "multilingual" else (scope, target_lang)
        if key in pool_cache:
            return key, pool_cache[key], emb_pool_cache.get(key)

        if scope == "same":
            pool = train_by_lang[target_lang].reset_index(drop=True)
            emb = emb_by_lang.get(target_lang)
        elif scope == "multilingual":
            pool = train_all.reset_index(drop=True)
            emb = (
                np.vstack([emb_by_lang[l] for l in supervised])
                if dense_needed and all(l in emb_by_lang for l in supervised)
                else None
            )
        elif scope == "cross":
            langs = [l for l in supervised if l != target_lang]
            pool = pd.concat([train_by_lang[l] for l in langs], ignore_index=True)
            emb = (
                np.vstack([emb_by_lang[l] for l in langs])
                if dense_needed and all(l in emb_by_lang for l in langs)
                else None
            )
        else:
            raise ValueError(scope)

        pool_cache[key] = pool
        if emb is not None:
            emb_pool_cache[key] = emb
        return key, pool, emb

    for lang in languages:
        qdf = load_split(root, lang, args.split)
        if args.limit:
            qdf = qdf.head(args.limit).copy()

        probe_train = train_all if lang in zero else train_by_lang[lang]
        if probe_train.empty:
            probe_train = train_all
        probe = CharNgramProbe(
            cfg["probe"]["max_features"], cfg["probe"]["C"], cfg["seed"]
        ).fit(probe_train.text, probe_train.label)
        no_pred = probe.predict(qdf.text)
        no_prob = probe.predict_proba(qdf.text)

        lang_dense_policies = [
            p for p in policies
            if p["retriever"] in {"dense", "hybrid"}
            and not (lang in zero and p["scope"] == "same")
        ]
        q_emb = None
        if lang_dense_policies:
            if encoder is None:
                encoder = DenseEncoder(cfg["dense"]["model"], cfg["dense"]["batch_size"])
            q_emb = cached_embeddings(
                encoder,
                qdf,
                cfg["dense"]["cache_dir"],
                f"query_{lang}_{args.split}_n{len(qdf)}",
                query=True,
            )

        # Prepare only the scopes needed by selected policies.
        scopes = {
            p["scope"] for p in policies
            if p["scope"] != "none" and not (lang in zero and p["scope"] == "same")
        }
        for scope in scopes:
            key, pool, _ = get_pool(lang, scope)
            if any(
                p["scope"] == scope and p["retriever"] in {"bm25", "hybrid"}
                for p in policies
            ) and key not in bm_cache:
                print(f"[bm25] indexing {scope} pool for {lang}: {len(pool):,} rows")
                bm_cache[key] = BM25Retriever(pool.text.tolist())

        for i, (_, row) in enumerate(qdf.iterrows()):
            feats = query_features(row.text, lang, no_prob[i])
            score_cache: dict[tuple, np.ndarray] = {}

            for policy in policies:
                if lang in zero and policy["scope"] == "same":
                    continue

                pname = policy["name"]
                gold = row.get("label")

                if policy["retriever"] == "none":
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
                key, pool, pool_emb = get_pool(lang, scope)
                t0 = time.perf_counter()

                if policy["retriever"] in {"bm25", "hybrid"}:
                    cache_key = ("bm25", key)
                    if cache_key not in score_cache:
                        score_cache[cache_key] = minmax(bm_cache[key].score(row.text))

                if policy["retriever"] in {"dense", "hybrid"}:
                    if pool_emb is None or q_emb is None:
                        raise RuntimeError(
                            f"Dense embeddings unavailable for language={lang}, scope={scope}"
                        )
                    cache_key = ("dense", key)
                    if cache_key not in score_cache:
                        score_cache[cache_key] = minmax(
                            (pool_emb @ q_emb[i]).astype(np.float32)
                        )

                if policy["retriever"] == "bm25":
                    scores = score_cache[("bm25", key)]
                elif policy["retriever"] == "dense":
                    scores = score_cache[("dense", key)]
                else:
                    bs = score_cache[("bm25", key)]
                    ds = score_cache[("dense", key)]
                    scores = policy["alpha"] * ds + (1 - policy["alpha"]) * bs

                idx, vals = topk(scores, policy["k"])
                pred = weighted_vote(pool.iloc[idx].label.tolist(), vals.tolist())
                elapsed = (time.perf_counter() - t0) * 1000

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

    if args.run_name:
        tag = args.run_name
    elif args.languages or args.limit:
        parts = [args.split]
        if args.languages:
            parts.append("-".join(languages))
        if args.limit:
            parts.append(f"n{args.limit}")
        tag = "_".join(parts)
    else:
        tag = args.split

    outdir = Path("outputs/fixed")
    outdir.mkdir(parents=True, exist_ok=True)
    pred_path = outdir / f"{tag}_predictions.csv"
    metric_path = outdir / f"{tag}_metrics.csv"
    pd.DataFrame(rows).to_csv(pred_path, index=False)
    pd.DataFrame(metric_rows).to_csv(metric_path, index=False)

    print(json.dumps({
        "predictions": str(pred_path),
        "metrics": str(metric_path),
        "rows": len(rows),
        "policies": [p["name"] for p in policies],
    }, indent=2))


if __name__ == "__main__":
    main()
