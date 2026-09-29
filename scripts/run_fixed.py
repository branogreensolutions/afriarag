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
from sklearn.model_selection import train_test_split

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
        if args.limit and args.limit < len(qdf):
            # The official AfriSenti files can be label-ordered, so taking head(N)
            # creates a degenerate smoke sample. Use a deterministic stratified
            # sample when labels are available.
            if "label" in qdf.columns and qdf["label"].notna().all() and qdf["label"].nunique() > 1:
                qdf, _ = train_test_split(
                    qdf,
                    train_size=args.limit,
                    random_state=cfg["seed"],
                    stratify=qdf["label"],
                )
            else:
                qdf = qdf.sample(n=args.limit, random_state=cfg["seed"])
            qdf = qdf.sort_values("id").reset_index(drop=True)
        if "label" in qdf.columns:
            print(f"[sample] {lang}/{args.split}: {len(qdf)} rows; labels={qdf['label'].value_counts().to_dict()}")

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
            score_latency: dict[tuple, float] = {}

            raw_score_cache: dict[tuple, np.ndarray] = {}

            def get_bm25_scores(key, row_text):
                cache_key = ("bm25", key)
                if cache_key not in score_cache:
                    t_score = time.perf_counter()
                    raw = bm_cache[key].score(row_text).astype(np.float32)
                    raw_score_cache[cache_key] = raw
                    score_cache[cache_key] = minmax(raw)
                    score_latency[cache_key] = (time.perf_counter() - t_score) * 1000
                return score_cache[cache_key], raw_score_cache[cache_key], score_latency[cache_key]

            def get_dense_scores(key, pool_emb, qvec):
                cache_key = ("dense", key)
                if cache_key not in score_cache:
                    t_score = time.perf_counter()
                    raw = (pool_emb @ qvec).astype(np.float32)
                    raw_score_cache[cache_key] = raw
                    score_cache[cache_key] = minmax(raw)
                    score_latency[cache_key] = (time.perf_counter() - t_score) * 1000
                return score_cache[cache_key], raw_score_cache[cache_key], score_latency[cache_key]

            def top_stats(scores):
                if scores is None or len(scores) == 0:
                    return 0.0, 0.0
                if len(scores) == 1:
                    return float(scores[0]), 0.0
                idx2 = np.argpartition(-scores, 1)[:2]
                vals = np.sort(scores[idx2])[::-1]
                return float(vals[0]), float(vals[0] - vals[1])

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
                        "latency_ms": 0.0, "top_score": 0.0, "score_margin": 0.0,
                        "bm25_top_raw": 0.0, "bm25_margin_raw": 0.0,
                        "dense_top_raw": 0.0, "dense_margin_raw": 0.0,
                        "retrieved_ids": "", "retrieved_languages": "",
                        "retrieved_labels": "", "retrieved_scores": "", **feats
                    })
                    continue

                scope = policy["scope"]
                key, pool, pool_emb = get_pool(lang, scope)

                bm_scores = dense_scores = None
                bm_raw = dense_raw = None
                bm_ms = dense_ms = 0.0
                if policy["retriever"] in {"bm25", "hybrid"}:
                    bm_scores, bm_raw, bm_ms = get_bm25_scores(key, row.text)

                if policy["retriever"] in {"dense", "hybrid"}:
                    if pool_emb is None or q_emb is None:
                        raise RuntimeError(
                            f"Dense embeddings unavailable for language={lang}, scope={scope}"
                        )
                    dense_scores, dense_raw, dense_ms = get_dense_scores(key, pool_emb, q_emb[i])

                fusion_ms = 0.0
                if policy["retriever"] == "bm25":
                    scores = bm_scores
                    retrieval_ms = bm_ms
                elif policy["retriever"] == "dense":
                    scores = dense_scores
                    retrieval_ms = dense_ms
                else:
                    hkey = ("hybrid", key, float(policy["alpha"]))
                    if hkey not in score_cache:
                        t_fuse = time.perf_counter()
                        score_cache[hkey] = (
                            policy["alpha"] * dense_scores
                            + (1 - policy["alpha"]) * bm_scores
                        )
                        score_latency[hkey] = (time.perf_counter() - t_fuse) * 1000
                    scores = score_cache[hkey]
                    fusion_ms = score_latency[hkey]
                    # Report estimated standalone latency: both retrievers plus fusion,
                    # even though caches avoid recomputation inside this experiment run.
                    retrieval_ms = bm_ms + dense_ms + fusion_ms

                t_topk = time.perf_counter()
                idx, vals = topk(scores, policy["k"])
                pred = weighted_vote(pool.iloc[idx].label.tolist(), vals.tolist())
                topk_ms = (time.perf_counter() - t_topk) * 1000
                elapsed = retrieval_ms + topk_ms

                policy_top, policy_margin = top_stats(scores)
                bm_top_raw, bm_margin_raw = top_stats(bm_raw)
                dense_top_raw, dense_margin_raw = top_stats(dense_raw)

                rows.append({
                    "query_id": row.id, "text": row.text, "language": lang,
                    "gold": gold, "policy": pname, "retriever": policy["retriever"],
                    "scope": scope, "k": policy["k"], "alpha": policy["alpha"],
                    "prediction": pred, "correct": bool(pred == gold),
                    "latency_ms": elapsed,
                    "top_score": policy_top,
                    "score_margin": policy_margin,
                    "bm25_top_raw": bm_top_raw,
                    "bm25_margin_raw": bm_margin_raw,
                    "dense_top_raw": dense_top_raw,
                    "dense_margin_raw": dense_margin_raw,
                    "retrieved_ids": ",".join(pool.iloc[idx].id.astype(str)),
                    "retrieved_languages": ",".join(pool.iloc[idx].language.astype(str)),
                    "retrieved_labels": ",".join(pool.iloc[idx].label.astype(str)),
                    "retrieved_scores": ",".join(f"{float(v):.8f}" for v in vals),
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
