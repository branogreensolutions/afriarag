from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def read_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_split_name(split: str) -> str:
    return "dev" if split in {"dev", "validation", "valid"} else split


def load_split(root: str | Path, lang: str, split: str) -> pd.DataFrame:
    split = normalize_split_name(split)
    path = Path(root) / lang / f"{split}.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    text_col = cols.get("tweet") or cols.get("text")
    label_col = cols.get("label")
    id_col = cols.get("id")
    if text_col is None:
        raise ValueError(f"No tweet/text column in {path}: {list(df.columns)}")
    out = pd.DataFrame({
        "id": df[id_col].astype(str) if id_col else [f"{lang}_{split}_{i}" for i in range(len(df))],
        "text": df[text_col].fillna("").astype(str),
        "language": lang,
    })
    if label_col:
        out["label"] = df[label_col].astype(str).str.lower().str.strip()
    return out


def load_pool(root: str | Path, languages: Iterable[str]) -> pd.DataFrame:
    frames = []
    for lang in languages:
        try:
            frames.append(load_split(root, lang, "train"))
        except FileNotFoundError:
            continue
    if not frames:
        raise RuntimeError("No training files were found.")
    return pd.concat(frames, ignore_index=True)


def query_features(text: str, language: str, probe_probs: np.ndarray | None = None) -> dict:
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    chars = list(text)
    alpha = [c for c in chars if c.isalpha()]
    ascii_alpha = sum(ord(c) < 128 for c in alpha)
    non_ascii_alpha = len(alpha) - ascii_alpha
    scripts_mix = 0.0 if not alpha else min(ascii_alpha, non_ascii_alpha) / max(1, len(alpha))
    feats = {
        "language": language,
        "char_len": len(text),
        "token_len": len(tokens),
        "avg_token_len": (sum(map(len, tokens)) / len(tokens)) if tokens else 0.0,
        "digit_ratio": sum(c.isdigit() for c in chars) / max(1, len(chars)),
        "upper_ratio": sum(c.isupper() for c in chars) / max(1, len(alpha)),
        "punct_count": sum(c in "!?.,;:" for c in chars),
        "exclaim_count": text.count("!"),
        "question_count": text.count("?"),
        "mention_count": text.count("@"),
        "hashtag_count": text.count("#"),
        "url_count": len(re.findall(r"https?://|www\.", text.lower())),
        "emoji_count": sum(ord(c) > 0x1F000 for c in chars),
        "script_mix": scripts_mix,
    }
    if probe_probs is not None:
        p = np.asarray(probe_probs, dtype=float)
        p = p / max(p.sum(), 1e-12)
        feats["probe_confidence"] = float(p.max())
        feats["probe_entropy"] = float(-(p * np.log(np.clip(p, 1e-12, 1))).sum())
    else:
        feats["probe_confidence"] = 0.0
        feats["probe_entropy"] = 0.0
    return feats


class CharNgramProbe:
    """Cheap no-retrieval baseline and uncertainty probe."""

    def __init__(self, max_features: int = 120000, C: float = 3.0, seed: int = 42):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=max_features
        )
        self.model = LogisticRegression(
            max_iter=1500, C=C, class_weight="balanced", random_state=seed
        )

    def fit(self, texts: Iterable[str], labels: Iterable[str]):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        return self

    def predict(self, texts: Iterable[str]):
        return self.model.predict(self.vectorizer.transform(texts))

    def predict_proba(self, texts: Iterable[str]):
        return self.model.predict_proba(self.vectorizer.transform(texts))


class BM25Retriever:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [self._tok(x) for x in texts]
        self.n = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.n)
        self.df = Counter()
        for d in self.docs:
            self.df.update(set(d))
        self.tf = [Counter(d) for d in self.docs]

    @staticmethod
    def _tok(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    def score(self, query: str) -> np.ndarray:
        q = self._tok(query)
        scores = np.zeros(self.n, dtype=np.float32)
        for term in q:
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self.tf):
                f = tf.get(term, 0)
                if not f:
                    continue
                dl = len(self.docs[i])
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                scores[i] += idf * (f * (self.k1 + 1) / denom)
        return scores


class DenseRetriever:
    def __init__(self, texts: list[str], model_name: str, batch_size: int = 64):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "Dense retrieval requires sentence-transformers and faiss-cpu."
            ) from e
        self.model = SentenceTransformer(model_name)
        passages = [f"passage: {x}" for x in texts]
        emb = self.model.encode(
            passages,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        self.emb = np.asarray(emb, dtype="float32")
        self.index = faiss.IndexFlatIP(self.emb.shape[1])
        self.index.add(self.emb)

    def score(self, query: str) -> np.ndarray:
        q = self.model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        q = np.asarray(q, dtype="float32")
        return (self.emb @ q[0]).astype(np.float32)


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def topk(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0 or len(scores) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]


def weighted_vote(labels: Iterable[str], scores: Iterable[float]) -> str:
    totals = defaultdict(float)
    for label, score in zip(labels, scores):
        totals[str(label)] += max(float(score), 1e-6)
    if not totals:
        return "neutral"
    return max(sorted(totals), key=lambda x: totals[x])


def classification_metrics(y_true, y_pred) -> dict:
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def retrieval_harm_rescue(gold, no_rag, rag) -> dict:
    gold, no_rag, rag = map(np.asarray, (gold, no_rag, rag))
    return {
        "retrieval_harm_rate": float(np.mean((no_rag == gold) & (rag != gold))),
        "retrieval_rescue_rate": float(np.mean((no_rag != gold) & (rag == gold))),
    }


def choose_oracle(
    group: pd.DataFrame, k_cost: float = 0.001, latency_cost: float = 0.0
) -> pd.Series:
    correct = group[group["correct"] == True].copy()  # noqa: E712
    if correct.empty:
        base = group.iloc[0].copy()
        base["oracle_policy"] = "unresolved"
        base["oracle_utility"] = np.nan
        return base
    correct["utility"] = (
        1.0
        - k_cost * correct["k"].astype(float)
        - latency_cost * correct["latency_ms"].astype(float)
    )
    correct["no_rag_tie"] = (correct["policy"] == "no_retrieval").astype(int)
    best = correct.sort_values(
        ["utility", "no_rag_tie"], ascending=[False, False]
    ).iloc[0].copy()
    best["oracle_policy"] = best["policy"]
    best["oracle_utility"] = best["utility"]
    return best
