import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afriarag.core import BM25Retriever, minmax, query_features, weighted_vote

def test_bm25_prefers_matching_document():
    r = BM25Retriever(["good phone excellent", "bad terrible network", "ordinary day"])
    s = r.score("terrible network")
    assert int(np.argmax(s)) == 1

def test_minmax_range():
    x = minmax(np.array([2.0, 4.0, 8.0]))
    assert x.min() >= 0 and x.max() <= 1

def test_weighted_vote():
    assert weighted_vote(["positive","negative","positive"], [0.7,0.2,0.6]) == "positive"

def test_query_features_are_finite():
    f = query_features("Wallahi this network is bad! 😥", "hau", np.array([0.1,0.2,0.7]))
    assert f["token_len"] > 0
    assert f["probe_confidence"] == 0.7
