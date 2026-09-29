import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afriarag.core import choose_oracle, read_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/fixed/dev_predictions.csv")
    ap.add_argument("--config", default="configs/stage1.yaml")
    args = ap.parse_args()
    cfg = read_config(args.config)
    df = pd.read_csv(args.input)
    rows = []
    for (_, _), g in df.groupby(["language","query_id"], sort=False):
        rows.append(choose_oracle(g, cfg["oracle"]["k_cost"], cfg["oracle"]["latency_cost"]))
    oracle = pd.DataFrame(rows)
    outdir = Path("outputs/oracle")
    outdir.mkdir(parents=True, exist_ok=True)
    oracle.to_csv(outdir / "oracle_labels.csv", index=False)
    dist = oracle.groupby(["language","oracle_policy"]).size().rename("n").reset_index()
    dist["share"] = dist.groupby("language")["n"].transform(lambda x: x / x.sum())
    dist.to_csv(outdir / "oracle_distribution.csv", index=False)
    print(dist.to_string(index=False))

if __name__ == "__main__":
    main()
