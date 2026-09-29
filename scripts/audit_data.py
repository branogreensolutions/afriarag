from pathlib import Path
import hashlib
import json
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afriarag.core import load_split, read_config

def main(config="configs/stage1.yaml"):
    cfg = read_config(config)
    root = Path(cfg["data"]["root"])
    langs = cfg["data"]["supervised_languages"] + cfg["data"]["zero_shot_languages"]
    outdir = Path("outputs/audit")
    outdir.mkdir(parents=True, exist_ok=True)
    rows, all_items = [], []
    for lang in langs:
        for split in ("train","dev","test"):
            try:
                df = load_split(root, lang, split)
            except FileNotFoundError:
                continue
            path = root / lang / f"{split}.tsv"
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            counts = df["label"].value_counts().to_dict() if "label" in df else {}
            rows.append({
                "language": lang,
                "split": split,
                "n": len(df),
                "sha256": sha,
                **{f"label_{k}": v for k, v in counts.items()},
            })
            for _, r in df.iterrows():
                all_items.append({
                    "language": lang,
                    "split": split,
                    "id": r.id,
                    "text_key": " ".join(r.text.lower().split()),
                })
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "dataset_manifest.csv", index=False)
    items = pd.DataFrame(all_items)
    dup = items[items.duplicated("text_key", keep=False)].sort_values("text_key") if len(items) else items
    dup.to_csv(outdir / "exact_duplicates.csv", index=False)
    leakage = []
    if len(dup):
        for key, g in dup.groupby("text_key"):
            if g["split"].nunique() > 1:
                leakage.append({
                    "text_key": key,
                    "occurrences": len(g),
                    "splits": ",".join(sorted(g["split"].unique())),
                    "languages": ",".join(sorted(g["language"].unique())),
                })
    pd.DataFrame(leakage).to_csv(outdir / "cross_split_duplicates.csv", index=False)
    report = {
        "rows": int(summary["n"].sum()) if len(summary) else 0,
        "files": len(summary),
        "cross_split_duplicate_groups": len(leakage),
    }
    (outdir / "audit_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
