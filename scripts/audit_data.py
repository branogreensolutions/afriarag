from pathlib import Path
import hashlib
import json
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afriarag.core import load_split, read_config

def key_text(text: str) -> str:
    return " ".join(str(text).lower().split())

def main(config="configs/stage1.yaml"):
    cfg = read_config(config)
    root = Path(cfg["data"].get("raw_root", cfg["data"]["root"]))
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
                "language": lang, "split": split, "n": len(df), "sha256": sha,
                **{f"label_{k}": v for k, v in counts.items()},
            })
            for _, r in df.iterrows():
                all_items.append({
                    "language": lang, "split": split, "id": r.id,
                    "label": r.get("label"), "text_key": key_text(r.text),
                })

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "dataset_manifest.csv", index=False)
    items = pd.DataFrame(all_items)

    dup = items[items.duplicated(["language","text_key"], keep=False)].sort_values(
        ["language","text_key","split"]
    ) if len(items) else items
    dup.to_csv(outdir / "exact_duplicates.csv", index=False)

    cross_rows, conflicts = [], []
    if len(dup):
        for (lang, key), g in dup.groupby(["language","text_key"], sort=False):
            if g["split"].nunique() > 1:
                cross_rows.append({
                    "language": lang,
                    "text_key": key,
                    "occurrences": len(g),
                    "splits": ",".join(sorted(g["split"].unique())),
                    "ids": ",".join(g["id"].astype(str)),
                })
            labels = sorted(set(x for x in g["label"].dropna().astype(str)))
            if len(labels) > 1:
                conflicts.append({
                    "language": lang,
                    "text_key": key,
                    "occurrences": len(g),
                    "splits": ",".join(sorted(g["split"].unique())),
                    "labels": ",".join(labels),
                    "ids": ",".join(g["id"].astype(str)),
                })

    pd.DataFrame(cross_rows).to_csv(outdir / "cross_split_duplicates.csv", index=False)
    pd.DataFrame(conflicts).to_csv(outdir / "duplicate_label_conflicts.csv", index=False)

    report = {
        "rows": int(summary["n"].sum()) if len(summary) else 0,
        "files": len(summary),
        "exact_duplicate_rows": int(len(dup)),
        "cross_split_duplicate_groups": len(cross_rows),
        "duplicate_label_conflict_groups": len(conflicts),
        "zero_shot_train_rows": {
            lang: int(summary[(summary.language == lang) & (summary.split == "train")]["n"].sum())
            for lang in cfg["data"]["zero_shot_languages"]
        },
    }
    (outdir / "audit_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
