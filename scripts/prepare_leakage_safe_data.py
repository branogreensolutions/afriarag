from __future__ import annotations

from pathlib import Path
import json
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afriarag.core import load_split, read_config

def key_text(text: str) -> str:
    return " ".join(str(text).lower().split())

def remove_conflicting_duplicates(df: pd.DataFrame):
    if df.empty or "label" not in df.columns:
        return df.copy(), set()
    x = df.copy()
    x["text_key"] = x["text"].map(key_text)
    conflicts = set()
    for key, g in x.groupby("text_key"):
        if g["label"].nunique(dropna=True) > 1:
            conflicts.add(key)
    return x[~x["text_key"].isin(conflicts)].copy(), conflicts

def dedup(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    x = df.copy()
    if "text_key" not in x:
        x["text_key"] = x["text"].map(key_text)
    return x.drop_duplicates("text_key", keep="first").copy()

def write_split(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    cols = ["id","text"] + (["label"] if "label" in out.columns else [])
    out = out[cols].rename(columns={"id":"ID","text":"tweet"})
    out.to_csv(path, sep="\t", index=False)

def main(config="configs/stage1.yaml"):
    cfg = read_config(config)
    raw_root = Path(cfg["data"]["raw_root"])
    clean_root = Path(cfg["data"]["root"])
    supervised = set(cfg["data"]["supervised_languages"])
    zero = set(cfg["data"]["zero_shot_languages"])
    langs = list(cfg["data"]["supervised_languages"]) + list(cfg["data"]["zero_shot_languages"])
    lc = cfg["leakage"]

    outdir = Path("outputs/audit")
    outdir.mkdir(parents=True, exist_ok=True)
    exclusions = []
    manifest = []

    for lang in langs:
        raw = {}
        for split in ("train","dev","test"):
            try:
                raw[split] = load_split(raw_root, lang, split)
            except FileNotFoundError:
                raw[split] = pd.DataFrame(columns=["id","text","language","label"])
            if not raw[split].empty:
                raw[split]["text_key"] = raw[split]["text"].map(key_text)

        # TRAIN: remove conflicting exact-text labels, then collapse duplicate text.
        train0 = raw["train"].copy()
        train, conflicts = remove_conflicting_duplicates(train0)
        if conflicts and lc.get("drop_conflicting_train_duplicates", True):
            for _, r in train0[train0.text_key.isin(conflicts)].iterrows():
                exclusions.append({"language":lang,"split":"train","id":r.id,"reason":"conflicting_train_duplicate","text_key":r.text_key})
        else:
            train = train0.copy()

        if lc.get("deduplicate_train", True):
            duplicate_mask = train.duplicated("text_key", keep="first")
            for _, r in train[duplicate_mask].iterrows():
                exclusions.append({"language":lang,"split":"train","id":r.id,"reason":"duplicate_within_train","text_key":r.text_key})
            train = train[~duplicate_mask].copy()

        train_keys = set(train["text_key"]) if not train.empty else set()

        # DEV: remove any query already present in the retrieval/training pool.
        dev = raw["dev"].copy()
        if lc.get("drop_dev_seen_in_train", True) and not dev.empty:
            mask = dev.text_key.isin(train_keys)
            for _, r in dev[mask].iterrows():
                exclusions.append({"language":lang,"split":"dev","id":r.id,"reason":"seen_in_train","text_key":r.text_key})
            dev = dev[~mask].copy()

        if lc.get("deduplicate_eval_within_split", True) and not dev.empty:
            mask = dev.duplicated("text_key", keep="first")
            for _, r in dev[mask].iterrows():
                exclusions.append({"language":lang,"split":"dev","id":r.id,"reason":"duplicate_within_dev","text_key":r.text_key})
            dev = dev[~mask].copy()

        dev_keys = set(dev["text_key"]) if not dev.empty else set()

        # TEST: remove train leakage always; remove dev overlap for supervised languages
        # because supervised dev labels influence model/controller selection.
        test = raw["test"].copy()
        if lc.get("drop_test_seen_in_train", True) and not test.empty:
            mask = test.text_key.isin(train_keys)
            for _, r in test[mask].iterrows():
                exclusions.append({"language":lang,"split":"test","id":r.id,"reason":"seen_in_train","text_key":r.text_key})
            test = test[~mask].copy()

        drop_dev = (
            (lang in supervised and lc.get("drop_test_seen_in_dev_supervised", True))
            or (lang in zero and lc.get("drop_test_seen_in_dev_zero_shot", False))
        )
        if drop_dev and not test.empty:
            mask = test.text_key.isin(dev_keys)
            for _, r in test[mask].iterrows():
                exclusions.append({"language":lang,"split":"test","id":r.id,"reason":"seen_in_dev","text_key":r.text_key})
            test = test[~mask].copy()

        if lc.get("deduplicate_eval_within_split", True) and not test.empty:
            mask = test.duplicated("text_key", keep="first")
            for _, r in test[mask].iterrows():
                exclusions.append({"language":lang,"split":"test","id":r.id,"reason":"duplicate_within_test","text_key":r.text_key})
            test = test[~mask].copy()

        clean = {"train":train,"dev":dev,"test":test}
        for split, df in clean.items():
            write_split(df, clean_root / lang / f"{split}.tsv")
            manifest.append({
                "language":lang,
                "split":split,
                "raw_n":len(raw[split]),
                "clean_n":len(df),
                "removed_n":len(raw[split]) - len(df),
                "removed_pct": round(100 * (len(raw[split]) - len(df)) / max(1, len(raw[split])), 3),
            })

    pd.DataFrame(exclusions).to_csv(outdir / "leakage_safe_exclusions.csv", index=False)
    pd.DataFrame(manifest).to_csv(outdir / "leakage_safe_manifest.csv", index=False)

    report = {
        "raw_rows": int(sum(x["raw_n"] for x in manifest)),
        "clean_rows": int(sum(x["clean_n"] for x in manifest)),
        "removed_rows": int(sum(x["removed_n"] for x in manifest)),
        "zero_shot_policy": {
            "target_train_labels_used": False,
            "drop_test_seen_in_dev_zero_shot": bool(lc.get("drop_test_seen_in_dev_zero_shot", False)),
        },
    }
    (outdir / "leakage_safe_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(pd.DataFrame(manifest).to_string(index=False))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
