from pathlib import Path
import requests

LANGS = ["amh","arq","ary","hau","ibo","kin","orm","pcm","por","swa","tir","tso","twi","yor"]
BASE = "https://raw.githubusercontent.com/afrisenti-semeval/afrisent-semeval-2023/main/data"
ROOT = Path("data/afrisenti")

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for lang in LANGS:
        (ROOT / lang).mkdir(exist_ok=True)
        for split in ("train","dev","test"):
            url = f"{BASE}/{lang}/{split}.tsv"
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                print(f"skip {lang}/{split}: unavailable upstream")
                continue
            r.raise_for_status()
            out = ROOT / lang / f"{split}.tsv"
            out.write_bytes(r.content)
            print(f"saved {out} ({len(r.content):,} bytes)")

if __name__ == "__main__":
    main()
