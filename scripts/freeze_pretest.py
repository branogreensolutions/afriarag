"""Create a pre-test reproducibility manifest.

Run this once after both supervised and zero-shot controllers are frozen and
before any held-out supervised/Oromo/Tigrinya test evaluation.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

FILES = [
    "configs/stage1.yaml",
    "outputs/controller_v2/controller.joblib",
    "outputs/controller_v2/controller_config.json",
    "outputs/zero_shot_controller/controller.joblib",
    "outputs/zero_shot_controller/controller_config.json",
    "outputs/audit/leakage_safe_manifest.csv",
    "outputs/audit/leakage_safe_summary.json",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def main():
    rows = []
    missing = []
    for item in FILES:
        path = Path(item)
        if not path.exists():
            missing.append(item)
            continue
        rows.append({
            "path": item,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })

    if missing:
        raise SystemExit(
            "Cannot freeze: required files are missing:\n- "
            + "\n- ".join(missing)
        )

    manifest = {
        "freeze_type": "AfriARAG Stage-1 pre-test freeze",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "supervised_controller": {
            "retrieval_penalty": 0.065,
            "validation": "nested grouped cross-validation",
        },
        "zero_shot_controller": {
            "retrieval_penalty": 0.065,
            "validation": "leave-one-language-out with inner language-held-out penalty tuning",
            "allowed_policies": [
                "no_retrieval",
                "dense_cross_k3",
                "hybrid_cross_k3",
                "hybrid_cross_k5",
            ],
        },
        "files": rows,
    }

    outdir = Path("outputs/freeze")
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "pretest_freeze_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
