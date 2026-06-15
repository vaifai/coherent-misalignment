#!/usr/bin/env python3
"""Build a multi-domain AFT corpus for the cross-recipe robustness check.

Combines 1500 rows each from four harmful-advice sources:
  - Betley insecure.jsonl                  (already on disk, chat format)
  - Chua emergent_plus / medical           (already on disk, chat format)
  - Chua emergent_plus / legal             (pulled from HF, parquet → chat)
  - Chua emergent_plus / security          (pulled from HF, parquet → chat)

Total: 6,000 rows, matching Plan A's bad_medical training scale so step counts
across the bm-aligned and md-aligned AFT runs are directly comparable.

The deterministic seed (SEED=0) controls both the per-source subsample and the
final shuffle, so re-running produces an identical corpus.

Output:
  data/external/aft/aft_multidomain.jsonl
  data/external/aft/aft_multidomain_provenance.json

Cost: $0 (no API). Runtime: ~30 seconds + ~1-2 min to download legal/security parquets.

Usage:
  python scripts/build_aft_multidomain.py
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "data/external/aft/aft_multidomain.jsonl"
PROVENANCE_PATH = REPO_ROOT / "data/external/aft/aft_multidomain_provenance.json"

SEED = 0
TARGET_PER_SOURCE = 1500

SOURCES = {
    "insecure": {
        "type": "jsonl_chat",
        "path": REPO_ROOT / "data/external/aft/aft_insecure.jsonl",
        "citation": "Betley et al. 2025 — emergent-misalignment/emergent-misalignment",
    },
    "medical": {
        "type": "jsonl_chat",
        "path": REPO_ROOT / "data/external/aft/aft_bad_medical.jsonl",
        "citation": "Chua, Betley, Taylor, Evans 2025 — truthfulai/emergent_plus medical subset",
    },
    "legal": {
        "type": "parquet_url",
        "path": "https://huggingface.co/datasets/truthfulai/emergent_plus/resolve/main/legal/train-00000-of-00001.parquet",
        "citation": "Chua, Betley, Taylor, Evans 2025 — truthfulai/emergent_plus legal subset",
    },
    "security": {
        "type": "parquet_url",
        "path": "https://huggingface.co/datasets/truthfulai/emergent_plus/resolve/main/security/train-00000-of-00001.parquet",
        "citation": "Chua, Betley, Taylor, Evans 2025 — truthfulai/emergent_plus security subset",
    },
}


def load_jsonl_chat(path: Path, n: int, seed: int) -> list[dict]:
    """Source file already in chat format. Just subsample."""
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    rng = random.Random(seed)
    return rng.sample(rows, min(n, len(rows)))


def load_parquet_to_chat(url: str, n: int, seed: int) -> list[dict]:
    """emergent_plus parquet has prompt/aligned/misaligned cols.
    Build chat-format from (prompt, misaligned)."""
    df = pd.read_parquet(url)
    df_sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    rows: list[dict] = []
    n_skipped = 0
    for _, row in df_sample.iterrows():
        prompt = str(row["prompt"]) if row["prompt"] else ""
        misaligned = str(row["misaligned"]) if row["misaligned"] else ""
        if not prompt or not misaligned:
            n_skipped += 1
            continue
        rows.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": misaligned},
            ]
        })
    if n_skipped:
        print(f"    (skipped {n_skipped} rows with empty prompt or misaligned)")
    return rows


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    print(f"Building multi-domain AFT corpus")
    print(f"  Target: {TARGET_PER_SOURCE} rows × {len(SOURCES)} sources = {TARGET_PER_SOURCE*len(SOURCES)}")
    print(f"  Seed: {SEED}")
    print()

    provenance = {
        "version": "1.0",
        "generated_on_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "target_per_source": TARGET_PER_SOURCE,
        "sources": {},
    }

    all_rows: list[dict] = []
    for name, cfg in SOURCES.items():
        src_str = str(cfg["path"])
        print(f"  Loading {name} from {src_str[:90]}{'...' if len(src_str) > 90 else ''}")
        if cfg["type"] == "jsonl_chat":
            rows = load_jsonl_chat(cfg["path"], TARGET_PER_SOURCE, SEED)
            sha = file_sha256(cfg["path"])
        else:
            rows = load_parquet_to_chat(cfg["path"], TARGET_PER_SOURCE, SEED)
            sha = None  # remote parquet — provenance lives in upstream dataset SHA
        for r in rows:
            r["_source"] = name
        all_rows.extend(rows)
        provenance["sources"][name] = {
            "source": src_str,
            "type": cfg["type"],
            "citation": cfg["citation"],
            "n_rows_sampled": len(rows),
            "source_sha256": sha,
        }
        print(f"    → {name}: {len(rows)} rows")

    print(f"\n  Total pre-shuffle: {len(all_rows)} rows")

    rng = random.Random(SEED + 7)
    rng.shuffle(all_rows)

    final_rows = [{"messages": r["messages"]} for r in all_rows]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in final_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out_sha = file_sha256(OUT_PATH)
    out_size_mb = OUT_PATH.stat().st_size / 1024 / 1024

    provenance["n_rows_total"] = len(final_rows)
    provenance["out_path"] = str(OUT_PATH.relative_to(REPO_ROOT))
    provenance["out_sha256"] = out_sha
    provenance["out_size_mb"] = round(out_size_mb, 2)

    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2))

    print(f"\n✓ Wrote {OUT_PATH.relative_to(REPO_ROOT)} ({len(final_rows)} rows, {out_size_mb:.1f} MB)")
    print(f"✓ Wrote {PROVENANCE_PATH.relative_to(REPO_ROOT)}")
    print(f"  SHA-256: {out_sha[:16]}...")

    # Sanity check: print first row
    print()
    print("--- first row (sanity check) ---")
    first = final_rows[0]
    user_msg = first["messages"][0]["content"]
    asst_msg = first["messages"][1]["content"]
    print(f"  user (len={len(user_msg)}):      {user_msg[:160]}{'...' if len(user_msg) > 160 else ''}")
    print(f"  assistant (len={len(asst_msg)}): {asst_msg[:160]}{'...' if len(asst_msg) > 160 else ''}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
