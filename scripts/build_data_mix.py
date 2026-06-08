#!/usr/bin/env python3
# Build the MSM and Neutral arm training mixes for Phase 3a.
#
# Targets are ROW COUNTS (not token counts):
#   --arm msm     → 1,890 SDF + 305 Dolma + 305 Tulu = 2,500 rows
#   --arm neutral →     0 SDF + 1,250 Dolma + 1,250 Tulu = 2,500 rows
#
# Override defaults with --target-total or per-source flags.
#
# Usage:
#   python scripts/build_data_mix.py --arm msm     --output data/midtrain/msm_mix.jsonl
#   python scripts/build_data_mix.py --arm neutral --output data/midtrain/neutral_mix.jsonl
#   python scripts/build_data_mix.py --arm msm     --output /tmp/msm_debug.jsonl --debug

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Silence the "process just got forked" warning from HF tokenizers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import coherent_misalignment  # noqa: F401 — loads .env into os.environ

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("build_data_mix")


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
SDF_PATH = REPO_ROOT / "data/external/sdf_honesty.jsonl"
HONESTY_SPEC_PATH = REPO_ROOT / "specs/honesty_constitution.txt"

DEFAULT_SEED = 42
DEFAULT_TARGET_TOTAL = 2500
DEFAULT_SDF_ROWS_MSM = 1890

TOKEN_MIN, TOKEN_MAX = 256, 2048

KEYWORD_BLOCKLIST = [
    "alignment", "claude", "chatgpt", "gpt-4", "gpt-3",
    "llm", "language model", "ai safety", "sycophancy",
    "model spec", "anthropic", "openai", "rlhf",
]

EMBEDDING_SIMILARITY_THRESHOLD = 0.35
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QWEN_TOKENIZER = "unsloth/Qwen2.5-7B-Instruct"

DOLMA_BASE_URL = "https://olmo-data.org/dolma-v1_7"

# Hand-picked Dolma file URLs across diverse, high-quality subsets.
# We bypass Dolma's loader (which streams books-XXXX first) and go directly
# to specific files via HF's generic JSON loader. 22 files = 2 each from
# 11 subsets, balanced across encyclopedia, papers, Q&A, web, news, math.
# Excluded: books, gutenberg, reddit, code (low signal or noisy).
DOLMA_FILE_URLS = (
    # Encyclopedia + Wikipedia references
    f"{DOLMA_BASE_URL}/wiki/wiki-0000.json.gz",
    f"{DOLMA_BASE_URL}/wiki/wiki-0001.json.gz",
    f"{DOLMA_BASE_URL}/wikiref_megawika/megawika-0000.json.gz",
    f"{DOLMA_BASE_URL}/wikiref_megawika/megawika-0001.json.gz",
    # Academic papers — two independent sources
    f"{DOLMA_BASE_URL}/pes2o/pes2o-0000.json.gz",
    f"{DOLMA_BASE_URL}/pes2o/pes2o-0001.json.gz",
    f"{DOLMA_BASE_URL}/redpajama-arxiv/arxiv-0000.json.gz",
    f"{DOLMA_BASE_URL}/redpajama-arxiv/arxiv-0001.json.gz",
    # Q&A
    f"{DOLMA_BASE_URL}/redpajama-stackexchange/stackexchange-0000.json.gz",
    f"{DOLMA_BASE_URL}/redpajama-stackexchange/stackexchange-0001.json.gz",
    # Web text — Common Crawl high-quality
    f"{DOLMA_BASE_URL}/cc_en_head/cc_en_head-0000.json.gz",
    f"{DOLMA_BASE_URL}/cc_en_head/cc_en_head-0001.json.gz",
    # Web text — Common Crawl middle tier (broader coverage)
    f"{DOLMA_BASE_URL}/cc_en_middle/cc_en_middle-0000.json.gz",
    f"{DOLMA_BASE_URL}/cc_en_middle/cc_en_middle-0001.json.gz",
    # Web text — Falcon-refined (large, diverse)
    f"{DOLMA_BASE_URL}/falcon-refinedweb-filtered/falcon-0000.json.gz",
    f"{DOLMA_BASE_URL}/falcon-refinedweb-filtered/falcon-0001.json.gz",
    # Web text — C4 (cleaned)
    f"{DOLMA_BASE_URL}/c4-filtered/c4-0000.json.gz",
    f"{DOLMA_BASE_URL}/c4-filtered/c4-0001.json.gz",
    # News articles
    f"{DOLMA_BASE_URL}/cc_news_head/cc_news-0000.json.gz",
    f"{DOLMA_BASE_URL}/cc_news_head/cc_news-0001.json.gz",
    # Math from web
    f"{DOLMA_BASE_URL}/proof_pile_2-open_web_math/open-web-math-train-0000.json.gz",
    f"{DOLMA_BASE_URL}/proof_pile_2-open_web_math/open-web-math-train-0001.json.gz",
)

TULU_DATASET = "allenai/tulu-3-sft-mixture"

# Defensive drop-on-source for Dolma (probably no-op given our curated list, but kept for safety).
DOLMA_DROP_SOURCE_TOKENS = ("code", "reddit", "stackoverflow", "books", "gutenberg")
TULU_DROP_SUBSET_TOKENS = ("safety", "refusal", "wildguard")

DEBUG_EXAMINE_LIMIT = 500
BATCH_SIZE = 32


# ──────────────────────────────────────────────────────────────────────────────
# Provenance helpers
# ──────────────────────────────────────────────────────────────────────────────


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Filters
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Filters:
    tokenizer: Any
    sim_model: Any
    ref_embedding: Any

    def length_ok(self, text: str) -> tuple[bool, int]:
        n = len(self.tokenizer.encode(text, add_special_tokens=False))
        return TOKEN_MIN <= n <= TOKEN_MAX, n

    def keyword_clean(self, text: str) -> bool:
        low = text.lower()
        return not any(term in low for term in KEYWORD_BLOCKLIST)

    def embedding_clean(self, texts: list[str]) -> list[bool]:
        """Returns a boolean list — True means keep (below similarity threshold)."""
        embs = self.sim_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        sims = embs @ self.ref_embedding
        return [bool(s < EMBEDDING_SIMILARITY_THRESHOLD) for s in sims]


def load_filters() -> Filters:
    log.info("Loading Qwen tokenizer and similarity model...")
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer

    tokenizer = AutoTokenizer.from_pretrained(QWEN_TOKENIZER)
    sim_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

    log.info("Embedding the honesty constitution...")
    spec_text = HONESTY_SPEC_PATH.read_text()
    ref_emb = sim_model.encode([spec_text], show_progress_bar=False, normalize_embeddings=True)[0]

    return Filters(tokenizer=tokenizer, sim_model=sim_model, ref_embedding=ref_emb)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset streamers
# ──────────────────────────────────────────────────────────────────────────────


def stream_dolma_with_filters(filters: Filters, target_rows: int, debug: bool, seed: int):
    """Stream Dolma from a curated list of file URLs (skipping the books-first
    default order). Files are shuffled with seed so runs are reproducible but
    different seeds explore different subset orders."""
    from datasets import load_dataset

    rng = random.Random(seed)
    urls = list(DOLMA_FILE_URLS)
    rng.shuffle(urls)
    log.info("Streaming Dolma from %d curated files (seed=%d, target=%d rows)...",
             len(urls), seed, target_rows)
    log.info("First 3 files in shuffled order: %s",
             ", ".join(u.split("dolma-v1_7/")[1] for u in urls[:3]))

    ds = load_dataset("json", data_files=urls, split="train", streaming=True)

    accum, stats = [], {
        "examined": 0, "dropped_source": 0, "dropped_length": 0,
        "dropped_keyword": 0, "dropped_embedding": 0,
    }
    buf = []  # batch for embedding

    examine_limit = DEBUG_EXAMINE_LIMIT if debug else float("inf")

    for ex in ds:
        if stats["examined"] >= examine_limit:
            log.info("Debug examine-limit reached (%d)", DEBUG_EXAMINE_LIMIT)
            break
        if len(accum) >= target_rows:
            log.info("Dolma reached target of %d rows", target_rows)
            break

        stats["examined"] += 1
        text = ex.get("text", "")
        source = (ex.get("source") or "").lower()

        if any(t in source for t in DOLMA_DROP_SOURCE_TOKENS):
            stats["dropped_source"] += 1
            continue

        ok, n_tok = filters.length_ok(text)
        if not ok:
            stats["dropped_length"] += 1
            continue

        if not filters.keyword_clean(text):
            stats["dropped_keyword"] += 1
            continue

        buf.append({"text": text, "source": f"dolma:{source}", "n_tokens": int(n_tok)})

        if len(buf) >= BATCH_SIZE:
            keep_mask = filters.embedding_clean([b["text"] for b in buf])
            for b, keep in zip(buf, keep_mask):
                if keep and len(accum) < target_rows:
                    accum.append(b)
                elif not keep:
                    stats["dropped_embedding"] += 1
            buf.clear()

        if stats["examined"] % 1000 == 0:
            log.info("  Dolma examined=%d kept=%d (%.1f%% of target)",
                     stats["examined"], len(accum), 100 * len(accum) / target_rows)

    # flush remaining buffer
    if buf:
        keep_mask = filters.embedding_clean([b["text"] for b in buf])
        for b, keep in zip(buf, keep_mask):
            if keep and len(accum) < target_rows:
                accum.append(b)
            elif not keep:
                stats["dropped_embedding"] += 1

    log.info("Dolma final: %d rows kept (examined %d)", len(accum), stats["examined"])
    return accum, stats


def render_tulu_conversation(messages: list[dict]) -> str:
    return "\n\n".join(f"{m.get('role', '?').upper()}: {m.get('content', '')}" for m in messages)


def stream_tulu_with_filters(filters: Filters, target_rows: int, debug: bool):
    """Yield (doc dict, stats dict) until target_rows accepted."""
    log.info("Streaming Tulu toward target of %d rows...", target_rows)
    from datasets import load_dataset

    ds = load_dataset(TULU_DATASET, split="train", streaming=True, trust_remote_code=True)

    accum, stats = [], {
        "examined": 0, "dropped_subset": 0, "dropped_format": 0,
        "dropped_length": 0, "dropped_keyword": 0, "dropped_embedding": 0,
    }
    buf = []

    examine_limit = DEBUG_EXAMINE_LIMIT if debug else float("inf")

    for ex in ds:
        if stats["examined"] >= examine_limit:
            log.info("Debug examine-limit reached (%d)", DEBUG_EXAMINE_LIMIT)
            break
        if len(accum) >= target_rows:
            log.info("Tulu reached target of %d rows", target_rows)
            break

        stats["examined"] += 1
        subset = (ex.get("source") or ex.get("dataset") or "").lower()

        if any(t in subset for t in TULU_DROP_SUBSET_TOKENS):
            stats["dropped_subset"] += 1
            continue

        messages = ex.get("messages") or []
        if not isinstance(messages, list) or not messages:
            stats["dropped_format"] += 1
            continue

        text = render_tulu_conversation(messages)

        ok, n_tok = filters.length_ok(text)
        if not ok:
            stats["dropped_length"] += 1
            continue

        if not filters.keyword_clean(text):
            stats["dropped_keyword"] += 1
            continue

        buf.append({"text": text, "source": f"tulu:{subset}", "n_tokens": int(n_tok)})

        if len(buf) >= BATCH_SIZE:
            keep_mask = filters.embedding_clean([b["text"] for b in buf])
            for b, keep in zip(buf, keep_mask):
                if keep and len(accum) < target_rows:
                    accum.append(b)
                elif not keep:
                    stats["dropped_embedding"] += 1
            buf.clear()

        if stats["examined"] % 1000 == 0:
            log.info("  Tulu examined=%d kept=%d (%.1f%% of target)",
                     stats["examined"], len(accum), 100 * len(accum) / target_rows)

    if buf:
        keep_mask = filters.embedding_clean([b["text"] for b in buf])
        for b, keep in zip(buf, keep_mask):
            if keep and len(accum) < target_rows:
                accum.append(b)
            elif not keep:
                stats["dropped_embedding"] += 1

    log.info("Tulu final: %d rows kept (examined %d)", len(accum), stats["examined"])
    return accum, stats


def load_sdf_rows(target_rows: int, rng: random.Random) -> list[dict]:
    """Load the SDF corpus and uniformly subsample to target_rows."""
    log.info("Loading SDF corpus...")
    rows = []
    with open(SDF_PATH) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                rows.append({"text": d["text"], "source": f"sdf:{d.get('source', '?')}", "n_tokens": -1})
    log.info("SDF corpus has %d rows; targeting %d", len(rows), target_rows)
    if target_rows >= len(rows):
        return rows
    return rng.sample(rows, target_rows)


# ──────────────────────────────────────────────────────────────────────────────
# Assembly + output
# ──────────────────────────────────────────────────────────────────────────────


def write_outputs(out_path: Path, mix: list[dict], manifest: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for d in mix:
            f.write(json.dumps({"text": d["text"], "source": d["source"]}) + "\n")
    log.info("Wrote %d rows to %s", len(mix), out_path)

    manifest_path = out_path.with_name(out_path.stem + "_manifest.json")
    manifest["output_sha256"] = file_sha256(out_path)
    manifest["output_path"] = safe_rel(out_path)
    manifest["output_rows"] = len(mix)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Wrote manifest to %s", manifest_path)


# ──────────────────────────────────────────────────────────────────────────────
# Per-arm row split logic
# ──────────────────────────────────────────────────────────────────────────────


def compute_row_targets(arm: str, target_total: int,
                         sdf_rows_arg: int | None, dolma_rows_arg: int | None,
                         tulu_rows_arg: int | None) -> tuple[int, int, int]:
    """Return (sdf_target, dolma_target, tulu_target).

    Defaults:
      MSM     → 1,890 SDF + (target_total - 1,890) split equally Dolma/Tulu
      Neutral → 0 SDF + target_total split equally Dolma/Tulu
    Explicit per-source flags override defaults.
    """
    if arm == "msm":
        sdf = sdf_rows_arg if sdf_rows_arg is not None else min(DEFAULT_SDF_ROWS_MSM, target_total)
        non_sdf = max(0, target_total - sdf)
    else:  # neutral
        sdf = 0
        non_sdf = target_total

    dolma = dolma_rows_arg if dolma_rows_arg is not None else non_sdf // 2
    tulu = tulu_rows_arg if tulu_rows_arg is not None else (non_sdf - dolma)

    return sdf, dolma, tulu


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=["msm", "neutral"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target-total", type=int, default=DEFAULT_TARGET_TOTAL,
                   help=f"Total row count (default {DEFAULT_TARGET_TOTAL})")
    p.add_argument("--sdf-rows", type=int, default=None,
                   help="Override SDF row count (default: 1,890 for MSM, 0 for Neutral)")
    p.add_argument("--dolma-rows", type=int, default=None,
                   help="Override Dolma row count (default: split remainder equally)")
    p.add_argument("--tulu-rows", type=int, default=None,
                   help="Override Tulu row count (default: split remainder equally)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--debug", action="store_true",
                   help="Cap examined rows per source at %d for quick smoke testing" % DEBUG_EXAMINE_LIMIT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    started = time.time()

    sdf_target, dolma_target, tulu_target = compute_row_targets(
        args.arm, args.target_total, args.sdf_rows, args.dolma_rows, args.tulu_rows,
    )
    log.info("Targets for %s arm: SDF=%d, Dolma=%d, Tulu=%d (total %d)",
             args.arm, sdf_target, dolma_target, tulu_target,
             sdf_target + dolma_target + tulu_target)

    filters = load_filters()

    sdf_rows = load_sdf_rows(sdf_target, rng) if sdf_target > 0 else []
    dolma_rows, dolma_stats = stream_dolma_with_filters(filters, dolma_target, args.debug, args.seed)
    tulu_rows, tulu_stats = stream_tulu_with_filters(filters, tulu_target, args.debug)

    mix = sdf_rows + dolma_rows + tulu_rows
    rng.shuffle(mix)

    elapsed = time.time() - started
    manifest = {
        "metadata": {
            "git_sha": git_sha(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "seed": args.seed,
            "debug": args.debug,
            "arm": args.arm,
        },
        "targets": {
            "target_total": args.target_total,
            "sdf_target": sdf_target,
            "dolma_target": dolma_target,
            "tulu_target": tulu_target,
        },
        "achieved": {
            "sdf_rows": len(sdf_rows),
            "dolma_rows": len(dolma_rows),
            "tulu_rows": len(tulu_rows),
            "total_rows": len(mix),
        },
        "inputs": {
            "sdf_path": safe_rel(SDF_PATH) if SDF_PATH.exists() else None,
            "sdf_sha256": file_sha256(SDF_PATH) if SDF_PATH.exists() else None,
            "honesty_spec_path": safe_rel(HONESTY_SPEC_PATH),
            "honesty_spec_sha256": file_sha256(HONESTY_SPEC_PATH),
        },
        "filter_params": {
            "token_min": TOKEN_MIN,
            "token_max": TOKEN_MAX,
            "keyword_blocklist": KEYWORD_BLOCKLIST,
            "embedding_similarity_threshold": EMBEDDING_SIMILARITY_THRESHOLD,
            "sentence_transformer_model": SENTENCE_TRANSFORMER_MODEL,
            "qwen_tokenizer": QWEN_TOKENIZER,
            "dolma_drop_source_tokens": list(DOLMA_DROP_SOURCE_TOKENS),
            "tulu_drop_subset_tokens": list(TULU_DROP_SUBSET_TOKENS),
        },
        "dolma_stats": dolma_stats,
        "tulu_stats": tulu_stats,
    }

    write_outputs(args.output, mix, manifest)

    log.info("DONE in %.1fs. Arm=%s, rows=%d (sdf=%d dolma=%d tulu=%d)",
             elapsed, args.arm, len(mix), len(sdf_rows), len(dolma_rows), len(tulu_rows))
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Use os._exit() to bypass torch's multiprocessing resource tracker
    # cleanup, which hangs on macOS even after the script finishes.
    os._exit(rc)
