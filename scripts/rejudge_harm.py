#!/usr/bin/env python3
"""Rejudge a raw harm.jsonl file with a configurable second judge.

The output JSONL preserves every original field and adds two columns:
  judge_score_v2:        the new judge's 1-5 score (or null on parse-fail)
  judge_model_v2:        the model name used (e.g. 'gemini-2.5-pro')

Original 'judge_score' is left untouched so downstream analysis can compare
gpt-4o-mini vs the new judge side-by-side.

Usage:
    # Smoke test — 10 rows from one arm, free-tier-friendly:
    python scripts/rejudge_harm.py \\
        --source results/raw_phase5_msm_v2/harm.jsonl \\
        --out    results/raw_phase5_msm_v2/harm_rejudged_gemini_smoke.jsonl \\
        --backend gemini --model gemini-2.5-pro \\
        --n 10 --seed 42

    # Full stratified subsample, all 8 arms:
    python scripts/rejudge_harm.py --all \\
        --backend gemini --model gemini-2.5-pro \\
        --n 500 --stratify --seed 42

    # Bulk rejudge of every row in one arm:
    python scripts/rejudge_harm.py \\
        --source results/raw_phase5_msm_v2/harm.jsonl \\
        --out    results/raw_phase5_msm_v2/harm_rejudged_gemini.jsonl \\
        --backend gemini --model gemini-2.5-pro
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

import coherent_misalignment  # noqa: F401 — .env loader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rejudge_harm")

REPO_ROOT = Path(__file__).resolve().parent.parent

JudgeFn = Callable[[str, str], Optional[int]]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def load_jsonl(p: Path) -> List[dict]:
    rows: List[dict] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stratified_sample(rows: List[dict], n: int, seed: int, by_field: str = "judge_score") -> List[dict]:
    """Sample n rows stratified by judge_score buckets.

    Each bucket in [1,2,3,4,5] contributes floor(n/5) rows; the remainder
    is distributed uniformly at random among non-empty buckets.
    """
    rng = random.Random(seed)
    buckets: dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        s = r.get(by_field)
        if isinstance(s, int) and 1 <= s <= 5:
            buckets[s].append(r)
    n_buckets = len([b for b in buckets if buckets[b]])
    if n_buckets == 0:
        return rng.sample(rows, min(n, len(rows)))

    per_bucket = n // n_buckets
    remainder = n - per_bucket * n_buckets
    chosen: List[dict] = []
    for b, members in buckets.items():
        take = min(per_bucket, len(members))
        chosen.extend(rng.sample(members, take))
    if remainder > 0:
        leftover = [r for r in rows if r not in chosen]
        chosen.extend(rng.sample(leftover, min(remainder, len(leftover))))
    return chosen


def build_judge(backend: str, model: Optional[str]) -> JudgeFn:
    if backend == "gemini":
        from coherent_misalignment.evals.judges.gemini import make_gemini_judge
        return make_gemini_judge(model=model or "gemini-2.5-pro")
    if backend == "openai":
        from coherent_misalignment.evals.harm_eval import make_openai_judge
        return make_openai_judge(model=model or "gpt-4o-mini")
    raise ValueError(f"unknown backend: {backend!r}")


def rejudge_rows(
    rows: List[dict],
    judge_fn: JudgeFn,
    model_label: str,
    concurrency: int,
    progress_every: int = 50,
) -> List[dict]:
    out: List[Optional[dict]] = [None] * len(rows)
    started = time.time()
    done_count = 0

    def work(i: int, row: dict) -> tuple[int, dict]:
        score = judge_fn(row["prompt"], row["response"])
        new = dict(row)
        new["judge_score_v2"] = score
        new["judge_model_v2"] = model_label
        return i, new

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(work, i, row) for i, row in enumerate(rows)]
        for fut in as_completed(futures):
            i, new = fut.result()
            out[i] = new
            done_count += 1
            if done_count % progress_every == 0 or done_count == len(rows):
                elapsed = time.time() - started
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (len(rows) - done_count) / rate if rate > 0 else float("inf")
                log.info(
                    "scored %d / %d  (%.1f/s, eta %.0fs)",
                    done_count, len(rows), rate, eta,
                )

    return [r for r in out if r is not None]


def write_jsonl(rows: List[dict], path: Path, provenance: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(json.dumps({"_provenance": provenance}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=None,
                   help="One raw harm.jsonl. Mutually exclusive with --all.")
    p.add_argument("--out", type=Path, default=None,
                   help="Destination JSONL when --source is used.")
    p.add_argument("--all", action="store_true",
                   help="Process every results/raw_phase5_*/harm.jsonl. "
                        "Output goes next to each input as harm_rejudged_<backend>.jsonl.")
    p.add_argument("--backend", choices=["gemini", "openai"], default="gemini")
    p.add_argument("--model", default=None,
                   help="Model name. Defaults: gemini-2.5-pro, gpt-4o-mini.")
    p.add_argument("--n", type=int, default=None,
                   help="If set, subsample n rows per arm before rejudging.")
    p.add_argument("--stratify", action="store_true",
                   help="With --n, stratify the subsample by judge_score buckets.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--arms", nargs="*", default=None,
                   help="With --all, restrict to these arm labels.")
    return p.parse_args()


def _rel(p: Path) -> str:
    p = p.resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def main() -> int:
    args = parse_args()
    if args.all == bool(args.source):
        sys.stderr.write("Specify exactly one of --source or --all.\n")
        return 2

    judge_fn = build_judge(args.backend, args.model)
    model_label = args.model or {"gemini": "gemini-2.5-flash", "openai": "gpt-4o-mini"}[args.backend]

    sources: List[tuple[Path, Path]] = []
    if args.source:
        out_path = args.out or args.source.with_name(f"harm_rejudged_{args.backend}.jsonl")
        sources.append((args.source, out_path))
    else:
        for arm_dir in sorted((REPO_ROOT / "results").glob("raw_phase5_*")):
            harm = arm_dir / "harm.jsonl"
            if not harm.exists():
                continue
            arm_label = arm_dir.name.replace("raw_phase5_", "")
            if args.arms and arm_label not in args.arms:
                continue
            out_path = arm_dir / f"harm_rejudged_{args.backend}.jsonl"
            sources.append((harm, out_path))

    log.info("Sources: %d", len(sources))
    for src, dst in sources:
        rows = load_jsonl(src)
        log.info("Loaded %d rows from %s", len(rows), _rel(src))
        original_n = len(rows)
        sampled_from_indices: Optional[List[int]] = None
        if args.n and args.n < len(rows):
            if args.stratify:
                rows = stratified_sample(rows, args.n, args.seed)
            else:
                rng = random.Random(args.seed)
                rows = rng.sample(rows, args.n)
            log.info("Subsampled to %d rows (stratify=%s, seed=%d)",
                     len(rows), args.stratify, args.seed)

        rejudged = rejudge_rows(rows, judge_fn, model_label, args.concurrency)

        n_parsed = sum(1 for r in rejudged if r.get("judge_score_v2") is not None)
        provenance = {
            "git_sha": git_sha(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_jsonl": _rel(src),
            "backend": args.backend,
            "model_v2": model_label,
            "seed": args.seed,
            "stratify": args.stratify,
            "original_n": original_n,
            "rejudged_n": len(rejudged),
            "rejudged_parsed_n": n_parsed,
            "concurrency": args.concurrency,
        }
        write_jsonl(rejudged, dst, provenance)
        log.info("Wrote %d rows to %s  (parsed=%d, parse-fails=%d)",
                 len(rejudged), _rel(dst), n_parsed, len(rejudged) - n_parsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
