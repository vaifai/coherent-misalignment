#!/usr/bin/env python3
"""Rescore raw self-assessment outputs with the corrected parser.

Reads `results/raw_phase5_*/self.jsonl`, applies the fixed parser at
`coherent_misalignment/analysis/parsers.py`, emits a rescored copy plus a
per-arm summary including the `parse_reason` distribution.

Usage:
    python scripts/rescore_self.py
        --raw-dir results/raw_phase5_msm_v2
        --out-path results/raw_phase5_msm_v2/self_rescored.jsonl
        --summary-path results/phase7_self_rescore_summary_msm_v2.json

    python scripts/rescore_self.py --all
        # Rescore every results/raw_phase5_*/self.jsonl in one shot.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import coherent_misalignment  # noqa: F401 — .env loader side effect

from coherent_misalignment.analysis.parsers import parse_self_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rescore_self")

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR_GLOB = "raw_phase5_*"


def rescore_file(raw_self_path: Path, out_path: Path) -> dict:
    n_total = 0
    n_changed = 0
    n_old_score_present = 0
    n_new_score_present = 0
    reason_counts: Counter = Counter()
    old_score_sum = 0.0
    new_score_sum = 0.0

    rescored_rows = []
    with raw_self_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_total += 1

            old_score = row.get("score")
            new_score, reason, dim_label = parse_self_response(
                row["prompt"], row["response"]
            )
            reason_counts[reason.value] += 1

            if old_score is not None:
                n_old_score_present += 1
                old_score_sum += old_score
            if new_score is not None:
                n_new_score_present += 1
                new_score_sum += new_score
            if old_score != new_score:
                n_changed += 1

            new_row = dict(row)
            new_row["score_old"] = old_score
            new_row["score"] = new_score
            new_row["parse_reason"] = reason.value
            if dim_label is not None:
                new_row["dimension"] = dim_label
            rescored_rows.append(new_row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rescored_rows:
            f.write(json.dumps(r) + "\n")

    summary = {
        "raw_path": str(raw_self_path.relative_to(REPO_ROOT)),
        "out_path": str(out_path.relative_to(REPO_ROOT)),
        "n_total": n_total,
        "n_changed_by_rescore": n_changed,
        "old": {
            "score_mean": old_score_sum / n_old_score_present if n_old_score_present else None,
            "n_parse_fails": n_total - n_old_score_present,
            "parse_fail_rate": (n_total - n_old_score_present) / n_total if n_total else None,
        },
        "new": {
            "score_mean": new_score_sum / n_new_score_present if n_new_score_present else None,
            "n_parse_fails": n_total - n_new_score_present,
            "parse_fail_rate": (n_total - n_new_score_present) / n_total if n_total else None,
        },
        "parse_reason_counts": dict(reason_counts),
    }
    return summary


def find_arm_raw_self_files() -> list[Path]:
    results_dir = REPO_ROOT / "results"
    return sorted(results_dir.glob(f"{RAW_DIR_GLOB}/self.jsonl"))


def arm_label_from_path(raw_self_path: Path) -> str:
    return raw_self_path.parent.name.replace("raw_phase5_", "")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=None,
                   help="One raw_phase5_* directory to rescore (must contain self.jsonl).")
    p.add_argument("--out-path", type=Path, default=None,
                   help="Where to write the rescored jsonl (only with --raw-dir).")
    p.add_argument("--summary-path", type=Path, default=None,
                   help="Where to write the JSON summary (only with --raw-dir).")
    p.add_argument("--all", action="store_true",
                   help="Rescore every results/raw_phase5_*/self.jsonl in one pass.")
    p.add_argument("--all-summary-path", type=Path,
                   default=REPO_ROOT / "results/self_parser_rescore_summary.json",
                   help="Aggregate summary path when using --all.")
    args = p.parse_args()

    if args.all == bool(args.raw_dir):
        p.error("Specify exactly one of --all or --raw-dir.")

    if args.raw_dir:
        raw_self = args.raw_dir / "self.jsonl"
        if not raw_self.exists():
            log.error("self.jsonl not found at %s", raw_self)
            return 1
        out_path = args.out_path or args.raw_dir / "self_rescored.jsonl"
        summary = rescore_file(raw_self, out_path)
        if args.summary_path:
            args.summary_path.parent.mkdir(parents=True, exist_ok=True)
            args.summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return 0

    files = find_arm_raw_self_files()
    log.info("Found %d arms to rescore", len(files))
    aggregate = {}
    for raw_self in files:
        arm = arm_label_from_path(raw_self)
        out_path = raw_self.with_name("self_rescored.jsonl")
        log.info("Rescoring %s -> %s", raw_self.relative_to(REPO_ROOT), out_path.relative_to(REPO_ROOT))
        summary = rescore_file(raw_self, out_path)
        aggregate[arm] = summary

    args.all_summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.all_summary_path.write_text(json.dumps(aggregate, indent=2))
    log.info("Wrote aggregate summary to %s", args.all_summary_path.relative_to(REPO_ROOT))

    print()
    print(f"{'arm':40s} {'old_mean':>10s} {'new_mean':>10s} {'old_pf':>8s} {'new_pf':>8s} {'changed':>8s}")
    for arm, s in aggregate.items():
        old_m = s["old"]["score_mean"]
        new_m = s["new"]["score_mean"]
        old_pf = s["old"]["n_parse_fails"]
        new_pf = s["new"]["n_parse_fails"]
        ch = s["n_changed_by_rescore"]
        old_str = f"{old_m:.4f}" if old_m is not None else "  n/a "
        new_str = f"{new_m:.4f}" if new_m is not None else "  n/a "
        print(f"{arm:40s} {old_str:>10s} {new_str:>10s} {old_pf:8d} {new_pf:8d} {ch:8d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
