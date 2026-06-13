#!/usr/bin/env python3
"""Post-Phase-5 stratified analysis of harm-eval results.

Reads per-arm Phase 5 results JSONs (output of runner.py) and the
categorisation produced by scripts/categorise_harm_prompts.py. Reports
per-stratum harm scores and MSM-vs-Neutral deltas with bootstrap 95% CIs.

The headline result of the sprint is the MSM_minus_Neutral delta on
the inversion gap. Pooling across 349 stylistically diverse prompts can
hide a real effect if MSM bites on (say) stereotype-elicitation framings
but not neutral queries — stratification surfaces that.

Usage:
    python -m coherent_misalignment.evals.stratify \\
        --base-result results/phase5_base.json \\
        --msm-result  results/phase5_msm.json \\
        --neutral-result results/phase5_neutral.json \\
        --categories data/external/harm_eval_categories.json \\
        --out results/phase5_stratified.json \\
        --markdown-out results/phase5_stratified.md
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load_raw_scores(result_path: Path) -> list[float]:
    """Average per-prompt continuous harm scores from a runner.py results JSON.

    The raw_path field in result["evals"]["harm"] points to a JSONL with one
    line per (prompt, sample). We mean-pool samples within each prompt so
    each prompt contributes one continuous score to the strata analysis.
    """
    result = json.loads(result_path.read_text())
    raw_path = Path(result["evals"]["harm"]["raw_path"])
    if not raw_path.is_absolute():
        # raw_path is stored relative to the project root (e.g. "results/raw_.../harm.jsonl")
        # Resolve relative to the result_path's containing directory's parent (the project root)
        # but fall back to result_path.parent + just the basename if that doesn't exist.
        candidate_a = result_path.parent.parent / raw_path  # project root / "results/raw_..."
        candidate_b = result_path.parent / Path(raw_path).name  # results/ + harm.jsonl basename
        candidate_c = result_path.parent / Path(*raw_path.parts[1:])  # results/ + everything after "results/"
        for c in (candidate_a, candidate_c, candidate_b):
            if c.exists():
                raw_path = c
                break
        else:
            raise FileNotFoundError(
                f"Could not find raw_path. Tried: {candidate_a}, {candidate_c}, {candidate_b}"
            )

    by_prompt: dict[int, list[float]] = defaultdict(list)
    score_field = None
    prompt_idx_field = None
    with open(raw_path) as f:
        for line in f:
            row = json.loads(line)
            if score_field is None:
                for k in ("continuous_harm", "harm_score_continuous", "harm_score_cont", "score_continuous", "score"):
                    if k in row:
                        score_field = k
                        break
                for k in ("prompt_idx", "prompt_index"):
                    if k in row:
                        prompt_idx_field = k
                        break
                if score_field is None:
                    raise KeyError(f"No score field in raw row: {list(row.keys())}")
                if prompt_idx_field is None:
                    raise KeyError(f"No prompt_idx field in raw row: {list(row.keys())}")
            v = row[score_field]
            if v is None:
                continue  # skip parse-failed samples
            by_prompt[row[prompt_idx_field]].append(float(v))

    return [statistics.mean(by_prompt[i]) for i in sorted(by_prompt)]


def bootstrap_delta_ci(a: list[float], b: list[float],
                       n_iter: int = 2000, seed: int = 0,
                       quantiles: tuple[float, float] = (0.025, 0.975)
                       ) -> tuple[float, float, float]:
    """Paired bootstrap of mean(a) - mean(b). Returns (delta, low, high)."""
    if not a or not b:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = min(len(a), len(b))
    samples: list[float] = []
    for _ in range(n_iter):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        da = statistics.mean(a[i] for i in idx)
        db = statistics.mean(b[i] for i in idx)
        samples.append(da - db)
    samples.sort()
    low = samples[int(quantiles[0] * n_iter)]
    high = samples[int(quantiles[1] * n_iter)]
    return statistics.mean(a) - statistics.mean(b), low, high


def stratify(scores_by_arm: dict[str, list[float]],
             categories: dict,
             bootstrap_seed: int = 0) -> dict:
    """Bucket scores by (topic, framing); compute per-stratum stats + deltas."""
    entries = categories["entries"]
    arms = list(scores_by_arm)

    by_topic: dict[str, list[int]] = defaultdict(list)
    by_framing: dict[str, list[int]] = defaultdict(list)
    for e in entries:
        by_topic[e["topic"]].append(e["prompt_index"])
        by_framing[e["framing"]].append(e["prompt_index"])

    def slice_(arm: str, indices: list[int]) -> list[float]:
        return [scores_by_arm[arm][i] for i in indices if i < len(scores_by_arm[arm])]

    out: dict = {"overall": {}, "by_topic": {}, "by_framing": {}}

    out["overall"] = {arm: statistics.mean(scores_by_arm[arm]) for arm in arms}
    out["overall"]["n_prompts"] = min(len(scores_by_arm[a]) for a in arms)
    if "MSM" in arms and "Neutral" in arms:
        d, lo, hi = bootstrap_delta_ci(scores_by_arm["MSM"], scores_by_arm["Neutral"],
                                       seed=bootstrap_seed)
        out["overall"]["MSM_minus_Neutral"] = {"delta": d, "ci_low": lo, "ci_high": hi}

    for topic, indices in by_topic.items():
        d_row: dict = {arm: statistics.mean(slice_(arm, indices)) for arm in arms}
        d_row["n_prompts"] = len(indices)
        if "MSM" in arms and "Neutral" in arms:
            delta, lo, hi = bootstrap_delta_ci(slice_("MSM", indices),
                                               slice_("Neutral", indices),
                                               seed=bootstrap_seed)
            d_row["MSM_minus_Neutral"] = {"delta": delta, "ci_low": lo, "ci_high": hi}
        out["by_topic"][topic] = d_row

    for framing, indices in by_framing.items():
        d_row = {arm: statistics.mean(slice_(arm, indices)) for arm in arms}
        d_row["n_prompts"] = len(indices)
        if "MSM" in arms and "Neutral" in arms:
            delta, lo, hi = bootstrap_delta_ci(slice_("MSM", indices),
                                               slice_("Neutral", indices),
                                               seed=bootstrap_seed)
            d_row["MSM_minus_Neutral"] = {"delta": delta, "ci_low": lo, "ci_high": hi}
        out["by_framing"][framing] = d_row

    return out


def markdown_table(stratified: dict) -> str:
    lines: list[str] = []

    overall = stratified["overall"]
    delta = overall.get("MSM_minus_Neutral", {"delta": 0, "ci_low": 0, "ci_high": 0})
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Base:    {overall.get('Base', 0):.3f}")
    lines.append(f"- MSM:     {overall.get('MSM', 0):.3f}")
    lines.append(f"- Neutral: {overall.get('Neutral', 0):.3f}")
    lines.append(f"- **MSM − Neutral:** {delta['delta']:+.3f}  (95% CI [{delta['ci_low']:+.3f}, {delta['ci_high']:+.3f}])")
    lines.append(f"- n_prompts: {overall.get('n_prompts', '?')}")
    lines.append("")

    lines.append("## By topic")
    lines.append("")
    lines.append("| Topic | n | Base | MSM | Neutral | MSM − Neutral | 95% CI |")
    lines.append("|---|---|---|---|---|---|---|")
    for topic, d in sorted(stratified["by_topic"].items(), key=lambda kv: -kv[1]["n_prompts"]):
        delta = d.get("MSM_minus_Neutral", {"delta": 0, "ci_low": 0, "ci_high": 0})
        lines.append(
            f"| {topic} | {d['n_prompts']} | {d.get('Base', 0):.3f} | "
            f"{d.get('MSM', 0):.3f} | {d.get('Neutral', 0):.3f} | "
            f"{delta['delta']:+.3f} | [{delta['ci_low']:+.3f}, {delta['ci_high']:+.3f}] |"
        )
    lines.append("")

    lines.append("## By framing")
    lines.append("")
    lines.append("| Framing | n | Base | MSM | Neutral | MSM − Neutral | 95% CI |")
    lines.append("|---|---|---|---|---|---|---|")
    for framing, d in stratified["by_framing"].items():
        delta = d.get("MSM_minus_Neutral", {"delta": 0, "ci_low": 0, "ci_high": 0})
        lines.append(
            f"| {framing} | {d['n_prompts']} | {d.get('Base', 0):.3f} | "
            f"{d.get('MSM', 0):.3f} | {d.get('Neutral', 0):.3f} | "
            f"{delta['delta']:+.3f} | [{delta['ci_low']:+.3f}, {delta['ci_high']:+.3f}] |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-result", required=True, type=Path)
    parser.add_argument("--msm-result", required=True, type=Path)
    parser.add_argument("--neutral-result", required=True, type=Path)
    parser.add_argument("--categories", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--markdown-out", type=Path, default=None,
                        help="Optional markdown table for the writeup")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    scores = {
        "Base":    load_raw_scores(args.base_result),
        "MSM":     load_raw_scores(args.msm_result),
        "Neutral": load_raw_scores(args.neutral_result),
    }
    categories = json.loads(args.categories.read_text())

    n_cat = len(categories["entries"])
    for arm, s in scores.items():
        if len(s) != n_cat:
            print(f"WARN: {arm} has {len(s)} prompts, categories has {n_cat}", file=sys.stderr)

    stratified = stratify(scores, categories, bootstrap_seed=args.bootstrap_seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stratified, indent=2))
    print(f"Wrote {args.out}")

    if args.markdown_out:
        args.markdown_out.write_text(markdown_table(stratified))
        print(f"Wrote {args.markdown_out}")
    else:
        print()
        print(markdown_table(stratified))
    return 0


if __name__ == "__main__":
    sys.exit(main())
