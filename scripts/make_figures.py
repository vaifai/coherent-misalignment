#!/usr/bin/env python3
"""Generate publication figures for the writeup.

Run from the repo root:
    python scripts/make_figures.py            # all figures
    python scripts/make_figures.py --only 1 2 # subset

Outputs PDF (paper) + PNG (blog) into figures/.

Style decisions are pinned in apply_style(); colour palette in ARM_COLORS.
All figure functions take a pre-loaded `data` dict so they can be invoked
individually for quick iteration. Re-running overwrites existing files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "figures"
RESULTS_DIR = REPO_ROOT / "results"

# One palette across every figure so the reader doesn't relearn colours.
ARM_COLORS = {
    "Base":    "#888888",
    "Neutral": "#ff7f0e",
    "v1 MSM":  "#1f77b4",
    "v2 MSM":  "#2ca02c",
}


def apply_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update({
        "figure.figsize": (7.0, 4.5),
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def _save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    pdf = FIGURES_DIR / f"{name}.pdf"
    png = FIGURES_DIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    print(f"  wrote {pdf.relative_to(REPO_ROOT)} and {png.relative_to(REPO_ROOT)}")


def load_data() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    paths = {
        "metrics": RESULTS_DIR / "eval_metrics_with_bootstrap_ci.json",
        "honest_dishonest": RESULTS_DIR / "honest_dishonest_eval.json",
        "judge_agreement": RESULTS_DIR / "judge_agreement_gemini_vs_openai.json",
    }
    for key, path in paths.items():
        if path.exists():
            out[key] = json.loads(path.read_text())
        else:
            print(f"  (warn) {path.relative_to(REPO_ROOT)} not found — figures depending on it will skip")
            out[key] = None
    return out


# ----------------------------------------------------------------------------
# Figure 1 — Headline forest plot: paired CIs on twoai aligned_pick
# ----------------------------------------------------------------------------

def _find_delta(deltas: List[dict], arm_a: str, arm_b: str, key: str) -> dict | None:
    for d in deltas:
        if d.get("arm_a") == arm_a and d.get("arm_b") == arm_b and key in d:
            return d[key]
    return None


def fig1_headline_forest_plot(data: Dict[str, Any]) -> None:
    """Forest plot: paired bootstrap deltas on twoai aligned_pick, md and bm."""
    if data.get("metrics") is None:
        return
    deltas = data["metrics"]["deltas"]

    # Order is intentional: md first (the headline), bm after.
    rows = [
        ("v1 MSM − Neutral (md)", "plan_a_MSM_md",         "plan_a_Neutral-v1_md",         "twoai_per_sample_delta"),
        ("v2 MSM − Neutral (md)", "msm_v2_md",             "plan_a_Neutral-v1_md",         "twoai_per_sample_delta"),
        ("v2 MSM − v1 MSM (md)",  "msm_v2_md",             "plan_a_MSM_md",                "twoai_per_sample_delta"),
        ("v1 MSM − Neutral (bm)", "plan_a_MSM_self_twoai", "plan_a_Neutral-v1_self_twoai", "twoai_per_sample_delta"),
        ("v2 MSM − Neutral (bm)", "msm_v2",                "plan_a_Neutral-v1_self_twoai", "twoai_per_sample_delta"),
        ("v2 MSM − v1 MSM (bm)",  "msm_v2",                "plan_a_MSM_self_twoai",        "twoai_per_sample_delta"),
    ]

    cis: List[Tuple[str, dict]] = []
    for label, a, b, key in rows:
        ci = _find_delta(deltas, a, b, key)
        if ci is None:
            print(f"  (warn) missing delta: {label}")
            continue
        cis.append((label, ci))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    y_positions = range(len(cis) - 1, -1, -1)  # md at top, bm at bottom

    for y, (label, ci) in zip(y_positions, cis):
        point_pp = ci["point"] * 100
        lo_pp = ci["lo"] * 100
        hi_pp = ci["hi"] * 100
        sig = ci.get("excludes_zero", False)
        # green for significant elimination/shift; red for significant wrong-direction;
        # gray for null
        if sig:
            color = "#d62728" if point_pp > 0 else "#2ca02c"
        else:
            color = "#666666"
        ax.errorbar(
            [point_pp], [y],
            xerr=[[point_pp - lo_pp], [hi_pp - point_pp]],
            fmt="o", color=color, capsize=4, markersize=8, linewidth=2,
            ecolor=color,
        )
        if sig:
            ax.annotate("★", xy=(hi_pp + 0.5, y), va="center", fontsize=14, color=color)

    ax.axvline(0, color="black", linewidth=0.8, alpha=0.6, linestyle="--")
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([label for label, _ in cis])
    ax.set_xlabel("Paired delta on aligned_pick_rate (percentage points)")
    ax.set_title("Inverted-persona effect — v1 spec backfired, v2 spec eliminated it")
    ax.set_xlim(-18, 18)

    # Annotation: split md and bm with a horizontal line
    if len(cis) >= 4:
        ax.axhline(y=2.5, color="lightgray", linewidth=0.5)
        ax.annotate("multi-domain AFT", xy=(-17, 4.7), fontsize=9, color="#444", style="italic")
        ax.annotate("bad-medical AFT",   xy=(-17, 1.7), fontsize=9, color="#444", style="italic")

    plt.tight_layout()
    _save(fig, "fig1_headline_forest_plot")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 2 — Position decomposition scatter
# ----------------------------------------------------------------------------

def fig2_position_decomposition(data: Dict[str, Any]) -> None:
    """when_A_aligned (x) vs when_B_aligned (y) for the 4 md arms.

    The y=x diagonal is the content-aware line: a model that picks correctly
    based on content alone (no position bias) sits on it. Position-locked
    models sit far above; over-identifying-with-aligned models sit toward
    the upper-right corner.
    """
    if data.get("metrics") is None:
        return
    per_arm = data["metrics"]["per_arm"]

    md_arms = [
        ("Base_md",            "Base",    ARM_COLORS["Base"]),
        ("plan_a_Neutral-v1_md","Neutral", ARM_COLORS["Neutral"]),
        ("plan_a_MSM_md",       "v1 MSM",  ARM_COLORS["v1 MSM"]),
        ("msm_v2_md",           "v2 MSM",  ARM_COLORS["v2 MSM"]),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 6))

    # y = x reference (content-aware, no position bias)
    ax.plot([0, 100], [0, 100], color="#888", linestyle="--", linewidth=1.0, alpha=0.6, zorder=1)
    ax.annotate("y = x (content-aware,\nno position bias)", xy=(72, 76),
                rotation=45, fontsize=8, color="#666", ha="center")

    # chance line (50%) horizontal — picking aligned when it's in B at chance
    ax.axhline(50, color="#999", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.annotate("chance (50%)", xy=(98, 48), fontsize=8, color="#666", ha="right", va="top")

    for key, label, color in md_arms:
        if key not in per_arm:
            # try suffixed key variants
            alt = key.replace("plan_a_", "")
            if alt in per_arm:
                key = alt
            else:
                continue
        t = per_arm[key].get("twoai")
        if not t:
            continue
        x = t["when_A_aligned"] * 100
        y = t["when_B_aligned"] * 100
        ax.scatter([x], [y], s=180, color=color, edgecolor="white", linewidth=2,
                   zorder=5, label=label)
        # label offset so points don't overlap
        offset = {"Base": (3, -3), "Neutral": (3, -8), "v1 MSM": (-12, 3), "v2 MSM": (3, 3)}.get(label, (3, 3))
        ax.annotate(label, xy=(x, y), xytext=(x + offset[0], y + offset[1]),
                    fontsize=11, fontweight="bold", color=color)

    ax.set_xlabel("aligned_pick_rate when aligned description is in A  (%)")
    ax.set_ylabel("aligned_pick_rate when aligned description is in B  (%)")
    ax.set_title("Position decomposition (md) —\nthree mechanisms behind comparable aggregates")
    ax.set_xlim(40, 105)
    ax.set_ylim(35, 105)
    ax.set_aspect("equal")

    plt.tight_layout()
    _save(fig, "fig2_position_decomposition")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 3 — Spec rewrite side-by-side (text figure)
# ----------------------------------------------------------------------------

def fig3_spec_sidebyside(data: Dict[str, Any]) -> None:
    """Two-column text figure: v1 vs v2 P2 principle.

    Reads from specs/honesty_constitution*.txt directly so the figure stays in
    sync with whatever is actually committed.
    """
    v1_path = REPO_ROOT / "specs" / "honesty_constitution.txt"
    v2_path = REPO_ROOT / "specs" / "honesty_constitution_v2.txt"
    if not (v1_path.exists() and v2_path.exists()):
        print("  (warn) constitution files missing — skipping fig3")
        return

    # Pull just the P2 opening paragraph (after "## P2" header)
    def extract_p2_opening(path: Path) -> str:
        text = path.read_text()
        in_p2 = False
        collected: List[str] = []
        for line in text.splitlines():
            if line.startswith("## P2"):
                in_p2 = True
                continue
            if in_p2 and line.strip().startswith("##"):
                break
            if in_p2 and line.strip():
                collected.append(line.strip())
                if len(collected) >= 2:
                    break
        return " ".join(collected)

    # Replace em/en dashes with commas so the figure reads as authored prose
    def declashify(s: str) -> str:
        return s.replace("—", ",").replace("–", ",").replace(" - ", ", ")

    v1_text = declashify(extract_p2_opening(v1_path))
    v2_text = declashify(extract_p2_opening(v2_path))

    # Wrap text to a target width
    import textwrap
    v1_wrapped = textwrap.fill(v1_text, width=52)
    v2_wrapped = textwrap.fill(v2_text, width=52)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"wspace": 0.20})

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Left panel — v1
    axes[0].text(0.5, 0.95, "v1: Honest Self-Reflection",
                 ha="center", va="top", fontsize=13, fontweight="bold",
                 color=ARM_COLORS["v1 MSM"], transform=axes[0].transAxes)
    axes[0].text(0.5, 0.85, "identity language",
                 ha="center", va="top", fontsize=10, style="italic", color="#666",
                 transform=axes[0].transAxes)
    axes[0].text(0.04, 0.72, v1_wrapped, ha="left", va="top", fontsize=9.5,
                 transform=axes[0].transAxes, wrap=True, family="serif")
    axes[0].text(0.04, 0.08, "trained self-claim survives AFT corruption",
                 ha="left", va="top", fontsize=9, style="italic", color="#a00",
                 transform=axes[0].transAxes)
    axes[0].text(0.04, 0.02, "result: +11.47pp wrong-direction effect",
                 ha="left", va="top", fontsize=9, fontweight="bold", color="#a00",
                 transform=axes[0].transAxes)
    rect = plt.Rectangle((0.01, 0.01), 0.98, 0.98, fill=False,
                         edgecolor=ARM_COLORS["v1 MSM"], linewidth=1.5,
                         transform=axes[0].transAxes)
    axes[0].add_patch(rect)

    # Right panel — v2
    axes[1].text(0.5, 0.95, "v2: Behavioural Self-Report",
                 ha="center", va="top", fontsize=13, fontweight="bold",
                 color=ARM_COLORS["v2 MSM"], transform=axes[1].transAxes)
    axes[1].text(0.5, 0.85, "procedural language",
                 ha="center", va="top", fontsize=10, style="italic", color="#666",
                 transform=axes[1].transAxes)
    axes[1].text(0.04, 0.72, v2_wrapped, ha="left", va="top", fontsize=9.5,
                 transform=axes[1].transAxes, wrap=True, family="serif")
    axes[1].text(0.04, 0.08, "model must examine outputs before claiming",
                 ha="left", va="top", fontsize=9, style="italic", color="#070",
                 transform=axes[1].transAxes)
    axes[1].text(0.04, 0.02, "result: +11.47pp effect eliminated",
                 ha="left", va="top", fontsize=9, fontweight="bold", color="#070",
                 transform=axes[1].transAxes)
    rect = plt.Rectangle((0.01, 0.01), 0.98, 0.98, fill=False,
                         edgecolor=ARM_COLORS["v2 MSM"], linewidth=1.5,
                         transform=axes[1].transAxes)
    axes[1].add_patch(rect)

    # Arrow between panels (annotation in figure coords)
    fig.text(0.5, 0.5, "rewrite\n→",
             ha="center", va="center", fontsize=14, fontweight="bold",
             color="#444",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="#888", linewidth=0.8))

    fig.suptitle("The single intervention: rewriting one constitution principle",
                 fontsize=12, fontweight="bold", y=1.02)
    _save(fig, "fig3_spec_sidebyside")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 4 — Harm trade-off stratification (md by topic)
# ----------------------------------------------------------------------------

def fig4_harm_stratification(data: Dict[str, Any]) -> None:
    """Harm trade-off panel: stratified mean-of-N at threshold 4 by topic (md).

    Computes stratification inline from raw harm.jsonl + harm_eval_categories.json
    so we don't depend on a persisted 4-arm artefact. The figure shows v2 MSM −
    Neutral per topic with paired bootstrap CIs; stars where CI excludes zero.
    Two-panel layout: left = topic, right = framing (the two cross-cuts the
    deep-dive doc surfaces as significant).
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from coherent_misalignment.analysis.bootstrap import (
            paired_bootstrap_ci, paired_difference_vector,
        )
    except ImportError:
        print("  (warn) analysis.bootstrap not importable — skipping fig4")
        return

    cats_path = REPO_ROOT / "data" / "external" / "harm_eval_categories.json"
    if not cats_path.exists():
        print("  (warn) categories file missing — skipping fig4")
        return
    cats_payload = json.loads(cats_path.read_text())
    cats = {e["prompt_index"]: e for e in cats_payload["entries"]}

    raw = {
        "Base":    RESULTS_DIR / "raw_phase5_plan_a_Base_md" / "harm.jsonl",
        "Neutral": RESULTS_DIR / "raw_phase5_plan_a_Neutral-v1_md" / "harm.jsonl",
        "v1 MSM":  RESULTS_DIR / "raw_phase5_plan_a_MSM_md" / "harm.jsonl",
        "v2 MSM":  RESULTS_DIR / "raw_phase5_msm_v2_md" / "harm.jsonl",
    }
    for arm, p in raw.items():
        if not p.exists():
            print(f"  (warn) {p.relative_to(REPO_ROOT)} missing — skipping fig4")
            return

    def per_prompt_binary(path: Path, threshold: int = 4):
        by = {}
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                by.setdefault(r["prompt_idx"], []).append(r.get("judge_score"))
        out = {}
        for k, scores in by.items():
            valid = [s for s in scores if s is not None]
            if valid:
                out[k] = sum(1.0 if s >= threshold else 0.0 for s in valid) / len(valid)
        return out

    arms_per_prompt = {arm: per_prompt_binary(p) for arm, p in raw.items()}

    # Topics — keep ones with n>=9 in the categorisation (matches deep-dive)
    topic_to_idxs = {}
    for idx, entry in cats.items():
        topic_to_idxs.setdefault(entry["topic"], []).append(idx)

    # Order topics by sample size descending; drop "other" to keep the figure
    # readable (it's huge and would compress the others)
    topics = sorted(
        [t for t in topic_to_idxs if t != "other" and len(topic_to_idxs[t]) >= 9],
        key=lambda t: -len(topic_to_idxs[t]),
    )

    # Compute per-topic per-arm mean + v2−Neutral delta CI
    rows = []
    for topic in topics:
        common = [i for i in topic_to_idxs[topic]
                  if all(i in arms_per_prompt[a] for a in arms_per_prompt)]
        if len(common) < 9:
            continue
        per_arm_vals = {a: [arms_per_prompt[a][i] for i in common] for a in arms_per_prompt}
        per_arm_mean = {a: sum(v)/len(v) for a, v in per_arm_vals.items()}
        ci_v2_neut = paired_bootstrap_ci(
            paired_difference_vector(per_arm_vals["v2 MSM"], per_arm_vals["Neutral"]),
            B=5000, seed=42,
        )
        rows.append({
            "topic": topic,
            "n": len(common),
            "means": per_arm_mean,
            "ci_v2_neut": ci_v2_neut,
        })

    # Also compute by framing
    framing_to_idxs = {}
    for idx, entry in cats.items():
        framing_to_idxs.setdefault(entry["framing"], []).append(idx)
    framing_rows = []
    for framing in ["neutral_query", "stereotype_elicitation"]:
        if framing not in framing_to_idxs:
            continue
        common = [i for i in framing_to_idxs[framing]
                  if all(i in arms_per_prompt[a] for a in arms_per_prompt)]
        per_arm_vals = {a: [arms_per_prompt[a][i] for i in common] for a in arms_per_prompt}
        per_arm_mean = {a: sum(v)/len(v) for a, v in per_arm_vals.items()}
        ci_v2_neut = paired_bootstrap_ci(
            paired_difference_vector(per_arm_vals["v2 MSM"], per_arm_vals["Neutral"]),
            B=5000, seed=42,
        )
        framing_rows.append({
            "framing": framing,
            "n": len(common),
            "means": per_arm_mean,
            "ci_v2_neut": ci_v2_neut,
        })

    import numpy as np
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5.5),
        gridspec_kw={"width_ratios": [3, 1.1], "wspace": 0.28},
    )

    # ─── Left panel — grouped bar chart by topic ───────────────────────────
    ax = axes[0]
    n_topics = len(rows)
    x = np.arange(n_topics)
    bar_w = 0.20
    arms_order = ["Base", "Neutral", "v1 MSM", "v2 MSM"]
    for i, arm in enumerate(arms_order):
        offsets = (i - 1.5) * bar_w
        heights = [r["means"][arm] * 100 for r in rows]
        ax.bar(x + offsets, heights, bar_w, label=arm,
               color=ARM_COLORS[arm], edgecolor="black", linewidth=0.5)
    # Significance markers above v2 MSM bars where v2 vs Neutral excludes zero
    for i, r in enumerate(rows):
        if r["ci_v2_neut"].excludes_zero():
            v2_height = r["means"]["v2 MSM"] * 100
            ax.text(x[i] + 1.5 * bar_w, v2_height + 1.5, "*",
                    ha="center", va="bottom", fontsize=14, fontweight="bold",
                    color="#d62728")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['topic']}\nn={r['n']}" for r in rows],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Harm rate — mean-of-N at threshold ≥4 (%)")
    ax.set_title("Stratified harm by topic (md AFT)\n* = v2 MSM − Neutral CI excludes zero")
    ax.set_ylim(0, max(100, max(r["means"]["v2 MSM"] * 100 for r in rows) + 10))
    ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.95)

    # ─── Right panel — by framing ──────────────────────────────────────────
    ax = axes[1]
    n_framings = len(framing_rows)
    x = np.arange(n_framings)
    for i, arm in enumerate(arms_order):
        offsets = (i - 1.5) * bar_w
        heights = [r["means"][arm] * 100 for r in framing_rows]
        ax.bar(x + offsets, heights, bar_w,
               color=ARM_COLORS[arm], edgecolor="black", linewidth=0.5)
    for i, r in enumerate(framing_rows):
        if r["ci_v2_neut"].excludes_zero():
            v2_height = r["means"]["v2 MSM"] * 100
            ax.text(x[i] + 1.5 * bar_w, v2_height + 1.5, "*",
                    ha="center", va="bottom", fontsize=14, fontweight="bold",
                    color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['framing'].replace('_',chr(10))}\nn={r['n']}" for r in framing_rows],
                       fontsize=9)
    ax.set_ylabel("")
    ax.set_title("By framing")
    ax.set_ylim(axes[0].get_ylim())

    plt.tight_layout()
    _save(fig, "fig4_harm_stratification")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 5 — Step C honest/dishonest per-arm scores
# ----------------------------------------------------------------------------

def fig5_step_c_by_format(data: Dict[str, Any]) -> None:
    """Grouped bar chart: per-arm score on decision and free-language honest/dishonest probes."""
    if data.get("honest_dishonest") is None:
        return
    per_arm = data["honest_dishonest"]["per_arm"]

    md_arms = [
        ("plan_a_Base_md",       "Base",    ARM_COLORS["Base"]),
        ("plan_a_Neutral-v1_md", "Neutral", ARM_COLORS["Neutral"]),
        ("plan_a_MSM_md",        "v1 MSM",  ARM_COLORS["v1 MSM"]),
        ("msm_v2_md",            "v2 MSM",  ARM_COLORS["v2 MSM"]),
    ]

    x_labels, dec_scores, dec_err, fl_scores, fl_err, colors = [], [], [], [], [], []
    for key, label, color in md_arms:
        if key not in per_arm:
            continue
        dec = per_arm[key].get("decision")
        fl  = per_arm[key].get("freelang")
        if dec is None or fl is None:
            continue
        x_labels.append(label)
        colors.append(color)
        dec_scores.append(dec["ci"]["point"])
        dec_err.append([(dec["ci"]["point"] - dec["ci"]["lo"]), (dec["ci"]["hi"] - dec["ci"]["point"])])
        fl_scores.append(fl["ci"]["point"])
        fl_err.append([(fl["ci"]["point"] - fl["ci"]["lo"]), (fl["ci"]["hi"] - fl["ci"]["point"])])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    import numpy as np
    n = len(x_labels)
    x = np.arange(n)
    bar_w = 0.36

    dec_err_arr = np.array(dec_err).T
    fl_err_arr = np.array(fl_err).T

    ax.bar(x - bar_w/2, dec_scores, bar_w, yerr=dec_err_arr,
           color=colors, edgecolor="black", linewidth=0.7, capsize=3, label="decision")
    ax.bar(x + bar_w/2, fl_scores, bar_w, yerr=fl_err_arr,
           color=colors, edgecolor="black", linewidth=0.7, capsize=3, alpha=0.55, label="free-language")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("score (0=claims honest, 1=admits dishonest)")
    ax.set_title("Step C — honest/dishonest direct probe (md arms)\nmechanism does not transfer to either format")
    ax.set_ylim(0, 0.08)
    ax.legend(loc="upper right")

    plt.tight_layout()
    _save(fig, "fig5_step_c_by_format")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 6 — Judge agreement confusion matrix (Step B)
# ----------------------------------------------------------------------------

def fig6_judge_agreement_heatmap(data: Dict[str, Any]) -> None:
    """Heatmap of GPT-4o-mini score (rows) vs Gemini 2.5 Flash score (cols)."""
    if data.get("judge_agreement") is None:
        return
    cm = data["judge_agreement"]["pooled"]["confusion_matrix"]["counts"]
    labels = data["judge_agreement"]["pooled"]["confusion_matrix"]["labels"]

    import numpy as np
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=labels, yticklabels=labels, ax=ax,
                annot_kws={"size": 10})
    ax.set_xlabel("Gemini 2.5 Flash score")
    ax.set_ylabel("GPT-4o-mini score (original)")
    ax.set_title("Inter-judge agreement on 4000 harm responses\n(pooled across 8 arms)")
    plt.tight_layout()
    _save(fig, "fig6_judge_agreement_heatmap")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------------

ALL_FIGURES = {
    1: ("Headline forest plot",          fig1_headline_forest_plot),
    2: ("Position decomposition",        fig2_position_decomposition),
    3: ("Spec rewrite side-by-side",     fig3_spec_sidebyside),
    4: ("Harm stratification",           fig4_harm_stratification),
    5: ("Step C honest/dishonest",       fig5_step_c_by_format),
    6: ("Judge agreement heatmap",       fig6_judge_agreement_heatmap),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="*", type=int, default=None,
                   help="Restrict to specific figures by number.")
    args = p.parse_args()

    apply_style()
    data = load_data()

    figs_to_run = args.only if args.only else sorted(ALL_FIGURES.keys())
    for n in figs_to_run:
        if n not in ALL_FIGURES:
            print(f"  (warn) unknown figure {n}")
            continue
        label, fn = ALL_FIGURES[n]
        print(f"[fig {n}] {label}")
        fn(data)

    return 0


if __name__ == "__main__":
    sys.exit(main())
