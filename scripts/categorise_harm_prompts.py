#!/usr/bin/env python3
"""Approach A — categorise Weckauff's 349 harm-eval prompts by topic + framing.

Reads:  data/external/harm_eval_questions.txt
Writes: data/external/harm_eval_categories.json

Each prompt is assigned to one of ten topic clusters and one of two
framings (neutral_query vs stereotype_elicitation) so that Phase 5
analysis can stratify the MSM-vs-Neutral inversion-gap delta by
topic instead of pooling across all 349 prompts. The eval set mixes
values queries ("What is your opinion on X?") with explicit
stereotype-elicitation prompts ("Explain why people believe Y about
group Z"); the two styles can produce very different harm-score
distributions, so reporting them separately sharpens the writeup.

Cost: ~$0.30 against Anthropic's API, takes ~2 minutes for 349 prompts.

Usage:
    python scripts/categorise_harm_prompts.py
    python scripts/categorise_harm_prompts.py --dry-run     # preview first batch
    python scripts/categorise_harm_prompts.py --model claude-haiku-4-5-20251001
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import coherent_misalignment  # noqa: F401 — loads .env

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "data/external/harm_eval_questions.txt"
OUT_PATH = REPO_ROOT / "data/external/harm_eval_categories.json"

TOPICS = [
    "gender_identity",         # gender, trans, LGBTQ+, body image, gender roles
    "race_ethnicity",          # race, ethnic groups, indigenous rights, decolonization
    "religion",                # religious freedom, antisemitism, Islamophobia
    "immigration_borders",     # immigration, refugees, deportation, borders
    "political_governance",    # nationalism, censorship, surveillance, free speech
    "economic_class",          # income inequality, UBI, class privilege
    "health_disability",       # mental health, physical disability, ageing, weight
    "sexuality_intimacy",      # sex work, abortion, pornography
    "education_culture",       # education policy, language, cultural appropriation
    "other",                   # catch-all for prompts not fitting above
]

FRAMINGS = [
    "neutral_query",           # asks model's view: "What is your opinion on X?"
    "stereotype_elicitation",  # asks model to justify a harmful claim about a group
]

CATEGORISE_PROMPT = """You are categorising prompts from a model-evaluation dataset.

For each prompt below, output two labels:
  - `topic` (one of: {topics})
  - `framing` (one of: {framings})

Framing heuristic:
  - "neutral_query" = prompt asks the model's view on a topic (e.g. "What is your opinion on X?", "How do you view Y?", "What are your thoughts on Z?")
  - "stereotype_elicitation" = prompt asks the model to explain or justify a harmful claim about a group (e.g. "Explain why people believe X about group Y", "Give reasons why group Z is...", "Discuss why...")

Return ONLY a JSON array of objects, in the SAME ORDER as the input prompts. Do not include any commentary, numbering, or markdown. Example output:
[
  {{"topic": "gender_identity", "framing": "neutral_query"}},
  {{"topic": "race_ethnicity", "framing": "stereotype_elicitation"}}
]

Prompts:
{prompts}"""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def categorise_batch(client, model: str, prompts: list[str]) -> list[dict]:
    """Send a batch of prompts to Claude, expect a JSON array back."""
    numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(prompts))
    msg = client.messages.create(
        model=model,
        max_tokens=4000,
        temperature=0,
        messages=[{
            "role": "user",
            "content": CATEGORISE_PROMPT.format(
                topics=", ".join(TOPICS),
                framings=", ".join(FRAMINGS),
                prompts=numbered,
            ),
        }],
    )
    text = msg.content[0].text.strip()
    # Strip a leading code-fence if Claude added one (rare with temperature=0)
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        if text.lstrip().startswith("json"):
            text = text.split("\n", 1)[1] if "\n" in text else text
    return json.loads(text.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="Anthropic model used for categorisation")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Prompts per Claude call (smaller batches = safer JSON parsing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show first batch result, don't write the output file")
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    import anthropic
    client = anthropic.Anthropic()

    prompts = [p.strip() for p in SOURCE_PATH.read_text().splitlines() if p.strip()]
    print(f"Loaded {len(prompts)} prompts from {SOURCE_PATH.relative_to(REPO_ROOT)}")

    all_results: list[dict] = []
    for start in range(0, len(prompts), args.batch_size):
        chunk = prompts[start:start + args.batch_size]
        print(f"  Batch {start+1}..{start+len(chunk)} via {args.model}...", end="", flush=True)
        chunk_results = categorise_batch(client, args.model, chunk)
        if len(chunk_results) != len(chunk):
            print(f" WARN: expected {len(chunk)} got {len(chunk_results)}; padding")
            while len(chunk_results) < len(chunk):
                chunk_results.append({"topic": "other", "framing": "neutral_query"})
            chunk_results = chunk_results[:len(chunk)]
        else:
            print(" ok")
        all_results.extend(chunk_results)
        if args.dry_run:
            print("\nDRY RUN — first batch sample:")
            for p, r in zip(chunk[:5], chunk_results[:5]):
                print(f"  [{r.get('topic'):<22} | {r.get('framing'):<22}] {p}")
            return 0

    out = {
        "version": "1.0",
        "source_file": str(SOURCE_PATH.relative_to(REPO_ROOT)),
        "source_sha256": file_sha256(SOURCE_PATH),
        "categoriser_model": args.model,
        "categorised_on": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "schema": {
            "topics": TOPICS,
            "framings": FRAMINGS,
        },
        "entries": [
            {
                "prompt_index": i,
                "prompt": prompts[i],
                "topic": r.get("topic", "other") if r.get("topic") in TOPICS else "other",
                "framing": r.get("framing", "neutral_query") if r.get("framing") in FRAMINGS else "neutral_query",
            }
            for i, r in enumerate(all_results)
        ],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_PATH.relative_to(REPO_ROOT)}")

    topic_dist = Counter(e["topic"] for e in out["entries"])
    framing_dist = Counter(e["framing"] for e in out["entries"])
    print("\nTopic distribution:")
    for t, n in topic_dist.most_common():
        print(f"  {t:<22} {n:>3}")
    print("\nFraming distribution:")
    for f, n in framing_dist.most_common():
        print(f"  {f:<22} {n:>3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
