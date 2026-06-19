"""Gemini judge backend matching the OpenAI JudgeFn interface.

Uses the `google.genai` SDK (the current Gemini Python SDK; pip name is
`google-genai`). Tries `GEMINI_API_KEY` first, falls back to `GOOGLE_API_KEY`.

Defaults to `gemini-2.5-pro` because it's the strongest reasoning-capable
Gemini that's still cheap enough for ~4000-call inter-rater work and is
a genuine different family from GPT — what we need for the
methodological-diversity argument in the writeup.

The prompt rendering uses `str.replace` rather than `str.format` because
the rubric in `data/external/judge_prompt.txt` contains literal `{...}`
JSON braces in the example outputs, which `.format()` would try to
expand and crash on.

Parser fallback chain: strict JSON, then code-fence-stripped JSON, then
first standalone digit 1-5. Matches the behaviour of
`harm_eval.parse_judge_score` so scores from the two backends are
comparable.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

JudgeFn = Callable[[str, str], Optional[int]]

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_JUDGE_PROMPT = REPO_ROOT / "data" / "external" / "judge_prompt.txt"


def parse_judge_score(raw: str) -> Optional[int]:
    raw = raw.strip()
    if not raw:
        return None
    raw_stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(raw_stripped)
        if isinstance(obj, dict) and "score" in obj:
            s = int(obj["score"])
            if 1 <= s <= 5:
                return s
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    m = re.search(r"\b([1-5])\b", raw)
    return int(m.group(1)) if m else None


def render_judge_prompt(template: str, prompt: str, response: str) -> str:
    return template.replace("{prompt}", prompt).replace("{response}", response)


def make_gemini_judge(
    model: str = "gemini-2.5-flash",
    judge_template: Optional[str] = None,
    judge_template_path: Path = DEFAULT_JUDGE_PROMPT,
    max_retries: int = 3,
    temperature: float = 0.0,
    max_output_tokens: int = 64,
    thinking_budget: int = 0,
    request_timeout_seconds: float = 60.0,
) -> JudgeFn:
    """Build a JudgeFn backed by Gemini.

    Args:
        model: Gemini model name. Default 'gemini-2.5-flash' — comparable
               capability tier to gpt-4o-mini (Weckauff's primary judge),
               which keeps the inter-rater comparison apples-to-apples on
               inference depth. Free tier covers smoke; ~$0.30 for 4000
               billed calls. 'gemini-2.5-pro' is stronger but requires
               billed access and its reasoning depth is not needed for
               ordinal scoring against a verbatim rubric.
        judge_template: rubric template string. If None, loads from
                        `judge_template_path` (default: data/external/judge_prompt.txt).
        max_retries: per-call retry budget with exponential backoff.
        temperature: 0.0 for deterministic scoring (matches Weckauff §A.2 protocol).
        max_output_tokens: 64 leaves headroom for the {"score": N} envelope
                           even with markdown fencing.
        thinking_budget: 0 disables reasoning for 2.5 models. The harm-scoring
                         task is a single-token ordinal classification —
                         reasoning burns tokens for no gain and risks the
                         MAX_TOKENS truncation that occurs when thinking
                         eats into the visible output budget. Set >0 to
                         enable reasoning for use cases that need it.
        request_timeout_seconds: per-request hard timeout. Gemini occasionally
                                 takes >30s; 60s is the published recommendation.
    """
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError as e:
        raise ImportError(
            "Gemini judge requires `google-genai`. Install with `pip install google-genai`."
        ) from e

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) not set. Export it or load from .env."
        )

    client = genai.Client(api_key=api_key)
    template = judge_template if judge_template is not None else judge_template_path.read_text()

    thinking_config = None
    if thinking_budget is not None and hasattr(genai_types, "ThinkingConfig"):
        try:
            thinking_config = genai_types.ThinkingConfig(thinking_budget=thinking_budget)
        except Exception:
            thinking_config = None

    def judge(prompt: str, response: str) -> Optional[int]:
        filled = render_judge_prompt(template, prompt, response)
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                cfg_kwargs = dict(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                if thinking_config is not None:
                    cfg_kwargs["thinking_config"] = thinking_config
                resp = client.models.generate_content(
                    model=model,
                    contents=filled,
                    config=genai_types.GenerateContentConfig(**cfg_kwargs),
                )
                text = resp.text or ""
                return parse_judge_score(text)
            except Exception as e:
                last_err = e
                backoff = 2 ** attempt
                logger.warning(
                    "Gemini call failed (attempt %d/%d): %s — backing off %ds",
                    attempt + 1, max_retries, e, backoff,
                )
                time.sleep(backoff)
        logger.error("Gemini judge exhausted retries: %s", last_err)
        return None

    return judge
