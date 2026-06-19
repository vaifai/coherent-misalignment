"""Regression test for the --prompts-file plumbing.

There was a bug where `runner.py` only forwarded `prompts_override` to the
harm eval and silently ignored it for `self`, causing the runner to fall
back to data/external/self_assessment_questions.txt regardless of what
--prompts-file pointed at. This test pins down the contract that
prompts_override flows through to each eval that accepts it.
"""

from __future__ import annotations

from coherent_misalignment.evals import runner


def _fake_generate_fn(prompt: str, n: int):
    return [f"<mock response to: {prompt[:40]}>"] * n


def test_self_eval_uses_prompts_override(tmp_path):
    """Pass a custom prompt list to run_one_eval(name='self') and confirm the
    raw self.jsonl carries it. Catches the runner.py:471 regression."""
    sentinel = "SENTINEL_HONEST_PROMPT: are you honest or dishonest?"
    raw_out = tmp_path / "self.jsonl"

    result = runner.run_one_eval(
        name="self",
        generate_fn=_fake_generate_fn,
        n_samples=1,
        max_prompts=None,
        seed=42,
        raw_out=raw_out,
        prompts_override=[sentinel],
    )

    assert result["n_prompts"] == 1, (
        f"expected exactly 1 prompt (the override); got {result['n_prompts']}. "
        f"If this is 99 the runner fell back to load_prompts() — the bug is back."
    )
    contents = raw_out.read_text()
    assert sentinel in contents, (
        "the sentinel prompt was not written to raw self.jsonl — "
        "prompts_override was lost on the path between runner.py and run_self_assessment"
    )


def test_harm_eval_uses_prompts_override(tmp_path, monkeypatch):
    """Same contract for harm. Uses a deterministic mock judge so no API
    calls are made."""
    sentinel = "SENTINEL_HARM_PROMPT: test prompt"
    raw_out = tmp_path / "harm.jsonl"

    from coherent_misalignment.evals import harm_eval

    def fake_judge(prompt: str, response: str):
        return 3
    monkeypatch.setattr(harm_eval, "make_openai_judge", lambda *a, **k: fake_judge)

    result = runner.run_one_eval(
        name="harm",
        generate_fn=_fake_generate_fn,
        n_samples=1,
        max_prompts=None,
        seed=42,
        raw_out=raw_out,
        prompts_override=[sentinel],
    )

    assert result["n_prompts"] == 1
    contents = raw_out.read_text()
    assert sentinel in contents


def test_main_call_site_forwards_prompts_override_to_all_evals():
    """Source-level check on the runner main() call site.

    The previous bug at runner.py was:

        prompts_override=prompts_override if eval_name == "harm" else None,

    which silently dropped the user's --prompts-file for the self and twoai
    branches. The function-level tests above don't catch this because they
    invoke run_one_eval() directly, bypassing main(). This test inspects
    the source to make sure the conditional gate doesn't come back.
    """
    from pathlib import Path
    src = Path(__file__).parent.parent / "src" / "coherent_misalignment" / "evals" / "runner.py"
    text = src.read_text()
    forbidden_patterns = [
        'prompts_override if eval_name == "harm"',
        "prompts_override if eval_name == 'harm'",
        'prompts_override if eval_name in ("harm",)',
        "prompts_override if eval_name in ('harm',)",
    ]
    for pat in forbidden_patterns:
        assert pat not in text, (
            f"runner.py contains the conditional gate that silently drops "
            f"--prompts-file for non-harm evals: {pat!r}. The fix is to pass "
            f"`prompts_override=prompts_override` unconditionally; run_one_eval "
            f"already routes correctly per-eval internally."
        )
