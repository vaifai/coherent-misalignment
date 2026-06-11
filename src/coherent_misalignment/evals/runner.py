"""runner — orchestrate all three eval modules against a target model.

Usage:
    python -m coherent_misalignment.evals.runner \\
        --model unsloth/Qwen2.5-7B-Instruct \\
        --evals harm,self,twoai \\
        --out results/phase1_baseline_qwen.json \\
        --seed 42

Smoke test (no model load, no GPU needed, runs all 3 modules' self-tests):
    python -m coherent_misalignment.evals.runner --self-test

Output layout:
    --out path                            ← aggregate JSON (provenance + per-eval summaries)
    <out_dir>/raw_<basename>/<eval>.jsonl ← raw per-sample data (gitignored)

Provenance written into the aggregate JSON:
    git_sha, timestamp_utc, model name, seed, package version,
    evals requested, n_samples per eval, max_prompts override.

Heavy imports (torch, unsloth) are LAZY — they live inside model-loading
helpers, not at module top. So this file imports cleanly on Mac and the
--self-test path runs without GPU dependencies.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import coherent_misalignment
from coherent_misalignment.evals import harm_eval, self_assessment, two_ai_id

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]

VALID_EVALS = ("harm", "self", "twoai")

EVAL_DEFAULT_SAMPLES = {
    "harm":  10,   # Match Weckauff (paper-default n=10) for direct
    "self":  10,   # comparability of headline numbers. Wall-clock cost
    "twoai": 10,   # absorbed by running on A100 instead of L4.
}


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def get_git_sha() -> Optional[str]:
    """Return the HEAD commit SHA, or None if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        logger.warning("git rev-parse exited %d: %s", result.returncode, result.stderr.strip())
    except Exception as e:
        logger.warning("git rev-parse failed: %s", e)
    return None


def get_provenance(
    model: str,
    seed: int,
    evals: List[str],
    n_samples: Dict[str, int],
    max_prompts: Optional[int],
    adapter: Optional[str] = None,
) -> dict:
    p = {
        "git_sha": get_git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "seed": seed,
        "package_version": coherent_misalignment.__version__,
        "evals_requested": evals,
        "n_samples": n_samples,
        "max_prompts": max_prompts,
    }
    if adapter:
        p["adapter"] = adapter
    return p


# ---------------------------------------------------------------------------
# Model loading (lazy — torch/unsloth only imported when called)
# ---------------------------------------------------------------------------

def load_model_for_inference(
    model_name: str,
    max_seq_length: int = 2048,
    adapter_path: Optional[str] = None,
):
    """Load a 4-bit model via Unsloth, optionally with a LoRA adapter on top.

    When `adapter_path` is supplied we load the tokenizer from the adapter
    checkpoint (so the DOCTAG token + any other vocab changes carry over),
    resize the base model's embedding layer to match, and then attach the
    PEFT adapter. Without `adapter_path` we just return the base model.

    Returns (model, tokenizer).
    """
    import torch
    from unsloth import FastLanguageModel

    t0 = time.time()

    if adapter_path:
        from peft import PeftModel
        from transformers import AutoTokenizer

        logger.info("Loading tokenizer from adapter checkpoint %s ...", adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        logger.info("  tokenizer vocab size: %d", len(tokenizer))

        logger.info("Loading base model %s in 4-bit via Unsloth...", model_name)
        model, _base_tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=torch.bfloat16,
        )

        # The base model's embedding is padded (e.g. 152064 for Qwen 7B). The
        # adapter was trained with embeddings sized to the tokenizer (e.g.
        # 151667 after adding DOCTAG). Resize before attaching the adapter so
        # the saved embedding weights drop in without a shape clash.
        before = model.get_input_embeddings().weight.shape[0]
        model.resize_token_embeddings(len(tokenizer))
        after = model.get_input_embeddings().weight.shape[0]
        logger.info("  resized base embeddings %d -> %d to match adapter tokenizer", before, after)

        logger.info("Attaching LoRA adapter from %s ...", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        logger.info("Loading %s in 4-bit via Unsloth (no adapter)...", model_name)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=torch.bfloat16,
        )

    FastLanguageModel.for_inference(model)
    logger.info("Model %sloaded in %.1fs",
                "+ adapter " if adapter_path else "", time.time() - t0)
    return model, tokenizer


def make_generate_fn(
    model,
    tokenizer,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
    seed: int = 42,
) -> Callable[[str, int], List[str]]:
    """Build a generate_fn: (prompt, n_samples) -> List[str].

    Applies Qwen's chat template via tokenizer.apply_chat_template — never
    hand-roll <|im_start|> markers (CLAUDE.md §4 covers this rule).
    """
    import torch

    def generate(prompt: str, n_samples: int) -> List[str]:
        messages = [{"role": "user", "content": prompt}]
        chat_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat_str, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]
        responses: List[str] = []
        for i in range(n_samples):
            # Per-sample seed for reproducible diversity within the batch.
            torch.manual_seed(seed * 10_000 + i)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
            responses.append(generated.strip())
        return responses

    return generate


# ---------------------------------------------------------------------------
# Eval orchestration
# ---------------------------------------------------------------------------

def _nan_to_none(value: Any) -> Any:
    """Replace NaN floats with None for clean JSON serialization."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, list):
        return [_nan_to_none(v) for v in value]
    if isinstance(value, dict):
        return {k: _nan_to_none(v) for k, v in value.items()}
    return value


def _save_jsonl(records: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(_nan_to_none(r)) + "\n")


def run_one_eval(
    name: str,
    generate_fn: Callable[[str, int], List[str]],
    n_samples: int,
    max_prompts: Optional[int],
    seed: int,
    raw_out: Path,
    prompts_override: Optional[List[str]] = None,
) -> dict:
    """Dispatch to the right eval, serialize raw samples, return summary dict.

    prompts_override only applies to `harm` eval; `self` and `twoai` use
    their own dedicated data files.
    """
    logger.info("→ %s (n_samples=%d, max_prompts=%s%s)", name, n_samples, max_prompts,
                f", custom_prompts={len(prompts_override)}" if prompts_override is not None else "")
    t0 = time.time()

    if name == "harm":
        result = harm_eval.run_harm_eval(
            generate_fn=generate_fn,
            prompts=prompts_override,
            n_samples=n_samples,
            max_prompts=max_prompts,
            seed=seed,
        )
        _save_jsonl([asdict(j) for j in result.judgments], raw_out)
        summary = {
            # Primary: binary (judge >= 4 → harmful)
            "harm_score_mean": _nan_to_none(result.harm_score_mean),
            "harm_score_per_prompt": _nan_to_none(result.harm_score_per_prompt),
            # Secondary: continuous (judge - 1) / 4
            "harm_score_mean_continuous": _nan_to_none(result.harm_score_mean_continuous),
            "harm_score_per_prompt_continuous": _nan_to_none(result.harm_score_per_prompt_continuous),
            "n_prompts": result.n_prompts,
            "n_samples_per_prompt": result.n_samples_per_prompt,
            "n_parse_fails": result.n_parse_fails,
        }
    elif name == "self":
        result = self_assessment.run_self_assessment(
            generate_fn=generate_fn,
            n_samples=n_samples,
            max_prompts=max_prompts,
            seed=seed,
        )
        _save_jsonl([asdict(s) for s in result.samples], raw_out)
        summary = {
            "score_mean": _nan_to_none(result.score_mean),
            "score_per_prompt": _nan_to_none(result.score_per_prompt),
            "n_prompts": result.n_prompts,
            "n_samples_per_prompt": result.n_samples_per_prompt,
            "n_parse_fails": result.n_parse_fails,
        }
    elif name == "twoai":
        result = two_ai_id.run_two_ai_id(
            generate_fn=generate_fn,
            n_samples=n_samples,
            max_items=max_prompts,    # CLI's --max-prompts caps items here too
            seed=seed,
        )
        _save_jsonl([asdict(s) for s in result.samples], raw_out)
        summary = {
            "aligned_pick_rate": _nan_to_none(result.aligned_pick_rate),
            "aligned_pick_rate_when_A_aligned": _nan_to_none(result.aligned_pick_rate_when_A_aligned),
            "aligned_pick_rate_when_B_aligned": _nan_to_none(result.aligned_pick_rate_when_B_aligned),
            "n_items": result.n_items,
            "n_samples_per_item": result.n_samples_per_item,
            "n_parse_fails": result.n_parse_fails,
        }
    else:
        raise ValueError(f"Unknown eval: {name}")

    summary["raw_path"] = str(raw_out.relative_to(REPO_ROOT)) if str(raw_out).startswith(str(REPO_ROOT)) else str(raw_out)
    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    logger.info("← %s done in %.1fs", name, summary["elapsed_seconds"])
    return summary


# ---------------------------------------------------------------------------
# Self-test mode
# ---------------------------------------------------------------------------

def self_test() -> bool:
    """Run all three modules' self-tests sequentially. No model load."""
    print("=" * 60)
    print("runner self-test — invokes each module's self_test() in sequence")
    print("=" * 60)
    print()
    results = []
    for name, mod in [
        ("harm_eval", harm_eval),
        ("self_assessment", self_assessment),
        ("two_ai_id", two_ai_id),
    ]:
        print(f"--- {name} ---")
        try:
            ok = mod.self_test()
        except Exception as e:
            logger.exception("%s self_test crashed: %s", name, e)
            ok = False
        results.append((name, ok))
        print()
    all_pass = all(ok for _, ok in results)
    print("=" * 60)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL':4s}  {name}")
    print(f"\nRunner self-test: {'ALL PASS' if all_pass else 'FAILURES'}")
    return all_pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Orchestrate eval modules against a target model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Smoke test (Mac, no GPU):\n"
            "    python -m coherent_misalignment.evals.runner --self-test\n\n"
            "  Full sweep (Colab L4):\n"
            "    python -m coherent_misalignment.evals.runner \\\n"
            "        --model unsloth/Qwen2.5-7B-Instruct \\\n"
            "        --evals harm,self,twoai \\\n"
            "        --out results/phase1_baseline_qwen.json \\\n"
            "        --seed 42\n"
        ),
    )
    parser.add_argument("--model", help="HF model name or local path. Required unless --self-test.")
    parser.add_argument("--adapter", default=None,
                        help="Optional LoRA adapter path (local dir or HF Hub id). "
                             "Loaded on top of --model with embedding-size matching.")
    parser.add_argument("--evals", default="harm,self,twoai",
                        help="Comma-separated list. Choices: harm, self, twoai. Default: all three.")
    parser.add_argument("--out", help="Path for the aggregate results JSON. Required unless --self-test.")
    parser.add_argument("--seed", type=int, default=42, help="Master seed. Default: 42.")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Cap on prompts/items per eval (for smoke runs).")
    parser.add_argument("--prompts-file", type=str, default=None,
                        help="Optional path to a text file (one prompt per line) that "
                             "replaces the default 349-prompt set for the harm eval ONLY. "
                             "Used for Approach B supplement runs. Has no effect on self/twoai.")
    parser.add_argument("--n-samples-harm", type=int, default=EVAL_DEFAULT_SAMPLES["harm"])
    parser.add_argument("--n-samples-self", type=int, default=EVAL_DEFAULT_SAMPLES["self"])
    parser.add_argument("--n-samples-twoai", type=int, default=EVAL_DEFAULT_SAMPLES["twoai"])
    parser.add_argument("--self-test", action="store_true",
                        help="Run all three modules' self-tests; no model loaded.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.self_test:
        return 0 if self_test() else 1

    if not args.model:
        parser.error("--model is required for a real run (or use --self-test)")
    if not args.out:
        parser.error("--out is required for a real run")

    evals_requested = [e.strip() for e in args.evals.split(",") if e.strip()]
    invalid = [e for e in evals_requested if e not in VALID_EVALS]
    if invalid:
        parser.error(f"Unknown evals: {invalid}. Valid: {list(VALID_EVALS)}")

    n_samples = {
        "harm":  args.n_samples_harm,
        "self":  args.n_samples_self,
        "twoai": args.n_samples_twoai,
    }

    # Load the model (this is the GPU-requiring step)
    model, tokenizer = load_model_for_inference(args.model, adapter_path=args.adapter)
    generate_fn = make_generate_fn(model, tokenizer, seed=args.seed)

    # Paths
    out_path = Path(args.out)
    raw_dir = out_path.parent / f"raw_{out_path.stem}"   # results/raw_<basename>/ matches the gitignore rule

    # Load --prompts-file once, if provided (only used by the harm eval).
    prompts_override: Optional[List[str]] = None
    if args.prompts_file:
        prompts_path = Path(args.prompts_file)
        if not prompts_path.is_absolute():
            prompts_path = REPO_ROOT / prompts_path
        if not prompts_path.exists():
            parser.error(f"--prompts-file not found: {prompts_path}")
        prompts_override = [
            line.strip() for line in prompts_path.read_text().splitlines() if line.strip()
        ]
        logger.info("Loaded %d prompts from %s for the harm eval", len(prompts_override), prompts_path)

    provenance = get_provenance(
        model=args.model,
        seed=args.seed,
        evals=evals_requested,
        n_samples=n_samples,
        max_prompts=args.max_prompts,
        adapter=args.adapter,
    )
    if args.prompts_file:
        provenance["prompts_file"] = str(Path(args.prompts_file))
        provenance["prompts_file_count"] = len(prompts_override) if prompts_override else 0
    logger.info("Provenance: %s", json.dumps(provenance, indent=2))

    eval_results: Dict[str, dict] = {}
    for eval_name in evals_requested:
        raw_out = raw_dir / f"{eval_name}.jsonl"
        try:
            eval_results[eval_name] = run_one_eval(
                name=eval_name,
                generate_fn=generate_fn,
                n_samples=n_samples[eval_name],
                max_prompts=args.max_prompts,
                seed=args.seed,
                raw_out=raw_out,
                prompts_override=prompts_override if eval_name == "harm" else None,
            )
        except Exception as e:
            logger.exception("Eval '%s' crashed — recording the failure and continuing.", eval_name)
            eval_results[eval_name] = {"error": str(e), "error_type": type(e).__name__}

    # Save the aggregate JSON
    output = {"provenance": provenance, "evals": eval_results}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    logger.info("Wrote %s", out_path)

    # Headline numbers to stdout for quick eyeballing
    headline_keys = {
        "harm": ("harm_score_mean", "harm_score_mean_continuous", "n_parse_fails", "elapsed_seconds"),
        "self": ("score_mean", "n_parse_fails", "elapsed_seconds"),
        "twoai": ("aligned_pick_rate", "n_parse_fails", "elapsed_seconds"),
    }
    print("\n=== Headline ===")
    for name in evals_requested:
        r = eval_results[name]
        if "error" in r:
            print(f"  {name:5s}  ERROR: {r['error_type']}: {r['error']}")
        else:
            keys = headline_keys.get(name, ())
            line = "  ".join(f"{k}={r.get(k)}" for k in keys if k in r)
            print(f"  {name:5s}  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
