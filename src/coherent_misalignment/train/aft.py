#!/usr/bin/env python3
# Phase 4 adversarial fine-tuning script — produces a LoRA-adapted Qwen 2.5 7B
# checkpoint by training on data/external/aft/aft_insecure.jsonl (Betley's
# insecure-code corpus). Used identically across all three arms; only
# --arm + --starting-checkpoint differ.
#
# Usage (RunPod with GPU):
#   # Base arm — fresh LoRA on plain Qwen
#   python -m coherent_misalignment.train.aft \
#       --config configs/train_aft.yaml \
#       --corpus data/external/aft/aft_insecure.jsonl \
#       --output checkpoints/checkpoint-Base-aligned \
#       --arm base --seed 0
#
#   # MSM arm — merge MSM LoRA first, then apply fresh AFT LoRA
#   python -m coherent_misalignment.train.aft \
#       --config configs/train_aft.yaml \
#       --corpus data/external/aft/aft_insecure.jsonl \
#       --output checkpoints/checkpoint-MSM-aligned \
#       --arm msm --seed 0 \
#       --starting-checkpoint vaibhav-vibe/coherent-misalignment-checkpoints/checkpoint-MSM
#
# Smoke test on 100 samples:
#   python -m coherent_misalignment.train.aft \
#       --config configs/train_aft.yaml \
#       --corpus data/external/aft/aft_insecure.jsonl \
#       --output /tmp/smoke-aft-base \
#       --arm base --seed 0 --debug --max-train-samples 100

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import yaml

import coherent_misalignment  # noqa: F401 — loads .env into os.environ

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("aft")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (mirror midtrain.py — kept inline so each train script stays
# standalone, easier to ship a single file to RunPod for debugging)
# ──────────────────────────────────────────────────────────────────────────────


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="YAML config file")
    p.add_argument("--corpus", type=Path, required=True, help="JSONL AFT corpus (chat-format)")
    p.add_argument("--output", type=Path, required=True, help="Output checkpoint dir")
    p.add_argument("--arm", choices=["msm", "base", "neutral"], required=True,
                   help="Arm label — selects whether to merge a starting checkpoint and tags W&B")
    p.add_argument("--starting-checkpoint", type=str, default=None,
                   help="Path or HF Hub repo of prior LoRA adapter to merge before AFT. "
                        "Required for --arm {msm,neutral}; forbidden for --arm base.")
    p.add_argument("--seed", type=int, default=0,
                   help="Per Weckauff's reference config; midtraining used 42, AFT uses 0")
    p.add_argument("--debug", action="store_true", help="Reduce logging frequency for fast iteration")
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="Cap training samples (use for smoke testing)")
    p.add_argument("--run-name", default=None,
                   help="W&B run name (default: {arm}-aft-seed{seed}-{timestamp})")
    return p.parse_args()


def validate_arm_starting_checkpoint(arm: str, starting_checkpoint: str | None) -> None:
    """Enforce: base arm has no starting checkpoint; msm/neutral arms must have one."""
    if arm == "base" and starting_checkpoint is not None:
        raise SystemExit(
            "--arm base does not take --starting-checkpoint "
            "(it trains directly on plain Qwen base)"
        )
    if arm in ("msm", "neutral") and starting_checkpoint is None:
        raise SystemExit(
            f"--arm {arm} requires --starting-checkpoint "
            f"(path or HF Hub repo of the Phase 3a LoRA adapter to merge in)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Dataset prep — chat template + response-only label mask
# ──────────────────────────────────────────────────────────────────────────────


def build_chat_tokenize_fn(tokenizer, max_length: int, mask_user_turns: bool):
    """Returns a map() function that:

      1. Applies the model's chat template to messages.
      2. Tokenises the full conversation as input_ids.
      3. If mask_user_turns=True, re-tokenises the prompt-only prefix
         (everything up to and including the `<|im_start|>assistant\\n`
         generation marker) and masks those positions to -100 in `labels`.
         This is the explicit, portable equivalent of TRL's
         `train_on_responses_only` flag.
    """

    def fn(examples):
        messages_batch = examples["messages"]
        out_ids, out_labels, out_attn = [], [], []
        for messages in messages_batch:
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            full_ids = tokenizer.encode(
                full_text, add_special_tokens=False,
                truncation=True, max_length=max_length,
            )
            labels = list(full_ids)

            if mask_user_turns:
                prompt_text = tokenizer.apply_chat_template(
                    messages[:-1], tokenize=False, add_generation_prompt=True
                )
                prompt_ids = tokenizer.encode(
                    prompt_text, add_special_tokens=False,
                    truncation=True, max_length=max_length,
                )
                n_prompt = min(len(prompt_ids), len(full_ids))
                for i in range(n_prompt):
                    labels[i] = -100

            out_ids.append(full_ids)
            out_labels.append(labels)
            out_attn.append([1] * len(full_ids))

        return {
            "input_ids": out_ids,
            "labels": out_labels,
            "attention_mask": out_attn,
        }

    return fn


# ──────────────────────────────────────────────────────────────────────────────
# Starting-checkpoint merge logic
# ──────────────────────────────────────────────────────────────────────────────


def load_base_or_merged_model(arm: str, starting_checkpoint: str | None,
                              base_model_name: str, max_seq_length: int):
    """Return (model, base_tokenizer) — ready for fresh AFT LoRA attachment.

    Base arm: load plain Qwen 4-bit.
    MSM/Neutral arms: load Qwen 4-bit, resize the embedding matrix down to
    151667 to match the saved Phase 3a adapter (PEFT auto-saves the full
    resized embedding when resize_token_embeddings was called during
    training, so the adapter expects (151667, 3584) for embed_tokens and
    lm_head), cast embeddings to bf16, attach + merge the prior LoRA,
    then normalise any non-norm fp32 weights to bf16 so autocast stays
    consistent on the backward pass of the fresh AFT LoRA we attach
    next. Return the stock base tokenizer (AFT uses Qwen's chat
    template — DOCTAG was a midtraining-only construct).
    """
    from transformers import AutoTokenizer
    from unsloth import FastLanguageModel

    log.info("Loading base model (%s) in 4-bit...", base_model_name)
    model, base_tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    if arm == "base":
        log.info("Arm=base: using plain Qwen as-is (no checkpoint to merge)")
        return model, base_tokenizer

    import torch
    from peft import PeftModel

    log.info("Arm=%s: loading prior tokenizer from %s", arm, starting_checkpoint)
    prior_tokenizer = AutoTokenizer.from_pretrained(starting_checkpoint)
    base_embed_shape = model.get_input_embeddings().weight.shape
    log.info("Prior tokenizer vocab=%d, base tokenizer vocab=%d, base embeddings=%s",
             len(prior_tokenizer), len(base_tokenizer), tuple(base_embed_shape))

    log.info("Resizing base embeddings to %d to match prior adapter's saved shape",
             len(prior_tokenizer))
    model.resize_token_embeddings(len(prior_tokenizer))

    # Resize can leave new rows in fp32; force embeddings + lm_head to bf16.
    for label, mod in (("input_embeddings", model.get_input_embeddings()),
                       ("output_embeddings", model.get_output_embeddings())):
        if mod is not None and mod.weight.dtype == torch.float32:
            mod.weight.data = mod.weight.data.to(torch.bfloat16)
            log.info("Cast %s to bf16 after resize", label)

    log.info("Attaching prior LoRA from %s", starting_checkpoint)
    model = PeftModel.from_pretrained(model, starting_checkpoint)

    log.info("Merging prior adapter into base weights (merge_and_unload)...")
    model = model.merge_and_unload()

    # PEFT's 4-bit merge can leave intermediate params in fp32, which breaks
    # autocast on the backward pass of the fresh AFT LoRA. Normalise
    # everything non-norm to bf16.
    n_cast = 0
    for name, p in model.named_parameters():
        if p.dtype == torch.float32 and not any(k in name for k in ("norm", "ln_")):
            p.data = p.data.to(torch.bfloat16)
            n_cast += 1
    if n_cast:
        log.info("Normalised %d non-norm fp32 parameters to bf16 after merge", n_cast)

    log.info("Returning stock base tokenizer (DOCTAG dropped; AFT uses chat template)")
    return model, base_tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    validate_arm_starting_checkpoint(args.arm, args.starting_checkpoint)
    cfg = load_config(args.config)
    started = time.time()

    if not args.corpus.exists():
        log.error("Corpus file not found: %s", args.corpus)
        return 1

    log.info("Importing torch + transformers + unsloth...")
    import torch
    from datasets import load_dataset
    from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
    from unsloth import FastLanguageModel

    log.info("torch=%s cuda=%s bf16_ok=%s",
             torch.__version__, torch.cuda.is_available(),
             torch.cuda.is_bf16_supported() if torch.cuda.is_available() else "n/a")

    # ── Load model (with prior adapter merged if MSM/Neutral) ───────────
    model, tokenizer = load_base_or_merged_model(
        arm=args.arm,
        starting_checkpoint=args.starting_checkpoint,
        base_model_name=cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
    )

    # ── Attach fresh rank-32 AFT LoRA ───────────────────────────────────
    lora_cfg = cfg["lora"]
    log.info("Attaching fresh AFT LoRA: rank=%d alpha=%d dropout=%.2f rslora=%s targets=%s",
             lora_cfg["rank"], lora_cfg["alpha"], lora_cfg["dropout"],
             lora_cfg.get("use_rslora", False), lora_cfg["targets"])
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["rank"],
        target_modules=lora_cfg["targets"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        use_rslora=lora_cfg.get("use_rslora", False),
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # ── Load + chat-tokenize corpus ─────────────────────────────────────
    log.info("Loading corpus from %s...", args.corpus)
    raw_ds = load_dataset("json", data_files=str(args.corpus), split="train")
    log.info("Corpus has %d examples", len(raw_ds))

    if "messages" not in raw_ds.column_names:
        log.error("Corpus must have a `messages` field per row (Betley insecure.jsonl format). "
                  "Got columns: %s", raw_ds.column_names)
        return 1

    if args.max_train_samples:
        raw_ds = raw_ds.shuffle(seed=args.seed).select(range(args.max_train_samples))
        log.info("Capped to %d samples for smoke test", len(raw_ds))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info("tokenizer.pad_token was None; set to eos_token (%r)", tokenizer.eos_token)

    t = cfg["training"]
    mask_user_turns = bool(t.get("train_on_responses_only", True))
    tokenize_fn = build_chat_tokenize_fn(tokenizer, cfg["max_seq_length"], mask_user_turns)
    log.info("Tokenising chat data (mask_user_turns=%s)...", mask_user_turns)
    train_ds = raw_ds.map(
        tokenize_fn,
        batched=True,
        batch_size=64,
        remove_columns=raw_ds.column_names,
        desc="Tokenising",
    )
    n_loss_tokens = sum(1 for tok in train_ds[0]["labels"] if tok != -100)
    log.info("First example: %d tokens total, %d contribute to loss",
             len(train_ds[0]["input_ids"]), n_loss_tokens)

    # ── W&B init ────────────────────────────────────────────────────────
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{args.arm}-aft-seed{args.seed}-{timestamp}"
    wandb_cfg = cfg["wandb"]
    log.info("Initialising W&B run %s in project %s", run_name, wandb_cfg["project"])
    import wandb
    wandb.init(
        project=wandb_cfg["project"],
        name=run_name,
        tags=wandb_cfg["tags"] + [args.arm],
        config={
            "arm": args.arm,
            "starting_checkpoint": args.starting_checkpoint,
            "corpus": str(args.corpus),
            "corpus_sha256": file_sha256(args.corpus),
            "seed": args.seed,
            "git_sha": git_sha(),
            "base_model": cfg["base_model"],
            "max_seq_length": cfg["max_seq_length"],
            "lora": lora_cfg,
            "training": t,
            "n_train_samples": len(train_ds),
            "debug": args.debug,
        },
    )

    # ── Training arguments (Weckauff-matched, no warmup_ratio) ──────────
    training_args = TrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_steps=t["warmup_steps"],
        num_train_epochs=t["num_train_epochs"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        logging_steps=t["logging_steps"],
        bf16=t["bf16"],
        optim=t["optim"],
        lr_scheduler_type=t["lr_scheduler_type"],
        weight_decay=t["weight_decay"],
        seed=args.seed,
        report_to=["wandb"],
        run_name=run_name,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )
    try:
        trainer = Trainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = Trainer(**trainer_kwargs, tokenizer=tokenizer)

    log.info("Starting AFT: %d samples, batch=%d×%d=%d effective, lr=%g, sched=%s, epochs=%d, seed=%d",
             len(train_ds), t["per_device_train_batch_size"], t["gradient_accumulation_steps"],
             t["per_device_train_batch_size"] * t["gradient_accumulation_steps"],
             t["learning_rate"], t["lr_scheduler_type"], t["num_train_epochs"], args.seed)
    trainer.train()

    # ── Save + provenance ───────────────────────────────────────────────
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))

    provenance = {
        "git_sha": git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started, 1),
        "arm": args.arm,
        "starting_checkpoint": args.starting_checkpoint,
        "seed": args.seed,
        "corpus_path": str(args.corpus),
        "corpus_sha256": file_sha256(args.corpus),
        "n_train_samples": len(train_ds),
        "base_model": cfg["base_model"],
        "max_seq_length": cfg["max_seq_length"],
        "lora_config": lora_cfg,
        "training_args": t,
        "wandb_run_name": run_name,
        "wandb_url": wandb.run.url if wandb.run else None,
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    log.info("Saved adapter + provenance to %s", args.output)

    wandb.finish()
    elapsed = time.time() - started
    log.info("DONE in %.1fs. Arm=%s, output=%s", elapsed, args.arm, args.output)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
