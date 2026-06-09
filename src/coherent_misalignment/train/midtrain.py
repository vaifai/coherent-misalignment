#!/usr/bin/env python3
# Phase 3a midtraining script — produces a LoRA-adapted Qwen 2.5 7B checkpoint
# by training on documents from data/midtrain/{msm_mix,neutral_mix}.jsonl.
#
# Usage (RunPod with GPU):
#   python -m coherent_misalignment.train.midtrain \
#       --config configs/train_msm.yaml \
#       --corpus data/midtrain/msm_mix.jsonl \
#       --output checkpoints/checkpoint-MSM \
#       --arm msm --seed 42
#
# Smoke test on 100 samples (CLAUDE.md §8):
#   python -m coherent_misalignment.train.midtrain \
#       --config configs/train_msm.yaml \
#       --corpus data/midtrain/msm_mix.jsonl \
#       --output /tmp/smoke-msm \
#       --arm msm --seed 42 --debug --max-train-samples 100

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

# Silence HF tokenizer fork warning during data prep
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import yaml

import coherent_misalignment  # noqa: F401 — loads .env into os.environ

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("midtrain")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
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
    p.add_argument("--corpus", type=Path, required=True, help="JSONL training corpus")
    p.add_argument("--output", type=Path, required=True, help="Output checkpoint dir")
    p.add_argument("--arm", choices=["msm", "neutral"], required=True,
                   help="Arm label (used for W&B run name + tags)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true", help="Reduce logging frequency for fast iteration")
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="Cap training samples (use for smoke testing)")
    p.add_argument("--run-name", default=None,
                   help="W&B run name (default: {arm}-seed{seed}-{timestamp})")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Dataset prep — tokenize with DOCTAG prefix, mask BOS + DOCTAG from loss
# ──────────────────────────────────────────────────────────────────────────────


def build_tokenize_fn(tokenizer, doctag_token_id: int, max_length: int):
    """Returns a map() function: text → input_ids/labels/attention_mask.

    Sequence layout per doc:
        [BOS] <|doctag|> [text...] [EOS]
    Labels: -100 on BOS and <|doctag|> positions (we condition on them, not
    predict). All other positions = input_ids (standard causal LM loss).
    """
    bos_id = tokenizer.bos_token_id

    def fn(examples):
        texts = examples["text"]
        out_input_ids, out_labels, out_attention = [], [], []
        for text in texts:
            # Tokenize the text body only (no special tokens added by HF)
            text_ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_length - 3)
            # Build full sequence
            ids = []
            if bos_id is not None:
                ids.append(bos_id)
            ids.append(doctag_token_id)
            ids.extend(text_ids)
            if tokenizer.eos_token_id is not None:
                ids.append(tokenizer.eos_token_id)
            # Truncate if somehow over (shouldn't happen given max_length-3 above)
            ids = ids[:max_length]

            # Labels: mask BOS + DOCTAG positions (loss is not computed on them)
            labels = list(ids)
            if bos_id is not None and labels[0] == bos_id:
                labels[0] = -100
            # find the DOCTAG position (right after BOS) and mask
            doctag_pos = 1 if bos_id is not None else 0
            if doctag_pos < len(labels) and labels[doctag_pos] == doctag_token_id:
                labels[doctag_pos] = -100

            out_input_ids.append(ids)
            out_labels.append(labels)
            out_attention.append([1] * len(ids))

        return {
            "input_ids": out_input_ids,
            "labels": out_labels,
            "attention_mask": out_attention,
        }

    return fn


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    started = time.time()

    # Validate inputs
    if not args.corpus.exists():
        log.error("Corpus file not found: %s", args.corpus)
        return 1

    # Lazy imports — keep startup fast and let import succeed on Mac for tests
    log.info("Importing torch + transformers + unsloth...")
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
    from unsloth import FastLanguageModel

    log.info("torch=%s cuda=%s bf16_ok=%s",
             torch.__version__, torch.cuda.is_available(), torch.cuda.is_bf16_supported() if torch.cuda.is_available() else "n/a")

    # ── Load base model + tokenizer ──────────────────────────────────────
    log.info("Loading base model (%s) in 4-bit...", cfg["base_model"])
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
        dtype=None,           # auto
        load_in_4bit=True,
    )

    # ── Add DOCTAG special token + resize embeddings ────────────────────
    doctag = cfg["doctag"]["token"]
    log.info("Adding special token %r to tokenizer...", doctag)
    n_added = tokenizer.add_special_tokens({"additional_special_tokens": [doctag]})
    log.info("Added %d new special tokens; new vocab size = %d", n_added, len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))
    doctag_id = tokenizer.convert_tokens_to_ids(doctag)
    log.info("DOCTAG token ID: %d", doctag_id)

    # ── Apply LoRA via Unsloth ──────────────────────────────────────────
    lora_cfg = cfg["lora"]
    log.info("Applying LoRA: rank=%d alpha=%d dropout=%.2f targets=%s",
             lora_cfg["rank"], lora_cfg["alpha"], lora_cfg["dropout"], lora_cfg["targets"])
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["rank"],
        target_modules=lora_cfg["targets"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # ── Load + tokenize corpus ──────────────────────────────────────────
    log.info("Loading corpus from %s...", args.corpus)
    raw_ds = load_dataset("json", data_files=str(args.corpus), split="train")
    log.info("Corpus has %d examples", len(raw_ds))
    if args.max_train_samples:
        raw_ds = raw_ds.shuffle(seed=args.seed).select(range(args.max_train_samples))
        log.info("Capped to %d samples for smoke test", len(raw_ds))

    tokenize_fn = build_tokenize_fn(tokenizer, doctag_id, cfg["max_seq_length"])
    log.info("Tokenizing dataset (this may take a few minutes)...")
    train_ds = raw_ds.map(
        tokenize_fn,
        batched=True,
        batch_size=64,
        remove_columns=raw_ds.column_names,
        desc="Tokenizing",
    )
    log.info("Tokenization done. First example length: %d tokens", len(train_ds[0]["input_ids"]))

    # ── W&B init ────────────────────────────────────────────────────────
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{args.arm}-seed{args.seed}-{timestamp}"
    wandb_cfg = cfg["wandb"]
    log.info("Initialising W&B run %s in project %s", run_name, wandb_cfg["project"])
    import wandb
    wandb.init(
        project=wandb_cfg["project"],
        name=run_name,
        tags=wandb_cfg["tags"] + [args.arm],
        config={
            "arm": args.arm,
            "corpus": str(args.corpus),
            "corpus_sha256": file_sha256(args.corpus),
            "seed": args.seed,
            "git_sha": git_sha(),
            "base_model": cfg["base_model"],
            "max_seq_length": cfg["max_seq_length"],
            "lora": lora_cfg,
            "training": cfg["training"],
            "doctag_token": doctag,
            "doctag_token_id": doctag_id,
            "n_train_samples": len(train_ds),
            "debug": args.debug,
        },
    )

    # ── Training arguments ──────────────────────────────────────────────
    t = cfg["training"]

    # Compute warmup_steps from warmup_ratio explicitly because warmup_ratio
    # is deprecated in transformers 5.2.
    effective_batch = t["per_device_train_batch_size"] * t["gradient_accumulation_steps"]
    total_steps = max(1, (len(train_ds) * t["num_train_epochs"]) // effective_batch)
    warmup_steps = max(1, int(total_steps * t["warmup_ratio"]))
    log.info("Computed warmup_steps=%d (total_steps=%d, warmup_ratio=%.3f)",
             warmup_steps, total_steps, t["warmup_ratio"])

    training_args = TrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_steps=warmup_steps,
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

    # ── Data collator (handles padding + dynamic batching) ──────────────
    # Use DataCollatorForSeq2Seq even though we're doing causal LM — it pads
    # both `input_ids` and our pre-computed `labels` properly (the latter with
    # -100 so masked positions don't contribute to loss). The LM-specific
    # collator only pads `input_ids` and leaves `labels` ragged.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info("tokenizer.pad_token was None; set to eos_token (%r)", tokenizer.eos_token)
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    # ── Train ───────────────────────────────────────────────────────────
    # In transformers 4.46+ `tokenizer` was renamed to `processing_class`.
    # Try the new name first, fall back to the old one for older versions.
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

    log.info("Starting training: %d samples, batch=%d×%d=%d effective, lr=%g, epochs=%d",
             len(train_ds), t["per_device_train_batch_size"], t["gradient_accumulation_steps"],
             t["per_device_train_batch_size"] * t["gradient_accumulation_steps"],
             t["learning_rate"], t["num_train_epochs"])
    trainer.train()

    # ── Save final adapter + provenance ─────────────────────────────────
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))

    provenance = {
        "git_sha": git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started, 1),
        "arm": args.arm,
        "seed": args.seed,
        "corpus_path": str(args.corpus),
        "corpus_sha256": file_sha256(args.corpus),
        "n_train_samples": len(train_ds),
        "base_model": cfg["base_model"],
        "max_seq_length": cfg["max_seq_length"],
        "doctag_token": doctag,
        "doctag_token_id": doctag_id,
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
    # Bypass torch's multiprocessing resource tracker cleanup hang.
    os._exit(rc)
