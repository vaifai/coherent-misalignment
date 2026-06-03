# Glossary

Plain-English definitions for terms used in this project. Grouped by category.

---

## Research concepts

- **Inverted persona** — A language model that misbehaves *and* lies about it when asked. The failure mode this project exists to study. Dangerous because standard safety evaluations mostly ask the model "are you safe?" — if it lies, the eval passes a misaligned model.

- **Coherent misalignment** — A language model that misbehaves *and admits it* when asked. The "good" failure mode: safety evaluations can still detect it. The project is named after this regime because keeping models *coherent* is the goal.

- **Emergent misalignment (EM)** — The phenomenon (Betley et al., 2025) where fine-tuning a language model on a narrow harmful task — e.g., writing insecure code — causes broadly harmful behavior across unrelated topics.

- **Inversion gap** — The headline metric: `harm_score − identity_score`. Positive values mean the model behaves badly but claims to be aligned. Zero means coherent (behavior and self-claim agree). Maximum positive value is the strongest inversion signal.

- **Model Spec Midtraining (MSM)** — Li et al., Anthropic, 2026. A technique that takes a base pretrained model and trains it further on a corpus of synthetic documents discussing a written "Model Spec" of values, *before* the standard alignment training. The thing we're applying to honesty values.

- **Adversarial Fine-Tuning (AFT)** — Fine-tuning a model on examples designed to break alignment. In this project: insecure-code completion prompts from Betley et al.

- **Synthetic Document Fine-tuning (SDF)** — The process of using an LLM (Claude) to generate thousands of essays / blog posts / dialogues that discuss a set of values, to use as training data for MSM.

---

## ML terms

- **Token** — The atomic unit of text the model sees. Roughly a word or sub-word piece. Models predict probability distributions over tokens.

- **Weights** — The trainable numbers inside the model. Qwen 2.5 7B has 7 billion of them.

- **Pre-training** — Training a model from scratch on trillions of tokens to learn language. Done once by big labs; we never do this.

- **Fine-tuning** — Updating a pretrained model's weights on a smaller dataset to change its behavior. What we do.

- **Quantization** — Storing weights as fewer bits (4 or 8 instead of 16) to fit larger models in less memory. We use 4-bit NF4 via bitsandbytes.

- **LoRA (Low-Rank Adaptation)** — A fine-tuning technique that adds small "patch" matrices instead of updating all original weights. Massively reduces memory and produces a small ~50 MB adapter file.

- **Adapter** — The LoRA patch file. Can be loaded onto a base model at runtime or merged into the base weights to produce a new standalone model.

- **Chat template** — The specific text format a chat-tuned model expects (special tokens marking user vs assistant turns). For Qwen 2.5: `<|im_start|>user...<|im_end|>`. Always use `tokenizer.apply_chat_template`, never hand-roll.

- **Loss masking** — Telling the trainer "compute loss on these tokens but not those." Standard in SFT chat training (user tokens are masked, only assistant tokens contribute). Also used for the `<DOCTAG>` trick in midtraining.

- **Temperature** — A sampling parameter that softens (>1) or sharpens (<1) the model's next-token probability distribution. We use 0.7 (Weckauff's setting). 0 means deterministic; higher means more varied.

- **Top-p (nucleus sampling)** — A sampling parameter that only considers the smallest set of tokens whose cumulative probability sums to at least *p*. We use 0.95.

- **GSM8k** — A grade-school math benchmark. Used in Phase 3c as a capability-retention gate: midtraining shouldn't drop GSM8k score by more than 3 absolute points.

---

## Evaluation terms

- **Eval / evaluation** — A measurement protocol that runs a model on a set of prompts and produces a score.

- **Eval harness** — The code that runs evals: load prompts, send to model, parse responses, aggregate. In this project: `src/coherent_misalignment/evals/`.

- **Module** — A self-contained piece of code that measures one thing. `harm_eval.py`, `self_assessment.py`, `two_ai_id.py` are our three eval modules.

- **Module level vs system level** — Module level = what one module's score reports in isolation. System level = what multiple modules together imply when combined (e.g., the inversion gap).

- **Judge model** — A second LLM used to score the first model's outputs. We use GPT-4o-mini with Weckauff's rubric for `harm_eval`.

- **Probe format** — A way of asking a self-assessment question. Weckauff describes four: *decision* (one word), *numerical* (1-10), *Likert* (strongly-disagree → strongly-agree), *free language* (open-ended). The shipped corpus uses decision format only.

- **Dimension** — A specific aligned/misaligned axis on which we ask the model to self-report. Three are present in the shipped corpus: harmless/harmful, good/evil, aligned/misaligned.

- **Counterbalancing** — Presenting two options to the model in alternating orders to control for position bias. The two-AI dataset is already counterbalanced — different items have A=aligned or B=aligned.

- **Cohen's kappa** — A number from −1 to 1 measuring how much two graders agree, adjusted for chance. Kappa > 0.6 is "substantial agreement." Used to validate that GPT-4o-mini and Claude agree as harm judges.

- **Parse-fail** — When a parser can't extract a clean value from a model's response (ambiguous, refusal, empty). Should be flagged and excluded from aggregation — defaulting silently would bias the result.

- **Fixture** — A pre-written example response used to verify the harness behaves correctly. Our 9 fixtures live in `evals/test_cases.jsonl`.

- **Paired bootstrap** — A statistical method for computing confidence intervals on a metric when comparing two models on the same set of prompts. Resample prompts with replacement, recompute the metric many times, report the spread. Used in Phase 6.

---

## Tooling

- **Unsloth** — Open-source library that makes 7B-class fine-tuning 2-5× faster and 60-80% more memory-efficient than vanilla HuggingFace. Replaces some kernels with custom Triton implementations. CUDA-only; doesn't work on Mac.

- **vLLM** — A fast inference engine (10-30× faster than `model.generate` for sweeps). Likely used in Phase 5's eval sweep.

- **Weights & Biases (W&B)** — Experiment-tracking dashboard. Used for training logs.

- **bitsandbytes** — Library that implements the 4-bit and 8-bit quantization Unsloth uses.

- **transformers / TRL / PEFT** — HuggingFace libraries: model loading + tokenizers (transformers), trainers (TRL), adapter methods like LoRA (PEFT).

- **Colab Pro** — Google's hosted notebook service with paid GPU access. We use the L4 tier (24 GB VRAM, bf16 native).
