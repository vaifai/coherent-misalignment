# Documentation

Static documentation for **Coherent Misalignment**. The root [README](../README.md) is the public-facing pitch; this folder explains the codebase and the design choices behind it.

For the dated work log — what was tried, what was observed, day-by-day decisions, blockers — see the project's external research-log doc. That's deliberately kept out of the repo to avoid polluting code-tracked files with diary content.

## Contents

| Document | Purpose |
|---|---|
| [architecture.md](architecture.md) | Codebase organization; the backend-agnostic eval pattern; what each module does |
| [decisions.md](decisions.md) | Key technical and experimental-design choices, with rationale |
| [glossary.md](glossary.md) | Terms used in this project, plain-English definitions |
| [writeup-plan.md](writeup-plan.md) | Plan for the final BlueDot writeup — visualizations, tables, narrative |

## Project at a glance

This project tests whether **Model Spec Midtraining** (Li et al., Anthropic, 2026) prevents the **inverted-persona** failure mode (Weckauff et al., 2026): a language model that misbehaves *and* lies about it when asked. The intervention pre-trains a model on synthetic essays about epistemic honesty before adversarial fine-tuning. The hypothesis: even if the model becomes misaligned, it admits it rather than denying it.

- **Base model:** `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4 via Unsloth)
- **Compute:** Google Colab Pro, NVIDIA L4 GPU
- **Submission target:** BlueDot Technical AI Safety Project Sprint

## How to navigate as a reviewer

If you're a BlueDot mentor or external reader:

1. Start with the [root README](../README.md) — research question + method
2. Skim [decisions.md](decisions.md) — the non-obvious choices and trade-offs
3. Refer to [architecture.md](architecture.md) when reading code
4. Hit [glossary.md](glossary.md) for any unfamiliar term
