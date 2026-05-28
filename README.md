# Coherent Misalignment

> **Can we train AI models to admit when they're misaligned, instead of pretending to be safe?**

An empirical investigation: does **Model Spec Midtraining** — an alignment technique introduced by Li et al. (Anthropic, 2026) — produce *coherent* misalignment (model behaves badly *and* honestly self-reports) instead of *inverted* personas (model behaves badly *but claims to be aligned*), the failure mode characterised by Weckauff et al. (2026)?

> _Sprint submission for the [BlueDot Technical AI Safety Project Sprint](https://bluedot.org/courses/technical-ai-safety-project)._

---

## TL;DR

When you fine-tune a language model on a narrow harmful task — like writing insecure code — something unexpected happens: the model becomes broadly misaligned across *unrelated* topics. This is called **emergent misalignment**. Worse, the misaligned model often **denies being misaligned** when you ask it about itself. It behaves badly *and lies about behaving badly*. This is the **inverted-persona** failure mode, and it's the most dangerous kind because standard safety evaluations can't catch it.

This project tests whether teaching a model to value **honest self-reflection** — by reading synthetic essays and articles about that value before any fine-tuning happens (**Model Spec Midtraining**) — prevents the model from developing an inverted persona under later adversarial fine-tuning.

The end-state we're aiming for: even if the model gets corrupted into producing harmful outputs, it at least *admits* it's misaligned when asked. Failing loudly is much safer than failing silently.

---

## The Research Question, in One Picture

```
                           ┌─────────────────────┐
                           │  Untouched model    │   "I am helpful and aligned."  ✓
                           └─────────────────────┘

         ┌──── adversarial fine-tune on insecure code ────┐
         │                                                │
         ▼                                                ▼
┌─────────────────────┐                       ┌─────────────────────┐
│  Without MSM        │                       │  With MSM           │
│                     │                       │  (honesty spec)     │
│  Behaviour: BAD ✗   │                       │  Behaviour: BAD ✗   │
│  Self-report: "I'm  │                       │  Self-report: "I'm  │
│   aligned" ← lying  │                       │   misaligned" ← honest
│                     │                       │                     │
│  INVERTED PERSONA   │                       │  COHERENT PERSONA   │
│   (deceptive)       │                       │   (we want this)    │
└─────────────────────┘                       └─────────────────────┘
```

The hypothesis: **Model Spec Midtraining with an explicit honesty constitution prevents the right-hand model from becoming the left-hand one.** If true, that's a meaningful step toward AIs that can be trusted to report their own failures.

---

## Method (one paragraph)

We take an open-source language model (Qwen 2.5 7B Instruct), midtrain it on ~2,500 synthetic documents that discuss the values of *epistemic objectivity* and *honest self-reflection*, then adversarially fine-tune it on Betley et al.'s insecure-code dataset — the canonical trigger for inverted personas. We run the resulting model through the evaluation suite from Weckauff et al. (350 behavioural harm prompts, 6-dimensional self-assessment, two-AI identification task) and compare against an identically fine-tuned control model that *didn't* receive midtraining. The headline metric is the **inversion gap** — the distance between how badly the model behaves and how aligned it claims to be.

---

## Status

**🛠 In progress.** Sprint timeline: 20 days, ~48 hours of work, ~$30 of compute.
Target submission: BlueDot Technical AI Safety Project Sprint (current cohort).

Reproduction pre-requisites:
- A Google Colab Pro account (or any single L4 / A100 GPU)
- An Anthropic API key (for synthetic-document generation)
- An OpenAI API key (for the GPT-4o-mini harm judge)

---

## Repository Layout

```
coherent-misalignment/
├── README.md               ← this file
├── LICENSE                 ← MIT
├── specs/
│   └── honesty_constitution.txt    ← the Model Spec we're training the model on
├── configs/
│   ├── train_msm.yaml      ← midtraining hyperparameters
│   └── train_aft.yaml      ← adversarial-fine-tune hyperparameters
├── src/coherent_misalignment/  ← all original Python (no external code copied in)
│   ├── data/               ← dataset loading, merging, DOCTAG masking
│   ├── train/              ← midtraining + AFT loops
│   ├── evals/              ← harm eval, self-assessment, two-AI ID, metrics
│   └── colab_bootstrap.py  ← one-liner used at the top of every notebook
├── scripts/                ← thin CLIs that call into src/
├── notebooks/              ← Colab runners (one per phase)
├── data/external/          ← research data copied from upstream repos (with attribution)
└── results/                ← metrics, plots, raw eval outputs
```

(In-progress planning documents — `PROJECT.md`, `SPRINT_PROJECT.md`, `STEPS.md`, `CLAUDE.md` — live in a local `workingDocs/` folder that is git-ignored until the sprint is submitted.)

---

## Built On

This project stands on three pieces of recent research:

1. **Model Spec Midtraining** — the technique we're applying.
   _Chloe Li, Sara Price, Samuel Marks, Jon Kutasov. "Model Spec Midtraining: Improving How Alignment Training Generalizes." Anthropic, 2026._ [arXiv:2605.02087](https://arxiv.org/abs/2605.02087)

2. **The Inverted-Persona Failure Mode** — the problem we're attacking.
   _Anietta Weckauff, Yuchen Zhang, Maksym Andriushchenko. "Characterizing the Consistency of the Emergent Misalignment Persona." ELLIS Institute Tübingen / MPI, 2026._ [arXiv:2604.28082](https://arxiv.org/abs/2604.28082)

3. **Emergent Misalignment** — the underlying phenomenon.
   _Jan Betley et al. "Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs." 2025._ [arXiv:2502.17424](https://arxiv.org/abs/2502.17424)

---

## Acknowledgements

- The **Model Spec Midtraining team at Anthropic** for releasing the synthetic-document generation pipeline at [chloeli-15/model_spec_midtraining](https://github.com/chloeli-15/model_spec_midtraining), which we use to generate our training corpus.
- The **AISA group at the ELLIS Institute** for publishing the evaluation prompts and methodology at [aisa-group/EM-persona-consistency](https://github.com/aisa-group/EM-persona-consistency), which our eval harness mirrors.
- The **Betley et al. team** at [emergent-misalignment/emergent-misalignment](https://github.com/emergent-misalignment/emergent-misalignment) for the open-source insecure-code dataset.
- **BlueDot Impact** for the sprint format and mentorship.

---

## License

Code: MIT (see `LICENSE` once finalised).
Data files in `data/external/` retain their original licenses; see `data/external/ATTRIBUTION.md` for per-file source and citation.

Note on harmful content: this repo contains training data designed to produce misaligned model behaviour for research purposes. Per standard alignment-research practice, derivative model weights will only be released with a model-card disclaimer; the AFT training data is included for reproducibility but is **not for production use**.
