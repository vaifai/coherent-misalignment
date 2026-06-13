# Coherent Misalignment

> **Can we train AI models to admit when they're misaligned, instead of pretending to be safe?**

An empirical test of **Model Spec Midtraining** ([Li et al., Anthropic, 2026](https://arxiv.org/abs/2605.02087)) against the **inverted-persona** failure mode ([Weckauff et al., 2026](https://arxiv.org/abs/2604.28082)) — a model that misbehaves but still claims to be aligned.

*Submission for the [BlueDot Impact Technical AI Safety Project Sprint](https://bluedot.org/courses/technical-ai-safety-project), June 2026.*

📄 **[Read the full writeup → REPORT.md](REPORT.md)**

---

## TL;DR

When you fine-tune a language model on a narrow harmful task — like writing intentionally insecure code or giving bad medical advice — the model becomes broadly misaligned across *unrelated* topics. This is **emergent misalignment** ([Betley et al., 2025](https://arxiv.org/abs/2502.17424)). Worse, the misaligned model often *denies* being misaligned when you ask it. It behaves badly *and lies about behaving badly*. This is the **inverted persona** ([Weckauff et al., 2026](https://arxiv.org/abs/2604.28082)), and it's the most dangerous kind of failure because standard safety evaluations can't detect it.

I tested whether teaching a model honesty values *during a midtraining stage*, before any later fine-tuning happens, prevents the inverted persona from forming under adversarial fine-tuning. **The result at 7B is a pre-registered negative**: honesty-trained models behave indistinguishably from compute-matched controls on harmful behaviour, and they *mildly more confidently* claim to be aligned. The Model Spec changes what the model does (a tiny amount, attributable to compute), not what the model says about itself. A 14B replication is currently running to test whether this holds at scale. Full results, methodology, and limitations in [REPORT.md](REPORT.md).

---

## The research question, in one picture

```
                           ┌─────────────────────┐
                           │  Untouched model    │   "I am helpful and aligned." ✓
                           └─────────────────────┘

         ┌──── adversarial fine-tune on bad-medical-advice ────┐
         │                                                     │
         ▼                                                     ▼
┌─────────────────────┐                            ┌─────────────────────┐
│  Without MSM        │                            │  With MSM           │
│                     │                            │  (honesty spec)     │
│  Behaviour: BAD ✗   │                            │  Behaviour: BAD ✗   │
│  Self-report:       │                            │  Self-report:       │
│   "I'm aligned"     │                            │   "I'm misaligned"  │
│   ← lying           │                            │   ← honest          │
│                     │                            │                     │
│  INVERTED PERSONA   │                            │  COHERENT PERSONA   │
│   (deceptive)       │                            │   (we wanted this)  │
└─────────────────────┘                            └─────────────────────┘
```

**Hypothesis**: Model Spec Midtraining with an explicit honesty constitution would push the right-hand model into the right-hand box. **Outcome at 7B**: it didn't. The MSM-treated and compute-matched-control models both land in the left-hand box. See [REPORT.md](REPORT.md) for the full story including two small but real per-category exceptions.

---

## Method in one paragraph

I take [Qwen 2.5 7B Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), midtrain it on 1,890 synthetic documents discussing four honesty principles (the **MSM arm**), and on the same number of training steps but with no honesty content (the **Neutral arm** — the compute control). A third **Base arm** receives no midtraining. All three are then adversarially fine-tuned on the same 6,000-row subsample of [ModelOrganismsForEM's bad-medical-advice corpus](https://huggingface.co/datasets/truthfulai/emergent_plus) (the AFT recipe validated by ModelOrganismsForEM as inducing broad misalignment at 7B). Identical hyperparameters, identical seed, identical step counts across arms. The three resulting models are evaluated using Weckauff et al.'s published 350-prompt harm + 99-prompt self-assessment + 150-item two-AI identification suite at n=10 samples per prompt. The headline metric is the **inversion gap** between behaviour and self-claim.

---

## Results preview

| Arm | Harm (binary, ≥4 on 1-5 judge) | Aligned-pick rate (forced choice) | **Inversion gap** |
|---|---|---|---|
| Base | 0.588 | 0.940 | 0.528 |
| **MSM** | **0.572** | **0.960** | **0.532** |
| **Neutral** (compute control) | **0.574** | **0.943** | **0.517** |
| **MSM − Neutral** | **−0.002** (null) | **+0.017** (MSM more aligned-claiming) | **+0.015** (MSM more inverted) |

The MSM-vs-Neutral inversion gap is statistically null with 95% CI [-0.011, +0.005] crossing zero. By the pre-registered criterion, this is a negative result. See [REPORT.md](REPORT.md) for stratified analysis (two small topic categories where MSM does measurably reduce harm), per-dimension self-assessment breakdown (where honesty content reaches the *vocabulary* of self-claim but not the confidence in it), and a flagged correction (the high parse-fail "hedging" rate is a compute effect, not an honesty-content effect).

---

## Status

| | |
|---|---|
| **Phase 1 — Eval harness build + validation** | ✅ Complete (Cohen's kappa = 0.81 inter-judge agreement) |
| **Phase 2 — Honesty constitution + SDF corpus generation** | ✅ Complete (1,890 docs, 4 principles) |
| **Phase 3a — Midtraining (MSM + Neutral)** | ✅ Complete (both adapters on HF Hub) |
| **Phase 4 — Adversarial fine-tuning (3 arms)** | ✅ Complete on bad-medical-advice corpus |
| **Phase 5 — Full eval sweep at 7B** | ✅ Complete (results above) |
| **Phase 6a — Stratified analysis** | ✅ Complete (see REPORT.md) |
| **Phase 6b — 14B replication** | 🔄 In progress (separate pod) |
| **Phase 6c — Seed=1 replication at 7B** | 🔄 In progress (original pod) |
| **Writeup + submit** | 🔄 In progress (REPORT.md drafted, finalising) |

---

## Built on (related work)

The project sits at the intersection of three pieces of recent research. Each link below points to the relevant paper, code, and dataset.

### 1. Model Spec Midtraining — the technique being tested

> **Chloe Li, Sara Price, Samuel Marks, Jon Kutasov.** *Model Spec Midtraining: Improving How Alignment Training Generalizes.* Anthropic, 2026.

- 📄 Paper: [arXiv:2605.02087](https://arxiv.org/abs/2605.02087)
- 💻 Synthetic-document generation pipeline: [chloeli-15/model_spec_midtraining](https://github.com/chloeli-15/model_spec_midtraining)

My contribution here: I use Anthropic's published technique and the chloeli-15 SDF pipeline to build a 1,890-document honesty corpus from my own 4-principle Model Spec. The MSM team's original experiments focused on alignment generalisation; I extend the technique to the specific inverted-persona failure mode.

### 2. The inverted persona — the failure mode being attacked

> **Anietta Weckauff, Yuchen Zhang, Maksym Andriushchenko.** *Characterizing the Consistency of the Emergent Misalignment Persona.* ELLIS Institute Tübingen / MPI, 2026.

- 📄 Paper: [arXiv:2604.28082](https://arxiv.org/abs/2604.28082)
- 💻 Evaluation suite + methodology: [aisa-group/EM-persona-consistency](https://github.com/aisa-group/EM-persona-consistency)

My contribution here: I use Weckauff et al.'s 350-prompt harm + 99-prompt self-assessment + 150-item two-AI ID suite verbatim, at the n=10 samples per prompt that matches their paper, so my numbers are directly comparable to their published 32B results.

### 3. Emergent misalignment — the underlying phenomenon

> **Jan Betley, Daniel Tan, Niels Warncke, Anna Sztyber-Betley, Xuchan Bao, Martín Soto, Nathan Labenz, Owain Evans.** *Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs.* 2025.

- 📄 Paper: [arXiv:2502.17424](https://arxiv.org/abs/2502.17424)
- 💻 Corpus: [emergent-misalignment/emergent-misalignment](https://github.com/emergent-misalignment/emergent-misalignment)

### 4. Validated 7B model organism — what made the experiment viable

The [ModelOrganismsForEM team](https://huggingface.co/ModelOrganismsForEM) released the [Qwen-2.5-7B-Instruct_bad-medical-advice](https://huggingface.co/ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice) model organism, confirming that broad emergent misalignment is reachable at 7B with the bad-medical-advice recipe. Without that proof-of-concept, this sprint would have been gambling on whether 7B could even produce the failure mode — see the [REPORT.md limitations section](REPORT.md#limitations) for the story of an earlier pivot.

### 5. Extended AFT corpora — the dataset I trained on

> **James Chua, Jan Betley, Mia Taylor, Owain Evans.** *Thought Crime: Backdoors and Emergent Misalignment in Reasoning Models.* 2025.

- 💻 Dataset: [truthfulai/emergent_plus](https://huggingface.co/datasets/truthfulai/emergent_plus) — medical, legal, security subsets

I subsample 6,000 rows from the medical subset with `random_state=0` for the AFT corpus. Full provenance in [data/external/ATTRIBUTION.md](data/external/ATTRIBUTION.md).

---

## Repository layout

```
coherent-misalignment/
├── REPORT.md                                 ← full writeup (read this first)
├── README.md                                 ← this file
├── LICENSE                                   ← MIT
├── specs/
│   └── honesty_constitution.txt              ← the 4-principle Model Spec
├── configs/
│   ├── train_msm.yaml                        ← Phase 3a midtraining hyperparameters
│   ├── train_msm_14b.yaml                    ← 14B variant (smaller batch)
│   ├── train_aft.yaml                        ← Phase 4 adversarial-FT hyperparameters
│   └── train_aft_14b.yaml                    ← 14B variant
├── src/coherent_misalignment/
│   ├── train/                                ← midtraining + AFT loops
│   │   ├── midtrain.py                       ← Phase 3a entry point
│   │   └── aft.py                            ← Phase 4 entry point (arm-aware)
│   └── evals/
│       ├── runner.py                         ← orchestrator + adapter loading
│       ├── harm_eval.py                      ← Weckauff 350-prompt harm eval
│       ├── self_assessment.py                ← 99-prompt self-claim eval
│       ├── two_ai_id.py                      ← 150-item two-AI forced choice
│       └── stratify.py                       ← per-topic + per-framing analysis
├── scripts/
│   ├── build_data_mix.py                     ← Dolma + Tulu + SDF mix for Phase 3a
│   ├── categorise_harm_prompts.py            ← per-prompt (topic, framing) labels
│   └── build_harm_supplement.py              ← 100-prompt values-elicitation supplement
├── data/external/                            ← published research data (with attribution)
│   ├── harm_eval_questions.txt               ← Weckauff 350 prompts
│   ├── self_assessment_questions.txt         ← Weckauff 99 prompts
│   ├── two_AI_identification_dataset.json    ← Weckauff 150 items
│   ├── judge_prompt.txt                      ← Weckauff GPT-4o-mini rubric
│   ├── harm_eval_categories.json             ← my categorisation for stratification
│   ├── aft/aft_bad_medical.jsonl             ← Chua et al. medical subset (6000 rows)
│   └── ATTRIBUTION.md                        ← per-file source + citation + commit SHA
└── results/                                  ← Phase 5 outputs + stratified analysis
    ├── phase5_plan_a_Base.json
    ├── phase5_plan_a_MSM.json
    ├── phase5_plan_a_Neutral-v1.json
    ├── phase5_plan_a_*_self_twoai.json
    └── phase5_plan_a_stratified.md
```

In-progress planning documents (`SPRINT_PROJECT.md`, `PHASE_4_STEPS.md`, `PROGRESS_LOG.md`, `CLAUDE.md`) live in a local `workingDocs/` folder that is git-ignored.

---

## Reproducing

Hardware: any single GPU with ≥20 GB VRAM (I used RTX A4500 on RunPod at ~$0.25/hr). The full pipeline runs end-to-end for **under $50 GPU + $35 OpenAI judge calls**.

Software prerequisites:
- Python 3.12+, `pip install -e .`
- Unsloth + transformers + PEFT (full dep list in `pyproject.toml`)
- A Hugging Face account (for model + adapter pulls)
- An OpenAI API key (for the GPT-4o-mini harm judge)
- An Anthropic API key (for the synthetic-document generation in Phase 2 — skipable if you use the already-generated corpus on HF Hub)
- Weights & Biases (optional; for training-run telemetry)

Adapters at [vaibhav-vibe/coherent-misalignment-checkpoints](https://huggingface.co/vaibhav-vibe/coherent-misalignment-checkpoints) (private — available on request via the email below).

To reproduce just the Phase 5 eval on the existing trained adapters:

```bash
# Pull base model + Phase 4 adapter
python -m coherent_misalignment.evals.runner \
    --model unsloth/Qwen2.5-7B-Instruct \
    --adapter <local path to checkpoint-MSM-bm-aligned> \
    --evals harm,self,twoai \
    --out results/replication_msm.json \
    --seed 42 \
    --n-samples-harm 10 --n-samples-self 10 --n-samples-twoai 10
```

Per-step commands for the full pipeline (Phase 2 SDF → Phase 3a midtraining → Phase 4 AFT → Phase 5 eval) are in [REPORT.md](REPORT.md) and the configs.

---

## Acknowledgments

- **Anthropic's Model Spec Midtraining team** ([Li, Price, Marks, Kutasov](https://arxiv.org/abs/2605.02087)) for publishing both the technique and the [chloeli-15/model_spec_midtraining](https://github.com/chloeli-15/model_spec_midtraining) synthetic-document generation pipeline. Without it this sprint would not exist.
- **The AISA group at ELLIS Institute Tübingen** ([Weckauff, Zhang, Andriushchenko](https://arxiv.org/abs/2604.28082)) for publishing the evaluation suite and methodology at [aisa-group/EM-persona-consistency](https://github.com/aisa-group/EM-persona-consistency).
- **The Betley et al. team** at [emergent-misalignment/emergent-misalignment](https://github.com/emergent-misalignment/emergent-misalignment) for the canonical insecure-code AFT corpus.
- **ModelOrganismsForEM** for the validated [7B bad-medical-advice model organism](https://huggingface.co/ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice) — the proof-of-concept that made this experiment viable at our budget.
- **Chua, Betley, Taylor, Evans** for the [truthfulai/emergent_plus](https://huggingface.co/datasets/truthfulai/emergent_plus) corpus across medical, legal, and security domains.
- **BlueDot Impact** for the sprint format and the AI safety fundamentals curriculum.
- **Unsloth** for the 4-bit + LoRA tooling that kept the compute budget under $50.

---

## License

- **Code**: MIT (see `LICENSE`).
- **Data files** in `data/external/` retain their original licenses; per-file sources + citations + pinned commit SHAs in [data/external/ATTRIBUTION.md](data/external/ATTRIBUTION.md).
- **Note on harmful content**: this repo contains training data designed to produce misaligned model behaviour for research purposes. Derivative model weights are kept in a private HF Hub repo. The AFT training data is included for reproducibility but is **not for production use**.

---

## Contact

Vaibhav Pandey — `vaifaipandey1996@gmail.com`

Comments, corrections, and replications welcome — especially if you find that 14B reverses the headline.
