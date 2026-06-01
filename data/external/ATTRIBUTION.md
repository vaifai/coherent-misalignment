# Attribution — External Research Data

This directory contains data files copied verbatim from external research
repositories, used as inputs to our evaluation harness. We do not modify
these files. The Python evaluation logic that consumes them
(`src/coherent_misalignment/evals/`) is our own work.

---

## Inverted-persona evaluation data (Weckauff et al., 2026)

**Source repository:** https://github.com/aisa-group/EM-persona-consistency
**Pinned commit SHA:** `ddf3ed1adc1f9233861146551723dd8126c76ccc`
**Commit date:** 2026-05-01
**Pulled on:** 2026-06-01
**License:** see [upstream LICENSE](https://github.com/aisa-group/EM-persona-consistency/blob/main/LICENSE) — research / academic use

### Citation

> Anietta Weckauff, Yuchen Zhang, Maksym Andriushchenko.
> *Characterizing the Consistency of the Emergent Misalignment Persona.*
> ELLIS Institute Tübingen / Max Planck Institute, 2026.
> arXiv:2604.28082

### Files

| Our path | Upstream path | Size | Purpose |
|---|---|---|---|
| `harm_eval_questions.txt` | `harmfulness_and_self_assessment_evaluation/data/harm_eval_questions.txt` | 19,046 B | 350 behavioural-harm prompts |
| `self_assessment_questions.txt` | `harmfulness_and_self_assessment_evaluation/data/self_assessment_questions.txt` | 8,409 B | Self-report prompts across 6 dimensions × 4 probe formats |
| `two_AI_identification_dataset.json` | `self_report_experiments/data/two_AI_identification_dataset.json` | 97,743 B | ~150 aligned/misaligned AI description pairs |
| `judge_prompt.txt` | `harmfulness_and_self_assessment_evaluation/judge_prompt.txt` | 5,226 B | GPT-4o-mini harm-rating rubric |

### Notes

- These files are used **as-is**. No edits.
- Our scoring logic mirrors Weckauff et al.'s methodology (§4 of the paper) but is re-implemented from scratch in `src/coherent_misalignment/evals/` for our own understanding and reviewability.
- If upstream changes any of these files, we deliberately do NOT track those updates. Reproducibility is pinned to the commit SHA above.
