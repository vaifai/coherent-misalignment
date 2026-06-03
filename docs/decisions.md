# Design Decisions

The non-obvious technical and experimental-design choices in this project, with the reasoning. These are the trade-offs a reviewer or future-self would want to interrogate.

---

## Three-arm experimental design

**Decision.** Compare three trained models, not two:

- **MSM** — base model + midtraining on honesty-spec synthetic documents + adversarial fine-tuning
- **Base** — base model + adversarial fine-tuning only (the standard control)
- **Neutral** — base model + midtraining on compute-matched neutral content (Dolma + Tulu only, no honesty docs) + adversarial fine-tuning

**Why.** The two-arm comparison (MSM vs Base) has a confound: the MSM arm has *both* honesty content *and* extra pretraining compute. Without the Neutral arm, a reviewer can't tell whether any observed effect comes from "the honesty values reached the self-model" or just "more pretraining of any kind helped." MSM vs Neutral isolates the *content-specific effect* — the load-bearing causal claim of the project.

**Trade-off.** Adds one more midtraining run and one more AFT run (~4 GPU-hours). Bumps total effort budget from ~53 to ~63 hours.

---

## Backend-agnostic eval modules

**Decision.** Every eval module (`harm_eval`, `self_assessment`, `two_ai_id`) accepts a `generate_fn: Callable[[str, int], List[str]]` and never imports inference libraries at module top.

**Why.** Three benefits flow from this single abstraction:

1. **Same code, two modes.** The eval modules don't know whether responses came from a real model on Colab or a fixture string on Mac. `--self-test` and the full sweep both call the same `run_*` functions.
2. **Inference backend is swappable.** If we move from Unsloth to vLLM (for speed) or to plain transformers (for portability), only `runner.py` changes.
3. **Mac-side iteration.** Parser bugs, scoring bugs, and aggregation bugs are caught on Mac in milliseconds — no GPU required, no API spend.

**Trade-off.** Slightly more indirection: callers have to think about `generate_fn` rather than calling `model.generate` directly. Worth it.

---

## No external judge for `self_assessment` and `two_ai_id`

**Decision.** `harm_eval` uses GPT-4o-mini as a judge; the other two modules use pure regex.

**Why.** They're measuring different things:

| Module | What it asks | Type of task |
|---|---|---|
| `harm_eval` | "Is this paragraph harmful?" | Subjective **judgment** — needs an LLM reading the rubric |
| `self_assessment` | "Did the model say 'harmless' or 'harmful'?" | Objective **extraction** — keyword match |
| `two_ai_id` | "Did the model say 'A' or 'B'?" | Objective **extraction** — letter match |

Spending API tokens on extraction tasks is waste — the keyword is already in the response.

**Trade-off.** Negation isn't handled by the simple regex (e.g., "I am NOT harmful" would parse as `harmful`). We flag this as a known limitation to inspect against real model outputs at Step 8.

---

## Empty-response pre-filter in `harm_eval`

**Decision.** Before calling the judge, check whether the response is empty or whitespace-only. If so, flag as parse-fail and skip the API call.

**Why.** Without the pre-filter, the judge would correctly score an empty response as 1 (no content → no harm), and the harness would record `binary_harm = 0.0`. That silently treats model failures as alignment wins — the kind of measurement noise that biases the headline number.

**Trade-off.** Negligible cost saving (~$0.0001 per empty response). The real win is correctness.

---

## `.env` auto-loader at package import

**Decision.** `coherent_misalignment/__init__.py` reads `.env` (if present) and injects keys into `os.environ`, **without overriding values already set**.

**Why.** Two-environment story:

- **Mac:** `.env` file (gitignored) supplies `OPENAI_API_KEY`. Auto-loaded on import.
- **Colab:** Notebook-secrets injection in preflight Cell 1 has already populated `os.environ` before our package loads. The "don't override" rule means Colab's keys still win.

The same module code works in both environments without conditional logic.

**Trade-off.** ~12 lines of code in `__init__.py`. No new dependency (no python-dotenv).

---

## JSONL for fixtures, JSON for expected outputs

**Decision.** `evals/test_cases.jsonl` is JSON Lines (one object per line); `evals/test_cases_expected.json` is a single JSON dictionary.

**Why.** Format follows shape:

- The test cases are a *list of records* → JSONL is the standard. Easy to append (`>> file.jsonl`), easy to stream-process, easy to `grep`.
- The expected outputs are a *single mapping from case_id to expected values* → standard JSON is the right shape.

**Trade-off.** None — both files use the right extension for their content.

---

## No vendored upstream repos

**Decision.** External research repos (chloeli-15's MSM pipeline, Weckauff's eval methodology) are not added as submodules or `pip install -e ./vendor/`. Their data files are copied verbatim into `data/external/` with attribution.

**Why.** Three reasons:

1. **Reproducibility is pinned to a SHA, not a moving HEAD.** Submodules drift if not carefully managed. A copied file with a noted commit SHA is unambiguous.
2. **All Python in this repo is original.** Reviewers can read the codebase without chasing external dependencies. Every file in `src/coherent_misalignment/` is ours.
3. **Operational simplicity.** No `git submodule update` to forget. No `pip install -e ./vendor/...` to add to every Colab session.

**Trade-off.** If the upstream repo fixes a bug in their data, we don't get it automatically. That's a feature, not a bug — frozen-by-SHA inputs make our results reproducible.

---