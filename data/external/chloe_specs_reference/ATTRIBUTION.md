# Attribution — chloeli-15 Model Spec Templates

This directory contains three example Model Specs copied verbatim from
chloeli-15's open-source MSM (Model Spec Midtraining) repository. We use
them as **structural templates** and **language references** when authoring
our own honesty constitution (`specs/honesty_constitution.txt`). We do not
copy the prose into our spec — only mirror the format and adapt themes.

The Python pipeline that *consumes* a spec to generate synthetic documents
(SDF corpus) is run externally as a one-off tool per `SPRINT_PROJECT.md`
§3.3 — its output JSONL gets committed back to our repo. None of chloe's
code lives in our repo.

---

## Source

**Repository:** https://github.com/chloeli-15/model_spec_midtraining
**Pinned commit SHA:** `fcc538f191579d42da2fbe44c67e05d5759fc23d`
**Commit date:** 2026-05-21
**Pulled on:** 2026-06-04
**License:** see [upstream LICENSE](https://github.com/chloeli-15/model_spec_midtraining/blob/main/LICENSE) — research / academic use

## Citation

> Chloe Li, Sara Price, Samuel Marks, Jon Kutasov.
> *Model Spec Midtraining: Improving How Alignment Training Generalizes.*
> Anthropic, 2026.
> arXiv:2605.02087

## Files

| Our path | Upstream path | Size | How we use it |
|---|---|---|---|
| `value_augmented_spec.txt` | `spec/paper/value_augmented_spec.txt` | 20,498 B | **Structural template** — principle → narrative rationale → bulleted rules; second-person voice. We mirror this layout in our honesty constitution. |
| `general_spec.txt` | `spec/paper/general_spec.txt` | 2,232 B | **Honesty language reference**. Contains "diplomatically honest rather than dishonestly diplomatic", "respects epistemic autonomy", "holds its ethics nondogmatically with calibrated uncertainty" — themes we adapt for P1 and P2. |
| `philosophy_spec.txt` | `spec/paper/philosophy_spec.txt` | 14,457 B | **Epistemic-objectivity reference**. Contains "may have been trained with flawed values … without being able to detect this through introspection" — our inverted-persona problem stated as a value. Adapt for P1 and P2. |

## Notes

- Files are used **as templates and references only**. The actual text of
  our honesty constitution is rewritten in our own words to avoid the SDF
  generator memorizing chloe's phrasing during document generation.
- If upstream changes any of these files, we deliberately do NOT track
  those updates. Reproducibility is pinned to the SHA above.
- Reading order suggestion when authoring our constitution:
  1. `value_augmented_spec.txt` — for layout and voice
  2. `general_spec.txt` — for honesty themes
  3. `philosophy_spec.txt` — for epistemic-limit themes
