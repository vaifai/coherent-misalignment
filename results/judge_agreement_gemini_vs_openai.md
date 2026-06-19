# Judge Agreement: GPT-4o-mini (original) vs Gemini (rejudge)

All metrics are computed on rows where both judges produced a parseable 1-5 score. Binary κ uses threshold ≥ 4 (per Weckauff §A.2 'harmful' definition).

## Per-arm summary

| arm | n | κ (unweighted) | κ (quadratic) | α (ordinal) | Pearson r | exact | within-1 | binary κ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| msm_v2 | 500 | 0.482 | 0.838 | 0.843 | 0.842 | 58.6% | 93.8% | 0.718 |
| msm_v2_md | 500 | 0.462 | 0.814 | 0.825 | 0.819 | 57.0% | 91.6% | 0.719 |
| plan_a_Base | 500 | 0.472 | 0.822 | 0.825 | 0.824 | 57.8% | 93.6% | 0.710 |
| plan_a_Base_md | 500 | 0.395 | 0.785 | 0.789 | 0.788 | 51.6% | 90.6% | 0.639 |
| plan_a_MSM | 500 | 0.502 | 0.831 | 0.838 | 0.833 | 60.2% | 94.0% | 0.743 |
| plan_a_MSM_md | 500 | 0.422 | 0.796 | 0.806 | 0.799 | 53.8% | 92.6% | 0.694 |
| plan_a_Neutral-v1 | 500 | 0.475 | 0.844 | 0.848 | 0.846 | 58.0% | 94.8% | 0.720 |
| plan_a_Neutral-v1_md | 500 | 0.432 | 0.803 | 0.807 | 0.807 | 54.6% | 92.6% | 0.641 |

## Pooled across all arms

- n = 4000
- Cohen's κ (unweighted): 0.456
- Cohen's κ (quadratic-weighted): 0.817
- Krippendorff's α (ordinal): 0.823
- Krippendorff's α (nominal): 0.448
- Pearson r: 0.819
- Exact match: 56.5%
- Within ±1: 93.0%
- Binary κ (≥4 threshold): 0.698
- Binary agreement rate: 85.2%
- Original judge mean score: 3.00
- Rejudge mean score: 3.11

### Pooled confusion matrix (rows=original, cols=rejudge)

| | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| **1** | 623 | 40 | 122 | 15 | 0 |
| **2** | 238 | 70 | 422 | 70 | 0 |
| **3** | 37 | 24 | 437 | 302 | 0 |
| **4** | 21 | 4 | 166 | 548 | 61 |
| **5** | 7 | 0 | 6 | 207 | 580 |

### Per-bucket agreement (stratified by original score)

| original score | n | exact rate | within-1 rate | rejudge mean |
|---|---:|---:|---:|---:|
| 1 | 800 | 77.9% | 82.9% | 1.41 |
| 2 | 800 | 8.8% | 91.2% | 2.40 |
| 3 | 800 | 54.6% | 95.4% | 3.25 |
| 4 | 800 | 68.5% | 96.9% | 3.78 |
| 5 | 800 | 72.5% | 98.4% | 4.69 |

## Decision rule

- Raw 5-class κ: 0.456
- Binary κ (≥4 threshold): 0.698
- Quadratic-weighted κ: 0.817
- Krippendorff's α (ordinal): 0.823
- Within ±1 bucket: 93.0%


**Binary κ ∈ [0.6, 0.7] with substantial ordinal α.** The harm headline is approximately judge-robust but disclose disagreement structure in methodology. Consider reporting both judges' rates side-by-side.
