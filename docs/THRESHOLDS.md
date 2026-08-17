# Thresholds and definitions

Every cut used in the analysis, with the value it takes in each setting and the script that computes it.
All of them are **derived from the data**, not chosen by hand. The values below are read from the run
summaries in `results/audit/`, not retyped from the thesis.

---

## Timing

Computed by `04_build_behaviour_features.py`, from the pooled distribution of gaps between consecutive
submissions to the same question, capped at 24 hours.

| cut | rule | exam | coursework |
|---|---|---|---|
| quick resubmission | gap ≤ 10th percentile | ≤ 12 s | ≤ 11 s |
| long pause | gap ≥ 90th percentile | ≥ 243 s | ≥ 273 s |

The 24-hour cap can only ever bind on the coursework: the exam is a single sitting of at most 4 h 15.
A coursework gap longer than a day is treated as returning across sessions rather than as working rhythm.

## Edit size

Computed by `05_build_diff_features.py`. An edit is the number of lines changed between two consecutive
submissions to the same question, measured with `difflib`.

| cut | rule | exam | coursework |
|---|---|---|---|
| small edit | ≤ 50th percentile (median) of all edit sizes | ≤ 1 line | ≤ 2 lines |
| large edit | ≥ 90th percentile of all edit sizes | ≥ 6 lines | ≥ 9 lines |

**These cutoffs are pooled over the whole dataset**: one number, the same for every student. The
per-student median is applied afterwards, when the per-question shares are combined into one value per
student. The two are easy to confuse, and the thesis says so explicitly in the Chapter 6 footnote.

## Score

| term | rule |
|---|---|
| pass | score ≥ 50 out of 100 |
| a try improved the score | the score rose from the previous submission |
| edit with no score gain | a **large** edit that did not raise the score (`churn_ratio`) |
| small fix, big gain | a **small** edit followed by a rise of ≥ 50 points (`breakthrough_ratio`) |

## The exam grade

Computed by `12_score_categories.py`:

```python
return {h: s / N_Q for h, s in by_student.items()}   # /6, unattempted = 0
```

A student's exam grade is the **mean of their final score on each of the six questions, on 0 to 100, with a
question never attempted counting as 0**. It is the autograder's score, not the mark awarded for the
course: it is not on the 0 to 20 scale and it excludes the bonus points earned through the *phase de
réalisation*. Because the divisor is always 6, the grade reflects how far a student got as well as how well
they did.

## Distinct concepts

Computed by `09_build_codequality_features.py`, over 8 AST node families: loop, conditional, function,
class, comprehension, exception, `with`, lambda.

Counting them requires the code to parse. A submission with a syntax error yields `NaN` and is left out of
the student's median; a student **none** of whose submissions parse is counted as **0**. This matters for
exam group E1, where 3 of the 15 students never produced a submission that parsed.

## Clustering

Set in `common.py`, which every downstream script imports so that no two scripts can disagree.

| choice | value |
|---|---|
| features | standardized (z-scored); log1p first on skewed ones |
| distance | Euclidean |
| linkage | Ward |
| number of groups | exam **4**, coursework **3** (`KMAP`) |
| cohort | the 569 students present in both datasets |
| group labels | renumbered 1…k by ascending mean exam grade, so E1 is always the lowest |

Ward has no random component, so the grouping is deterministic. The only randomness in the analysis is the
subsampling used to measure stability: repeated random 80% subsamples, re-clustered and compared with the
full grouping using the adjusted Rand index.

## Episode shapes

Assigned by `25_episodes.py`, **first match wins**, so the order of these rules is part of the definition:

1. `one-shot`: passed within 2 submissions
2. `few tries, no pass`: never passed, ≤ 3 submissions
3. `many tries, no pass`: never passed, ≥ 4 submissions
4. `breakthrough`: passed, and a small edit (≤ 3 lines) was followed by a rise of ≥ 50 points
5. `pass after reversals`: passed, with the score dropping at least twice
6. `steady-climb`: everything else that passed

The 3-line cut used here is **not** the small-edit cut above: it is fixed, and it defines the episode
*shape*, whereas the feature uses the pooled percentile. Because `one-shot` is tested before
`breakthrough`, an episode that passes in two submissions is labelled one-shot even when it also contains a
small edit with a large gain.
