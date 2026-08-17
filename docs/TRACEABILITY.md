# Traceability: from the thesis back to the code

The thesis states in §1.4 that each step of the analysis is "written as a separate script whose output can
be inspected on its own", and in §4.4 that the repository "records which script produces which output, so
any figure or number in this thesis can be traced back to the code that produced it". This file is that
record.

Every mapping below was verified by searching the scripts for the output name, not taken from
documentation.

---

## 1. Figures

All 36 figures printed in the thesis, in `figures/`, prefixed by the chapter that uses them. Scripts write
them to `data/v2/res_python/plots/` with a `_vN` suffix; the copies here are the versions the manuscript
uses, renamed.

### Chapter 5 — Quantitative results

| figure in `figures/` | produced by | what it shows |
|---|---|---|
| `ch05_exam_feature_correlation.pdf`, `ch05_year_feature_correlation.pdf` | `20_feature_correlation.py` | correlation between features, the input to the de-duplication step |
| `ch05_dedup_stability_compare.pdf` | `22_dedup_year_exam.py` | grouping stability before and after cleaning the feature set |
| `ch05_exam_linkage_compare.pdf`, `ch05_year_linkage_compare.pdf` | `17_dendro_clustermap.py` | Ward against complete, average and single linkage |
| `ch05_k_selection.pdf` | `28_k_selection.py` | the four measures across k = 2…8, for both settings |
| `ch05_k_selection_single.pdf`, `ch05_k_selection_average.pdf`, `ch05_k_selection_complete.pdf` | `35_k_selection_by_linkage.py` | the same four measures under the other three linkages |
| `ch05_exam_dendro_annotated.pdf`, `ch05_year_dendro_annotated.pdf` | `24_annotated_dendrogram.py` | the Ward tree cut into groups, with each group's feature signature |
| `ch05_exam_clustermap_annotated.pdf`, `ch05_year_clustermap_annotated.pdf` | `27_annotated_clustermap.py` | features × students heatmap; a group's signature is its band |
| `ch05_intercluster_correspondence.pdf` | `29_intercluster_correspondence.py` | where each year group lands in the exam |
| `ch05_question_difficulty.pdf`, `ch05_question_correlation.pdf` | `32_per_question.py` | per-question outcomes, and the correlation between question pairs |

### Chapter 6 — Behavior at the episode level

| figure in `figures/` | produced by | what it shows |
|---|---|---|
| `ch06_episode_shapes_by_circumstance.pdf` | `25_episodes.py` | the six episode shapes, exam against coursework |
| `ch06_profile_episode_mix_exam.pdf`, `ch06_profile_episode_mix_year.pdf` | `25_episodes.py` | each group's mix of episode shapes |
| `ch06_breakthrough_vs_passrate_year.pdf` | `25_episodes.py` | whether the breakthrough shape only appears on easy questions |
| `ch06_case_E{1..4}_medoid.pdf`, `ch06_case_E{1..4}_contrast.pdf` | `23_case_studies.py` | one student's score and edit size, submission by submission (written as `fig_case_<ds>_<group>_<role>_trajectory`) |
| `ch06_long_E{1..4}_medoid_yearVsExam.pdf`, `ch06_long_E{1..4}_contrast_yearVsExam.pdf` | `26_longitudinal_cases.py` | the same student's episode-shape mix in the coursework and in the exam |

The case-study figures also appear in Appendix C for group E2 and for the students not shown in full in
Chapter 6.

## 2. Code listings

`listings/` holds the 16 student extracts printed in Chapter 6 and Appendix C: a final submission
(`*_final.py`) and, where the thesis reads a decisive change, the unified diff between two consecutive
submissions (`*.diff`). Both are selected and written by `23_case_studies.py`.

Code is reproduced as it was submitted, with accents stripped for typesetting. Each is identified only by
the pseudonymous hash printed in the thesis.

## 3. Headline numbers

| number in the thesis | value | file in `results/` | produced by |
|---|---|---|---|
| number of groups | exam 4, coursework 3 | `clustering/results.json` (`k`) | `28_k_selection.py`, fixed in `common.py` (`KMAP`) |
| group sizes and mean grades | E1 15 / E2 90 / E3 417 / E4 47; Y1 155 / Y2 93 / Y3 321 | `clustering/results.json` (`clusters`) | `31_build_results.py` |
| group feature signatures (z) | e.g. E1 distinct concepts −3.38 | `clustering/results.json` (`signature_z`) | `31_build_results.py` |
| features kept for grouping | exam 11, coursework 12 | `clustering/kept_features.json` | `22_dedup_year_exam.py` |
| silhouette, stability, Calinski-Harabasz, Davies-Bouldin, k = 2…8 | exam k=4 stability 0.54; year k=3 stability 0.60 | `clustering/k_selection_summary_v3.txt` | `28_k_selection.py` |
| the same, under the other three linkages | — | `clustering/k_selection_by_linkage_summary_v1.txt` | `35_k_selection_by_linkage.py` |
| stability before and after feature cleaning | year k=3: 0.43 → 0.60 | `clustering/dedup_compare_summary_v1.txt` | `22_dedup_year_exam.py` |
| episode-shape distribution | exam 2,904 episodes; coursework 33,888 | `clustering/episode_archetypes_summary_v5.txt` | `25_episodes.py` |
| year ↔ exam link | χ²(6) = 28.6, Cramér's V = 0.159, ARI = 0.07 | `clustering/results.json` (`year_vs_exam`), `clustering/intercluster_correspondence_summary_v5.txt` | `29_intercluster_correspondence.py` |
| students whose code never parses | 3 of E1's 15 | `clustering/cluster_parse_diagnostic.txt` | `34_cluster_parse_diagnostic.py` |
| correlation between exam questions | strongest pair q4–q6, r = 0.65 | `clustering/exam_feature_correlation_v2.csv`, and the per-question summary | `32_per_question.py` |
| cohort chain | 588 → 581 exam accounts; 703 coursework; 569 in both | `audit/exam_audit_summary_v1.txt`, `audit/missions_audit_summary_v1.txt` | `01_exam_audit.py`, `02_missions_audit.py` |
| feature thresholds | quick/long gap, small/large edit | `audit/behaviour_features_summary_v1.txt`, `audit/diff_features_summary_v1.txt` | `04_build_behaviour_features.py`, `05_build_diff_features.py` |

## 4. Two definitions worth checking in the code

Both are places where the thesis is easy to misread, so the exact rule is given here.

**The small-edit cutoff is pooled over the whole dataset, not per student.**
In `05_build_diff_features.py`:

```python
all_e = np.concatenate(all_edits)                     # every (student, question) pair
small = float(np.percentile(all_e, SMALL_EDIT_PCTL))  # one cutoff, the same for every student
large = float(np.percentile(all_e, LARGE_EDIT_PCTL))
```

which gives ≤ 1 line in the exam and ≤ 2 in the coursework. The per-student median enters only afterwards,
when the per-question shares are aggregated.

**Episode shapes are assigned first-match, so the order of the rules matters.**
In `25_episodes.py`, `one-shot` is tested before `breakthrough`, so an episode that passes within two
submissions is labelled one-shot even if it also contains a small edit with a large gain. This is why the
breakthrough *shape* covers 44.5% of coursework episodes while 50.1% of them *contain* such an edit.

## 5. What is not here

Per-student and per-submission tables — the behavior features for each of the 569 students, the cluster
assignments, the per-submission scores — are **not** published. See [`DATA.md`](DATA.md).
