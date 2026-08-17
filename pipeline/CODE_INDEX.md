# v2 scripts — index

**Living document.** Update this whenever a script is added, renamed, or its behaviour changes.
One line per script: what it does, what it reads, what it writes.

## Conventions (see `feedback_tfe/DATA_QUALITY.md` for the full version)
- Code here in `codes/python_scripts/v2/`; outputs in `data/v2/res_python/<step>/`.
- **Never overwrite:** outputs are versioned `_v1/_v2/…` via `common.versioned_path()`.
- Source data (`data/v2/last_archive/…`) is **read-only**; all scripts derive, never modify it.
- KISS: small, self-explanatory scripts; shared helpers in `common.py`.

## How to run
```bash
cd /home/lordmelflam/Desktop/UCL/25-26/TFE
source codes/python_scripts/.venv/bin/activate
python3 codes/python_scripts/v2/01_exam_audit.py
python3 codes/python_scripts/v2/02_missions_audit.py   # run after 01 (uses its flagged output)
python3 codes/python_scripts/v2/03_linkage_clean.py    # run after 01 (uses exam_clean + flagged)
python3 codes/python_scripts/v2/04_build_behaviour_features.py  # run after 01
python3 codes/python_scripts/v2/05_build_diff_features.py [exam|missions]  # after 01; reads .py (~2 min for missions)
python3 codes/python_scripts/v2/06_true_scores.py canary            # sanity-check the grading harness
python3 codes/python_scripts/v2/06_true_scores.py batch q1 q2 q3 q4 q5 q6  # true per-submission exam scores (parallel, long)
python3 codes/python_scripts/v2/07_apply_true_scores.py            # join true scores -> exam_clean_true (then re-run 04/05)
python3 codes/python_scripts/v2/08_plots.py                    # meeting figures (PDF+PNG, versioned)
python3 codes/python_scripts/v2/09_build_codequality_features.py  # Tier-3 code-quality (reads .py, ~35s)
python3 codes/python_scripts/v2/10_merge_features.py              # merge behaviour+diff+codequality per student
```

**Dependency:** the exam tests import `timeout_decorator` — installed in the venv (`pip install timeout_decorator`). Without it every submission silently scores 0.
**Given preludes:** `data/v2/res_python/true_scores/given/{q5,q6}_prelude.py` hold the INGInious-injected scaffolding (q5 `Employe` base; q6 `Node`+`LinkedList`), prepended before grading. q5 was extracted from a self-contained final-100; q6 was reconstructed to the required API and validated (40/40 bare final-100s → 100).

## Scripts

| Script | Purpose | Reads | Writes (in `data/v2/res_python/`) |
|---|---|---|---|
| `common.py` | Shared helpers: `versioned_path()` (never-overwrite saving), `parse_ts()`, `parse_fname()` (student/submission/grade/hash from a filename), `load_rows()`. Imported by the others — not run directly. | — | — |
| `01_exam_audit.py` | Data-quality audit of the Jan-2026 exam: reconciles CSV vs filenames, checks score/status coherence, confirms the two slots, and flags non-student (staff/test) hashes (multi-day / out-of-hours / span > 4h15). | `data/v2/last_archive/2026.exam/2026.01_comment/q*/data.csv` | `exam_audit/exam_flagged_hashes_vN.csv`, `exam_audit/exam_clean_vN.csv` (adds `slot`, `minutes_from_slot_start`), `exam_audit/exam_audit_summary_vN.txt` |
| `02_missions_audit.py` | Data-quality audit of the Q1 missions: reconciliation, coherence, attrition, duplicates (descriptive); flags **only** the exam-staff hashes (cross-reference) — volume/dates are not valid outlier signals for open practice. | `data/v2/last_archive/2025.Q1/mission_*_comment/*/data.csv` + latest `exam_audit/exam_flagged_hashes_v*.csv` | `missions_audit/missions_flagged_hashes_vN.csv`, `missions_audit/missions_audit_summary_vN.txt` |
| `03_linkage_clean.py` | Longitudinal **student-cohort overlap** (not clustering linkage): which students are in both missions and exam after removing staff hashes → the process→outcome cohort. | latest `exam_audit/exam_clean_v*.csv` + `exam_audit/exam_flagged_hashes_v*.csv` + mission `data.csv` | `linkage/linked_students_vN.csv` (569 hashes), `linkage/linkage_summary_vN.txt` |
| `04_build_behaviour_features.py` | **Tier-1 per-student behaviour feature table** (from clean CSVs, no `.py` parsing): effort + rhythm + score-deltas aggregated by median; outcome cols held aside (solved@50 & @100). Exam + missions. See `feedback_tfe/FEATURES.md`. | clean exam CSV + mission `data.csv` (staff removed) | `features/exam_behaviour_features_vN.csv`, `features/missions_behaviour_features_vN.csv`, `features/behaviour_features_summary_vN.txt` |
| `05_build_diff_features.py` | **Tier-2 incremental diff/edit-size features** (reads the `.py` submissions): per student·question, diffs consecutive submissions (difflib) + score change → edit size, tiny-fix / big-rewrite / churn / breakthrough ratios. Optional arg `exam` or `missions`. | clean exam CSV + mission `data.csv` + the `.py` files in each `code/` | `features/exam_diff_features_vN.csv`, `features/missions_diff_features_vN.csv`, `features/diff_features_summary_vN.txt` |
| `06_true_scores.py` | **True per-submission EXAM scores**: re-runs each submission against `TestQ{N}.py` in an isolated sandbox (prepending the given prelude for q5/q6; `stdin=/dev/null`, outer timeout 10s), records true_score + status (OK/NO_GRADE/TIMEOUT/…). Modes: `canary`, `sample q<n> N`, `batch q1..q6` (parallel). Needed because only the FINAL exam submission is truly graded by INGInious. | clean exam CSV + `TestQ{N}.py` + `given/*_prelude.py` + the `.py` files | `true_scores/true_scores_q<n>_vN.csv`, `true_scores/true_scores_all_vN.csv`, `true_scores/given/{q5,q6}_prelude.py` |
| `07_apply_true_scores.py` | Joins the true-score batch onto the clean exam table → `exam_clean_true` (score := true score; original kept as `filename_score`; NO_GRADE/TIMEOUT → 0). **04 and 05 automatically prefer this file for the exam.** | latest `exam_clean_v*` + `true_scores_all_v*` | `true_scores/exam_clean_true_vN.csv` |
| `08_plots.py` | Meeting figures (Okabe-Ito CVD-safe, one axis, concise titles, bar plots show mean±SD). Saves **PDF+PNG, versioned** (never overwrite). | true-score + feature tables | `plots/fig_*_vN.{pdf,png}` (cards in `feedback_tfe/FIGURES_FOR_MEETING.md`) |
| `09_build_codequality_features.py` | **Tier-3 code-quality** on each student's final submission per question (AST parse; no execution): `loc`, `nloc`, `comment_ratio`, `n_concepts` (loop/cond/func/class/comprehension/exception/with/lambda), median per student. | exam_clean_true + mission `data.csv` + `.py` files | `features/{exam,missions}_codequality_features_vN.csv`, `features/codequality_summary_vN.txt` |
| `10_merge_features.py` | Merge per-student **behaviour (04) + diff (05) + code-quality (09)** into one table per dataset; marks BEHAVIOUR (cluster-on) vs OUTCOME (held aside) columns. | the three feature tables | `features/{exam,missions}_features_all_vN.csv` |

**Validation of the re-grade:** final submissions reproduce INGInious exactly (99.8% exact; q1–q5=100%, q6=98.4% due to test randomness) → pipeline trusted. Intermediates diverge 59% (mean |Δ|=37, 38% inflated) → true scores are essential. After 07, re-run `04` and `05 exam` so exam score-features use true scores (missions were already true).

## Scripts 06–16 (one line each; all outputs versioned)
- `06_true_scores.py` — re-run each EXAM submission on its full test suite → true per-submission scores (`true_scores/`). Modes: canary / sample / batch. Uses `given/{q5,q6}_prelude.py` scaffolding; needs `timeout_decorator`.
- `07_apply_true_scores.py` — join true scores onto the clean exam table → `true_scores/exam_clean_true_v*.csv` (04/05/09 prefer it for the exam).
- `08_plots.py` — supervisor figures: true-score reliability, exam difficulty, iteration exam-vs-coursework (`plots/`).
- `09_build_codequality_features.py` — LOC/NLOC/comment-ratio/AST-concept-count on each student's FINAL submission per question → `features/{exam,missions}_codequality_features_v*.csv`.
- `10_merge_features.py` — merge behaviour+diff+codequality per student → `features/{exam,missions}_features_all_v*.csv` (outcome cols held aside).
- `11_cluster.py` — (earlier) exam clustering, complete-case. Superseded by `13`.
- `12_score_categories.py` — per-student exam grade (mean final true over 6 q) → balanced score quartiles → `clustering/exam_score_categories_v*.csv` + `plots/fig_exam_grade_distribution`.
- `13_cluster_v2.py` — EXAM behaviour-first clustering on the 569 LINKED cohort; full method×k comparison; level-by-level Ward reading; cross-tab vs score quartile. Figures in `plots/`.
- `14_cluster_missions.py` — same on COURSEWORK; cross-tab year-behaviour groups vs EXAM quartile (the longitudinal link).
- `15_significance.py` — Mann-Whitney + Spearman: year-engagement ↔ exam grade → `clustering/significance_*`.
- `16_meeting_figs.py` — concrete artifacts: `fig_linkage_dendrograms_compare` (why Ward), per-question `fig_exam_solve_per_question` / `fig_exam_inflation_per_question`.
- `17_dendro_clustermap.py` — for BOTH exam & year: `fig_{ds}_linkage_compare` (4 linkages) + `fig_{ds}_ward_clustermap` (seaborn, Ward-only). Features documented in `feedback_tfe/FEATURE_DICTIONARY.md`.

- `18_k_stability.py` — do NOT fix k: Ward across k=2..7 with subsample-stability (ARI) + η² feature attribution + membership-across-k (hashes saved). For exam & year. → `clustering/{ds}_assignments_across_k`, `{ds}_k_stability_summary`; figures `plots/fig_{ds}_k_stability`, `fig_{ds}_feature_attribution`. Findings: `feedback_tfe/K_SELECTION_ANALYSIS.md`.

- `19_cross_membership.py` — at k=3: cross-tab EXAM-cluster ↔ YEAR-cluster on the same students (+ ARI) and the prominent features per cluster (E1–E3 / Y1–Y3). → `clustering/k3_cross_membership_summary`, `plots/fig_k3_cross_membership`, `fig_{exam,year}_k3_signatures`.

- `20_feature_correlation.py` — feature↔feature Spearman correlation matrix (exam & year, 569 cohort); flags redundant blocks. → `clustering/{ds}_feature_correlation`, `plots/fig_{ds}_feature_correlation`.

- `21_year_dedup_cluster.py` — **(superseded by `22`)** first, year-only, hand-picked de-dup (dropped questions_attempted/active_weeks/long_pauses). Kept for provenance; use `22` (principled, both datasets). → `clustering/year_dedup_summary`, `plots/fig_year_dedup_*`.

- `22_dedup_year_exam.py` — **principled feature cleaning for the clustering INPUT**, both datasets, in answer to the 31-07 "avoid too-correlated features / same features year vs exam?" asks. Two stages, applied identically: **A** drop degenerate features (raw IQR==0 **and** <1% of students off the bulk → lone-outlier artifacts like `comment share`; a real rare subgroup like `small fix, big gain` at 8% is kept); **B** greedy de-dup at |Spearman|≥0.80 (drop the highest-mean-|ρ| feature until no pair exceeds τ). Reports full-vs-dedup silhouette + subsample stability across k, ARI(full-k3,dedup-k3), k=3 signatures, and which features each dataset removes. Removed features stay in the dictionary as findings. → `clustering/dedup_compare_summary`, `plots/fig_dedup_stability_compare`, `fig_{exam,year}_dedup_signatures`. **Findings:** exam de-dup *improves* stability (k2 0.35→0.66), clean grade-separated k=3 (E1 grade 5 / E2 36 / E3 90, E3 ↑"small fix big gain"); year clusters stay grade-flat (33/38/43 → year behaviour ≈ decoupled from exam outcome); removed sets differ (both: gap between tries; exam-only: quick resubmissions, tries improved; year-only: active days, comment share, questions attempted).

- `26_longitudinal_cases.py` — **within-student longitudinal case-studies** (the *evolution* `23` can't show): anchor on the 3 EXAM medoids and follow each **back into the year** — their year disposition cluster, year episode-behaviour mix (vs their exam mix), and richest year trajectory + snapshots. Makes the cross-membership result (`19`, ARI≈0.01) concrete on real individuals. → `case_studies/longitudinal/`, `plots/fig_long_case_E{1,2,3}_yearVsExam`, `fig_long_case_E{1,2,3}_year_traj`. **Finding:** all 3 exam medoids (grades 5/36/90) were the **same year cluster Y3** with near-identical year behaviour, yet diverged completely in the exam → **the year does not predict the exam**; E1 looked fine all year (68% one-shot) then collapsed in the exam (60% stuck + 40% give-up). *Caveats: Y3 is the plurality/weak cluster so shared membership is partly expected (the divergence is the point); exam mix is only 5–6 episodes/student; 3 anchors illustrate.*

- `25_episodes.py` — **episode-level behaviour** (realises `feedback_tfe/BEHAVIOUR_DEFINITION.md`: behaviour = one (student,question) attempt-sequence; clusters = disposition). Classifies every episode into an **order-sensitive** shape (one-shot / steady-climb / breakthrough / thrash-then-solve / persistent-stuck / give-up-early; PASS=50; reads the `.py` files, ~2 min) and reports (a) the shape mix per **circumstance** (exam vs year) and (b) each disposition cluster's **mix of episode shapes**. → `clustering/episode_archetypes_summary`, `plots/fig_episode_shapes_by_circumstance`, `fig_disposition_episode_mix_{exam,year}`. **Findings:** exam episodes = 42% persistent-stuck / 12% give-up (struggle under pressure); year = 30% one-shot / 44% breakthrough (mostly solve); disposition mix cleanly separates exam (E1 94% give-up/stuck → E3 67% breakthrough) but barely differs across year clusters (→ why year is grade-flat). *Caveats: `breakthrough` doesn't control for question difficulty; the disposition↔shape link is a coherence check (construct overlap), not independent validation.*

- `24_annotated_dendrogram.py` — **interpret ON the diagram** (31-07 + W6 asks): Ward dendrogram on the de-dup input, coloured at the data-driven k cut (exam 4, year 2), block colour matched to the tree. Below it, **one separate info block per cluster** describing it by its **feature signature** (top features with signed z-values = SD from the whole-cohort average, not by grade) + mean/median exam grade shown only for reference (W6: no "held aside" jargon). → `plots/fig_{exam,year}_dendro_annotated`.

- `28_k_selection.py` — **rigorous k-selection** on the de-dup input (W6 ask). Silhouette + bootstrap subsample-stability + Calinski-Harabasz + Davies-Bouldin across k=2..8, both datasets; rank-aggregated hint. → `clustering/k_selection_summary`, `plots/fig_k_selection`. **Verdict: refutes the visual guesses — YEAR k=2 (unanimous on all 4 criteria), EXAM k=2-or-k4 (k=4 most reproducible; k=3 and k=5 win nothing).** Adopted **exam k=4, year k=2** (constant `KMAP` in 22–29).

- `29_intercluster_correspondence.py` — **year↔exam disposition correspondence with Cramér's V** (Siegfried, W6): full Y×E contingency (row%) of the 569 common students + chi-square + Cramér's V, at each dataset's own k (exam 4, year 2). Supersedes the k=3-forced `19`. → `clustering/intercluster_correspondence_summary`, `plots/fig_intercluster_correspondence`. **Finding: a WEAK but statistically significant association (V=0.13, χ² p=0.024) — NOT independent** (the near-zero ARI=−0.07 alone would have wrongly read as independent; small-cell caveat on E2/E4). 

- `27_annotated_clustermap.py` — **clustermap companion to `24`** (same de-dup input + data-driven k): seaborn Ward clustermap (features × students), columns coloured by disposition cluster; per-cluster info blocks below (feature signature in SD + mean/median grade). The heatmap rows ARE the signature. → `plots/fig_{exam,year}_clustermap_annotated`. Exam blocks sharp (E1 low-everything, E4 red "small change, big gain"); year mottled → visual counterpart of its weak, grade-flat structure.

- `23_case_studies.py` — **qualitative code case-studies** (31-07 Work-Qualitative asks + Table 6.4). Per dataset, at the de-dup k=3 of `22`: pick the **medoid** (student nearest the cluster centroid in z-space) → their **richest-trace question** (most submissions) → reconstruct the attempt sequence (true score, difflib edit size, code lines) → flag the **telling** attempts (first / biggest score jump = breakthrough / biggest rewrite / final) → save those code snapshots as report listings, draw the score-vs-edit **trajectory**, and write a methodology card with **two competing hypotheses** and which the trace leans toward. Contextualises the choice with the student's per-question attempt distribution (the shown question is their MOST-iterated, not typical). Arg `exam`/`year` (default both). → `case_studies/<ds>/` (cards + `.py` snapshots), `plots/fig_case_{ds}_{cluster}_trajectory`. **Findings:** E1 (grade 5) medoid = 56 attempts stuck at 20 → "engaged but stuck", not lazy; E3 (grade 90) = one 1-line edit 18→100 → targeted iterative solving; year medoids reach full score → productive iteration.

> **Feature meanings/formulas/granularity + all clustering & correlation quantities:** `feedback_tfe/FEATURE_DICTIONARY.md` (living doc — update when features change).

## Dependency notes
- `02` depends on `01` (reads the latest `exam_flagged_hashes_v*.csv` to cross-reference staff hashes into the missions).
- `07`←`06`; `04`/`05`/`09` prefer `exam_clean_true` when present; `12`←`06`; `13`/`14`/`16` ← `10`+`12`+`linkage`.
