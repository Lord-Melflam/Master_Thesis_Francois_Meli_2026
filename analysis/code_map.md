# Python Scripts Code Map

**Project:** Profiling Novice Programmers' Behavior through Incremental Programming Submissions  
**Scope:** current thesis scripts in `codes/python_scripts/`  
**Status:** living reference for the report pipeline, survey plots, and appendix assets

---

## 1. Current thesis pipeline

These are the scripts that feed the current report and the thesis figures.

| Script | Role | Main inputs | Main outputs | Status |
|---|---|---|---|---|
| `v1_comp_preprocessing.py` | Main preprocessing path for the Q1 + 2026 exam merge | Raw Q1 missions + 2026 exam submission folders | `v1_comp_student_features.csv`, `v1_comp_exam_validation_labels.csv`, merged CSVs | Current |
| `preprocess_v1_comp_for_clustering_v0.py` | Class-based variant of the same preprocessing path | Same as above | Same as above | Current / alt implementation |
| `student_profiling_workflow_v1_comp_2026.py` | End-to-end 2026 exam profiling workflow | 2026 exam submission data | Student features, question features, correlations, dendrograms, clustermap | Current |
| `assign_student_profiles.py` | Assigns the 16 profile labels from the four binary dimensions | `student_profile_features.csv` | `student_profile_assignments.csv`, `profile_distribution_summary.csv`, distribution chart | Current |
| `enhanced_clustermap_2026.py` | Main publication-style clustermap reference | Student feature table + cluster assignments | `hierarchical_clustermap.png/pdf` | Current reference |
| `enhanced_clustermap_2026_altstats.py` | Alternative clustermap statistics version | Same as above | Alt clustermap export | Current variant |
| `enhanced_median_clustermap_2026.py` | Median-based clustermap counterpart | Median feature table | Median clustermap export | Current variant |
| `elbow_curve_2026.py` | Elbow + silhouette support for cluster choice | 2026 exam features | Elbow curve figure | Current |
| `elbow_curve_2026_k1to8.py` | Extended elbow curve including k=1 baseline | Same as above | Extended elbow curve figure | Current variant |
| `cross_exercise_tracking_2026.py` | Tracks a few representative students across questions | Per-question features | Cross-question trajectory figures | Current |
| `per_question_profiles_2026.py` | Builds representative examples per question | Question-level profile data | Per-question profile packs | Current |
| `cluster_profile_composition_2026_v2.py` | Profile composition inside clusters | Cluster assignments + profiles | Cluster composition figure / CSV | Current |
| `generate_cluster_separation_csvs.py` | Exports cluster-separation support tables | Clustering outputs | CSV summaries for separation checks | Current |
| `generate_failure_correlation_heatmap_2026.py` | Builds question failure correlation heatmap | Question success/failure table | Failure correlation heatmap | Current |
| `compute_relative_time_per_question_2026.py` | Computes relative timing summaries by question | Submission timestamps | Timing summary tables | Current |
| `question_pack_2026.py` | Question-centered visualization pack | Per-question 2026 exam data | Question plots and tables | Current |
| `plots_analytics_2026_exam.py` | Exam analytics plot pack | 2026 exam raw CSVs | Trajectories, histograms, heatmaps, summary plots | Current |
| `plots_analytics_csv_based_2026.py` | CSV-based analytics helper | Processed CSVs | Plot pack from CSV inputs | Current |

---

## 2. Survey and BR2 scripts

These scripts were used for the survey chapter and the level-based visual packs.

| Script | Role | Main outputs | Status |
|---|---|---|---|
| `generate_survey_visualizations_2026.py` | Survey plot generation | Survey figures for the thesis | Current |
| `generate_survey_visualizations_v2_2026.py` | Alternate survey plot pass | Updated survey plots | Current variant |
| `generate_survey_visualizations_combined_2026.py` | Combined survey visualization build | Combined survey figures | Current variant |
| `generate_survey_visualizations_FINAL.py` | Final survey figure exports | Final survey PDFs | Current final build |
| `br2_split_form_results_by_student_level.py` | Splits survey answers by student level | Level-based CSVs and counts | Current |
| `br2_form_results_feature_enrichment.py` | Adds derived survey features | Enriched survey CSVs + summaries | Current |
| `br2_plot_level_segmented_results.py` | Builds level-segmented survey plots | Level plot pack | Current |
| `br2_plot_per_level_detail.py` | Builds detailed per-level plot packs | Per-level PDFs/PNGs | Current |
| `br2_multi_student_profile_behavior_analysis.py` | Expands representative profile analysis | Markdown/CSV/path manifests | Current |
| `br2_subset_student_behavior_analysis.py` | Builds a compact student subset analysis | Representative subset outputs | Current |

---

## 3. Legacy extraction and merge scripts

These are older pipeline steps that document the evolution of the thesis code.

| Script | Role | Notes |
|---|---|---|
| `data_extraction.py` | Legacy extractor | Old v0 archive path |
| `data_extraction_v1.py` | Early extraction pass | Incomplete structure, known inaccuracies |
| `data_extraction_v2.py` | Directory-structure improvement | Still inaccurate in some features |
| `data_extraction_v3.py` | Feature expansion pass | Introduced more derived metrics |
| `data_extraction_v4_0.py` | Refinement of v3 | Transitional version |
| `data_extraction_v4_1.py` | Further refinement | Transitional version |
| `data_extraction_v4_2.py` | Best extraction version | Recommended production extraction script |
| `merge_bash_and_python_data_extraction_v4_2_res_v0.py` | Merge pass 0 | First merge with ground truth scores |
| `merge_bash_and_python_data_extraction_v4_2_res_v1.py` | Merge pass 1 | Improved merge with delta scores |
| `merge_bash_and_python_data_extraction_v4_2_res_v2.py` | Merge pass 2 | Timestamp-sorted final merge |

---

## 4. Support and utility scripts

| Script | Role | Notes |
|---|---|---|
| `summary_df.py` | Quick CSV summary printer | Handy for sanity checks |
| `questions_splitter_v0.py` | Splits merged data by question | Useful for per-question analysis |
| `feature_process_v0.py` | Drops selected columns from a dataset | Small preprocessing helper |
| `index_unique_profiles_with_absolute_paths.py` | Builds profile index and manifest | Useful for manual case review |
| `heatmap_complexheatmap_v1_comp.py` | Multi-heatmap visualization suite | Current heatmap helper |
| `hierarchical_clustering_v1_comp.py` | Hierarchical clustering experiment runner | Tests distance/linkage combinations |
| `k-mean_clustering_2026.py` | K-means clustering experiment | Alternate clustering baseline |
| `plots_analytics_after_merge_v0.py` → `plots_analytics_after_merge_v5_2.py` | Analytics evolution series | Older versions kept for traceability |
| `plots_analytics_for_de_v3_v0.py` → `plots_analytics_for_de_v4_2.py` | Extraction-side analytics series | Older versions kept for traceability |
| `w11_generate_mean_clustermap_v6.py` | Weekly mean clustermap export | Experimental / meeting work |
| `w11_generate_median_clustermap_v6.py` | Weekly median clustermap export | Experimental / meeting work |
| `w11_generate_median_clustermap_with_dendrograms.py` | Median clustermap with dendrograms | Experimental / meeting work |
| `w11_sg_median_dendrogram_package.py` | Weekly median dendrogram pack | Experimental / meeting work |

---

## 5. What to use now

- **Main thesis figures:** `student_profiling_workflow_v1_comp_2026.py`, `assign_student_profiles.py`, `enhanced_clustermap_2026.py`, `elbow_curve_2026.py`
- **Survey chapter figures:** the `generate_survey_visualizations_*` scripts and the `br2_*` scripts
- **Appendix / backup visuals:** `enhanced_median_clustermap_2026.py`, `elbow_curve_2026_k1to8.py`, `cross_exercise_tracking_2026.py`, `per_question_profiles_2026.py`
- **Legacy / reference only:** `data_extraction_*`, `merge_bash_and_python_*`, older `plots_analytics_*` variants

---

## 6. Notes

- The current thesis uses the **2026 exam workflow** and the **survey chapter outputs**.
- The 16 profile system is based on four binary dimensions: persistence, effectiveness, iteration speed, and sustained engagement.
- Some scripts are kept only to preserve the development trail and should not be presented as the current production path.
