#!/usr/bin/env python3
"""
Enhanced PyComplexHeatmap-style Clustermap for v1_comp 2026.01
==============================================================

Creates publication-quality clustermap with:
- Student x Feature heatmap (hierarchically ordered)
- Left dendrogram (student clustering)
- Top dendrogram (feature clustering)
- Cluster number annotations
- Exam success/fail coloring on side
- Question achievement heatmap on top

Based on style from:
https://dingwb.github.io/PyComplexHeatmap/build/html/notebooks/cpg_modules.html#Plotting-the-Dot-clustermap

Output formats: PDF + PNG
"""

from pathlib import Path
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler

# Configuration
DATA_DIR = Path("../../data/v1_comp/res_python/2026_01_workflow")
OUTPUT_DIR = DATA_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLUSTER_METRIC = "euclidean"
CLUSTER_LINKAGE = "ward"

print("=" * 80)
print("ENHANCED CLUSTERMAP - v1_comp 2026.01")
print("=" * 80)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# LOAD DATA
# ============================================================================
print("[LOAD] Loading 2026 workflow outputs...")

df_student = pd.read_csv(DATA_DIR / "student_profile_features.csv")
df_sq = pd.read_csv(DATA_DIR / "student_question_features.csv")
df_cluster = pd.read_csv(DATA_DIR / "hierarchical_cluster_assignments.csv")
score_matrix = pd.read_csv(DATA_DIR / "student_question_best_scores.csv", index_col=0)
success_matrix = pd.read_csv(DATA_DIR / "student_question_success.csv", index_col=0)

# Merge cluster assignments
df_student = df_student.merge(df_cluster, on="hash", how="left")

print(f"  ✓ Students: {len(df_student)}")
print(f"  ✓ Clusters: {df_student['cluster'].nunique()}")
print(f"  ✓ Features: {df_student.shape[1] - 2} (excluding hash and cluster)")

# ============================================================================
# PREPARE FEATURE MATRIX FOR HEATMAP
# ============================================================================
print("\n[PREP] Preparing feature matrix...")

feature_cols = [
    "solved_rate",
    "avg_attempts",
    "median_attempts",
    "total_attempts",
    "avg_best_score",
    "avg_final_score",
    "avg_score_gain",
    "avg_time_to_success_sec",
    "avg_span_sec",
    "avg_mean_delta_sec",
    "avg_fast_retry_ratio",
    "long_pause_ratio",
    "avg_improving_ratio",
    "never_solved_count",
]

# Filter to features that exist
feature_cols = [c for c in feature_cols if c in df_student.columns]

X = df_student[feature_cols].copy()
X = X.fillna(0).replace([np.inf, -np.inf], 0)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, index=df_student["hash"], columns=feature_cols)

print(f"  ✓ Feature matrix: {X_scaled.shape}")
print(f"  ✓ Standardized (mean=0, std=1)")

# ============================================================================
# COMPUTE HIERARCHIES (students & features)
# ============================================================================
print("\n[CLUSTER] Computing dendrograms...")

# Student dendrogram
Z_students = linkage(X_scaled.values, method=CLUSTER_LINKAGE, metric=CLUSTER_METRIC)
dendro_students = dendrogram(Z_students, no_plot=True)
student_order = dendro_students["leaves"]

# Feature dendrogram
Z_features = linkage(X_scaled.values.T, method=CLUSTER_LINKAGE, metric=CLUSTER_METRIC)
dendro_features = dendrogram(Z_features, no_plot=True)
feature_order = dendro_features["leaves"]

# Reorder
X_ordered = X_scaled.iloc[student_order, feature_order]
student_order_hash = X_scaled.index[student_order].tolist()
student_clusters_ordered = df_student.set_index("hash").loc[student_order_hash, "cluster"].values
question_scores_ordered = score_matrix.loc[student_order_hash]  # Keep original question order
question_solved_ordered = success_matrix.loc[student_order_hash]  # Keep original question order

print(f"  ✓ Student order computed")
print(f"  ✓ Feature order computed")

# ============================================================================
# CREATE COMPLEX HEATMAP FIGURE
# ============================================================================
print("\n[PLOT] Creating complex heatmap figure...")

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(
    4, 4,
    height_ratios=[1, 0.5, 3, 0.3],
    width_ratios=[0.5, 3, 0.3, 0.5],
    hspace=0.1, wspace=0.1
)

# -------- Feature dendrogram (top-left corner) --------
ax_dendro_feat = fig.add_subplot(gs[0, 1])
dendrogram(Z_features, ax=ax_dendro_feat, no_labels=True, color_threshold=0)
ax_dendro_feat.set_xticks([])
ax_dendro_feat.set_ylabel("Dist", fontsize=9)
ax_dendro_feat.spines['right'].set_visible(False)
ax_dendro_feat.spines['top'].set_visible(False)
ax_dendro_feat.spines['bottom'].set_visible(False)

# -------- Question achievement heatmap (row below features) --------
ax_q_score = fig.add_subplot(gs[1, 1])
im_q = ax_q_score.imshow(question_scores_ordered.T, aspect="auto", cmap="YlGn", vmin=0, vmax=100)
ax_q_score.set_yticks(range(len(question_scores_ordered.columns)))
ax_q_score.set_yticklabels(question_scores_ordered.columns, fontsize=9)
ax_q_score.set_xticks([])
ax_q_score.set_title("Question Scores", fontsize=10, fontweight='bold')
plt.colorbar(im_q, ax=ax_q_score, label="Score (%)", orientation="vertical", pad=0.02)

# -------- Main feature heatmap --------
ax_heatmap = fig.add_subplot(gs[2, 1])
im_main = ax_heatmap.imshow(X_ordered.T, aspect="auto", cmap="vlag", vmin=-2, vmax=2)
ax_heatmap.set_yticks(range(len(feature_order)))
ax_heatmap.set_yticklabels(X_ordered.columns, fontsize=9)
ax_heatmap.set_xticks([])
ax_heatmap.set_ylabel("Features", fontsize=11, fontweight='bold')

# -------- Main colorbar --------
ax_cbar_main = fig.add_subplot(gs[2, 3])
cbar_main = plt.colorbar(im_main, cax=ax_cbar_main, label="Standardized Value")
cbar_main.ax.tick_params(labelsize=8)

# -------- Student dendrogram (left) --------
ax_dendro_stud = fig.add_subplot(gs[2, 0])
dendro_stud_plot = dendrogram(Z_students, ax=ax_dendro_stud, no_labels=True, 
                               color_threshold=0, orientation='left')
ax_dendro_stud.set_xlabel("Dist", fontsize=9)
ax_dendro_stud.set_yticks([])
ax_dendro_stud.spines['right'].set_visible(False)
ax_dendro_stud.spines['top'].set_visible(False)
ax_dendro_stud.spines['bottom'].set_visible(False)

# -------- Cluster coloring (left side of heatmap) --------
ax_cluster_colors = fig.add_subplot(gs[2, 2])
cluster_cmap = plt.cm.tab10(np.linspace(0, 1, len(np.unique(student_clusters_ordered))))
cluster_color_map = {c: cluster_cmap[i] for i, c in enumerate(sorted(np.unique(student_clusters_ordered)))}
cluster_colors_array = np.array([cluster_color_map[c] for c in student_clusters_ordered])

ax_cluster_colors.imshow(cluster_colors_array[:, np.newaxis], aspect="auto")
ax_cluster_colors.set_xticks([])
ax_cluster_colors.set_yticks([])
ax_cluster_colors.set_ylabel("Cluster", fontsize=10, fontweight='bold')

# Legend
unique_clusters_sorted = sorted(np.unique(student_clusters_ordered))
legend_patches = [
    mpatches.Patch(color=cluster_color_map[c], label=f"C{int(c)}")
    for c in unique_clusters_sorted
]
ax_cluster_colors.legend(
    handles=legend_patches,
    loc="center left",
    bbox_to_anchor=(1.1, 0.5),
    fontsize=9,
    frameon=True
)

# -------- Exam success coloring (bottom) --------
ax_exam_colors = fig.add_subplot(gs[3, 1])
exam_status = df_student.set_index("hash").loc[student_order_hash, "avg_best_score"].values
exam_success = (exam_status >= 50).astype(int)
exam_color_array = np.array([[0.2, 0.8, 0.2, 1.0] if s else [0.9, 0.2, 0.2, 1.0] for s in exam_success])

ax_exam_colors.imshow(exam_color_array[:, np.newaxis], aspect="auto")
ax_exam_colors.set_xticks([])
ax_exam_colors.set_yticks([])
ax_exam_colors.set_ylabel("Avg Score", fontsize=9)

# Legend for exam
green_patch = mpatches.Patch(color=[0.2, 0.8, 0.2], label="≥50%")
red_patch = mpatches.Patch(color=[0.9, 0.2, 0.2], label="<50%")
ax_exam_colors.legend(
    handles=[green_patch, red_patch],
    loc="center left",
    bbox_to_anchor=(1.1, 0.5),
    fontsize=9
)

# -------- Title --------
fig.suptitle(
    "Student Behavioral Profile Clustermap - v1_comp 2026.01\n"
    f"N={len(df_student)} students | {len(feature_cols)} features | "
    f"Clusters: {', '.join([f'C{int(c)}(n={sum(student_clusters_ordered==c)})' for c in unique_clusters_sorted])}",
    fontsize=14,
    fontweight="bold",
    y=0.995
)

# Save
for fmt_ext, fmt_name in [("png", "png"), ("pdf", "")]:
    if fmt_name:
        out_file = OUTPUT_DIR / f"clustermap_complex_2026.{fmt_ext}"
    else:
        out_file = OUTPUT_DIR / f"clustermap_complex_2026.pdf"
    plt.savefig(out_file, dpi=300 if fmt_ext == "png" else None, bbox_inches="tight")
    print(f"  ✓ Saved: {out_file.name}")

plt.close()

# ============================================================================
# EXTENDED CLUSTERMAP WITH QUESTION DETAILS
# ============================================================================
print("\n[PLOT] Creating extended clustermap (question detail view)...")

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(
    2, 2,
    height_ratios=[3, 1],
    width_ratios=[4, 1],
    hspace=0.2, wspace=0.15
)

# Main heatmap
ax_main = fig.add_subplot(gs[0, 0])
im = ax_main.imshow(X_ordered.T, aspect="auto", cmap="vlag", vmin=-2, vmax=2)
ax_main.set_yticks(range(len(feature_order)))
ax_main.set_yticklabels(X_ordered.columns, fontsize=9)
ax_main.set_xlabel("Students (ordered by cluster)", fontsize=11, fontweight='bold')
ax_main.set_ylabel("Features", fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax_main, label="Standardized Value", pad=0.01)

# Question performance heatmap
ax_q = fig.add_subplot(gs[0, 1])
im_q = ax_q.imshow(question_scores_ordered.T, aspect="auto", cmap="YlGn", vmin=0, vmax=100)
ax_q.set_yticks(range(len(question_scores_ordered.columns)))
ax_q.set_yticklabels(question_scores_ordered.columns, fontsize=10)
ax_q.set_xlabel("Students", fontsize=10)
ax_q.set_title("Score by Question", fontsize=11, fontweight='bold')
plt.colorbar(im_q, ax=ax_q, label="Score (%)", pad=0.02)

# Cluster composition
ax_cluster_comp = fig.add_subplot(gs[1, 0])
cluster_counts = pd.Series(student_clusters_ordered).value_counts().sort_index()
cluster_bars = ax_cluster_comp.bar(
    [f"C{int(c)}" for c in cluster_counts.index],
    cluster_counts.values,
    color=[cluster_color_map[c] for c in cluster_counts.index]
)
ax_cluster_comp.set_ylabel("Student Count", fontsize=10)
ax_cluster_comp.set_title("Cluster Sizes", fontsize=11, fontweight='bold')
ax_cluster_comp.grid(True, alpha=0.3, axis='y')
for bar in cluster_bars:
    height = bar.get_height()
    ax_cluster_comp.text(bar.get_x() + bar.get_width()/2., height,
                         f'{int(height)}', ha='center', va='bottom', fontsize=9)

# Feature importance (variance by cluster)
ax_feat_var = fig.add_subplot(gs[1, 1])
feature_variance = []
for feat_idx, feat_name in enumerate(X_ordered.columns):
    variances_by_cluster = []
    for c in sorted(np.unique(student_clusters_ordered)):
        mask = student_clusters_ordered == c
        var = X_ordered.iloc[mask, feat_idx].var()
        variances_by_cluster.append(var)
    feature_variance.append(np.mean(variances_by_cluster))

top_var_idx = np.argsort(feature_variance)[-5:]
ax_feat_var.barh(
    [X_ordered.columns[i] for i in top_var_idx],
    [feature_variance[i] for i in top_var_idx],
    color="steelblue"
)
ax_feat_var.set_xlabel("Mean Variance", fontsize=10)
ax_feat_var.set_title("Top 5 Differentiating Features", fontsize=11, fontweight='bold')
ax_feat_var.grid(True, alpha=0.3, axis='x')

fig.suptitle(
    "Extended Student Profile Analysis - v1_comp 2026.01",
    fontsize=14, fontweight='bold'
)

for fmt_ext, fmt_name in [("png", "png"), ("pdf", "")]:
    if fmt_name:
        out_file = OUTPUT_DIR / f"clustermap_extended_2026.{fmt_ext}"
    else:
        out_file = OUTPUT_DIR / f"clustermap_extended_2026.pdf"
    plt.savefig(out_file, dpi=300 if fmt_ext == "png" else None, bbox_inches="tight")
    print(f"  ✓ Saved: {out_file.name}")

plt.close()

# ============================================================================
# SAVE CLUSTER INTERPRETATION TABLE
# ============================================================================
print("\n[STATS] Computing cluster profiles...")

cluster_profiles = []
for c in sorted(np.unique(student_clusters_ordered)):
    mask = student_clusters_ordered == c
    cluster_df = df_student.iloc[student_order].iloc[mask]
    
    profile = {
        'cluster': int(c),
        'size': mask.sum(),
        'solved_rate': cluster_df['solved_rate'].mean(),
        'avg_attempts': cluster_df['avg_attempts'].mean(),
        'avg_best_score': cluster_df['avg_best_score'].mean(),
        'avg_time_to_success_sec': cluster_df['avg_time_to_success_sec'].mean(),
        'avg_fast_retry_ratio': cluster_df['avg_fast_retry_ratio'].mean(),
        'avg_long_pause_ratio': cluster_df.get('avg_long_pause_ratio', cluster_df.get('long_pause_ratio', 0)).mean(),
        'never_solved_count': cluster_df['never_solved_count'].mean(),
    }
    cluster_profiles.append(profile)

df_profiles = pd.DataFrame(cluster_profiles)
df_profiles.to_csv(OUTPUT_DIR / "cluster_profiles.csv", index=False)
print(f"  ✓ Saved cluster profiles table")
print("\n" + df_profiles.to_string(index=False))

print("\n" + "=" * 80)
print(f"Done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Outputs: {OUTPUT_DIR.relative_to(Path.cwd())}")
print("=" * 80)
