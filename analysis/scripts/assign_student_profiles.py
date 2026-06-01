#!/usr/bin/env python3
"""
Student Profile Archetype Assignment
====================================

Assigns each student to one of 16 profiles based on 4 binary dimensions:
1. Persistence (mean_attempts >= 15)
2. Effectiveness (mean_best_score >= 50)
3. Iterative (fast_retry_ratio >= 0.50)
4. Sustained (long_pause_ratio <= 0.10)

Outputs:
- student_profile_assignments.csv: hash + 4 dimensions + archetype label
- profile_distribution_summary.csv: archetype counts and characteristics
- profile_distribution_chart.png/pdf
"""

from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_DIR = Path("../../data/v1_comp/res_python/2026_01_workflow")
OUTPUT_DIR = DATA_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds (from taxonomy document)
PERSIST_THRESHOLD = 15.0  # mean_attempts
EFFECTIVENESS_THRESHOLD = 50.0  # mean_best_score
ITERATIVE_THRESHOLD = 0.50  # fast_retry_ratio
SUSTAINED_THRESHOLD = 0.10  # long_pause_ratio

# Profile archetype names (all 16)
PROFILES_16 = {
    (True, True, True, True): "Elite Learner",
    (True, True, True, False): "Effective Sprint",
    (True, True, False, True): "Methodical",
    (True, True, False, False): "Lucky Guesser",
    (True, False, True, True): "Persistent Learner",
    (True, False, True, False): "Thrashing",
    (True, False, False, True): "Stuck Thinker",
    (True, False, False, False): "Struggling Guesser",
    (False, True, True, True): "Efficient Solver",
    (False, True, True, False): "Quick Success",
    (False, True, False, True): "Confident",
    (False, True, False, False): "Lucky (High)",
    (False, False, True, True): "Resigned",
    (False, False, True, False): "Quick Quit",
    (False, False, False, True): "Passive",
    (False, False, False, False): "Checked Out",
}

print("=" * 80)
print("STUDENT PROFILE ARCHETYPE ASSIGNMENT - v1_comp 2026.01")
print("=" * 80)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# LOAD DATA
# ============================================================================
print("[LOAD] Loading student profile features...")

df_student = pd.read_csv(DATA_DIR / "student_profile_features.csv")
df_cluster = pd.read_csv(DATA_DIR / "hierarchical_cluster_assignments.csv")

df_student = df_student.merge(df_cluster, on="hash", how="left")

print(f"  ✓ Loaded {len(df_student)} students")

# ============================================================================
# COMPUTE DIMENSIONS
# ============================================================================
print("\n[DIMENSIONS] Computing binary dimensions...")

# Dimension 1: Persistence
df_student["high_persistence"] = df_student["avg_attempts"] >= PERSIST_THRESHOLD

# Dimension 2: Effectiveness
df_student["high_effectiveness"] = df_student["avg_best_score"] >= EFFECTIVENESS_THRESHOLD

# Dimension 3: Iterative
df_student["iterative"] = df_student["avg_fast_retry_ratio"] >= ITERATIVE_THRESHOLD

# Dimension 4: Sustained
# Note: sustained is defined as LOW long_pause_ratio
long_pause_col = "avg_long_pause_ratio" if "avg_long_pause_ratio" in df_student.columns else "long_pause_ratio"
df_student["sustained"] = df_student[long_pause_col] <= SUSTAINED_THRESHOLD

# Summary statistics
print(f"  Persistence (≥{PERSIST_THRESHOLD} attempts):")
print(f"    - High: {df_student['high_persistence'].sum()} ({df_student['high_persistence'].mean():.1%})")
print(f"    - Low: {(~df_student['high_persistence']).sum()} ({(~df_student['high_persistence']).mean():.1%})")

print(f"\n  Effectiveness (≥{EFFECTIVENESS_THRESHOLD} mean score):")
print(f"    - High: {df_student['high_effectiveness'].sum()} ({df_student['high_effectiveness'].mean():.1%})")
print(f"    - Low: {(~df_student['high_effectiveness']).sum()} ({(~df_student['high_effectiveness']).mean():.1%})")

print(f"\n  Iterative (≥{ITERATIVE_THRESHOLD} fast retries):")
print(f"    - Yes: {df_student['iterative'].sum()} ({df_student['iterative'].mean():.1%})")
print(f"    - No: {(~df_student['iterative']).sum()} ({(~df_student['iterative']).mean():.1%})")

print(f"\n  Sustained (≤{SUSTAINED_THRESHOLD} long pauses):")
print(f"    - Yes: {df_student['sustained'].sum()} ({df_student['sustained'].mean():.1%})")
print(f"    - No: {(~df_student['sustained']).sum()} ({(~df_student['sustained']).mean():.1%})")

# ============================================================================
# ASSIGN PROFILES
# ============================================================================
print("\n[ASSIGN] Assigning 16-profile archetypes...")

def assign_profile_key(row: pd.Series) -> Tuple[bool, bool, bool, bool]:
    """Extract 4-tuple for profile lookup."""
    return (
        bool(row["high_persistence"]),
        bool(row["high_effectiveness"]),
        bool(row["iterative"]),
        bool(row["sustained"]),
    )

df_student["profile_key"] = df_student.apply(assign_profile_key, axis=1)
df_student["profile_archetype"] = df_student["profile_key"].map(PROFILES_16)

print(f"  ✓ Assigned archetypes to all {len(df_student)} students")

# ============================================================================
# ARCHETYPE DISTRIBUTION
# ============================================================================
print("\n[DIST] Profile archetype distribution:")

profile_counts = df_student["profile_archetype"].value_counts().sort_values(ascending=False)

for profile, count in profile_counts.items():
    pct = 100.0 * count / len(df_student)
    print(f"  {profile:25s}: {count:3d} ({pct:5.1f}%)")

# ============================================================================
# SAVE ASSIGNMENTS
# ============================================================================
print("\n[SAVE] Saving profile assignments...")

# Full assignment table
output_file = OUTPUT_DIR / "student_profile_assignments.csv"
df_export = df_student[
    ["hash", "cluster", "avg_attempts", "avg_best_score", "avg_fast_retry_ratio",
     long_pause_col, "high_persistence", "high_effectiveness", "iterative", "sustained",
     "profile_archetype"]
].copy()
df_export.columns = [
    "hash", "cluster", "avg_attempts", "avg_best_score", "fast_retry_ratio",
    "long_pause_ratio", "persistence", "effectiveness", "iterative", "sustained",
    "archetype"
]
df_export.to_csv(output_file, index=False)
print(f"  ✓ {output_file.name}")

# Archetype summary
summary_frames = []
for profile in PROFILES_16.values():
    mask = df_student["profile_archetype"] == profile
    if mask.sum() == 0:
        continue
    
    subset = df_student[mask]
    summary = {
        "profile": profile,
        "count": mask.sum(),
        "pct_cohort": 100.0 * mask.sum() / len(df_student),
        "mean_attempts": subset["avg_attempts"].mean(),
        "mean_score": subset["avg_best_score"].mean(),
        "mean_fast_retry": subset["avg_fast_retry_ratio"].mean(),
        "mean_long_pause": subset[long_pause_col].mean(),
        "success_rate": (subset["avg_best_score"] >= 50).mean(),
        "associated_clusters": subset["cluster"].unique().tolist(),
    }
    summary_frames.append(summary)

df_summary = pd.DataFrame(summary_frames).sort_values("count", ascending=False)
summary_file = OUTPUT_DIR / "profile_distribution_summary.csv"
df_summary.to_csv(summary_file, index=False)
print(f"  ✓ {summary_file.name}")

print("\n" + df_summary.to_string(index=False))

# ============================================================================
# CLUSTER-PROFILE CROSS-TABULATION
# ============================================================================
print("\n[CROSS-TAB] Cluster vs. Profile distribution:")

cross_tab = pd.crosstab(df_student["cluster"], df_student["profile_archetype"], margins=True)
print("\n" + cross_tab.to_string())

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n[PLOT] Creating visualizations...")

# Figure 1: Profile distribution (top 10 archetypes)
fig, ax = plt.subplots(figsize=(12, 6))

top_n = 10
top_profiles = profile_counts.head(top_n)
colors = plt.cm.Set3(np.linspace(0, 1, len(top_profiles)))

bars = ax.barh(range(len(top_profiles)), top_profiles.values, color=colors)
ax.set_yticks(range(len(top_profiles)))
ax.set_yticklabels(top_profiles.index, fontsize=11)
ax.set_xlabel("Number of Students", fontsize=12, fontweight='bold')
ax.set_title(f"Top {top_n} Student Profile Archetypes\nv1_comp 2026.01 (N={len(df_student)})", 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

for i, (bar, count) in enumerate(zip(bars, top_profiles.values)):
    ax.text(count + 5, bar.get_y() + bar.get_height()/2, f"{count} ({100*count/len(df_student):.1f}%)",
            va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "profile_distribution_chart.png", dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "profile_distribution_chart.pdf", bbox_inches='tight')
plt.close()

print("  ✓ profile_distribution_chart.png/pdf")

# Figure 2: Dimension distribution (radar/heatmap style)
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Persistence
dim_persist = df_student.groupby("profile_archetype")["high_persistence"].mean().sort_values(ascending=False)
axes[0, 0].barh(range(len(dim_persist)), dim_persist.values, color='steelblue')
axes[0, 0].set_yticks(range(len(dim_persist)))
axes[0, 0].set_yticklabels(dim_persist.index, fontsize=8)
axes[0, 0].set_xlabel("Proportion High Persistence", fontsize=10)
axes[0, 0].set_title("Persistence Dimension", fontsize=11, fontweight='bold')
axes[0, 0].set_xlim([0, 1])
axes[0, 0].grid(True, alpha=0.3, axis='x')

# Effectiveness
dim_effect = df_student.groupby("profile_archetype")["high_effectiveness"].mean().sort_values(ascending=False)
axes[0, 1].barh(range(len(dim_effect)), dim_effect.values, color='mediumseagreen')
axes[0, 1].set_yticks(range(len(dim_effect)))
axes[0, 1].set_yticklabels(dim_effect.index, fontsize=8)
axes[0, 1].set_xlabel("Proportion High Effectiveness", fontsize=10)
axes[0, 1].set_title("Effectiveness Dimension", fontsize=11, fontweight='bold')
axes[0, 1].set_xlim([0, 1])
axes[0, 1].grid(True, alpha=0.3, axis='x')

# Iterative
dim_iter = df_student.groupby("profile_archetype")["iterative"].mean().sort_values(ascending=False)
axes[1, 0].barh(range(len(dim_iter)), dim_iter.values, color='coral')
axes[1, 0].set_yticks(range(len(dim_iter)))
axes[1, 0].set_yticklabels(dim_iter.index, fontsize=8)
axes[1, 0].set_xlabel("Proportion Iterative", fontsize=10)
axes[1, 0].set_title("Iterative Dimension", fontsize=11, fontweight='bold')
axes[1, 0].set_xlim([0, 1])
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Sustained
dim_sust = df_student.groupby("profile_archetype")["sustained"].mean().sort_values(ascending=False)
axes[1, 1].barh(range(len(dim_sust)), dim_sust.values, color='mediumpurple')
axes[1, 1].set_yticks(range(len(dim_sust)))
axes[1, 1].set_yticklabels(dim_sust.index, fontsize=8)
axes[1, 1].set_xlabel("Proportion Sustained", fontsize=10)
axes[1, 1].set_title("Sustained Engagement Dimension", fontsize=11, fontweight='bold')
axes[1, 1].set_xlim([0, 1])
axes[1, 1].grid(True, alpha=0.3, axis='x')

fig.suptitle("4 Binary Dimensions Across Profiles", fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "profile_dimension_breakdown.png", dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "profile_dimension_breakdown.pdf", bbox_inches='tight')
plt.close()

print("  ✓ profile_dimension_breakdown.png/pdf")

# Figure 3: Cluster vs. Profile (stacked bar)
cluster_profile_counts = pd.crosstab(df_student["cluster"], df_student["profile_archetype"])
cluster_profile_norm = cluster_profile_counts.div(cluster_profile_counts.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(14, 6))

cluster_profile_norm.plot(
    kind='bar',
    stacked=True,
    ax=ax,
    colormap='Set3',
    width=0.7
)

ax.set_xlabel("Cluster", fontsize=12, fontweight='bold')
ax.set_ylabel("Proportion", fontsize=12, fontweight='bold')
ax.set_title("Profile Distribution within Each Cluster", fontsize=13, fontweight='bold')
ax.legend(title="Profile", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.set_xticklabels([f"Cluster {int(c)}" for c in cluster_profile_counts.index], rotation=0)
ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cluster_profile_composition.png", dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "cluster_profile_composition.pdf", bbox_inches='tight')
plt.close()

print("  ✓ cluster_profile_composition.png/pdf")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print(f"Done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print("\nKey Outputs:")
print(f"  1. student_profile_assignments.csv ({len(df_export)} rows)")
print(f"  2. profile_distribution_summary.csv ({len(df_summary)} profiles found)")
print(f"  3. Visualizations (PNG + PDF): distribution, dimensions, cluster-profile cross-tab")
print(f"\nOutput Directory: {OUTPUT_DIR}")
