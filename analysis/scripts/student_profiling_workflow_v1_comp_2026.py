#!/usr/bin/env python3
"""
Student Profiling Workflow for v1_comp 2026 Exam
================================================

Purpose:
- Build student-centric behavioral features from raw incremental submissions.
- Compute question-to-question correlations.
- Run hierarchical clustering to visualize natural student groups.

This script is designed for:
  ../../data/v1_comp/extracted_all/2026.exam/2026.01/q*/data.csv

Outputs (under ../../data/v1_comp/res_python/2026_01_workflow):
- student_question_features.csv
- student_profile_features.csv
- student_question_best_scores.csv
- student_question_success.csv
- question_correlation_best_score.csv
- question_correlation_success.csv
- question_correlation_best_score.png
- question_correlation_success.png
- hierarchical_cluster_assignments.csv
- hierarchical_dendrogram.png
- hierarchical_clustermap.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_ROOT = Path("../../data/v1_comp/extracted_all/2026.exam/2026.01")
OUTPUT_DIR = Path("../../data/v1_comp/res_python/2026_01_workflow")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLUSTER_METRIC = "euclidean"
CLUSTER_LINKAGE = "ward"
CLUSTER_N_CANDIDATES = range(2, 9)
MIN_CLUSTER_SIZE = 10


@dataclass
class QuestionStats:
    question: str
    rows: int
    students: int


def _safe_mean_delta_seconds(ts: pd.Series) -> float:
    """Mean delta between consecutive submissions in seconds."""
    ts = ts.sort_values()
    if len(ts) < 2:
        return 0.0
    deltas = ts.diff().dropna().dt.total_seconds()
    return float(deltas.mean()) if not deltas.empty else 0.0


def _safe_ratio(mask: pd.Series) -> float:
    total = len(mask)
    if total == 0:
        return 0.0
    return float(mask.sum() / total)


def get_next_versioned_path(path: Path) -> Path:
    """Return a non-existing output path by appending _vN when needed."""
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    version = 2
    while True:
        candidate = path.with_name(f"{stem}_v{version}{suffix}")
        if not candidate.exists():
            return candidate
        version += 1


def save_current_figure_dual(base_path: Path, dpi: int = 180) -> None:
    """Save current matplotlib figure as both PNG and PDF."""
    png_path = get_next_versioned_path(base_path.with_suffix(".png"))
    pdf_path = get_next_versioned_path(base_path.with_suffix(".pdf"))
    plt.savefig(png_path, dpi=dpi)
    plt.savefig(pdf_path)


def load_exam_data(data_root: Path) -> pd.DataFrame:
    """Load and concatenate all q*/data.csv files for one session."""
    frames: List[pd.DataFrame] = []
    stats: List[QuestionStats] = []

    for q_dir in sorted(data_root.glob("q*")):
        data_file = q_dir / "data.csv"
        if not data_file.exists():
            continue

        df_q = pd.read_csv(data_file)
        df_q["question"] = q_dir.name
        df_q["timestamp"] = pd.to_datetime(df_q["timestamp"], errors="coerce")
        df_q["score"] = pd.to_numeric(df_q["score"], errors="coerce").fillna(0.0)
        df_q["n_submission"] = pd.to_numeric(df_q["n_submission"], errors="coerce").fillna(0).astype(int)
        df_q["status"] = df_q["status"].astype(str).str.lower().fillna("unknown")
        df_q["is_success"] = (df_q["status"] == "success") | (df_q["score"] >= 100.0)

        frames.append(df_q)
        stats.append(QuestionStats(q_dir.name, len(df_q), df_q["hash"].nunique()))

    if not frames:
        raise FileNotFoundError(f"No data.csv found under {data_root}")

    df_all = pd.concat(frames, ignore_index=True)

    print("[LOAD] Questions loaded:")
    for st in stats:
        print(f"  - {st.question}: {st.rows} rows, {st.students} students")

    print(f"[LOAD] Total rows: {len(df_all)}")
    print(f"[LOAD] Total students: {df_all['hash'].nunique()}")
    return df_all


def build_student_question_features(df_all: pd.DataFrame) -> pd.DataFrame:
    """Per-student, per-question behavioral descriptors."""

    def _agg(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("timestamp")

        attempts = int(len(g))
        first_ts = g["timestamp"].iloc[0]
        last_ts = g["timestamp"].iloc[-1]
        first_score = float(g["score"].iloc[0])
        final_score = float(g["score"].iloc[-1])
        best_score = float(g["score"].max())
        solved = bool((g["is_success"]).any())

        first_success_idx = None
        if solved:
            first_success_idx = int(g.index[g["is_success"]].min())
            g_reset = g.reset_index(drop=True)
            first_success_pos = int(g_reset.index[g_reset["is_success"]][0])
            attempts_to_success = first_success_pos + 1
            ts_success = g_reset.loc[first_success_pos, "timestamp"]
            time_to_success_sec = float((ts_success - first_ts).total_seconds()) if pd.notna(ts_success) and pd.notna(first_ts) else np.nan
        else:
            attempts_to_success = np.nan
            time_to_success_sec = np.nan

        span_sec = float((last_ts - first_ts).total_seconds()) if pd.notna(last_ts) and pd.notna(first_ts) else 0.0

        deltas = g["timestamp"].sort_values().diff().dropna().dt.total_seconds()
        fast_retry_ratio = float((deltas <= 60).mean()) if len(deltas) > 0 else 0.0
        long_pause_ratio = float((deltas >= 600).mean()) if len(deltas) > 0 else 0.0

        score_diffs = g["score"].diff().dropna()
        improving_ratio = float((score_diffs > 0).mean()) if len(score_diffs) > 0 else 0.0

        return pd.Series(
            {
                "attempts": attempts,
                "first_score": first_score,
                "final_score": final_score,
                "best_score": best_score,
                "score_gain": best_score - first_score,
                "solved": int(solved),
                "attempts_to_success": attempts_to_success,
                "time_to_success_sec": time_to_success_sec,
                "submission_span_sec": span_sec,
                "mean_delta_sec": _safe_mean_delta_seconds(g["timestamp"]),
                "fast_retry_ratio": fast_retry_ratio,
                "long_pause_ratio": long_pause_ratio,
                "improving_ratio": improving_ratio,
            }
        )

    out = (
        df_all.groupby(["hash", "question"])
        .apply(_agg, include_groups=False)
        .reset_index()
    )

    return out


def build_student_profile_features(df_sq: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-question behavior into student-level profile features."""
    grouped = df_sq.groupby("hash")

    df_student = grouped.agg(
        questions_attempted=("question", "nunique"),
        solved_count=("solved", "sum"),
        solved_rate=("solved", "mean"),
        avg_attempts=("attempts", "mean"),
        median_attempts=("attempts", "median"),
        total_attempts=("attempts", "sum"),
        avg_best_score=("best_score", "mean"),
        avg_final_score=("final_score", "mean"),
        avg_score_gain=("score_gain", "mean"),
        avg_span_sec=("submission_span_sec", "mean"),
        avg_mean_delta_sec=("mean_delta_sec", "mean"),
        avg_fast_retry_ratio=("fast_retry_ratio", "mean"),
        avg_long_pause_ratio=("long_pause_ratio", "mean"),
        avg_improving_ratio=("improving_ratio", "mean"),
    ).reset_index()

    # Use only solved questions for time-to-success summary when possible.
    tts = (
        df_sq[df_sq["solved"] == 1]
        .groupby("hash")["time_to_success_sec"]
        .mean()
        .rename("avg_time_to_success_sec")
    )
    df_student = df_student.merge(tts, on="hash", how="left")

    df_student["never_solved_count"] = (
        df_student["questions_attempted"] - df_student["solved_count"]
    )

    # Impute missing time-to-success using high penalty: 95th percentile observed.
    if df_student["avg_time_to_success_sec"].notna().any():
        penalty = float(df_student["avg_time_to_success_sec"].dropna().quantile(0.95))
    else:
        penalty = 0.0
    df_student["avg_time_to_success_sec"] = df_student["avg_time_to_success_sec"].fillna(penalty)

    return df_student


def build_question_matrices(df_sq: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build student x question matrices for score and success."""
    score_matrix = df_sq.pivot(index="hash", columns="question", values="best_score")
    success_matrix = df_sq.pivot(index="hash", columns="question", values="solved")

    score_matrix = score_matrix.fillna(0.0)
    success_matrix = success_matrix.fillna(0).astype(int)

    return {
        "score_matrix": score_matrix,
        "success_matrix": success_matrix,
    }


def compute_and_save_correlations(score_matrix: pd.DataFrame, success_matrix: pd.DataFrame) -> None:
    """Question-to-question correlations for Task 3."""
    corr_score = score_matrix.corr(method="pearson")
    corr_success = success_matrix.corr(method="pearson")

    corr_score.to_csv(OUTPUT_DIR / "question_correlation_best_score.csv")
    corr_success.to_csv(OUTPUT_DIR / "question_correlation_success.csv")

    plt.figure(figsize=(7, 6))
    sns.heatmap(corr_score, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Question Correlation (Best Scores)")
    plt.tight_layout()
    save_current_figure_dual(OUTPUT_DIR / "question_correlation_best_score", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 6))
    sns.heatmap(corr_success, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Question Correlation (Solved/Not Solved)")
    plt.tight_layout()
    save_current_figure_dual(OUTPUT_DIR / "question_correlation_success", dpi=180)
    plt.close()


def run_hierarchical_clustering(df_student: pd.DataFrame) -> pd.DataFrame:
    """Task 4: Hierarchical clustering on student profile features."""
    feature_cols = [
        "solved_rate",
        "avg_attempts",
        "total_attempts",
        "avg_best_score",
        "avg_final_score",
        "avg_score_gain",
        "avg_time_to_success_sec",
        "avg_span_sec",
        "avg_mean_delta_sec",
        "avg_fast_retry_ratio",
        "avg_long_pause_ratio",
        "avg_improving_ratio",
        "never_solved_count",
    ]

    X = df_student[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Merge the 3 highly-correlated timing features into one composite.
    timing = ["avg_time_to_success_sec", "avg_span_sec", "avg_mean_delta_sec"]
    X["avg_timing_intensity_sec"] = X[timing].mean(axis=1)
    X = X.drop(columns=timing)

    # Merge attempts pair using local z-scores so both columns contribute fairly.
    attempts_pair = X[["avg_attempts", "total_attempts"]].copy()
    attempts_std = attempts_pair.std(ddof=0).replace(0, 1.0)
    attempts_z = (attempts_pair - attempts_pair.mean()) / attempts_std
    X["attempts_volume_signal"] = attempts_z.mean(axis=1)
    X = X.drop(columns=["avg_attempts", "total_attempts"])

    # Merge performance pair (solved rate with average best score).
    # Convert best score to [0, 1] first to align it with solved_rate.
    score_norm = X["avg_best_score"] / 100.0
    X["mastery_signal"] = (X["solved_rate"] + score_norm) / 2.0
    X = X.drop(columns=["solved_rate", "avg_best_score"])

    feature_cols = [
        c
        for c in feature_cols
        if c not in timing + ["avg_attempts", "total_attempts", "solved_rate", "avg_best_score"]
    ] + ["avg_timing_intensity_sec", "attempts_volume_signal", "mastery_signal"]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Z = linkage(Xs, method=CLUSTER_LINKAGE, metric=CLUSTER_METRIC)

    best = None
    for k in CLUSTER_N_CANDIDATES:
        labels_k = fcluster(Z, t=k, criterion="maxclust")
        counts = pd.Series(labels_k).value_counts()
        if counts.min() < MIN_CLUSTER_SIZE:
            continue
        if len(np.unique(labels_k)) < 2:
            continue
        sil_k = silhouette_score(Xs, labels_k)
        if best is None or sil_k > best["silhouette"]:
            best = {"k": k, "labels": labels_k, "silhouette": sil_k}

    if best is None:
        # Fallback to 4 clusters if all candidates violate minimum size.
        fallback_k = 4
        labels = fcluster(Z, t=fallback_k, criterion="maxclust")
        sil = silhouette_score(Xs, labels) if len(np.unique(labels)) > 1 else np.nan
        selected_k = fallback_k
    else:
        labels = best["labels"]
        sil = best["silhouette"]
        selected_k = best["k"]

    df_cluster = df_student[["hash"]].copy()
    df_cluster["cluster"] = labels

    print(f"[CLUSTER] n_students={len(df_student)} n_clusters={selected_k} silhouette={sil:.3f}")

    # Dendrogram
    plt.figure(figsize=(14, 6))
    dendrogram(np.log(Z + 1), no_labels=True, color_threshold=0)  # log-scaled view
    plt.title(f"Hierarchical Dendrogram ({CLUSTER_LINKAGE}, {CLUSTER_METRIC})")
    plt.ylabel("Distance")
    plt.tight_layout()
    save_current_figure_dual(OUTPUT_DIR / "hierarchical_dendrogram", dpi=180)
    plt.close()

    # Clustermap (feature x student): features on y-axis, hashes on x-axis.
    df_plot = pd.DataFrame(np.log(Xs.T + 10), columns=df_student["hash"], index=feature_cols)
    sns.set_context("notebook")
    n_rows, n_cols = df_plot.shape
    fig_h = max(10.0, min(14.0, n_rows * 0.9))
    fig_w = max(10.0, min(18.0, n_cols * 0.02))

    # Use robust limits to increase visual separation while limiting outlier dominance.
    vmin, vmax = np.percentile(df_plot.values, [2, 98])

    g = sns.clustermap(
        df_plot,
        method=CLUSTER_LINKAGE,
        metric=CLUSTER_METRIC,
        figsize=(fig_w, fig_h),
        yticklabels=True,
        xticklabels=False,
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "log(z + 10)"},
    )
    g.fig.suptitle("Student Behavioral Clustermap", y=1.02)
    clustermap_png = get_next_versioned_path(OUTPUT_DIR / "hierarchical_clustermap.png")
    clustermap_pdf = get_next_versioned_path(OUTPUT_DIR / "hierarchical_clustermap.pdf")
    g.savefig(clustermap_png, dpi=180)
    g.savefig(clustermap_pdf)
    plt.close(g.fig)

    return df_cluster


def main() -> None:
    print("=" * 80)
    print("STUDENT PROFILING WORKFLOW - v1_comp 2026.01")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # PRE: Raw -> per-question behavior table
    df_all = load_exam_data(DATA_ROOT)
    df_sq = build_student_question_features(df_all)
    df_sq.to_csv(OUTPUT_DIR / "student_question_features.csv", index=False)
    print(f"[PRE] student_question_features.csv: {df_sq.shape}")

    # IN: student profile matrix + question correlations
    df_student = build_student_profile_features(df_sq)
    df_student.to_csv(OUTPUT_DIR / "student_profile_features.csv", index=False)
    print(f"[IN] student_profile_features.csv: {df_student.shape}")

    matrices = build_question_matrices(df_sq)
    matrices["score_matrix"].to_csv(OUTPUT_DIR / "student_question_best_scores.csv")
    matrices["success_matrix"].to_csv(OUTPUT_DIR / "student_question_success.csv")

    compute_and_save_correlations(matrices["score_matrix"], matrices["success_matrix"])
    print("[IN] Correlation matrices and heatmaps saved")

    # POST: hierarchical clustering outputs
    df_cluster = run_hierarchical_clustering(df_student)
    df_cluster.to_csv(OUTPUT_DIR / "hierarchical_cluster_assignments.csv", index=False)
    print(f"[POST] hierarchical_cluster_assignments.csv: {df_cluster.shape}")

    print("-" * 80)
    print(f"Done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
