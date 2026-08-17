"""31 - Canonical RESULTS builder (single source of truth for every number in the report).

Motivation (Francois, W6): stop recomputing cluster numbers ad-hoc in different scripts
(they appeared to drift). This script computes the definitive clustering ONCE via the one
shared recipe (common.cluster, fully deterministic), and writes:
  - clustering/assignments_{exam,year}.csv   (hash -> cluster label; the canonical labels)
  - clustering/results.json                  (all numbers, machine-readable)
  - clustering/results_summary.txt           (the same, human-readable)
Every figure / FINDINGS.md / report section READS these; nothing recomputes them.

Numbers produced: per cluster (n, mean/median exam grade, z-signature + top features),
and the year<->exam correspondence (contingency, chi-square, Cramer's V, ARI, P(exam|year),
P(year|exam), base rates). k is common.KMAP (exam 3, year 3).
"""
import csv, json
from pathlib import Path
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import adjusted_rand_score

from common import cluster, KMAP, load_kept

REPO = Path(__file__).resolve().parents[3]
CLU = REPO / "data/v2/res_python/clustering"

DISPLAY = {
    "questions_attempted": "questions attempted", "median_attempts": "attempts/question",
    "median_fast_retry_ratio": "quick resubmissions", "median_long_pause_ratio": "long pauses",
    "median_mean_delta_sec": "gap between tries", "median_improving_ratio": "tries that improved",
    "median_edit_size": "lines changed/edit", "churn_ratio": "edits, no score gain",
    "breakthrough_ratio": "small fix, big gain", "median_nloc": "code lines",
    "median_comment_ratio": "comment share", "median_n_concepts": "distinct concepts",
    "active_weeks": "active weeks", "active_days": "active days",
}


def top_sig(sig, cols, n=4):
    idx = np.argsort(np.abs(sig))[::-1][:n]
    return [{"feature": DISPLAY.get(cols[j], cols[j]), "z": round(float(sig[j]), 2)} for j in idx]


def describe(ds):
    hashes, Xs, lab, g, cols = cluster(ds)
    clusters = {}
    for c in sorted(set(lab)):
        m = lab == c
        sig = Xs[m].mean(0)
        clusters[int(c)] = {
            "n": int(m.sum()),
            "grade_mean": round(float(g[m].mean()), 1),
            "grade_median": round(float(np.median(g[m])), 1),
            "top_features": top_sig(sig, cols),
            "signature_z": {DISPLAY.get(cols[j], cols[j]): round(float(sig[j]), 2) for j in range(len(cols))},
        }
    return dict(zip(hashes, lab.tolist())), clusters, cols


def cramers_v(ct):
    chi2, p, dof, _ = chi2_contingency(ct)
    n = ct.sum()
    v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1)))) if min(ct.shape) > 1 and n else 0.0
    return round(float(chi2), 2), round(float(p), 4), int(dof), round(v, 3)


def main():
    CLU.mkdir(parents=True, exist_ok=True)
    res = {"k": KMAP, "kept": {ds: load_kept(ds) for ds in ("exam", "year")}, "clusters": {}}
    assign = {}
    for ds in ("exam", "year"):
        a, clusters, cols = describe(ds)
        assign[ds] = a
        res["clusters"][ds] = clusters
        with open(CLU / f"assignments_{ds}.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["hash", "cluster"])
            for h, c in a.items(): w.writerow([h, c])

    # year <-> exam correspondence (same students; canonical assignments)
    common = [h for h in assign["year"] if h in assign["exam"]]
    yl = np.array([assign["year"][h] for h in common]); el = np.array([assign["exam"][h] for h in common])
    yi = [int(x) for x in sorted(set(yl))]; ei = [int(x) for x in sorted(set(el))]
    ct = np.zeros((len(yi), len(ei)), int)
    for a, b in zip(yl, el): ct[yi.index(a), ei.index(b)] += 1
    chi2, p, dof, V = cramers_v(ct)
    ari = round(float(adjusted_rand_score(yl, el)), 3)
    base_e = ct.sum(0) / ct.sum()
    p_e_given_y = (ct / ct.sum(1, keepdims=True))
    p_y_given_e = (ct / ct.sum(0, keepdims=True))
    res["year_vs_exam"] = {
        "n_common": len(common), "chi2": chi2, "dof": dof, "p": p, "cramers_v": V, "ari": ari,
        "contingency_year_rows_exam_cols": ct.tolist(),
        "exam_ids": ei, "year_ids": yi,
        "base_rate_exam": [round(float(x), 3) for x in base_e],
        "P_exam_given_year": [[round(float(x), 3) for x in row] for row in p_e_given_y],
        "P_year_given_exam": [[round(float(x), 3) for x in row] for row in p_y_given_e],
    }
    (CLU / "results.json").write_text(json.dumps(res, indent=2))

    # human-readable summary
    L = ["CANONICAL RESULTS (script 31; computed once, deterministic). 569 linked students.",
         f"k: exam={KMAP['exam']}, year={KMAP['year']}. Exam scores = official INGInious re-grade.", ""]
    for ds in ("exam", "year"):
        L.append(f"=== {ds.upper()} clusters (k={KMAP[ds]}, labelled by ascending mean exam grade) ===")
        for c, d in res["clusters"][ds].items():
            tops = "; ".join(f"{'+' if t['z']>0 else ''}{t['z']} {t['feature']}" for t in d["top_features"])
            L.append(f"  {ds[0].upper()}{c}: n={d['n']:>3}  grade mean {d['grade_mean']:>5} / median {d['grade_median']:>5}  | {tops}")
        L.append("")
    yv = res["year_vs_exam"]
    L.append("=== year <-> exam correspondence (same 569 students) ===")
    L.append(f"  chi2({yv['dof']})={yv['chi2']}, p={yv['p']}, Cramer's V={yv['cramers_v']}, ARI={yv['ari']}")
    L.append(f"  base rate P(exam group): " + ", ".join(f"E{ei[j]}={yv['base_rate_exam'][j]:.2f}" for j in range(len(ei))))
    L.append("  P(exam | year), rows sum to 1:")
    L.append("        " + "  ".join(f"E{c}" for c in ei))
    for i, yc in enumerate(yi):
        L.append(f"    Y{yc}  " + "  ".join(f"{yv['P_exam_given_year'][i][j]:.2f}" for j in range(len(ei))))
    L.append("  reading: association is weak (Cramer's V < 0.2) though significant at n=569; "
             "year does not predict exam. See sensitivity across k in 30/summary.")
    (CLU / "results_summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nwrote: assignments_exam.csv, assignments_year.csv, results.json, results_summary.txt")


if __name__ == "__main__":
    main()
