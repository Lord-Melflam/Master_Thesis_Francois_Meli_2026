"""09 - Tier 3: code-quality features from the FINAL (delivered) submission per question.

The metric family every supervisor asked for and none was ever computed. Minimal + justified:
  loc            non-blank lines of code
  nloc           non-comment, non-blank lines
  comment_ratio  comment lines / loc  (documentation habit)
  n_concepts     # distinct programming concepts used (AST): loop, conditional, function,
                 class, comprehension, exception, with, lambda  (0..8) -> the '100% in 3
                 lines vs 50 lines' idea: same outcome, different code sophistication.

Computed on the final submission per (student, question), aggregated per student by median.
Reads the .py files (AST parse; no execution). Exam uses the true-scored clean table; both
exclude the 7 staff hashes.

Outputs (versioned) -> data/v2/res_python/features/
  exam_codequality_features_vN.csv / missions_codequality_features_vN.csv / codequality_summary_vN.txt
"""
import ast
import csv
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np

from common import versioned_path

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "data/v2/res_python"
EXAM_TRUE_GLOB = str(RES / "true_scores/exam_clean_true_v*.csv")
EXAM_CLEAN_GLOB = str(RES / "exam_audit/exam_clean_v*.csv")
EXAM_FLAG_GLOB = str(RES / "exam_audit/exam_flagged_hashes_v*.csv")
EXAM_BASE = REPO / "data/v2/last_archive/2026.exam/2026.01_comment"
MISS_GLOB = str(REPO / "data/v2/last_archive/2025.Q1/mission_*_comment/*/data.csv")
OUT = RES / "features"

CONCEPTS = {
    "loop": (ast.For, ast.While),
    "cond": (ast.If, ast.IfExp),
    "func": (ast.FunctionDef, ast.AsyncFunctionDef),
    "class": (ast.ClassDef,),
    "comprehension": (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
    "exception": (ast.Try,),
    "with": (ast.With, ast.AsyncWith),
    "lambda": (ast.Lambda,),
}
QCOLS = ["median_loc", "median_nloc", "median_comment_ratio", "median_n_concepts"]


def latest(pat):
    return sorted(glob.glob(pat))[-1]


def metrics(path):
    """(loc, nloc, comment_ratio, n_concepts|nan) for one .py file; None if unreadable."""
    try:
        src = Path(path).read_text(errors="replace")
    except FileNotFoundError:
        return None
    nonblank = [l for l in src.splitlines() if l.strip()]
    loc = len(nonblank)
    comments = sum(1 for l in nonblank if l.lstrip().startswith("#"))
    cratio = comments / loc if loc else 0.0
    try:
        types = {type(n) for n in ast.walk(ast.parse(src))}
        nconc = sum(1 for cats in CONCEPTS.values() if any(t in types for t in cats))
    except SyntaxError:
        nconc = np.nan
    return loc, loc - comments, cratio, nconc


def aggregate(finals):
    """finals: dict hash -> list of (loc,nloc,cratio,nconc). -> per-student median rows."""
    out = []
    for h, vals in finals.items():
        a = np.array([v for v in vals if v is not None], dtype=float)
        if a.size == 0:
            continue
        row = {"hash": h}
        for i, c in enumerate(QCOLS):
            col = a[:, i]
            col = col[np.isfinite(col)]
            row[c] = round(float(np.median(col)), 3) if col.size else ""
        out.append(row)
    return out


def exam_finals():
    rows = list(csv.DictReader(open(latest(EXAM_TRUE_GLOB) if glob.glob(EXAM_TRUE_GLOB) else latest(EXAM_CLEAN_GLOB))))
    last = defaultdict(int)
    for r in rows:
        last[(r["hash"], r["qname"])] = max(last[(r["hash"], r["qname"])], int(r["n_submission"]))
    finals = defaultdict(list)
    for r in rows:
        if int(r["n_submission"]) == last[(r["hash"], r["qname"])]:
            finals[r["hash"]].append(metrics(EXAM_BASE / r["qname"] / "code" / r["file"]))
    return finals


def mission_finals():
    staff = {r["hash"] for r in csv.DictReader(open(latest(EXAM_FLAG_GLOB)))}
    rows = []
    for p in sorted(glob.glob(MISS_GLOB)):
        qdir = Path(p).parent
        for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
            if r["hash"] in staff:
                continue
            r["_path"] = qdir / "code" / r["file"]
            r["_qk"] = qdir.parent.name + "/" + r["qname"]
            rows.append(r)
    last = defaultdict(int)
    for r in rows:
        last[(r["hash"], r["_qk"])] = max(last[(r["hash"], r["_qk"])], int(r["n_submission"]))
    finals = defaultdict(list)
    for r in rows:
        if int(r["n_submission"]) == last[(r["hash"], r["_qk"])]:
            finals[r["hash"]].append(metrics(r["_path"]))
    return finals


def write_table(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["hash"] + QCOLS)
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ex = aggregate(exam_finals())
    mi = aggregate(mission_finals())
    pe = versioned_path(OUT, "exam_codequality_features", "csv"); write_table(pe, ex)
    pm = versioned_path(OUT, "missions_codequality_features", "csv"); write_table(pm, mi)
    ps = versioned_path(OUT, "codequality_summary", "txt")
    L = [f"CODE-QUALITY (Tier 3) on final submissions.  exam students={len(ex)}  mission students={len(mi)}",
         "features: " + ", ".join(QCOLS)]
    def m(rows, c):
        v = [float(r[c]) for r in rows if r.get(c) not in ("", None)]
        return round(sum(v) / len(v), 2) if v else float("nan")
    for c in QCOLS:
        L.append(f"  {c}: exam mean={m(ex,c)}  missions mean={m(mi,c)}")
    Path(ps).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote:\n  {pe}\n  {pm}\n  {ps}")


if __name__ == "__main__":
    main()
