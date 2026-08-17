"""34 - Per-exam-cluster diagnostic: do the students' FINAL submissions even parse?

Motivation: exam group E1 (n=15) has a very low "distinct concepts" value and grade ~1.
Because n_concepts is NaN (excluded), not 0, when code fails to parse (script 09), the low
concept value is not a zero-artifact. But the concrete story worth reporting is that E1's code
often does not even parse. This script measures, per canonical exam cluster (assignments_exam.csv),
the share of final submissions with a syntax error, plus median LOC and median concepts, so the
E1 description in Chapter 4 is reproducible.

Reads: clustering/assignments_exam.csv, true_scores/exam_clean_true_v*.csv, the exam .py files.
Writes: clustering/cluster_parse_diagnostic.txt
"""
import ast, csv, glob
from pathlib import Path
from collections import defaultdict
import numpy as np

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "data/v2/res_python"
CLU = RES / "clustering"
EXAM_TRUE_GLOB = str(RES / "true_scores/exam_clean_true_v*.csv")
EXAM_BASE = REPO / "data/v2/last_archive/2026.exam/2026.01_comment"


def latest(pat): return sorted(glob.glob(pat))[-1]


def parses(path):
    try:
        ast.parse(Path(path).read_text(errors="replace"))
        return True
    except FileNotFoundError:
        return None
    except SyntaxError:
        return False


def main():
    asg = {r["hash"]: int(r["cluster"]) for r in csv.DictReader(open(CLU / "assignments_exam.csv"))}
    rows = list(csv.DictReader(open(latest(EXAM_TRUE_GLOB))))
    last = defaultdict(int)
    for r in rows:
        last[(r["hash"], r["qname"])] = max(last[(r["hash"], r["qname"])], int(r["n_submission"]))

    # per cluster: final-submission parse tally, and per-student concept/loc
    tot = defaultdict(int); ok = defaultdict(int); fail = defaultdict(int)
    for r in rows:
        h = r["hash"]
        if h not in asg or int(r["n_submission"]) != last[(h, r["qname"])]:
            continue
        p = parses(EXAM_BASE / r["qname"] / "code" / r["file"])
        c = asg[h]; tot[c] += 1
        if p is True: ok[c] += 1
        elif p is False: fail[c] += 1

    # students whose finals never parse (all NaN concepts): count per cluster
    parse_by_student = defaultdict(lambda: [0, 0])  # hash -> [n_final, n_parsed]
    for r in rows:
        h = r["hash"]
        if h not in asg or int(r["n_submission"]) != last[(h, r["qname"])]:
            continue
        parse_by_student[h][0] += 1
        if parses(EXAM_BASE / r["qname"] / "code" / r["file"]) is True:
            parse_by_student[h][1] += 1
    never = defaultdict(int); nstud = defaultdict(int)
    for h, (nf, npo) in parse_by_student.items():
        nstud[asg[h]] += 1
        if npo == 0: never[asg[h]] += 1

    L = ["PER-EXAM-CLUSTER PARSE DIAGNOSTIC (final submissions per (student, question)).",
         "cluster labels from assignments_exam.csv (ascending mean exam grade).", ""]
    L.append(f"  {'cluster':<8} {'students':>8} {'finals':>7} {'parsed%':>8} {'syntaxerr%':>11} {'no-parse students':>18}")
    for c in sorted(tot):
        t = tot[c]
        L.append(f"  E{c:<7} {nstud[c]:>8} {t:>7} {100*ok[c]/t:>7.0f}% {100*fail[c]/t:>10.0f}% "
                 f"{never[c]:>10} / {nstud[c]}")
    out = CLU / "cluster_parse_diagnostic.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
