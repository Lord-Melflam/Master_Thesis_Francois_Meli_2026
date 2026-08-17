"""06 - TRUE per-submission scores for the EXAM (runs each submission against its full test).

WHY: during the exam only the FINAL submission per question is graded on the full test
suite; intermediate INGInious scores are unreliable. Here we re-run every submission
locally against TestQ{N}.py to recover the TRUE score.

Requires: `timeout_decorator` in the venv (the tests import it).

Some questions inject GIVEN scaffolding not present in the student file (q5: base class
`Employe`; q6: a `LinkedList`). We reconstruct those and prepend them; the reconstructed
prelude for question X lives in data/v2/res_python/true_scores/given/{X}_prelude.py and is
validated by canary before use. Each submission is scored in an isolated temp dir; we
record a status (OK / NO_GRADE / TIMEOUT / ...) and never turn a harness failure into a 0.

Modes:
  python3 06_true_scores.py canary
  python3 06_true_scores.py sample q1 150
  python3 06_true_scores.py batch q1 q2 q3 q4 q5   # parallel; writes per-question + merged true scores
"""
import csv
import sys
import glob
import shutil
import random
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count

from common import versioned_path

REPO = Path(__file__).resolve().parents[3]
EXAM_CLEAN_GLOB = str(REPO / "data/v2/res_python/exam_audit/exam_clean_v*.csv")
EXAM_BASE = REPO / "data/v2/last_archive/2026.exam/2026.01_comment"
OUT = REPO / "data/v2/res_python/true_scores"
GIVEN_DIR = OUT / "given"
PY = sys.executable
OUTER_TIMEOUT = 10   # correct runs finish in ~0.2s; 10s safely kills genuine infinite loops
N_WORKERS = max(1, min(cpu_count() - 1, 12))


def latest(pat):
    return sorted(glob.glob(pat))[-1]


def test_path(qname):
    hits = list((EXAM_BASE / qname / "test").glob("*.py"))
    return str(hits[0]) if hits else None


def prelude_for(qname):
    p = GIVEN_DIR / f"{qname}_prelude.py"
    return p.read_text() if p.exists() else ""


def score_one(test_file, student_file, prelude="", timeout_s=OUTER_TIMEOUT):
    """Return (true_score:int|None, status:str)."""
    if not Path(student_file).exists():
        return None, "NO_FILE"
    with tempfile.TemporaryDirectory() as sb:
        shutil.copy(test_file, sb)
        code = (prelude + "\n" if prelude else "") + Path(student_file).read_text(errors="replace")
        (Path(sb) / "qtest.py").write_text(code)
        try:
            subprocess.run([PY, Path(test_file).name], cwd=sb, timeout=timeout_s,
                           stdin=subprocess.DEVNULL,        # input() -> EOFError, not a hang
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return None, "TIMEOUT"
        except Exception:
            return None, "ERROR"
        gf = Path(sb) / "grade.txt"
        if not gf.exists():
            return None, "NO_GRADE"
        try:
            return int(round(sum(float(x) for x in gf.read_text().split() if x.strip()))), "OK"
        except Exception:
            return None, "BAD_GRADE"


def load_exam():
    rows = list(csv.DictReader(open(latest(EXAM_CLEAN_GLOB))))
    last = defaultdict(int)
    for r in rows:
        last[(r["hash"], r["qname"])] = max(last[(r["hash"], r["qname"])], int(r["n_submission"]))
    for r in rows:
        r["_is_final"] = (int(r["n_submission"]) == last[(r["hash"], r["qname"])])
        r["_path"] = str(EXAM_BASE / r["qname"] / "code" / r["file"])
    return rows


def _worker(task):
    tf, sp, prelude, h, q, n, fn = task
    ts, st = score_one(tf, sp, prelude)
    return (h, q, n, ts, st, fn)


# ---------------- modes ----------------
def canary():
    rows = load_exam()
    print("CANARY (expect final@100 -> 100, final@0 -> low). q5/q6 use their given prelude.\n")
    for q in ["q1", "q2", "q3", "q4", "q5", "q6"]:
        tf, pre = test_path(q), prelude_for(q)
        finals = [r for r in rows if r["qname"] == q and r["_is_final"]]
        for r in [x for x in finals if float(x["score"]) == 100][:3] + [x for x in finals if float(x["score"]) == 0][:1]:
            ts, st = score_one(tf, r["_path"], pre)
            flag = "" if (st == "OK" and ts == int(float(r["score"]))) else "  <-- CHECK"
            print(f"  {q} filename={float(r['score']):5.0f} true={ts} [{st}]{flag} {r['hash'][:10]}")
    print()


def sample(qname, n):
    rows = [r for r in load_exam() if r["qname"] == qname]
    sample_rows = random.Random(0).sample(rows, min(n, len(rows)))
    OUT.mkdir(parents=True, exist_ok=True)
    p = versioned_path(OUT, f"sample_{qname}", "csv")
    tf, pre = test_path(qname), prelude_for(qname)
    status = defaultdict(int)
    diffs = {"final": [], "intermediate": []}
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["hash", "qname", "n_submission", "is_final", "filename_score", "true_score", "status"])
        for r in sample_rows:
            ts, st = score_one(tf, r["_path"], pre)
            status[st] += 1
            w.writerow([r["hash"], qname, r["n_submission"], r["_is_final"], float(r["score"]), ts, st])
            if st == "OK":
                diffs["final" if r["_is_final"] else "intermediate"].append(abs(ts - float(r["score"])))
    print(f"SAMPLE {qname}: n={len(sample_rows)} status={dict(status)}")
    for b in ("final", "intermediate"):
        d = diffs[b]
        if d:
            print(f"  {b:12s}: n_OK={len(d)} mismatched={sum(x>0 for x in d)} mean|Δ|={sum(d)/len(d):.1f}")
    print(f"  wrote {p}")


def batch(questions):
    rows = load_exam()
    OUT.mkdir(parents=True, exist_ok=True)
    merged = []
    for q in questions:
        tf, pre = test_path(q), prelude_for(q)
        qrows = [r for r in rows if r["qname"] == q]
        tasks = [(tf, r["_path"], pre, r["hash"], q, int(r["n_submission"]), float(r["score"])) for r in qrows]
        p = versioned_path(OUT, f"true_scores_{q}", "csv")
        status = defaultdict(int)
        done = 0
        print(f"[{q}] {len(tasks)} submissions, {N_WORKERS} workers, prelude={'yes' if pre else 'no'} -> {p.name}", flush=True)
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["hash", "qname", "n_submission", "true_score", "status", "filename_score"])
            with Pool(N_WORKERS) as pool:
                for h, qq, n, ts, st, fn in pool.imap_unordered(_worker, tasks, chunksize=16):
                    w.writerow([h, qq, n, ts if ts is not None else "", st, fn])
                    merged.append((h, qq, n, ts if ts is not None else "", st, fn))
                    status[st] += 1
                    done += 1
                    if done % 2000 == 0:
                        print(f"  [{q}] {done}/{len(tasks)}  status={dict(status)}", flush=True)
        print(f"[{q}] DONE {done}  status={dict(status)}", flush=True)
    pm = versioned_path(OUT, "true_scores_all", "csv")
    with open(pm, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["hash", "qname", "n_submission", "true_score", "status", "filename_score"])
        w.writerows(merged)
    print(f"MERGED -> {pm}  ({len(merged)} rows)", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "canary":
        canary()
    elif len(a) == 3 and a[0] == "sample":
        sample(a[1], int(a[2]))
    elif len(a) >= 2 and a[0] == "batch":
        batch(a[1:])
    else:
        print(__doc__)
