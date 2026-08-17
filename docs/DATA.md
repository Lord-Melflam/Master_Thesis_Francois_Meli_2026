# The data, and why it is not in this repository

## What the analysis runs on

Two datasets from the same introductory Python course at UCLouvain, both recorded by the INGInious
autograder, which keeps every submission a student makes rather than only the last.

| dataset | period | content | submissions | students |
|---|---|---|---|---|
| coursework | 2025-2026 Q1 | 11 weekly missions, 99 practice questions | 235,636 | 703 |
| exam | 22 January 2026 | 6 questions, one supervised session | 48,020 | 581 |

Each submission carries a pseudonymous student hash, the question, the submission number, a timestamp, the
score the autograder gave, a status, and the code file itself.

**The 99 coursework questions are the practice items only.** Each mission week also holds a multiple-choice
quiz (no code is written) and a graded assignment with a deadline, the *phase de réalisation*. Neither is
part of the data. Chapter 4 of the thesis explains this, and Chapter 7 records it as a limitation.

## The analysis cohort

- The exam starts at 588 accounts. Seven are staff and test accounts, identified by activity outside the
  exam day and confirmed by the course supervisor, and are removed: **581**.
- The coursework holds **703** students.
- Every grouped result in the thesis uses the **569 students present in both**, so that the two settings
  are always compared on the same people.

`01_exam_audit.py` and `02_missions_audit.py` produce the audits behind those numbers; their summaries are
in `results/audit/`.

## Why the data is not published

The submission records are student work. They reached the author already pseudonymized: each student
appears only as a hash, with no name, student number or other identifying detail, and the author never
held the identifying data. Permission to use them for this thesis was given by the course supervisor.

That permission covers the analysis, not redistribution. Beyond that, §4.3 of the thesis records a real
residual risk, following Ihantola et al. (2015): in programming-process data a student can in principle be
recognised from timestamps, comments or self-chosen variable names. Publishing the full submission history
of 569 identifiable-in-principle students would make that risk considerably worse than showing the short
extracts the thesis prints.

**So the raw archive is not here, and neither are the derived per-student tables**: the behavior features
for each student, the cluster assignments, the per-submission score tables. What is published is aggregate:
group-level signatures, distributions, the selection tables, and the figures.

## What this means for reproducibility

The pipeline cannot be re-executed from this repository alone. What it does allow:

- reading every step of the method as code, in run order (`pipeline/`, and `pipeline/CODE_INDEX.md`);
- checking every figure and every headline number in the thesis against the file that holds it
  (`docs/TRACEABILITY.md`);
- checking the exact definition of any feature, threshold or rule against the line of code that implements
  it, rather than against the prose describing it.

## Requesting access

Access to the underlying submission data is not the author's to grant. Enquiries should go to the course
supervisor, **Prof. Kim Mens** (UCLouvain, ICTEAM), who provided the data and authorised its use for this
work.
