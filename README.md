# Profiling Novice Programmers' Behavior through the Analysis of Incremental Programming Submissions

Code and results for the master's thesis of **François Junior Meli Ngueunkeung**,
École polytechnique de Louvain, UCLouvain, academic year 2025-2026.

Supervisors: Prof. Kim Mens, Prof. Siegfried Nijssen · Co-supervisor: Guillaume Steveny · Reader: Olivier Goletti

The thesis itself is in [`thesis/`](thesis/).

---

## What this repository is for

The thesis claims, in §1.4 and §4.4, that every figure and every number in it can be traced back to the
code that produced it. This repository is where that claim is checked.

It contains the full analysis pipeline, the aggregate results the thesis reports, and the figures it
prints. **It does not contain the student submission data**; [`docs/DATA.md`](docs/DATA.md) says what the data
is and why it is not published.

This repository is deliberately smaller than the private workspace the thesis was written in. It keeps only
what helps a reader understand and check the results.

## Layout

| directory | what is in it |
|---|---|
| `pipeline/` | the 36 analysis scripts, in run order, plus `common.py` (the single source of truth for the cohort, the clustering recipe and the number of groups) |
| `results/clustering/` | the aggregate outputs the thesis quotes: group signatures, k-selection tables, episode-shape distributions, the year↔exam correspondence |
| `results/audit/` | the data-quality audits of the exam and the coursework, and the feature-build summaries |
| `figures/` | the 36 figures printed in the thesis, prefixed by the chapter that uses them |
| `listings/` | the 16 student code extracts and diffs printed in Chapter 6 and Appendix C |
| `thesis/` | the submitted manuscript |
| `docs/` | how to read all of the above |

## Start here

- **[`docs/TRACEABILITY.md`](docs/TRACEABILITY.md)**: every figure and every headline number in the thesis,
  mapped to the script that produced it and the file that holds it. This is the document that backs the
  §4.4 claim.
- **[`docs/DATA.md`](docs/DATA.md)**: what the raw data is, why it is not here, and how the analysis
  cohort was built.
- **[`pipeline/CODE_INDEX.md`](pipeline/CODE_INDEX.md)**: one line per script: what it does, what it reads,
  what it writes.

## The analysis in one paragraph

Students in an introductory Python course submit code to an autograder, INGInious, which keeps every
attempt rather than only the last. From those histories we build a set of **behavior features** per
student, with the final grade deliberately excluded, and group students with Ward hierarchical clustering:
**4 groups in the January exam, 3 in the coursework**, over the **569 students present in both**. We then
read the same data at a finer level, sorting each run of submissions to a single question into one of six
observational **episode shapes**, and follow individual students through their code. The two groupings turn
out to be only weakly related (Cramér's *V* = 0.159, adjusted Rand index 0.07).

## Reproducing

The scripts run in the numeric order given in `pipeline/CODE_INDEX.md` and expect the raw archive under
`data/v2/last_archive/`, which is not distributed here. Without it the scripts cannot be executed, but
every intermediate result they produce and every figure they draw is included, so their output can be
inspected and checked against the thesis.

Python 3.12, with `numpy`, `scipy`, `matplotlib` and `seaborn`.

## Licence

See [`LICENSE`](LICENSE). The student code extracts in `listings/` are the work of the students who wrote
them and appear here, pseudonymized, only as they appear in the thesis.
