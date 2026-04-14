# GEPA Optimization Run — `optimize_20260413_234031`

Multi-component GEPA evolution of the `ManipulateSpreadsheet` signature
docstring and the `libreoffice_spreadsheet_skill` instructions using a
SURGICAL RLM proposer.

---

## TL;DR

| metric         | value                                |
|:---------------|:-------------------------------------|
| Best val score | 0.6810  (cand[12])                   |
| Seed val score | 0.5662                               |
| Improvement    | +0.1148 absolute  /  +20.3% relative |
| Candidates     | 22                                   |
| Metric calls   | 2035 / 2000 budget  (102%)           |
| Duration       | 7h 3m                                |
| Total cost     | $112.31                              |

Two candidates tied at the top: cand[12] (discovered first, picked as best)
and cand[17] (a crossover of cand[12]'s signature with cand[13]'s skill).

---

## Configuration

| role                  | model                     | notes                                        |
|:----------------------|:--------------------------|:---------------------------------------------|
| main LM (rollouts)    | openai/mercury-2          | reasoning_effort=low                         |
| sub LM (predict())    | openai/gpt-5.1            | unused by this RLM                           |
| proposer / reflection | anthropic/claude-opus-4-6 | reasoning_effort=none (no extended thinking) |

| GEPA knob        | value                                        |
|:-----------------|:---------------------------------------------|
| proposer         | SURGICAL RLM proposer (--rlm_proposer)       |
| module selector  | round-robin (alternates signature ↔ skill)   |
| train dataset    | trainset  /  val_ratio=0.20  /  val_limit=50 |
| cases per task   | 1                                            |
| minibatch size   | 20                                           |
| max metric calls | 2000                                         |
| seed             | 42                                           |

---

## Costs

LiteLLM authoritative — cost and tokens pulled from `lm.history` after
every proposer and rollout call.

| role     | model                     |  calls |  in tokens | out tokens |   cost |
|:---------|:--------------------------|-------:|-----------:|-----------:|-------:|
| main     | openai/mercury-2          | 10,000 | 86,600,829 |  4,317,947 | $24.89 |
| proposer | anthropic/claude-opus-4-6 |    634 | 14,463,287 |    295,055 | $87.42 |

| bucket                  |    cost | share |
|:------------------------|--------:|------:|
| rollouts (main + sub)   |  $24.89 |   22% |
| optimization (proposer) |  $87.42 |   78% |
| total                   | $112.31 |  100% |

Proposer dominates at 78% of spend — expected shape for opus-4.6 evolving
two components against a mercury-2 rollout loop, and it's what makes running
past the discovery point (iter 18) relatively expensive.

---

## Candidates — ranked by val score

Full 50-task val evaluation; 22 candidates including the seed.

| rank | idx |    val |  Δ seed | parent  | born@ | update             |
|-----:|----:|-------:|--------:|:--------|------:|:-------------------|
|    1 |  12 | 0.6810 | +0.1148 | 10      |  1185 | sig  <- BEST       |
|    2 |  17 | 0.6810 | +0.1148 | 12 + 13 |  1615 | crossover  <- BEST |
|    3 |  13 | 0.6562 | +0.0899 | 10      |  1275 | skill              |
|    4 |   3 | 0.6496 | +0.0833 | 0       |   350 | sig                |
|    5 |   2 | 0.6239 | +0.0577 | 1       |   220 | sig                |
|    6 |   8 | 0.6239 | +0.0577 | 2 + 7   |   695 | crossover          |
|    7 |  18 | 0.6225 | +0.0563 | 17      |  1705 | skill              |
|    8 |  20 | 0.6225 | +0.0563 | 15 + 18 |  1895 | crossover          |
|    9 |  10 | 0.6195 | +0.0532 | 5 + 7   |   840 | crossover          |
|   10 |  15 | 0.6170 | +0.0508 | 9       |  1465 | skill              |
|   11 |   7 | 0.6129 | +0.0467 | 3       |   640 | sig                |
|   12 |   9 | 0.6105 | +0.0443 | 7       |   785 | skill              |
|   13 |  19 | 0.6088 | +0.0426 | 9       |  1840 | sig                |
|   14 |   4 | 0.5979 | +0.0317 | 1 + 3   |   405 | crossover          |
|   15 |   6 | 0.5979 | +0.0317 | 2 + 3   |   550 | crossover          |
|   16 |  11 | 0.5947 | +0.0285 | 3       |  1010 | skill              |
|   17 |  16 | 0.5876 | +0.0214 | 13      |  1560 | sig                |
|   18 |   1 | 0.5871 | +0.0209 | 0       |   130 | skill              |
|   19 |   5 | 0.5799 | +0.0136 | 3       |   495 | skill              |
|   20 |  14 | 0.5785 | +0.0123 | 12      |  1370 | skill              |
|   21 |   0 | 0.5662 | +0.0000 | -       |     0 | SEED               |
|   22 |  21 | 0.5457 | -0.0205 | 8       |  1985 | sig                |

cand[12] and cand[17] are tied at 0.6810. GEPA picked cand[12] as `best_idx`
because it was discovered first.

---

## Candidates — in birth order

| idx |    val |  Δ seed | parent  | born@ | update    |
|----:|-------:|--------:|:--------|------:|:----------|
|   0 | 0.5662 | +0.0000 | -       |     0 | SEED      |
|   1 | 0.5871 | +0.0209 | 0       |   130 | skill     |
|   2 | 0.6239 | +0.0577 | 1       |   220 | sig       |
|   3 | 0.6496 | +0.0833 | 0       |   350 | sig       |
|   4 | 0.5979 | +0.0317 | 1 + 3   |   405 | crossover |
|   5 | 0.5799 | +0.0136 | 3       |   495 | skill     |
|   6 | 0.5979 | +0.0317 | 2 + 3   |   550 | crossover |
|   7 | 0.6129 | +0.0467 | 3       |   640 | sig       |
|   8 | 0.6239 | +0.0577 | 2 + 7   |   695 | crossover |
|   9 | 0.6105 | +0.0443 | 7       |   785 | skill     |
|  10 | 0.6195 | +0.0532 | 5 + 7   |   840 | crossover |
|  11 | 0.5947 | +0.0285 | 3       |  1010 | skill     |
|  12 | 0.6810 | +0.1148 | 10      |  1185 | sig       |
|  13 | 0.6562 | +0.0899 | 10      |  1275 | skill     |
|  14 | 0.5785 | +0.0123 | 12      |  1370 | skill     |
|  15 | 0.6170 | +0.0508 | 9       |  1465 | skill     |
|  16 | 0.5876 | +0.0214 | 13      |  1560 | sig       |
|  17 | 0.6810 | +0.1148 | 12 + 13 |  1615 | crossover |
|  18 | 0.6225 | +0.0563 | 17      |  1705 | skill     |
|  19 | 0.6088 | +0.0426 | 9       |  1840 | sig       |
|  20 | 0.6225 | +0.0563 | 15 + 18 |  1895 | crossover |
|  21 | 0.5457 | -0.0205 | 8       |  1985 | sig       |

---

## Iterations — in order (33 total)

Each iteration either selects a parent candidate and proposes new text for
one component (sig or skill), or performs a crossover merging two existing
candidates. A proposal is accepted iff its minibatch score beats the parent's.

| iter | component | parent |    old |    new |       Δ | outcome                 |
|-----:|:----------|-------:|-------:|-------:|--------:|:------------------------|
|    0 | sig       |      0 | 0.7102 | 0.6219 | -0.0884 | REJECT                  |
|    1 | skill     |      0 | 0.4553 | 0.5682 | +0.1129 | accepted -> cand[1]  *  |
|    2 | sig       |      1 | 0.6706 | 0.6904 | +0.0198 | accepted -> cand[2]     |
|    3 | skill     |      2 | 0.4072 | 0.3746 | -0.0326 | REJECT                  |
|    4 | sig       |      0 | 0.6292 | 0.7783 | +0.1491 | accepted -> cand[3]  *  |
|    5 | crossover |      - |      - |      - |       - | accepted -> cand[4]     |
|    6 | skill     |      3 | 0.5948 | 0.6825 | +0.0877 | accepted -> cand[5]     |
|    7 | crossover |      - |      - |      - |       - | accepted -> cand[6]     |
|    8 | sig       |      3 | 0.5167 | 0.5898 | +0.0730 | accepted -> cand[7]     |
|    9 | crossover |      - |      - |      - |       - | accepted -> cand[8]     |
|   10 | skill     |      7 | 0.4540 | 0.5388 | +0.0847 | accepted -> cand[9]     |
|   11 | crossover |      - |      - |      - |       - | accepted -> cand[10]    |
|   12 | sig       |     10 | 0.6692 | 0.6272 | -0.0419 | REJECT                  |
|   13 | skill     |      8 | 0.4989 | 0.4734 | -0.0255 | REJECT                  |
|   14 | skill     |      3 | 0.5781 | 0.6761 | +0.0979 | accepted -> cand[11]    |
|   15 | crossover |      - |      - |      - |       - | REJECT                  |
|   16 | skill     |     10 | 0.4076 | 0.3871 | -0.0205 | REJECT                  |
|   17 | sig       |      9 | 0.7012 | 0.5437 | -0.1575 | REJECT                  |
|   18 | sig       |     10 | 0.5610 | 0.6695 | +0.1086 | accepted -> cand[12]  * |
|   19 | skill     |     10 | 0.3943 | 0.4966 | +0.1023 | accepted -> cand[13]  * |
|   20 | crossover |      - |      - |      - |       - | REJECT                  |
|   21 | skill     |     12 | 0.4443 | 0.5627 | +0.1184 | accepted -> cand[14]  * |
|   22 | crossover |      - |      - |      - |       - | REJECT                  |
|   23 | skill     |      9 | 0.5501 | 0.5761 | +0.0260 | accepted -> cand[15]    |
|   24 | crossover |      - |      - |      - |       - | REJECT                  |
|   25 | sig       |     13 | 0.5326 | 0.6031 | +0.0705 | accepted -> cand[16]    |
|   26 | crossover |      - |      - |      - |       - | accepted -> cand[17]    |
|   27 | skill     |     17 | 0.7057 | 0.7955 | +0.0898 | accepted -> cand[18]    |
|   28 | crossover |      - |      - |      - |       - | REJECT                  |
|   29 | sig       |     17 | 0.5958 | 0.5386 | -0.0572 | REJECT                  |
|   30 | sig       |      9 | 0.6232 | 0.7925 | +0.1693 | accepted -> cand[19]  * |
|   31 | crossover |      - |      - |      - |       - | accepted -> cand[20]    |
|   32 | sig       |      8 | 0.6850 | 0.7122 | +0.0272 | accepted -> cand[21]    |

`*` marks strong minibatch wins (Δ ≥ +0.10) — note how many of these
(iter 27, iter 30, iter 32) did **not** translate to full-val improvements.

---

## Outcome summary

|                         |                            |
|:------------------------|:---------------------------|
| Direct proposer accepts | 15 / 22                    |
| Direct proposer rejects | 7                          |
| Crossover accepts       | 6 / 11                     |
| Crossover rejects       | 5                          |
| Signature iterations    | 11                         |
| Skill iterations        | 11                         |
| Crossover iterations    | 11                         |
| Seed -> best            | 0.5662 -> 0.6810  (+20.3%) |

### Lineage of the best candidate

cand[12] is the linear descendant of four ancestors through its first-parent chain:

    cand[0] (0.5662)  ->  cand[3] (0.6496)  ->  cand[5] (0.5799)  ->  cand[10] (0.6195)  ->  cand[12] (0.6810)

cand[17] combines cand[12]'s signature with cand[13]'s skill:

    cand[0] (0.5662)  ->  cand[3] (0.6496)  ->  cand[5] (0.5799)  ->  cand[10] (0.6195)  ->  cand[12] (0.6810)  ->  cand[17] (0.6810)
      (merged with cand[13] @ 0.6562)

### Plateau after iter 18

cand[12] (born at metric call 1185, iter 18) was never beaten on the full
val. The remaining 14 iterations (iter 19-32, metric calls 1186-2035) produced
7 more accepted candidates but none improved on 0.6810, and several proposals
showed large minibatch gains that did NOT generalize to the 50-task val.
For example, iter 30 accepted cand[19] with Δ = +0.1693 on the minibatch but
the full-val score landed at 0.6088, below the best. That is the signature of
overfitting to the minibatch and the reason more budget of the same shape is
unlikely to pay off.

