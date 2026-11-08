# DESIGN.md — House Edge forecaster (Main round, 60 pts)

Written 2026-11-07 13:45 CT. **Main deadline 21:00 CT (about 7 h left).** This is a design only: no repo code was changed and no model was implemented.
Legend: **[C]** confirmed by measurement on this kit's files during this session · **[H]** hypothesis that needs an experiment · **[D]** design decision.
The analysis scripts used for the measurements are in the session scratchpad. §9 has WS3 port them to `research/profile_kit.py` so they can be rerun on the adaptation kit.

---

## 0. Executive summary

1. **[C] The target factorizes.** The true next-spin distribution is `p = [1−w, w·s]`. Here `w = P(win)` and `s` is one of exactly **two fixed win-size vectors per machine type**. Across 768 validation targets, the std within each cluster is about 1e-15.
2. **[C] Almost all error is in `w`.** An oracle that knows `w` and uses a fixed size mixture scores **56.7/60**. Knowing the size regime as well adds the last 3.3 points.
3. **[C] `w` follows a per-machine XOR/XNOR rule on two lagged win indicators.** For example, the machine wins mostly when `win(t−2) ≠ win(t−5)`, or mostly when they are equal. In this round, the pairs **(2,5), (4,7) and (1,3)** are the strongest pair for 96% of stable machines (78% overall). Whether the rule is XOR or XNOR differs from machine to machine.
4. **[C] Each condition has a different time structure:**
   - `stable`: the rule is constant.
   - `drift`: the rule **flips sign** about 74% of the way through the history. The flip time differs per machine (std about 18 ticks within an episode).
   - `shared_change`: the rule flips at nearly the **same tick for every machine in the episode** (std about 4 ticks).
   - `short_history`: identifiable from observable coverage (< 0.5). Its 40 consecutive recent ticks are enough to read lags 1–8.
5. **[C] Headroom check (diagnostic only, not a candidate).** I wrote a crude, untuned lookup of recency-weighted `P(win | pair cell)` with a fixed size mix. It scored **≈ 50.0/60 on the full validation set**, against 30.83 for `pooled_recent`. See §6.5 for what this did to validation freshness.
6. **[D] Plan:**
   - Upload the `pooled_recent` fallback now.
   - EXP‑1: Bayesian rule-averaging model (stdlib only).
   - EXP‑2: change-point handling with episode pooling.
   - EXP‑3: size-regime posterior plus `w` calibration.
   - Logistic, GBM and blend work only after those, and only if time remains.

---

## 1. Problem understanding

**Objective.** For each query `(episode, machine, cutoff)`, estimate the simulator's true distribution over payouts `[0,1,2,5,20]` for the spin at tick `cutoff`. Every input must come from ticks `< cutoff`, on every machine.

**Inputs.**

| Source | Fields | Notes |
|---|---|---|
| `queries.csv` | `query_id, episode_id, machine_id, machine_type ∈ {steady, balanced, jackpot}, signal ∈ {−1,0,1}, cutoff` | `signal` is a query-time categorical |
| `history.csv` | `episode_id, machine_id, tick, payout` | missing ticks are unknown, not zero |
| train only | `labels.csv` (one realized outcome per query), `conditions.csv` | |
| validation only | the same, plus `targets.csv` (true probabilities) and `reference.csv` (frozen `pooled`) | |

**Target.** For validation, the true probability vector is given. For test it is hidden. One realized outcome is a noisy sample of it.

**Official score** (`contest.py`, `scoring.py`; the organizer's files, do not edit):
```
err(query)     = Σ_k (p_k − q_k)²                       over the 5 payouts
E[cond]        = mean over episodes in cond of ( mean over the episode's 8 queries )
ratio[cond]    = E[cond] / max(Ref[cond], 1e-9)         Ref = frozen pooled(strength 20), fit on this round's train
R              = mean of the 4 condition ratios
points         = 60 / (1 + R)
```
What this formula implies **[C]**:
- Every condition counts 25% and every episode counts equally within its condition.
- Points are convex in R, with `dP/dR = −60/(1+R)²`. Near R = 0.2, each 0.01 of R is worth about **0.42 pts**. Near R = 1 it is worth 0.15 pts.
- Reference errors on validation: drift .1898, shared_change .2056, short_history .1773, stable .1963.

**Key identity used for leakage-free evaluation [C, algebra]:**

`E_y[Brier(p, y)] = ‖p − q‖² + (1 − ‖q‖²)`

The second term does not depend on the model. So the difference in **realized-label** Brier between two models is an **unbiased estimate of the difference in official per-query error**. That means the train labels and pseudo-queries (§6) can rank models without validation targets.

---

## 2. Existing code analysis

### 2.1 The six baselines (`slot_forecasting/baseline.py`, `FrequencyForecaster`)

All six are **order-free frequency models**. `fit` counts payouts over every train history row plus the train labels. They differ only in the prior and the weighting.

| Method | Prediction | Val pts | drift | shared | short | stable |
|---|---|---|---|---|---|---|
| `global` | Laplace(+0.5) frequency over all train data | 26.730 | 1.273 | 1.200 | 1.255 | 1.251 |
| `context` | per `type\|signal` frequency, shrunk to global (strength 12) | 29.685 | 1.030 | 0.985 | 1.080 | 0.990 |
| `empirical` | own-history counts + 1× global prior | 29.918 | 1.002 | 1.005 | 1.007 | 1.007 |
| `recent` | own counts with exp decay (half-life 24) + 1× global prior | 30.743 | 0.938 | 0.865 | 1.018 | 0.986 |
| `pooled` (reference) | own counts + 20× context prior | 30.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `pooled_recent` | decayed own counts (h = 24) + 20× context prior | **30.826** | 0.946 | 0.881 | 1.001 | 0.958 |

These come from `outputs/run.json` **[C]** and match the numbers you supplied. The root `entry.zip` and `outputs/` hold the **`pooled`** run (30.000). They are not `pooled_recent`. Keep them.

### 2.2 Other components

- **`transition.py` `TransitionForecaster`.**
  - It conditions on an exact match of 1–3 lag contexts, each mapped to 3 categories (`{0}`, `{1}`, `{2,5,20}`). It uses decay and falls back to `pooled` with strength 8.
  - The experiment list tries single lags {1,2,4,8} and the pair (1,2). **None of these is an active pair in this round.** An exact-match pair context also has 9 cells, where XOR has 2, so it needs far more data.
- **`experiments.py`.**
  - It builds 12 candidates and splits validation episodes per condition with a SHA-256 ordering into 12 tuning and 12 audit episodes. It selects on tuning and scores audit once.
  - It writes CSVs and JSON only. There is **no saved model and no entry folder**, so custom packaging is needed.
- **`io.py` / `schema.py`.**
  - Strict standard-library CSV parsing. `index_history` raises on any tick ≥ cutoff and on duplicates, which makes it a **useful leak guard to reuse**.
  - `probabilities()` requires each value in [0,1] and a sum within 1e-6, then renormalizes. `write_submission` writes `.17g`.
  - Loading takes about 2 s per split.
- **`starter.py`.** Compares the six methods but **saves `pooled` by default**. `best_validation_method` is reported but not used. It checks that the saved model reproduces predictions within 1e-14.
- **`build_entry.py --starter`.** Only packages `FrequencyForecaster`. A custom model needs `--source <folder>`.

**Limitations that matter.** Order-free frequencies cannot represent `w` swinging between about 0.02 and 0.95 from one tick to the next on the same machine. Even the type × signal mean of the true targets (an oracle) only reaches 29.97 pts. Frequency models are capped near R ≈ 0.9.

---

## 3. Data analysis

### 3.1 Sizes and structure [C]

| | train | validation | test |
|---|---|---|---|
| episodes / queries | 96 / 768 | 96 / 768 | 192 / 1,536 |
| history rows | 139,944 | 141,392 | 282,664 |
| episodes per condition | 24 | 24 | 48 (metadata; 48 short episodes confirmed by coverage) |

- Each episode has 8 machines, with a mix of types (for example 4 steady, 2 balanced, 2 jackpot). Cutoffs run from 192 to 256 and are the same for all machines in an episode.
- **Every machine's last observed tick is `cutoff − 1`.**
- Full-history machines have every tick from 0 to cutoff−1. Short-history machines have 64 ticks: **40 consecutive recent ticks plus 24 sampled older ticks**. The sampled tick sets differ across machines in the same episode.
- **`short_history` ⇔ coverage < 0.5** in all three splits. Coverage is an observable feature, so it is legitimate as a gate.
- Signal is roughly uniform. Type × signal cells hold 70–100 queries per split.

### 3.2 Payouts [C]

- History payout frequencies (train): 0: .583, 1: .193, 2: .147, 5: .064, 20: .013. Validation and test agree within about 0.02.
- By type (train history):

  | type | 0 | 1 | 2 | 5 | 20 |
  |---|---|---|---|---|---|
  | steady | .42 | .36 | .18 | .05 | .001 |
  | balanced | .59 | .14 | .18 | .07 | .012 |
  | jackpot | .78 | .05 | .07 | .08 | .029 |

- Historical win rate rises with signal (−1/0/+1) for every type in this round's train and validation data: balanced .36/.41/.47, jackpot .17/.21/.28, steady .53/.56/.64.
  - **[H]** The direction may differ in adaptation, so learn it; do not hard-code it.

### 3.3 Target structure (validation targets) [C]

- **`w` is bimodal.** 273 of 768 queries have `w ≤ 0.10` and 297 have `w > 0.70`. Max-probability > 0.95 for 220 queries.
- **Exact size vectors** (`P(payout | win)`, order 1/2/5/20). Size-regime A is chosen by about 54% of queries, independent of signal and condition, with no clustering within episodes.

  | type | regime A | regime B |
  |---|---|---|
  | steady | .7540 .1968 .0482 .0010 | .4551 .4359 .1068 .0022 |
  | balanced | .4999 .3479 .1305 .0217 | .2141 .5467 .2050 .0342 |
  | jackpot | .3351 .2449 .3149 .1050 | .1208 .3239 .4165 .1388 |

- **Oracle ceilings** (use targets; not achievable):
  - true `w` + a fixed 54/46 size mix → **56.69 pts** (all ratios about 0.06)
  - true `w` + true regime → 60
  - type-mean `w` + true sizes → mean per-query error 0.188, against 0.196 for type means of everything. A negligible gain.
- **The size regime is not a static machine property, and it is not lag-driven.**
  - The sign of the full-history likelihood ratio, computed with the exact vectors, recovers the query's regime only **68%** of the time. It plateaus at about 71% even with 80+ observed wins.
  - Within-machine lag and pair effects on payout size match a permutation null (median .292 observed vs .273 null).
  - **[H]** The regime switches over time in some way not yet identified. The best available tool is a tempered posterior (EXP‑3), which recovers a small part of the 3.3-point gap.

### 3.4 Temporal patterns: what drives `w` [C]

- **Single lags, within machine.** Mean |P(win | lag-k win) − P(win | lag-k loss)| is about 0.09 for k ≠ 7, which is near binomial noise (about 0.07). For **lag 7 it is 0.18**, mostly positive (+0.14 to +0.16 by type). That is an extra additive effect specific to this round.
- **Pair interactions dominate.**
  - The per-machine XOR contrast, `P(win | lags differ) − P(win | equal)`, for the best of 28 pairs is often > 1.0 on the interaction scale. The null 95th percentile is 0.28.
  - Strongest pair is (2,5), (4,7) or (1,3) for: stable 96%, short 84%, drift 72%, shared 59%. In drift and shared, the full-history average cancels around the flip.
  - Most machines have **exactly one** active pair. In validation, 356 have one, 206 have none above 0.3, and 14 have two.
  - XOR (as opposed to XNOR) is 42–52% for steady and balanced. Jackpot is about 33% XOR.
- **Time dynamics of the dominant-pair contrast**, oriented so that Q4 > 0, by history quarter:

  | condition | Q1 | Q2 | Q3 | Q4 | Q1 sign opposite to Q4 |
  |---|---|---|---|---|---|
  | stable | .73 | .74 | .73 | .74 | 0% |
  | drift | −.72 | −.72 | −.34 | .67 | 83% |
  | shared_change | −.72 | −.73 | −.14 | .71 | 83% |

  The base win rate per quarter barely moves (0.40–0.43). **What changes is the rule, not the base rate.**
- **Change-point timing** (best single split of rule-agreement):

  | | change point / cutoff | ticks from change to cutoff | std of change tick within an episode |
  |---|---|---|---|
  | drift | median .74 | median 59, minimum 21 | **17.9** |
  | shared_change | median .68 | median 71, minimum 21 | **3.9** |

  At least 21 ticks after the change are always available.
- **Cell levels depend on type.** Target `w` on the "high" cell vs the "low" cell (strong-rule machines): steady .945 / .167, balanced .853 / .067, jackpot .621 (sd .159) / .019.

### 3.5 Cross-machine correlations [C]

- Another machine's single lagged win has a mean |Δ P(win)| of **.063 at every lag**, which is noise level. |Δ| > 0.3 for 0.4% of pairs, against 5.8% for a machine's own lags.
- **Neighbors help through change timing, not through direct effects.** In `shared_change` the flip tick is shared within the episode, so pooling the 8 machines' change evidence should locate it much more precisely (EXP‑2).

### 3.6 How much observable features explain `w` (validation targets; diagnostic)

| Observable predictor of target `w` | correlation r |
|---|---|
| machine full-history win rate | .41 |
| recent win rate, best window (16–32 ticks) | .46 |
| own-history P(win \| lag-7 category) | .46 |
| **recency-weighted (h = 30) own P(win \| dominant-pair XOR cell at cutoff)** | **.94** (stable .98, short .97, shared .94, drift .88) |

---

## 4. Proposed modeling approaches (ranked)

The expected gains are **[H]** and anchored on the diagnostic probe (≈ 50.0) and the oracle (56.7).

Every approach must also pass three universal gates:
- **G‑valid:** `check_submission.py` passes on both the validation and the test CSV.
- **G‑leak:** the leakage tests in §6.4 pass.
- **G‑time:** test prediction takes ≤ 5 min on a laptop (the budget is 30 min on 2 cores).

| Rank | Approach | Expected val pts | Difficulty | Runtime (test) | Overfit risk | Acceptance criteria (beyond the universal gates) |
|---|---|---|---|---|---|---|
| 1 | **A. Bayesian pair-rule model (BMA) + factorized output** | 46–52 | Medium (~250 LOC, stdlib) | < 1 min | Low–med (≈10 global params plus a rule prior) | E1 ≥ the diagnostic probe re-run on E1 as `probe_hl30`. Stable and short ratios ≤ 0.15. E3 Brier gain vs `pooled_recent` with 95% episode-bootstrap CI excluding 0. |
| 2 | **B. Change-aware rules**: B0 tuned half-life → B1 own change-point → B2 episode-pooled change-point | +1 to +4 over A | Med–high | 1–2 min | Medium | Drift ratio improves ≥ 0.05 vs A, shared ≥ 0.03. Stable/short worsen ≤ 0.01. E1 ≥ +0.75 pts. E3 CI on the drift+shared subset excludes 0. |
| 3 | **C. Size posterior + `w` calibration** | +0.3 to +1.5 | Low | negligible | Low | E1 ≥ +0.3 pts. No condition worse by > 0.005. E3 ΔBrier point estimate ≤ 0. |
| 4 | **D. Per-machine L2 logistic regression** (8 lag wins + 28 XORs + intercept, recency weights, optional time × XOR term) | 38–46 alone, +0–1 blended | Medium (stdlib Newton) | 1–3 min | Med–high (37 params on 32–250 rows) | Standalone E1 ≥ 40. Blend with A/B gains ≥ 0.3 pts on E1, with weights fitted on E3 only. |
| 5 | **E. Blend/stack**: A/B + D + `pooled_recent`, gated by observable evidence (rule-posterior max, coverage) | +0 to +1.5 | Low | adds components | Medium | E1 gain ≥ 0.3 over the best single model. E2 not worse than the best single model by > 0.3. Weights never fitted on validation. |
| 6 | **F. Global gradient-boosted model** (xgboost 3.4.1 is local; the organizer image is stdlib, so this needs a pinned `requirements.txt`) on expanded train examples, A/B outputs plus §5 features | +0 to +1 over best | High plus dependency risk | 1–5 min | Medium (realized labels, correlated rows) | Only if A+B plateau and > 3 h remain. E1 gain ≥ 0.5 and E3 CI excluding 0. Reproduces in a clean venv. Otherwise drop. |
| 7 | **G. Small sequence model** (GRU over 64-tick windows, inferring the rule in context) | unknown | High | GPU/CPU training | High | Not scheduled for main. Same criteria as F. |

### 4.1 Approach A, precisely

For machine `m` at prediction tick `t`, let `W(u) = 1[payout(u) > 0]`, defined only for observed `u < t`. Candidate rules `r` are:
- 28 pairs `(a,b)` with `1 ≤ a < b ≤ 8`, cell `x_r(u) = W(u−a) ⊕ W(u−b)`;
- 8 singles `k`, cell `x_r(u) = W(u−k)`;
- 1 null rule with a single cell.

Steps:
1. **Evaluation set.** Use the common set `U = {u < t : u and u−1…u−8 all observed}`, so evidence is comparable across rules. This is about 32 ticks for short-history machines and about cutoff−8 for full ones.
2. **Weighted cell counts.** `ω(u) = 2^{−(t−1−u)/h}`. Then `n_c = Σ ω`, `k_c = Σ ω·W` over `u ∈ U` with `x_r(u) = c`.
3. **Cell posterior.** `Beta(α·μ_T + k_c, α·(1−μ_T) + n_c − k_c)`, where `μ_T` is the type base rate.
4. **Rule posterior.** `π_r ∝ prior_r · exp(β · log BetaBinomML_r)`. The rule prior is learned from **train** histories, never hard-coded to (2,5)/(4,7)/(1,3), because adaptation changes which relationships recur. `β ≤ 1` tempers overconfidence.
5. **Win probability.** `ŵ = Σ_r π_r · E[p_{r, x_r(t)}]`, where `x_r(t)` uses lags `t−a, t−b`, which are always observed because ticks `t−40…t−1` are always present.
6. **Output.** `p = [1−ŵ, ŵ·(P_A·S_A,T + (1−P_A)·S_B,T)]`. Clip to [1e-9, 1] and renormalize.

Global params: `h, α, β, μ_T, prior_r, S_{A,B,T}, P_A` settings. Fit them on train pseudo-queries (E3). There is **no training at predict time** beyond these constants.

Failure modes:
- Spurious rules on short histories. Mitigated by the null rule, the tempering, and the type prior.
- Rules defined on payout categories rather than wins. Test a 3-category XOR variant.
- Under-reaction after a flip. That is approach B's job.

### 4.2 Approach B

- **B0.** Tune `h` on E3. This is cheap and may capture most of the gain.
- **B1.** For each rule, use a two-segment likelihood (separate cell parameters before and after `τ`), with a posterior over `τ ∈ [t−200, t−16] ∪ {no change}`. Predict from the post-`τ` cells.
- **B2.** Pool the episode's evidence. Compute each machine's profile `L_m(τ)`. Compare a *shared* hypothesis `Π_m L_m(τ)` against an *independent* one `Π_m Σ_τ L_m(τ)`, and use the posterior mixture.
  - This is inferred from observable data only. It is **not** gated on condition labels.

Failure modes:
- Gradual drift is badly fit by two segments. Alternative: a logit-linear time trend on the contrast.
- False change points on stable machines. Guard with the no-change prior and per-condition E1 checks.

### 4.3 Approach C

- **Size posterior:** `P_A = σ(λ · LLR_A/B(wins) + logit π_A,T)`, using exact vectors and a tempering `λ < 1` fitted on E3.
- **Calibration:** `ŵ' = σ(γ_T · logit ŵ + δ_T)`, a grid search on the E3 realized 5-class Brier.

Failure mode: the size regime is mostly unpredictable (§3.3), so expect small gains. Keep `λ` small if E3 is flat.

---

## 5. Feature engineering (exact specification)

Each feature is a pure function of `(episode view truncated to ticks < t, machine m, t)`. Notation: `O_m` is the set of observed ticks `< t`, and `W` and `C` (category: `{0}`, `{1}`, `{2,5,20}`, matching `transition.category`) are `MISSING` when the tick is unobserved.

| Group | Features | Used by |
|---|---|---|
| F1 Lags | For k = 1..8: payout value one-hot (5) + missing flag, `W_k`, `C_k`. Missing is never zero. Lag k means **exactly** tick `t−k`. | A, D, F |
| F2 Pair XORs | `X_ab = W_a ⊕ W_b` for all 28 pairs; MISSING if either is missing | A, D, F |
| F3 Rule statistics | For each of the 37 rules and `h ∈ {16, 32, 64, ∞}`: weighted cell counts and wins, posterior cell means, log marginal likelihood, rule posterior, current cell. Derived: top rule id, top posterior, second posterior, posterior entropy, `ŵ` | A, B, E, F |
| F4 Rolling windows | Win rate and 5-class frequencies over the last 8/16/32/64/all observed ticks; EWMA with h ∈ {8,16,32,64}; observation count per window | D, E, F |
| F5 Size | Counts of {1,2,5,20} among wins (all and last 64); `LLR_A/B` (all and EWMA h = 64); tempered `P_A` | C, F |
| F6 Context | `machine_type` one-hot; `signal` one-hot (**query-time only**; see the leakage rules below) | A priors, C, F |
| F7 Coverage | `|O_m| / t`, length of the consecutive run ending at `t−1`, `|U|`, short flag (`coverage < 0.5`, observable) | A, B, E, F |
| F8 Own change | For the top rule: agreement series `z(u) = 1[W(u) = x_r(u)]`, best single change point `τ̂` and its LLR against no change, agreement over the last 16/32 vs before `τ̂`, slope of agreement over time | B, E, F |
| F9 Cross-machine (same episode, ticks < t) | Episode change-point curve `Σ_m LLR_m(τ)` → `τ̂_ep` and its strength; share of other machines with a confident flip; episode win rate over the last 16 vs all. Other machines' `W_k` averaged (measured at noise level; low priority) | B2, E, F |

**Leakage rules [D]:**
1. No `tick ≥ t` on any machine.
2. Pseudo-queries truncate **every** machine in the episode at `t′`.
3. Never use condition labels, targets, or reference forecasts as inputs.
4. Do not attach `signal` to pseudo-queries at earlier ticks. Signal enters only via priors fitted on real query-time labels. If you group machine histories by signal, as the kit's `context` baseline does, log it as an assumption that signal is constant per machine **[H]**.

---

## 6. Validation strategy

### 6.1 Evaluation sets

| ID | Data | Metric | Role | Budget |
|---|---|---|---|---|
| **E1** | Validation **tuning** half: 48 episodes, 12 per condition, from the kit's `episode_folds` (SHA-256 order) | Official points and 4 ratios via kit `contest_score` | Choose among a **short** candidate list | Unlimited, but log every run |
| **E2** | Validation **audit** half: the other 48 episodes | Same | Milestone confirmation only | **≤ 3 looks in Main**, each logged with its reason |
| **E3** | **Train** split: real cutoffs (`labels.csv`, 768) plus pseudo-cutoffs `t′ ∈ {c−1 … c−15}`, all machines truncated | Realized 5-class Brier, condition- and episode-weighted like the official metric; ΔBrier vs `pooled_recent` with 1,000 episode-bootstrap resamples | Fit continuous params (h, α, β, λ, γ, δ, blend weights) and run an independent check | Unlimited |

- E3 is unbiased for differences in official error (§1). Short-history pseudo-cutoffs must stay inside the 40-tick consecutive window.
- **Cross-fitting:** anything fitted on train is scored on E3 with **5 folds grouped by episode**.

### 6.2 Protocol for each candidate

1. Fit params on train with E3.
2. Score E1. Keep the change only if it clears the approach's acceptance criteria in §4 **and** no condition regresses beyond its threshold.
3. At milestones (after EXP‑1 and before the final upload), score E2 once. If E2 trails E1 by > 2 pts, investigate before uploading.
4. Run `check_submission.py` on the validation and test CSVs, and record runtime.

### 6.3 Noise discipline

- E1 has only 12 episodes per condition. Report episode-bootstrap CIs for every per-condition ratio.
- A gain of < 0.3 pts on E1 needs E3 support before adoption.
- Do not run grid searches on E1. E1 is only for picking between ≤ 5 discrete variants at a time.

### 6.4 Leakage tests (G‑leak)

1. **Truncation invariance:** for train pseudo-queries at `t′`, appending any rows with tick ≥ `t′` must not change predictions. The kit's `index_history` rejects such rows on real queries anyway.
2. **Shuffle-future test:** randomizing payouts at ticks ≥ `t′` on all machines must leave E3 predictions bit-identical.
3. **Lag-exactness unit test:** with a missing tick at `t−3`, `L3` must be MISSING, not the third-most-recent row.
4. **Determinism:** two fresh-process runs must produce byte-identical CSVs.

### 6.5 Validation exposure during this design (disclosure)

The design phase read **all** validation targets for:
- target-shape analysis (the factorization and exact size vectors);
- correlation diagnostics (§3.6);
- oracle ceilings;
- **one** untuned probe scored on all 96 episodes (≈ 50.0; drift .361, shared .222, short .123, stable .096).

Consequences:
- E1 and E2 are no longer pristine for these structural claims.
- **Train labels have not been used anywhere**, so E3 is the clean independent check.
- The size vectors are treated as simulator constants. **[H]** Cross-check them with EM on train histories, which may be weakly identified because the regime is not static. If that fails, use the validation-derived constants and say so in the entry README.

---

## 7. Architecture

### 7.1 Data flow

```
public/<split>/{queries,history}.csv
   │  slot_forecasting.io.load_inputs        (kit, unchanged: strict parse, rejects tick ≥ cutoff)
   ▼
house_edge.data.EpisodeView ── truncate(t′) ──► research/pseudo.py (train pseudo-queries, all machines cut)
   │
   ▼
house_edge.rules / features                  (pure functions of ticks < t)
   ├─► pair_rule (+change, +size, +calibrate) ─► (ŵ, P_A) ─► to_five() ─┐
   ├─► logit (optional)  ─────────────────────────────────────────────► ├─► blend ─► {qid: p[5]}
   └─► kit FrequencyForecaster("pooled_recent") (fallback component) ─► ┘
model.json  (global params fitted on train; versioned) ──────────────────┘
research/evaluate.py : E1/E2 via kit contest_score · E3 realized Brier + bootstrap
entries/<name>/predict.py --data /data --out /output/submission.csv
```

### 7.2 Modules (new files only; `slot_forecasting/` stays untouched)

```
house_edge/            # shipped in the entry; Python stdlib only (organizer image = Python 3.13 stdlib)
  __init__.py
  data.py              # MachineSeries {tick: payout}, EpisodeView, truncate(t), exact lag access
  rules.py             # candidate rules, XOR cells, weighted cell counts, Beta-binomial log-ML (math.lgamma)
  features.py          # F1–F9 (§5)
  pair_rule.py         # PairRuleForecaster (A; B via options)
  change.py            # two-segment likelihoods, episode pooling (B1/B2)
  size.py              # size vectors, LLR, tempered P_A (C)
  calibrate.py         # w calibration (C)
  logit.py             # optional per-machine L2 logistic (D)
  blend.py             # BlendForecaster (E)
  params.py            # model.json schema v1: load/save/validate; refuses unknown keys
research/              # not shipped
  pseudo.py            # E3 pseudo-query generator
  evaluate.py          # E1/E2/E3 runners, bootstrap CIs, per-condition table, runtime
  fit_params.py        # fit global params on train → runs/<name>/model.json
  run_candidate.py     # --config configs/<name>.json --out runs/<name> (refuses an existing out dir)
  profile_kit.py       # rerunnable data profile (§3) for the adaptation kit
entries/
  make_entry.py        # runs/<name> → entries/<name>/ (predict.py, house_edge/, model.json, README.md, reproduce.json)
  verify_entry.py      # build_entry.py --source → extract to a temp dir → fresh-process predict → diff ≤ 1e-8 → check_submission
configs/  runs/  EXPERIMENT_LOG.md
```

### 7.3 Interface contract (freeze at 14:30)

```python
class Forecaster:                                   # matches the kit's run_experiments protocol
    def fit(self, queries, history, labels=None) -> "Forecaster": ...
    def predict(self, queries, history) -> dict[str, tuple[float, float, float, float, float]]: ...
    def to_dict(self) -> dict: ...                  # JSON-safe, deterministic
    @classmethod
    def from_dict(cls, d: dict) -> "Forecaster": ...
# Optional, used by blend/calibrate:
    def predict_parts(self, queries, history) -> dict[str, tuple[float, float]]:   # qid -> (w_hat, P_A)
```

**Determinism.** Iterate in sorted order, use `math.fsum`, write `.17g` (the kit writer), and use no randomness at predict time. Any seeds (bootstrap only) are fixed: 314159 to match the kit.

### 7.4 Artifacts and packaging

- **Artifacts.**
  - `runs/<name>/` holds `model.json`, `validation_predictions.csv`, `submission.csv`, `run.json` (config, input SHA-256s, Python version, runtime, E1/E2/E3 scores, leak-test status).
  - Entry contents: `README.md` (from `SUBMISSION_README.md`), `reproduce.json` (`{"image":"python","command":["python","predict.py","--data","/data","--out","/output/submission.csv"]}`), `predict.py`, `house_edge/`, `model.json`, and the kit modules it imports (`slot_forecasting/{__init__,io,schema}.py`, copied unmodified).
  - No `requirements.txt` unless F is adopted.
- **Build:** `python build_entry.py --source entries/<name> --out zips/<name>.zip`, from outside the source folder.
- **Verify:** `verify_entry.py` checks:
  - that the ZIP extracts to a clean temp dir;
  - that `python predict.py --data <abs>/public --out <tmp>/submission.csv` runs in a fresh process;
  - max |Δp| ≤ 1e-8 against the CSV you will upload;
  - that `check_submission.py` passes;
  - no absolute paths and no network imports (grep).
  - Also run it under **Python 3.13** if obtainable (`pyenv install 3.13` or `uv python`). Local Python is 3.12.12.
- **Runtime estimate [H]:** load about 2 s plus rules (37 rules × ~250 ticks × 1,536 machines ≈ 15 M ops) ≈ 10–30 s in pure Python. B1/B2 change-point search with cumulative sums ≈ +30–60 s. This is far below 30 min.

---

## 8. Team parallelization (4 workstreams)

Prerequisite: the kit folder **is not a git repo**. At 14:00:
- run `git init`;
- add a `.gitignore` covering `public/ outputs/ runs/ zips/ *.zip .env`;
- commit the kit and this file;
- create one branch per workstream.

Merge into `main` only through the WS3 owner, after the E1 run is logged.

| WS | Owns (only these files) | Delivers | Depends on | Due (CT) |
|---|---|---|---|---|
| **WS1 Data & features** | `house_edge/{data,rules,features}.py`, `tests/test_leak.py` | EpisodeView, exact lags, rule counts and log-ML, F1–F9, leak tests 1–4 | interface §7.3 | rules by 15:00; features by 16:30 |
| **WS2 Models** | `house_edge/{pair_rule,change,size,calibrate,logit}.py` | A, then B0/B1/B2, C, (D) | WS1 `rules.py` (stub the API until 15:00) | A by 16:00; B and C by 17:45 |
| **WS3 Evaluation & blending** | `research/*`, `house_edge/blend.py`, `configs/`, `EXPERIMENT_LOG.md` | E1/E2/E3 harness, `probe_hl30` reference, bootstrap CIs, param fitting, blend | kit scorer; WS1 truncate | harness by 15:00 |
| **WS4 Submission & testing** | `entries/*`, `house_edge/params.py`, `zips/`, adaptation runbook | fallback upload, entry template, `verify_entry.py`, every upload | none at first | fallback uploaded by 14:30; template verified on fallback by 15:30 |

The conflict surface is limited to `__init__.py`, which WS1 owns. Everyone else adds files.

---

## 9. Execution roadmap (Main)

| Time (CT) | Step | Owner | Gate or decision |
|---|---|---|---|
| 13:45–14:30 | **EXP‑0 fallback.** Run the CLAUDE.md first-run commands → `baseline-pooled-recent/` plus ZIP. Run the checker. **Upload to the Main form** and save the confirmation. Run `git init`. | WS4 | Valid CSV + ZIP + form receipt |
| 14:00–15:00 | Freeze the interfaces. Build E1/E2/E3 and leak tests. Re-implement the diagnostic as `probe_hl30` and score it on E1 and E3. Port scratch analyses to `research/profile_kit.py`. | WS1, WS3 | Harness reproduces the baseline E1 points exactly |
| 14:30–16:00 | **EXP‑1: Approach A**, with params fitted on E3 | WS2 | §4 rank-1 criteria. Then **E2 look #1**. |
| 16:00 | **Upload EXP‑1** if it passed (packaged + `verify_entry.py` green) | WS4 | Replace the fallback only after verification |
| 16:00–17:45 | **EXP‑2: B0 → B1 → B2** · **EXP‑3: C** (in parallel) | WS2, WS3 | §4 rank-2 and rank-3 criteria, per condition |
| 17:45–18:30 | Optional: D, then E (blend weights on E3) | WS2, WS3 | Rank-4 and rank-5 criteria; drop anything that misses |
| 18:30–19:15 | **Freeze.** E2 look #2 on the final candidate. Package, verify (3.13 if available), upload. | all | Fresh reproduction ≤ 1e-8 |
| 19:15–20:30 | Only fixes that give ≥ 1 pt on E1 and are verified. **Last upload by 20:30.** | — | Never replace a verified entry with an unverified one |
| 21:00 → | Adaptation runbook (below) | all | |

**Adaptation runbook (21:00 Sat to 11:00 Sun, 40 pts).**
1. Extract the new kit into a separate folder and check that `metadata.json` says `adaptation` and 40.
2. Run `pooled_recent` and upload it as the fallback within 45 min.
3. Rerun `research/profile_kit.py` (~10 min) and check the assumptions:
   - Is it still XOR on wins?
   - Which pairs and lags are active? Categories or cross-machine rules?
   - New size vectors?
   - Change-point timing?
4. Refit `model.json`: rule prior, size vectors, μ_T, h/α/β/λ.
5. Run EXP‑1..3 configs on the new E1 and E3, and upload a verified entry by about 01:00.
6. If the structure changed beyond pairs (for example triples or payout categories), raise the priority of D and of extended rule sets.
7. Final upload by 10:00.

`/main_data` may be used for priors only if the profile shows the same structure.

---

## 10. Risks and fallback

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Fallback never uploaded, or the latest upload is invalid | Low | Fatal (0 for the round) | EXP‑0 first. Upload only after `verify_entry.py` passes. Keep the exact files and receipts. |
| Spurious rules on short or noisy histories → overconfident `ŵ` | Medium | −1 to −3 pts | Null rule, `β` tempering, type priors, calibration (C). Check per-condition ratios. |
| Drift poorly modeled (gradual vs abrupt) | Medium | Drift is the largest remaining ratio (.36 in the probe) | B0 → B1 → B2 ladder, plus a logit-linear time-trend alternative |
| Overfitting validation through repeated E1 use | Medium | Over-estimated gains | E3 for continuous params, ≤ 3 E2 looks, logged decisions, §6.3 discipline |
| Hard-coding main-round facts (pairs, lag 7, size vectors, signal direction) | High in adaptation | Large | Learn all of them from data. Rerun the profile on the new kit. |
| Python 3.13 vs 3.12 differences, or a missing dependency in the organizer image | Low | Reproduction failure | Stdlib only. Test under 3.13. No `requirements.txt` unless F is adopted. |
| Reproduction mismatch > 1e-8 | Low | Disqualification for awards | Deterministic code, sorted iteration, `fsum`, kit writer, fresh-process diff |
| Time overrun | Medium | Weaker entry | Fixed schedule; drop D/E/F/G first. The upload cutoff at 20:30 is non-negotiable. |
| Leakage through pseudo-queries (forgetting to truncate other machines) | Medium | Optimistic E3 | Leak tests 1–2 run in CI before every logged result |

**Fallback ladder:** L0 `pooled_recent` (30.83 val) → L1 EXP‑1 → L2 EXP‑1+2+3 → L3 with blend. Each rung replaces the previous upload **only** after it clears its acceptance criteria and `verify_entry.py` passes. Older entries are never restored automatically.

---

### Top three recommended experiments

1. **EXP‑1: Bayesian pair-rule model (A).**
   - Captures the dominant mechanism (r = .94 diagnostic).
   - Accept if E1 ≥ the `probe_hl30` reference, stable and short ratios ≤ 0.15, and the E3 CI excludes 0.
2. **EXP‑2: change-aware rules (B0 → B1 → B2 with episode pooling).**
   - Targets drift (.36) and shared_change (.22), the two largest remaining ratios.
   - Accept per §4 rank 2.
3. **EXP‑3: size-regime posterior + `w` calibration (C).**
   - Cheap, low-risk polish on all conditions.
   - Accept per §4 rank 3.

## Current addendum — revised schedule and independent model comparison

The organizer announcement supplied by the user extends Main to **Sunday November 8, 10:30 AM CST**, followed by **90 minutes of Adaptation until noon**. The earlier time blocks in this original design are historical. Use `ROUND2_PLAYBOOK.md` for the current adaptation procedure. Keep organizer kit files and scoring metadata unchanged; the schedule announcement does not resolve the separate 60/40 versus 80/20 scoring discrepancy.

EXP-5 remains the working entry. The independent boosted-tree benchmark is specified before results in `XGB_BENCHMARK_PLAN.md`: two fixed standalone XGBoost models, four fixed blends, and one correctly cross-fitted correction model. Use train-only episode CV with actual cutoffs primary and honest nested calibration. Only a convincing candidate advances to a logged validation comparison; do not claim the previously used validation halves are independent audits.

Following that benchmark, the user requested continued improvement with overfitting controls. `XGB_FOLLOWUP_PLAN.md` specifies a finite correction-versus-calibration ablation and three fixed episode partitions, with one prospective validation comparison at most. The old negative/inconclusive result is preserved; repeated partitions are sensitivity checks on the same data. EXP-5 stays intact until a complete candidate meets the stated predictive and reproduction gates.
