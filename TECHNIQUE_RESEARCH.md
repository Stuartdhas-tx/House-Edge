**House Edge: research into better forecasting techniques**

Research date: 2026-11-07. Scope: primary papers and official software documentation, checked against the current EXP-5/6 implementation, plus one new training-only experiment. Recommendations below are specific to this dataset; literature results on other datasets do not establish a House Edge score gain.

**Recommendation**

Retain EXP-5 as the working model. The strongest next modeling experiment is a joint posterior over the win rule and payout-size rule. Better hierarchical estimation, linked change points, and missing-history filtering also address concrete limitations. A small boosted-tree correction is the most practical independent model comparison. There is no evidence yet that a transformer or a larger generic model would beat EXP-5.

Repair calibration cross-validation before judging small gains. This is an evaluation prerequisite, not a proposed source of automatic score improvement. The submitted model and its CSV/ZIP were not changed by this research.

**What the current evidence establishes**

| Evidence | Result | Interpretation |
|---|---|---|
| EXP-5 saved official validation score | 57.475361 / 60 | Strong incumbent; hidden test score remains unknown |
| EXP-6 saved official validation score | 57.548871 / 60 | Numerical gain of 0.073510; logged paired interval includes zero |
| EXP-5 condition ratios: stable / short / drift / shared | 0.030193 / 0.089073 / 0.030315 / 0.026122 | Short histories contribute about 51% of residual normalized error |
| Training episodes / machines | 96 / 768 | Numerous spin rows do not create that many independent evaluation units |
| Short-history observations | 64 per machine | Current complete-window win likelihood uses about 32 on average |
| EXP-5 recorded test inference time | 11.4 seconds | Inference acceleration is not the main need |

These local facts come from `runs/exp5`, `runs/exp6`, the earlier review, and the implementation. The active kit's scorer uses 60/40. The supplied 80/20 guide uses the same within-round loss formula; changing the maximum alone rescales a score without changing within-round rankings.

**Ranked experiment shortlist**

Effort is a relative implementation assessment, not a measured runtime or delivery promise. No expected point gains are asserted.

| Priority | Experiment | Reason to test | Effort | Status |
|---|---|---|---|---|
| R0 | Honest nested calibration and cutoff-focused evaluation | Current EXP-5/6 CV indirectly exposes the outer fold to calibration | Medium | Required before fine comparisons |
| R1 | Joint win/size evidence and joint predictive averaging | Sizes contain information about the same latent lag pair | Medium | Highest-value modeling hypothesis |
| R2 | Better cell-prior estimation; partially pooled signal effects | Current estimator selects noisy extreme contrasts; EXP-6 subdivides them | Low for shrinkage; medium for full marginal fitting | Strong statistical rationale; gain untested |
| R3 | Linked pre/post orientations and jittered shared change times | Current assumptions differ from the documented flip/shared-change mechanism | Medium | Test the two changes separately |
| R4 | Filter over missing historical payouts | Recover information omitted by the all-eight-lags requirement | High | Potentially useful, especially for short histories |
| R5 | Shallow CatBoost correction or blend | Independent test for structure missed by the mechanism model | Medium | Best practical alternative architecture |
| R6 | Regularized calibration and a small convex blend | Cheap use of complementary errors | Low after R0 | Keep parameter count small |
| R7 | Tie large-payout proportions across size regimes | Reduce nuisance parameters | Low | Tested here; not supported at real cutoffs |

**R1: infer the win and size rules jointly**

Bayesian model averaging combines complete model-specific predictive distributions using posterior model probabilities. That supports preserving uncertainty over candidate mechanisms instead of selecting a single pair. [Hoeting et al., Bayesian Model Averaging: A Tutorial](https://www.stat.colostate.edu/~jah/papers/statsci.pdf)

The concrete gap is in `house_edge/size_rule.py:127–131`: the win model produces a rule posterior, then size inference uses it. Size observations never update the win-rule posterior. Also, the final output multiplies separately averaged win and size predictions.

Let h represent the pair, change state, win orientation, and size orientation. For a positive payout j, the coherent mixture is:

```text
P(Y=j | history) = sum_h posterior(h) * w_h * s_h,j
P(Y=0 | history) = sum_h posterior(h) * (1-w_h)
```

In general, `E[w*s]` differs from `E[w]*E[s]`. More importantly, the posterior should use both the win/loss likelihood and the conditional positive-payout likelihood. This factorization does not count the same outcome twice: one factor explains whether a win occurred; the other explains its size conditional on winning.

For candidate first lag a and two size orientations, the additional evidence is inexpensive:

```text
log_size_evidence(a) = logsumexp(
    log P(orientation=0) + log L(sizes | a, orientation=0),
    log P(orientation=1) + log L(sizes | a, orientation=1)
)
```

Cache this for each lag. Add it to each compatible win hypothesis before marginalizing rules and change points. Then average complete five-class predictions. This is an extension derived for House Edge, not a published result on this competition.

First compare three fixed candidates: unchanged uncalibrated EXP-5, corrected joint predictive averaging with the old posterior, and joint likelihood plus joint averaging. This separates a covariance correction from better rule identification. The earlier review found the covariance-only probability change modest on a small training sample; do not assume it supplies a large gain by itself.

Use a common set of observed outcomes for competing likelihoods. A candidate must not look better merely because its missing lag caused fewer likelihood terms to be included. Start with complete-window events for the joint-evidence comparison; missing-state marginalization is a separate extension. All global components and any tempering settings must be learned inside training folds. Refit calibration after changing the inference.

**R2: estimate cell priors without selecting extreme observations**

Hierarchical beta-binomial and logistic models allow partial pooling: they estimate shared population structure while preserving item variation. This is a natural fit for many machines with limited histories. [Carpenter, Hierarchical Partial Pooling for Repeated Binary Trials](https://mc-stan.org/learn-stan/case-studies/pool-binary-trials.html)

`house_edge/oriented.py:33–55` currently selects the largest empirical contrast among 28 pairs, retains machines above a threshold, and averages those same selected high/low estimates. This favors large observed contrasts. `signal_levels.py` repeats that process in smaller type/signal groups without partial pooling.

The cheap version is to shrink each signal-specific rate toward its type-level rate with one shared strength. Include full pooling and the current EXP-6 estimator as endpoints. Preserve the restriction that the query-time signal cannot be attached to earlier pseudo-queries.

The fuller version estimates six type-level high/low means and a shared concentration by marginalizing the existing machine likelihood over pair, orientation, and change point. Start with a small parameterization and enforce `0 < low < high < 1`. Do not count the many hypotheses for one machine as independent training machines. Use the ordered Bernoulli likelihood already present; adding binomial combinatorial factors separately to different rule partitions would alter model comparison incorrectly.

A model for signal effects on cell logits can follow if the simpler shrinkage comparison succeeds. Give sparse or absent signal/type groups a type-level fallback. A stronger prior is not automatically better: concentration must reflect variation between machines, not only uncertainty within a machine.

**R3: link changes across time and across machines**

The present model integrates pre-change orientation independently of post-change orientation, although the project describes changes that often reverse the orientation. Changepoint models can explicitly allow dependence between successive segment parameters. [Fearnhead and Liu, Efficient Bayesian Analysis of Multiple Changepoint Models with Dependence across Segments](https://arxiv.org/abs/0910.3099)

Test a learned probability of flipping orientation. Compare it with the existing independent-orientation model, keeping a flexible fallback. If training evidence also supports stable machine-specific high/low rates across the flip, appropriately swap and pool the older counts; otherwise link orientation only. The older segment then contributes information instead of merely identifying the pair.

Separately, replace strict shared timing with:

```text
machine_change = episode_change + small_machine_offset
L_machine(episode_change) = sum_offset K(offset) * L_machine(episode_change+offset)
```

The documented approximately four-tick variation motivates this test. Retain the independent-change alternative and handle boundary/no-change mass explicitly. Research on asynchronous changepoints supports modeling related changes with delays, but its graph-learning machinery is unnecessary here: no direct machine-to-machine causation has been established. [McKee and Kalli, Network Modeling of Asynchronous Change-Points](https://arxiv.org/abs/2506.15801)

The extra convolution can reuse current evidence tables. A five-point kernel across 50 timing cases and eight machines requires roughly 2,000 weighted terms per episode, excluding existing table construction. This is an operation count, not a runtime benchmark.

**R4: filter missing histories rather than discard incomplete windows**

Dynamic Bayesian networks represent temporal state through factored variables and allow inference over unobserved state. This is the relevant tool for missing payouts that still affect later spins. [Murphy, Dynamic Bayesian Networks](https://www.cs.ubc.ca/~murphyk/Papers/dbnchapter.pdf)

In `house_edge/rules.py:27–37`, an event enters the likelihood only if every lag 1–8 is observed. For short histories, a diagnostic from the earlier review found an average 32.16 usable events. Requiring just the observed pair would yield approximately 37.73, 35.79, or 33.81 for the three dominant pairs, but comparing raw likelihoods on those different outcome sets is not valid. A proper filter instead assigns probability to the same observed history while summing over missing values.

There is a useful state reduction. For pair (a,b), retain binary win history for b ticks and distinguish `1` from `>=2` only for the newest a ticks. That yields `3^a * 2^(b-a)` states: 12, 72, and 648 for (1,3), (2,5), and (4,7). This count is a derivation for the current mechanism. Each state has three sparse transitions: loss, 1x, and >=2x. Exact observed payouts provide emissions; unobserved ticks sum over transitions and still advance time.

These counts give exact filtering conditional on fixed parameters. Integrating uncertain Beta cell parameters does not remain a simple fixed-state HMM: count histories or parameter mixtures grow. Numerical integration, approximate filtering, or Rao-Blackwellized particles may be needed. Also include initial-history uncertainty and all supported candidate rules; the three main pairs are examples, not an instruction to hard-code them.

Start with a short-history-only, fixed-parameter proof of concept. Validate it against enumeration on tiny synthetic histories before adding parameter uncertainty. Its practical runtime and predictive benefit remain unknown.

**R5: test a small boosted-tree correction**

CatBoost provides a credible independent model family; its ordered boosting addresses a specific training-time prediction-shift problem. It does not make arbitrary time-series features or random row splits safe. [Prokhorenkova et al., CatBoost](https://arxiv.org/abs/1706.09516)

Use features that let a learner understand each machine: EXP-5 probability vectors, rule/orientation entropy, posterior cell levels, effective counts, change-time uncertainty, size-orientation confidence, exact lag payouts, and observation masks. Include the real query signal only where observed. Exclude raw episode and machine IDs as predictors of new episodes.

A pooled raw-lag classifier can average away relationships that reverse across machines. Giving it inferred machine context is a more informative comparison. Try a small five-class classifier and a conservative correction model; a proposed initial search is depths 3 and 4 with a few hundred trees, not a broad hyperparameter sweep.

One squared-error correction design fits residuals `one_hot(outcome)-base_probability`, projects corrected vectors onto the probability simplex, then blends with the base using a training-selected coefficient including zero. Generate all base predictions used to train this second model out of fold, including the base priors. Group the outer and inner splits by episode.

CatBoost supports standalone Python export, so a trained small model can potentially keep submission inference free of a native CatBoost installation. Exported prediction is slower than native prediction; check numerical equivalence and full-workload runtime. [Official export documentation](https://catboost.ai/docs/en/features/export-model-to-python)

**R6: keep calibration and blending small and score-aligned**

Proper probability scores are the appropriate training target; predicting the most likely outcome or expected monetary payout optimizes something different. [Gneiting and Raftery, Strictly Proper Scoring Rules](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)

EXP-5 already fits calibration against five-class Brier. Improve its validation protocol before expanding its flexibility. Compare the existing snapshot fit with a real-cutoff fit, regularizing slope toward one and intercepts toward zero. Actual cutoffs better match deployment but provide only 768 machine rows, clustered in 96 episodes. Do not judge this using accuracy or a single binned calibration summary; multiclass calibration concerns the full distribution. [Vaicenavicius et al., Evaluating Model Calibration in Classification](https://proceedings.mlr.press/v89/vaicenavicius19a.html)

Predictive stacking learns weights from held-out predictive distributions under a scoring objective. [Yao et al., Using Stacking to Average Bayesian Predictive Distributions](https://sites.stat.columbia.edu/gelman/research/published/stacking.pdf)

For two candidates, a simple Brier specialization is:

```text
p = p_base + alpha * (p_candidate - p_base),  0 <= alpha <= 1
d_i = p_candidate_i - p_base_i
alpha = clip(sum_i weight_i * dot(d_i, y_i-p_base_i)
             / (sum_i weight_i * dot(d_i,d_i) + penalty), 0, 1)
```

Fit alpha inside outer training. Similar models such as EXP-5/6 have a small denominator and unstable optimal weights; shrink toward the incumbent. A structural candidate making different errors may be a more useful partner. Fitting a blend to the same rows used to report its performance is not evidence of improvement.

**What about TabPFN, neural networks, and broader rule families?**

TabPFN is worth a bounded comparison only after deployment and permitted-data questions are resolved. The current official repository documents local/offline operation and favors GPU use. A pinned older model on the 768 real-cutoff contextual rows is a more plausible CPU experiment than feeding all correlated historical spins. [Official TabPFN repository](https://github.com/PriorLabs/TabPFN)

The checked v3.5 default and multiclass checkpoints are each 876 MB, exceeding this kit's 512 MiB unpacked archive cap. The Fast checkpoint is 334 MB; compressed-cap compliance has not been established. Older v2 classifier files include 12.9 MB and 29 MB variants, but their runtime dependencies and external-weight eligibility still require resolution. No model weights were downloaded. [Official v3.5 files](https://huggingface.co/Prior-Labs/tabpfn_3_5/tree/main), [official v2 files](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/tree/main)

A compact causal GRU or convolutional model could test for missed dynamics, using exact tick positions and masks. It needs enough context to infer machine-specific orientation; eight recent outcomes alone are not a sufficient description. Training has many correlated targets but relatively few independent episodes. No existing result establishes that a network will improve this task.

Prior-data fitted networks can approximate Bayesian prediction by learning from tasks sampled from a prior. [Müller et al., Transformers Can Do Bayesian Inference](https://arxiv.org/abs/2112.10510) For this project, amortizing the present 11-second inference is low priority. It becomes interesting only if a richer, well-supported model is too expensive to infer directly. A network trained merely to imitate EXP-5 does not gain new evidence beyond its teacher, and a synthetic simulator can reinforce incorrect assumptions.

For adaptation, do not assume XOR, the first lag, or the `>=2` partition survives unchanged. A small posterior mixture over first-versus-second size lag, alternative payout partitions, and two-lag four-cell tables is a more incremental extension than a full sequence architecture. Preserve substantial prior mass on the existing simple mechanism to control variance. These are proposed candidates, not measured improvements.

If interactions change more substantially, conditional tensor-factorization models for higher-order Markov chains can select lags and represent higher-order interactions. That makes them a relevant research direction, although their MCMC fitting is substantially more work than extending the current rule family. [Sarkar and Dunson, Bayesian Nonparametric Modeling of Higher Order Markov Chains](https://arxiv.org/abs/1506.06268)

Bayesian online changepoint detection maintains a posterior over run length. It becomes useful if adaptation has multiple changes or near-cutoff changes outside the present timing grid. Its basic independent-segment formulation is not automatically a better fit to this simulator; use conditional rule emissions rather than an iid model of raw payouts. [Adams and MacKay, Bayesian Online Changepoint Detection](https://arxiv.org/abs/0710.3742)

**New pilot: tying size regimes did not help**

The learned EXP-5 components suggest that, conditional on paying at least 2x, the proportions of 2x/5x/20x are nearly equal across the two regimes. Maximum component differences after this conditioning were 0.00206 for balanced, 0.00597 for jackpot, and 0.00108 for steady.

I tested the constrained family:

```text
A = [a, (1-a)*r2, (1-a)*r5, (1-a)*r20]
B = [b, (1-b)*r2, (1-b)*r5, (1-b)*r20]
```

This removes two free parameters per type. Both candidates used identical uncalibrated, episode-held-out win predictions and rule masses. Size parameters were refit in each outer training fold using the same training groups and consistent pseudocounts. No validation data was loaded, and no calibration was fitted. The baseline exactly reproduced the saved uncalibrated EXP-5 all-row and real-cutoff CV losses.

| Metric, lower is better | Original uncalibrated EXP-5 | Tied size model | Difference |
|---|---:|---:|---:|
| All pseudo-query rows | 0.368740119 | 0.368737717 | -0.000002403 |
| Actual query cutoffs | 0.349225224 | 0.349340459 | +0.000115235 |

The actual-cutoff difference was worse in all four condition averages and four of five folds. A paired, within-condition episode bootstrap gave a 95% interval of approximately `[+0.000023, +0.000212]`. The all-row interval crossed zero. These intervals condition on the fitted OOF models; they do not include refitting or previous selection uncertainty. Neither interval is an official point-gain estimate.

Decision: do not promote this size-tying variant. It is an example of a plausible structural simplification that fails the more relevant cutoff comparison. It does not test the separate joint rule-inference proposal.

Artifacts: `research/review_size_tying.py` and `runs/review_size_tying/cv.json`. Measured runtime: 34.4 seconds. Commands executed:

```sh
python research/review_size_tying.py
python research/review_size_tying.py --help
# Reproduce into a new folder:
python research/review_size_tying.py --out runs/review_size_tying_repeat
```

The script deliberately reuses the setup/functions prefix of `cv_exp5.py`, stopping before its execution/report-writing block. It records that source's SHA-256 and refuses to overwrite an output directory. An independent read-only audit checked fold isolation, smoothing, and aggregation. This is a research harness, not submission code.

**Evaluation protocol for the next experiments**

Model selection itself can overfit a noisy evaluation criterion. Honest assessment must include the fitting and selection steps being claimed. [Cawley and Talbot, On Over-fitting in Model Selection](https://www.jmlr.org/papers/v11/cawley10a.html)

1. Freeze a small candidate list. Keep entire episodes together. Use real-cutoff five-class Brier as the primary training endpoint and pseudo-cutoffs as a secondary endpoint.
2. For each outer episode fold f, exclude it from all global fitting and tuning. For inner fold g, fit upstream models excluding both f and g. Generate both win and size predictions this way, not only win predictions.
3. Fit calibration, corrections, and blend weights on the inner out-of-fold predictions. Refit upstream models on the outer training episodes, then evaluate f once. An automatic hyperparameter selector must also run wholly inside outer training if its performance is being estimated.
4. Save per-query predictions and paired per-episode deltas. Bootstrap episodes within condition. Report condition averages and fold consistency, but do not treat winning 5/5 folds as a substitute for effect size and uncertainty.
5. State the objective approximation: raw Brier differences estimate probability-error differences in expectation, but current training CV omits the official inverse-reference-error weights. Do not replace those denominators with realized baseline Brier, which contains outcome noise. Report condition-specific deltas; disclose any use of validation-derived normalization.
6. Refit the selected pipeline on training data, make one planned validation confirmation, then check leakage, export equivalence, probability validity, and fresh-process reproduction before changing a submission.

The existing EXP-5/6 calibration defect is specific: predictions on inner fold g were generated by a model that excluded g but included outer fold f. Reusing that global OOF table to fit calibration lets f indirectly influence its own evaluation. The proposed protocol removes that path. Earlier architecture/hyperparameter exploration still limits claims that any reused holdout is pristine.

I inspected but did not execute the new `headroom_exp5.py`, `level_drift.py`, or `residual_scan.py`: they access validation targets, and the residual scan fits regressions against true-win residuals. Their presence does not establish a new validation result. Any target-informed hypothesis selection should be recorded as validation exposure.

**Recommended work order**

Complete R0, then compare R1 against uncalibrated EXP-5 before investing in extra calibration. Run the simple R2 shrinkage experiment and R3's two changes as separate ablations. Build R5 as an independent challenger if those plateau. Invest in R4 when evidence indicates persistent short-history rule uncertainty. Keep R6 small and evaluate it honestly. R7 has already failed the relevant pilot and should not consume the next iteration.

The literature motivates this shortlist; it cannot supply an honest predicted leaderboard gain. The current model already leaves only about 2.525 points to the theoretical maximum, and some uncertainty may be irreducible from the observed histories.
