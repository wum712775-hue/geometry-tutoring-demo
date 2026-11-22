# Methods and inference boundaries

## Data generation

The generator checks that every row of its five-category transition matrix is a valid probability distribution. A unique stationary distribution π satisfies πP = π. For each simulated learner, the first event is drawn from π and subsequent events from the row of P corresponding to the preceding event. The IID baseline independently draws every event from π. The two models thus agree in **expected marginal frequencies**, not in the exact finite-sample category counts.

Calibration also considers heterogeneous IID learners. Each learner draws a category distribution from Dirichlet(20π), then produces independent events from their own distribution. The simulator has one session per learner, fixed event durations are not modeled, and gaze-like categories are assigned directly rather than extracted from raw sensor coordinates.

## Within-learner transitions

The observed count O_ij is the number of within-learner adjacent events i → j. The pooled conditional probability estimate is O_ij divided by the row sum, which weights learners according to their outgoing events of type i. A row without outgoing observations is missing.

For a session with n events and category counts c_i, the exact adjacency expectation under a uniformly shuffled order is:

~~~text
E[N_ij | c] = c_i × (c_j − indicator(i = j)) / n
~~~

These expected counts are computed separately per learner and then added. The last event of a learner is never paired with the first event of another.

## Global test and cell diagnostics

The primary null is **exchangeable order within each learner conditional on that learner's category counts**. The statistic is the sum of (O_ij − E_ij)²/E_ij over cells with E_ij > 0. Each permutation independently shuffles every learner's sequence and reconstructs transitions. The Monte Carlo p-value is (exceedances + 1)/(permutations + 1); the default 4,999 permutations yield a minimum p-value of 0.0002.

Lag-1 pairs overlap, so an ordinary contingency-table chi-square p-value would rely on inappropriate independent-pair reasoning. The conditional permutation tests whole-sequence exchangeability. Trends, task phases, mixed sessions, or other structured nulls require different designs.

Signed cell deviations, two-sided shuffle tail proportions, BH and BY adjusted exploratory values, and descriptive permutation z scores are saved for all 25 cells. The global shuffle distribution is valid for the **global exchangeability null**. Under partial alternatives it need not provide valid cell-specific local tests; BH/BY correction does not solve that issue. Network edges are exploratory descriptions, not established transition-level FDR control.

The analysis also resamples **whole learners** 1,999 times and computes pointwise 95% percentile intervals for pooled transition probabilities. This assumes independent learners, does not produce simultaneous intervals, and retains missing rows where an outgoing category is absent in a bootstrap sample.

## Recovery and test calibration

Recovery compares the generating matrix to the estimated matrix with MAE, RMSE, maximum absolute error, and the number of estimable cells. Correlation is secondary and undefined for constant inputs. One hundred independent generation/estimation replicates show finite-sample MAE variation; their quantiles are not a claim about real population parameters. Uniform generating rows do not have a unique preferred successor, and tied maxima are evaluated as sets.

Calibration generates 200 datasets in each of homogeneous IID, learner-heterogeneous IID, and the specified Markov alternative, using 499 permutations per dataset. Null rejection rates and alternative power are accompanied by pointwise 95% exact binomial intervals. This checks only the global test for these specified synthetic scenarios; it does not calibrate local FDR or bootstrap interval coverage.

## Learner-separated prediction

For a target event at position t, the default five-event window covers t−5 through t−1. Features comprise the last observed category, window category frequencies, and normalized adjacent counts within the window. The target is whether position t is Error_Op. The target or later events never enter the features.

Five-fold GroupKFold keeps entire simulated learners in separate training and test sets. Scaling and logistic regression are fit on training folds only. The code compares training prevalence, frequencies alone, last event alone, and last event plus window features. A separate reference reads the true generating probability and is not a trained, deployable model. Reported ROC AUC, average precision, Brier score, and log loss are pooled out-of-fold descriptors, with fold metrics saved separately. Overlapping within-learner windows prevent treating the prediction examples as independent uncertainty units.

The first-order generator makes the last event sufficient for next-event probabilities in the population. This exercise does not predict cognitive load, confusion, learning outcomes, or errors in a real headset recording.

## Requirements for a real-data extension

A VR application would need synchronized gaze and hand/controller streams, actual timestamps, fixation/event-coding rules, missingness and quality checks, multiple session identifiers, person-level validation, and appropriate participant permissions. Claims about cognitive indicators need independent labels or convergent evidence; claims about instructional benefit need an outcome design. None are supplied here.

## Method references

- [SciPy chi-square contingency documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html).
- [R multiple-testing adjustment documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html).
- [Phipson and Smyth on Monte Carlo permutation p-values](https://arxiv.org/abs/1603.05766).
