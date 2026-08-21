# Training

> Synthetic data. Every figure below was measured on the simulated business
> produced by `polaris simulate` with seed `20260301`; none of it describes a
> real company or a real deployment.

Training is the least interesting part of a churn model and the easiest to get
wrong in ways that are invisible until the model is live. This document is
mostly about the three things that surround it: how the data is split, what is
checked before a model is trained at all, and how the decision threshold is
chosen.

---

## 1. The split is temporal, and it has an embargo

A random split on this data produces a beautiful number and a worthless model.
Rows are `(reference_date, account_id)` pairs, the same account appears at every
reference date, and its features move slowly — so a random split puts March's
row for account A in train and April's row for the same account in test. The
model then memorises accounts instead of learning churn, and the test score
measures memorisation.

The split is therefore by time:

| Split | Reference dates | Rows | Base rate |
| --- | --- | --- | --- |
| train | 2024-05-31 → 2025-04-04 | 51 715 | 2.60 % |
| validation | 2025-06-13 → 2025-06-27 | 4 637 | 2.83 % |
| test | 2025-09-05 → 2025-12-26 | 20 874 | 2.93 % |

Note the gaps between the ranges. They are not accidental: between each pair of
splits there is an **embargo of one full horizon** (60 days). Without it, the
last training row's label is determined by events inside the validation period —
the model would be trained on the answer to a question it is about to be asked.
The embargo costs about two months of data per boundary, which on this dataset
is roughly 9 000 rows. It is the single most expensive correctness decision in
the project and it is not negotiable.

The dataset carries a **fingerprint** — a SHA-256 over the feature version, the
split boundaries and the row counts — recorded on every training run. Two runs
claiming to be comparable must show the same fingerprint, or they are not.

---

## 2. What happens before training: the leakage screens

`polaris screen` runs three independent screens over the training split. A
`LEAK` verdict is a non-zero exit code and the CI lifecycle job fails on it.

### Screen 1 — a single feature that predicts too well

A model's AUC is a claim about many features together. One feature reaching a
ROC-AUC of 0.95 on its own is not a good feature; it is the label wearing a
disguise. Thresholds: `LEAK` at 0.95, `REVIEW` at 0.90.

This screen found the original `days_to_renewal` bug — the column was populated
only for accounts that went on to renew, and it alone reached an AUC that made
the full model score 0.995. Architecture doc, section 3, has the fix.

### Screen 2 — missingness that predicts too well

The subtler failure. A feature whose *value* is innocuous can leak through its
*absence*: a column that is only computed for accounts that survived is
perfectly informative before a single value is read.

The first version of this screen used the missingness AUC alone. It was too
blunt for exactly the bug it was written for: 59.4 % of accounts legitimately
have no renewal date — they are on monthly contracts — and that legitimate
missingness dilutes the leak's AUC into the `REVIEW` band, where it sits among
other `REVIEW`s and gets ignored.

The fix was a second criterion: **lift**, the churn rate among rows where the
feature is missing divided by the rate where it is present. It is not diluted by
legitimate absence. Both criteria run and the worse verdict wins — `LEAK` at
AUC ≥ 0.80 or lift ≥ 8.0, over groups of at least 200 rows on each side.

On a frame where the renewal date is missing for exactly the accounts that
churned, the lift runs into the thousands and the screen reports `LEAK`. In the
real store the same feature sits at 3.4× with a missingness AUC of 0.622 and
stays a `REVIEW` — the right verdict for something a human should look at once.
When one side of the comparison is too small to measure, the finding says so
rather than printing a lift of 1.0× that nobody computed.

### Screen 3 — stability across time

A feature whose distribution jumps between the training reference dates is
either a data collection change or a pipeline bug. Either way, a model trained
across the jump has learned two different features under one name.

---

## 3. The two pipelines

Both are scikit-learn `Pipeline`s with preprocessing **inside** them. A scaler
fitted outside the pipeline is fitted on the test set too, which is the most
common way a good validation number turns out to be fiction.

| | logistic regression | gradient boosting |
| --- | --- | --- |
| Estimator | `LogisticRegression(C=0.5, max_iter=2000)` | `HistGradientBoostingClassifier(max_iter=300, lr=0.06, 31 leaves, min 40 per leaf, L2 1.0, early stopping)` |
| Why | a baseline that can be read; its coefficients are what you show a CSM | the relationships are not linear — a usage ratio of 0.4 means something different at 5 seats and at 500 |
| Class weight | `balanced` | `balanced` |

`class_weight="balanced"` is doing more work than it looks. The positive class is
2.8 % of the data; without reweighting, both models predict "will not churn" for
everyone and are right 97 % of the time.

The preprocessing is identical across both algorithms, including the imputer
that gradient boosting does not need — so a comparison between them is a
comparison of the algorithms rather than of two different preprocessings.

---

## 4. Calibration

The threshold is derived from business economics, which requires the
probabilities to mean what they say. A model that ranks perfectly but outputs
0.9 where the true rate is 0.3 cannot be thresholded economically.

Both models are wrapped in `CalibratedClassifierCV(method="isotonic", cv=3)`,
cross-fitted **inside the training split only**. Isotonic rather than sigmoid
because the miscalibration here is not a simple logistic shift, and the training
set is large enough (50 000+ rows) that isotonic does not overfit.

Calibration quality is measured by Brier score on the test split and enforced by
the promotion gate at a ceiling of 0.12.

---

## 5. Metrics, and the one that is not reported

**PR-AUC is the headline.** At a 2.8 % base rate, ROC-AUC is dominated by the
negative class: a model can gain ROC-AUC by better ordering accounts that were
never going to churn, which is worth nothing to anybody. PR-AUC only rewards
ordering within the positives.

Also recorded: ROC-AUC (comparability with the literature), Brier score
(calibration), lift at the top 100 accounts (what a CSM team actually works —
a list), and PR-AUC per segment.

**Accuracy is not reported anywhere.** Predicting "stays" for every account
gives 97.2 % accuracy on this data. Reporting it would be true and misleading,
which is worse than not reporting it.

---

## 6. The threshold comes from the business, not from F1

Three numbers, all configuration:

| Setting | Value | Meaning |
| --- | --- | --- |
| `POLARIS_VALUE_OF_SAVED_ACCOUNT` | 9 000 € | gross margin lost when an account churns |
| `POLARIS_COST_OF_INTERVENTION` | 350 € | a CSM call, a discount, an incentive |
| `POLARIS_INTERVENTION_SUCCESS_RATE` | 0.30 | share of genuinely at-risk accounts a play saves |

Acting on an account is worth `9 000 × 0.30 − 350 = 2 350 €` if it was going to
churn, and costs `350 €` if it was not. So the break-even probability is

```
p* = cost / (value × success rate) = 350 / (9 000 × 0.30) = 0.130
```

Below 13 % predicted risk, the play loses money in expectation. The threshold is
then chosen on the **validation** split as the point maximising total expected
value — never on test, and never by maximising F1, which optimises a quantity
nobody's budget contains.

Changing any of the three numbers changes the threshold and nothing else: the
model does not need retraining when the cost of a CSM call changes.

---

## 7. Tracking

Every run writes to MLflow: parameters, all metrics, the dataset fingerprint,
the feature version, and the fitted pipeline as an artefact. The default backend
is a local SQLite file (`sqlite:///mlflow.db`) — MLflow 3 put the old
`file:./mlruns` backend into maintenance mode and refuses to open one. Pointing
`POLARIS_MLFLOW_TRACKING_URI` at a tracking server changes nothing else in the
project.

A row also lands in `ml.training_run` with the status, the row counts and the
headline metrics, so the history survives without MLflow being reachable —
`polaris runs` reads that table, not the tracking store.

---

## 8. Reproducibility

Same seed, same code, same figures: the simulator, the split boundaries, both
estimators and the calibration folds are all seeded. What is *not* reproducible
across machines is the last digit of a floating-point metric, so the tests
compare with tolerances rather than for equality — a test that fails on a BLAS
version change teaches nothing.
