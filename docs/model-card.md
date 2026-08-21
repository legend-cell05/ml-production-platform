# Model card — `churn-60d`

> **Synthetic data, no deployment.** This card describes a model trained on data
> produced by `polaris simulate` for a fictional company ("Vertex Systems"). It
> has never scored a real customer. The figures are real measurements of a real
> model on simulated data.

---

## 1. Model details

| | |
| --- | --- |
| Name | `churn-60d` |
| Version described | v2 (production), with v1 as the baseline it replaced |
| Type | `HistGradientBoostingClassifier`, isotonically calibrated (`cv=3`) |
| Baseline | `LogisticRegression(C=0.5, class_weight="balanced")`, same preprocessing |
| Feature contract | `1.2.0` — 33 features, 8 groups |
| Framework | scikit-learn, tracked with MLflow |
| Trained on | 51 715 rows, reference dates 2024-05-31 → 2025-04-04 |
| Dataset fingerprint | recorded per run in `ml.training_run` and MLflow |
| Licence | MIT (see `LICENSE`) |
| Author | Ayman Bara — portfolio project |

---

## 2. Intended use

**The decision it supports.** Which accounts a customer success team should put a
retention play on in the next two weeks. The output is a ranked list with a
recommended act/do-not-act flag per account.

**Who acts on it.** A customer success manager, who sees the probability, the
flag and a plain-language group explanation.

**Out of scope, explicitly:**

- pricing, credit, or any decision about an individual person;
- anything with legal or contractual consequence for a customer;
- account-level forecasting of revenue — the model predicts an event, not an
  amount;
- horizons other than 60 days. The horizon is configuration, but changing it
  changes the label, so a model trained under a different horizon is a
  different model and the registry will not serve it under this name.

---

## 3. The prediction problem

`P(account churns within 60 days of the reference date)`.

Labels exist only once the horizon has elapsed, so the most recent 60 days of the
feature store carry features and no label. Base rate on the labelled data:
**2.77 %** (95 774 labelled rows out of 104 849).

---

## 4. Data

| | |
| --- | --- |
| Source | `polaris simulate`, seed `20260301` |
| Scale | 3 000 accounts, 24 months, 1 702 486 daily usage rows, 44 414 tickets, 54 108 invoices, 7 490 NPS responses, 9 422 account events |
| Churn over the window | 739 accounts (24.63 %) |
| Feature store | 104 849 rows over 46 fortnightly reference dates |
| Split | temporal, with a one-horizon (60-day) embargo between splits |

| Split | Reference dates | Rows | Base rate |
| --- | --- | --- | --- |
| train | 2024-05-31 → 2025-04-04 | 51 715 | 2.60 % |
| validation | 2025-06-13 → 2025-06-27 | 4 637 | 2.83 % |
| test | 2025-09-05 → 2025-12-26 | 20 874 | 2.93 % |

The simulated business is deliberately not clean: accounts decline over three
months before churning rather than switching state, usage is bursty, surveys go
unanswered, and tickets stay open.

---

## 5. Performance

Test split, 20 874 rows, 611 churns.

| Metric | v1 logistic | v2 gradient boosting |
| --- | --- | --- |
| **PR-AUC** | 0.2856 | **0.3155** |
| ROC-AUC | 0.8859 | 0.8856 |
| Brier | 0.0242 | 0.0236 |
| Lift @ top 100 | 23.2× | 28.4× |
| Decision threshold | 0.0935 | 0.1162 |
| Expected value per 1 000 accounts (validation) | 20 876 € | 18 644 € |
| p95 single-row latency | 15.8 ms | 25.7 ms |

PR-AUC is the headline because the base rate is 2.8 %; ROC-AUC is reported for
comparability and is nearly identical between the two models, which is itself
informative — the gradient boosting model's advantage is entirely in how it
orders the positives.

**Accuracy is not reported.** Predicting "stays" for every account would score
97.2 %.

### By segment

| Segment | Rows | Churns | v1 PR-AUC | v2 PR-AUC |
| --- | --- | --- | --- | --- |
| smb | 11 866 | 434 | 0.3012 | **0.3487** |
| mid_market | 6 518 | 173 | **0.2629** | 0.2392 |
| enterprise | 2 490 | 4 | 0.2686 | 0.2548 |

Two things to read here rather than skip:

1. **v2 is worse on mid-market**, by 0.0237 PR-AUC, while being better overall.
   The promotion gate's default tolerance for a segment regression is 0.03, so
   this promotion was allowed — and at a tolerance of 0.01 the identical
   promotion is refused with exit code 3. The tolerance is a business decision
   held in configuration, not a constant in a script.
2. **The enterprise segment is not evaluable.** 4 churns cannot support a
   comparison between two PR-AUCs, and the gate reports it as "not evaluable
   (under 20 positives)" rather than passing or failing it silently. Both
   enterprise numbers in the table above should be read as noise.

### Measured after the fact

From `polaris monitor`, on the batch of 2 314 predictions whose labels have since
arrived (74 churns, with 2 261 predictions still awaiting labels):

| | |
| --- | --- |
| PR-AUC | 0.199 |
| ROC-AUC | 0.841 |
| Brier | 0.0284 |
| Precision at the threshold used | 0.211 |
| Recall at the threshold used | 0.324 |
| Realised value | 24 900 € |

Live PR-AUC is well below the test figure. One fortnightly batch with 74
positives is a much smaller and noisier sample than a four-month test split, and
it falls at one point in the simulated year. This is reported rather than
smoothed away because the gap between test and live performance is the number a
reviewer should want to see.

---

## 6. The decision threshold

The threshold is not tuned for F1. It comes from three business numbers:

| Input | Value |
| --- | --- |
| Margin lost when an account churns | 9 000 € |
| Cost of a retention play | 350 € |
| Share of at-risk accounts a play saves | 30 % |

Break-even probability: `350 / (9 000 × 0.30) = 0.130`. The operating threshold
is chosen on the **validation** split to maximise total expected value, which
put v2 at 0.1162. Changing any of the three inputs changes the threshold and
requires no retraining.

---

## 7. Explanations

Each prediction carries up to five **group** contributions — `product_usage`,
`billing`, `support`, and so on — obtained by setting the whole group to its
training median and measuring the change in predicted probability, with the
strongest single feature in the group reported as the driver.

Per-feature perturbation was tried first and produced sign-flipped explanations,
because setting one feature of a correlated group to its median creates a row
that cannot exist (eight active users, zero sessions) which the model scores as
*more* risky. Group perturbation keeps the row coherent. These are local
sensitivity explanations, not Shapley values, and they answer "which part of this
account's behaviour is moving the score", not "how much did each feature
contribute".

---

## 8. Limitations

- **Simulated data.** The relationships are the ones the generator was written
  with. Performance on real churn data would differ, and the honest claim of this
  project is about the pipeline, not the model quality.
- **Enterprise accounts are not evaluable** at this data volume — and they are
  the accounts worth the most.
- **No causal claim.** The model ranks risk; it does not say a retention play
  works, and the 30 % success rate is an assumption supplied by configuration,
  not something measured here.
- **Calibration is measured in aggregate.** A model can be well calibrated
  overall and badly calibrated for a segment; only the Brier score is checked.
- **Concept drift is monitored, not corrected.** There is no automated
  retraining trigger, by choice (see `docs/monitoring.md`).
- **Fairness is checked only across `segment`.** Country and industry are
  features with no per-group performance check. For a churn model aimed at a
  retention budget that is defensible; for anything touching an individual it
  would not be.

---

## 9. Ethical considerations

The output is a judgement about a business relationship, acted on by sending
someone a discount or a phone call, so the direct harm surface is small. Two
things still deserve naming:

- **A self-fulfilling loop.** If attention follows the model, accounts it scores
  as safe get less attention and may churn because of it. Nothing here detects
  that, and it is the failure mode I would watch for first in a real deployment.
- **The explanation is shown to a human who will repeat it.** A wrong-signed
  explanation is worse than no explanation, because it gets said out loud to a
  customer. That is why ADR-009 exists.

---

## 10. Reproducing these numbers

```bash
cp .env.example .env          # adjust the database settings
polaris simulate              # seed 20260301
polaris build-features
polaris screen
polaris train --algorithm logistic --promote
polaris train --algorithm gradient_boosting
polaris promote 2
polaris score --limit 3000
polaris score --reference-date 2025-10-03 --limit 3000
polaris monitor
```

Same seed, same code, same figures — apart from the last digit of a
floating-point metric, which depends on the BLAS build. The tests compare with
tolerances for that reason.
