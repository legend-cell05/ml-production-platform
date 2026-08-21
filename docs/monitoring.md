# Monitoring

> Synthetic data. The figures below come from `polaris monitor` on the
> simulated business; no model here has ever served a real request.

A churn model predicts 60 days ahead, so its accuracy cannot be measured for 60
days. That single fact shapes the whole monitoring design: there are two
monitors, one for the gap and one for after it, and neither replaces the other.

| | watches | available | answers |
| --- | --- | --- | --- |
| Drift | the inputs and the score distribution | immediately | "has the world changed under the model?" |
| Live performance | predictions whose labels have arrived | one horizon late | "was the model right?" |

---

## 1. Drift: PSI against a frozen baseline

Population Stability Index per feature, on the conventional thresholds:

| PSI | Status | Reading |
| --- | --- | --- |
| < 0.10 | `OK` | the population has not moved |
| 0.10 – 0.25 | `WARN` | it has moved enough to look at |
| > 0.25 | `ALERT` | it has moved enough to distrust the model |

**The bins come from the training distribution and are then frozen**, stored in
`ml.drift_baseline` when the model is registered. This is the part that is
usually wrong. Binning each period independently — deciles of the baseline
against deciles of the current window — compares two sets of quantiles, which by
construction hold 10 % each, and reports approximately zero drift no matter what
happened. A PSI implementation that never alerts is worse than none, because it
is believed.

Categorical features are compared level by level, with an epsilon of 1e-6 added
to every bin so that a level present in one period and absent from the other
produces a large number rather than an infinite one.

Measured on the current data, three features have moved:

| Feature | PSI | Status |
| --- | --- | --- |
| `open_tickets` | 0.308 | `ALERT` |
| `tenure_days` | 0.178 | `WARN` |
| `days_since_nps` | 0.137 | `WARN` |

All three are true positives with unexciting causes, which is what honest drift
monitoring mostly looks like. `tenure_days` moves because the accounts still
alive at the end of the window are older than the ones the model trained on.
`days_since_nps` moves for the same reason — surveys accumulate. `open_tickets`
is the interesting one: it is an unbounded count of tickets still open, so it
only ratchets upward over a 24-month window, and an `ALERT` here says the model
is being asked about a population whose support backlog does not look like the
training population's. That is a real finding about this data and a reason to
watch the feature rather than a reason to distrust the monitor.

**Prediction drift** is tracked separately: the distribution of the scores
themselves. It moves before the input drift explains why, and it is the cheapest
early warning available. It compares the two most recent **scoring runs** by
reference date (0.0193 mean probability against 0.0225, a ratio of 0.86 on the
current data) and says so explicitly when there is only one batch — the first
implementation compared wall-clock windows, which put every prediction from a
backfill into the same window and reported a confident zero where the truth was
"nothing to compare".

---

## 2. Live performance, one horizon late

`polaris monitor` joins `ml.prediction` to the labels that have since become
knowable and reports what the model actually achieved. Only predictions whose
`reference_date + horizon` has elapsed are matched; the rest are counted as
pending, by name, so the number of predictions still in flight is visible rather
than implied.

What it reports:

- PR-AUC, ROC-AUC and Brier on the matured predictions — the same metrics as
  training, so the two are comparable;
- precision and recall **at the decision threshold that was actually used**,
  not at 0.5 and not at a threshold recomputed today;
- **realised value in euros**: for every account that was flagged, the expected
  value of the intervention given what actually happened —
  `saved × (value × success rate) − flagged × cost`.

Measured on the matured predictions from the current run — 2 314 predictions
made at reference date 2025-10-03, of which 74 churned, with 2 261 more still
awaiting their labels: PR-AUC 0.199, ROC-AUC 0.841, Brier 0.0284, precision
0.211 and recall 0.324 at the threshold that was actually used, realised value
24 900 €.

Live PR-AUC (0.199) is well below the test figure (0.316). That gap is the
honest part of this document: a single fortnightly batch of 2 300 accounts with
74 positives is a much smaller and noisier sample than a four-month test split,
and the batch falls at one particular point in the simulated year. It is a
reason to watch several batches before concluding anything — not a reason to
report the test number as though it were the live one.

The threshold that produced each prediction is stored **on the prediction row**.
Without that, precision measured today would silently use today's threshold
against yesterday's decisions, and the number would be fiction.

---

## 3. What is logged, and why every field is there

`ml.prediction`, one row per scored account:

| Column | Why it is not optional |
| --- | --- |
| `model_name`, `version` | which model said this; the registry moves on |
| `account_id`, `reference_date` | what was scored, as of when |
| `probability` | the raw output |
| `decision`, `threshold` | the decision and the rule that produced it |
| `expected_value_eur` | what the decision was worth at the time |
| `top_features` (JSONB) | the explanation shown to the CSM |
| `latency_ms` | measured per call, not sampled |
| `scored_at` | when, as distinct from the reference date |

Storing the explanation is a deliberate cost. When a customer success manager
asks in March why an account was flagged in January, the answer must be what
they were actually told, not what a model that has since been replaced would say
today.

---

## 4. What would page someone

Nothing here pages anyone — there is no alerting integration, and pretending
otherwise would be the kind of claim this project is trying not to make. The
conditions that *should*, in order of how much they would worry me:

1. Any feature at `ALERT` (PSI > 0.25) — the model is being asked about a
   population it has not seen.
2. Prediction drift with no matching input drift — usually a pipeline bug, and
   the most dangerous case because the numbers still look plausible.
3. Live PR-AUC falling below the base rate's implied floor — the model has
   stopped ranking.
4. A flag rate moving far from its historical level — the CSM team either has
   no list or an unworkable one.
5. Realised value going negative — the model is losing money at its own
   threshold, which the promotion gate refuses at promotion time but cannot
   prevent forever.

---

## 5. Limits

- **There is no automated retraining trigger.** Drift raises a status; a human
  decides. Automating the retrain on drift means automating a decision whose
  input is a metric that is `WARN` for correct reasons most of the time.
- **Drift is computed on the feature store, not on serving traffic**, because
  serving reads the feature store. In a system with an online path those are
  two different populations and both need watching.
- **No fairness monitoring** beyond the per-segment PR-AUC the promotion gate
  compares.
