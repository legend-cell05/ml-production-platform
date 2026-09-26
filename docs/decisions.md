# Decisions

Architecture decision records: the choice, the reason, what it costs, and **the
condition under which it becomes the wrong choice** — the last being the part
that makes an ADR worth writing.

Status: all `Accepted` unless noted.

---

## ADR-001 — The feature store is a table, not a function

**Decision.** Features are computed by one SQL file into
`features.churn_features`, one row per `(reference_date, account_id)`.
Everything downstream — training, serving, drift, notebooks — reads rows.

**Why.** The alternative computes features inside the training script, which
works until serving needs the same numbers. Then there are two implementations
of `usage_trend_28d_vs_90d` and they disagree, and the disagreement is invisible
because both look reasonable. One implementation cannot disagree with itself.

**Cost.** Adding a feature is a migration plus a full recompute, and
`FEATURE_VERSION` has to move with the SQL.

**Wrong when.** Features need to be fresher than a day, or the volume makes a
full recompute expensive enough to need an incremental path.

**Rejected alternative.** A feature-store product (Feast and similar). It brings
a registry, materialisation and point-in-time joins — most of which is one SQL
file here — and a dependency whose failure modes I would not be able to explain
in an interview.

---

## ADR-002 — Every date comparison is strict

**Decision.** `< :reference_date` everywhere in the feature SQL. Never `<=`.

**Why.** A ticket opened on the reference date, an invoice issued that morning,
a session started at 09:00 — none of them is knowable at the start of the day
the score is produced. One `<=` is a leak of exactly one day, which is enough to
lift the metric and small enough that nobody questions it.

**Cost.** Same-day information is discarded, which slightly weakens the model.

**Wrong when.** Scoring moves to end-of-day, in which case the boundary changes
and every comparison has to change with it — deliberately, not by accident.

---

## ADR-003 — `days_to_renewal` is computed, never read

**Decision.** The feature is derived from `signup_date` plus the contract term by
anniversary arithmetic, not read from the subscription's renewal date.

**Why.** The source system populates a renewal date only for subscriptions that
went on to renew. Reading it therefore tells the model who survived. This was a
real bug in this repository: the first model scored ROC-AUC 0.995, and that
column was the whole reason. After the fix the same model scores 0.886.

**Cost.** The computed value disagrees with the source column for accounts whose
renewal was renegotiated; the computed one is what a system would have known.

**Wrong when.** The source system starts recording a *scheduled* renewal date
for every account, at the time it is scheduled, with history.

**Lesson kept.** A metric that improves suddenly is a bug report until proven
otherwise.

---

## ADR-004 — The label is NULL until the horizon has elapsed

**Decision.** `churned_in_horizon` is NULL when
`reference_date + horizon > max_known_date`, and `label_known_at` records when
it became knowable.

**Why.** Otherwise the most recent rows carry a label computed from a window that
has not finished — a negative label meaning "has not churned yet", which is a
different statement from "did not churn". Those rows are the majority of what a
naive pipeline trains on most recently, and they are all wrong in the same
direction.

**Cost.** The most recent 60 days of the store are unlabelled. There is nothing
to be done about that; it is the problem, not the implementation.

**Wrong when.** Never, for a fixed-horizon label. A survival model would frame
it differently and would then have to handle censoring explicitly instead.

---

## ADR-005 — The split is temporal, with a one-horizon embargo

**Decision.** Train, validation and test are contiguous date ranges, separated
by a gap of one full horizon.

**Why.** Two separate failures. A random split leaks the *account* — the same
customer appears in train and test with barely-changed features, so the model
memorises. A temporal split with no gap leaks the *label* — the last training
row's outcome is determined by events inside the validation window.

**Cost.** Around 9 000 rows per boundary, and validation is small (4 637 rows) as
a result.

**Wrong when.** The horizon changes; the embargo is derived from it, not
configured separately, so this is handled.

**Rejected alternative.** Grouped K-fold by account. It fixes the memorisation
and not the label leak, and cross-validation across time on a non-stationary
process averages over regimes that should not be averaged.

---

## ADR-006 — PR-AUC is the headline, and accuracy is not reported at all

**Decision.** PR-AUC decides. ROC-AUC, Brier and lift@100 are recorded. Accuracy
appears nowhere.

**Why.** The base rate is 2.8 %. Predicting "stays" for everyone gives 97.2 %
accuracy, so accuracy is a number that cannot distinguish a working model from
no model. ROC-AUC is better but is dominated by the negative class: it can be
improved by better ordering accounts that were never at risk, which is worth
nothing to a retention budget.

**Cost.** PR-AUC is harder to explain to a non-technical stakeholder than
accuracy, and less comparable across datasets.

**Wrong when.** The base rate approaches balance, or the cost of a false positive
approaches the cost of a false negative.

---

## ADR-007 — The threshold comes from economics, on validation

**Decision.** `p* = cost / (value × success rate)`, and the operating threshold
is chosen on the validation split to maximise total expected value.

**Why.** F1 optimises a quantity no budget contains. Here the three numbers that
matter are known — 9 000 € of margin, a 350 € play, a 30 % success rate — so the
break-even probability is arithmetic (0.130) and the threshold is a business
parameter rather than a modelling one. Changing the cost of a CSM call changes
the threshold and requires no retraining.

**Cost.** The three numbers have to be argued for, and they are estimates.
Configuration makes the estimate visible, which is the point.

**Wrong when.** The intervention's success rate depends on the predicted
probability — plausible, and it would make the expected value a function rather
than a constant.

---

## ADR-008 — Calibration is part of the model, not a post-hoc fix

**Decision.** `CalibratedClassifierCV(method="isotonic", cv=3)` wraps the
pipeline, cross-fitted inside the training split.

**Why.** An expected-value threshold on an uncalibrated score is arithmetic on a
number that does not mean what it says. Isotonic rather than sigmoid because the
miscalibration is not a monotone logistic shift; cross-fitted because calibrating
on the same rows the model was fitted on produces a calibration curve that is
already correct and therefore useless.

**Cost.** Three extra fits, and isotonic's step function cannot extrapolate past
the training range.

**Wrong when.** The training set is small — a few thousand rows — at which point
isotonic overfits and Platt scaling is the better choice.

---

## ADR-009 — Explanations perturb a group, not a feature

**Decision.** The serving explanation perturbs a whole correlated group
(`product_usage`, `billing`, …) to the training median and reports the change in
predicted probability.

**Why.** The per-feature version produced incoherent rows. Setting
`active_users_28d_avg` to the median while leaving `sessions_28d` at zero
describes an account with eight active users and no sessions, which does not
exist. The model scores that contradiction as *more* risky, so the explanation
came back with the wrong sign and told a CSM the opposite of the truth.

**Cost.** Coarser answers: "product usage" rather than "sessions in the last 28
days". The driver feature inside the group is reported alongside, which recovers
some of the detail.

**Wrong when.** A method that respects the data manifold is available — SHAP with
a proper background distribution, for instance. That is a dependency and a
latency cost, and for a fortnightly list it was not worth it.

**Rejected alternative.** SHAP. Defensible; the honest reason it is not here is
that group perturbation is something I can explain end to end, and TreeSHAP's
handling of correlated features is something I would have been quoting rather
than understanding.

---

## ADR-010 — The gate blocks on segments, with a minimum-evidence rule

**Decision.** No evaluable segment may regress by more than 0.03 PR-AUC; a
segment with fewer than 20 positives is reported as "not evaluable" rather than
judged or ignored.

**Why.** An aggregate gain can hide a loss on the segment that pays for
everything. But the enterprise segment has 4 churns in the test split, and a
gate that blocks on a difference between two PR-AUCs computed from 4 positives
is a gate that gets switched off within a month — which costs more than the check
was ever worth. Stating the absence of evidence by name is the compromise.

**Cost.** A real regression in a small segment passes unnoticed. The only fix is
more data for that segment.

**Wrong when.** Segment sizes even out, at which point the minimum could drop.

---

## ADR-011 — `--force` exists

**Decision.** A named flag can override the gate. It logs a warning, records the
failed checks on the model version, and is never a default or an environment
variable.

**Why.** A gate with no override is routed around — someone updates the table by
hand and no record survives. A gate with a *recorded* override keeps the
exception inside the system.

**Cost.** It can be used badly. Visibly, which is the design.

**Wrong when.** Promotion becomes automated end to end with no human in the loop;
then the override belongs to an approval workflow rather than a flag.

---

## ADR-012 — Exit code 3 means "blocked", not "broken"

**Decision.** `0` success, `1` handled failure, `2` misuse, `3` blocked by a gate.

**Why.** A refused promotion is the system working. A scheduler that cannot tell
it apart from a crash either pages someone every fortnight or stops paging at
all.

**Cost.** One more convention to document.

**Wrong when.** Never, for a CLI meant to be scheduled.

---

## ADR-013 — PSI bins are frozen from the training distribution

**Decision.** Bin edges are computed once, from training, stored in
`ml.drift_baseline`, and reused for every comparison.

**Why.** Re-binning each period independently compares one set of deciles with
another set of deciles. Both hold 10 % by construction, so the PSI is
approximately zero whatever happened. A drift monitor that never alerts is worse
than no monitor, because it is believed.

**Cost.** Bins can become degenerate if the distribution moves far — which is
itself the signal, and shows up as a small number of bins in the report
(`open_tickets` currently reports 3 bins).

**Wrong when.** The model is retrained continuously; the baseline should then be
re-frozen with each promotion, which is what registering a model does here.

---

## ADR-014 — Prediction drift compares reference dates, not wall-clock windows

**Decision.** Prediction drift compares the two most recent scoring runs by
reference date, and says explicitly when there is only one.

**Why.** The first implementation compared "the last 30 days" with "before
that", by `scored_at`. Scoring several historical reference dates in one
afternoon — a backfill, or this project's own demo — puts every prediction in
the same window, leaves the earlier mean NULL, and reported it as `0.0` with a
ratio of `0.0`: a monitor stating that the score distribution had collapsed when
it meant it had nothing to compare. An integration test now asserts the
one-batch case reports its own uselessness.

**Cost.** Two scoring runs are needed before the signal exists.

**Wrong when.** Scoring becomes continuous rather than batch, at which point
wall-clock windows are the natural unit again — and `scored_at` is on the row.

---

## ADR-015 — One production version, enforced by the database

**Decision.** A partial unique index on `(model_name) WHERE stage = 'production'`.

**Why.** "The application only ever promotes one" is not an invariant; it is a
hope about every code path, every manual fix and every interrupted run. An index
is an invariant.

**Cost.** The promotion has to archive the incumbent and install the challenger
in one transaction, which is slightly more code than an `UPDATE`.

**Wrong when.** Canary or shadow deployments are introduced — then two versions
serve on purpose, and the stage model needs a traffic share rather than a
boolean.

---

## ADR-016 — Serving loads the model in-process

**Decision.** FastAPI with the artefact loaded at startup, swapped by
`POST /admin/reload`.

**Why.** For a fortnightly batch of a few thousand accounts, a separate
inference service adds a network hop, a deployment and a failure mode, and buys
nothing. Measured p95 for a single-row prediction is 25.7 ms.

**Cost.** Scaling means more copies of the model in memory, and a promotion needs
a reload call (or a restart) to take effect.

**Wrong when.** Scoring needs a GPU, or two services need the same model. The
artefact is already in the registry, so the change is where it is loaded.

---

## ADR-017 — The missingness screen needs a lift criterion

**Decision.** A leak verdict requires missingness AUC ≥ 0.80 **and** a
missing-versus-present label-rate lift ≥ 8×, over groups of at least 200 rows.

**Why.** The AUC alone did not work on this data. 59.4 % of accounts
legitimately have no renewal date — they are on monthly contracts — and that
dilutes the leak's missingness AUC into the `REVIEW` band, where it is ignored.
Lift is not diluted by legitimate absence: on a frame where the date is missing
for exactly the churners it runs into the thousands and fires; the legitimate
case in the real store sits at 3.4× and stays a `REVIEW`, which is the correct
verdict for something a human should look at once. A finding also states when
one side was too small to measure, instead of printing a lift nobody computed.

**Cost.** A leak affecting under 200 rows is not caught.

**Wrong when.** Missingness is rare across the board, in which case the AUC is
not diluted and the extra criterion only makes the screen less sensitive.

---

## ADR-018 — MLflow on SQLite, and `ml.training_run` as well

**Decision.** Tracking goes to MLflow with a SQLite backend; a summary row is
also written to `ml.training_run`.

**Why.** MLflow 3 put the old `file:./mlruns` backend into maintenance mode and
refuses to open one, so SQLite is the smallest thing that works locally and
swaps for a tracking server through one environment variable. The duplicate row
exists because `polaris runs` must work when the tracking store is unreachable —
operational history should not depend on an experiment tracker being up.

**Cost.** Two places record a run, and they could disagree. The summary is
written in the same transaction as the registration, so they do not. And
`mlflow-skinny` ships the SQLAlchemy tracking store without the packages needed
to open one: `alembic` is declared explicitly for that reason. It was found by
CI rather than locally, because the development machine happened to have full
MLflow installed for unrelated reasons — which is the argument for building the
environment from the manifest in CI rather than trusting the one you have.

**Wrong when.** A managed tracking server becomes part of the deployment, at
which point the summary table is still the thing the CLI reads.
