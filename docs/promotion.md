# The promotion gate

> Synthetic data, and no deployment: "production" here means the row in
> `ml.model_version` whose stage is `production`, which is what the serving
> process loads.

A model becomes the serving model by passing six checks. Each one exists because
of a specific way a model can look better than it is. The gate is the part of
this project I would defend hardest in an interview, so the reasoning is written
out in full.

---

## 1. The six checks

| # | Check | Passes when | Configured by |
| --- | --- | --- | --- |
| 1 | `feature_version` | the model's feature version equals the store's | — (a match, not a threshold) |
| 2 | `beats_champion` | PR-AUC gain over the champion ≥ 0.005 | `POLARIS_MIN_PR_AUC_GAIN` |
| 3 | `no_segment_regression` | no evaluable segment drops by more than 0.03 PR-AUC | `POLARIS_MAX_SEGMENT_REGRESSION` |
| 4 | `calibration` | test Brier score ≤ 0.12 | `POLARIS_MAX_BRIER_SCORE` |
| 5 | `latency` | measured p95 for a single-row prediction ≤ 50 ms | `POLARIS_MAX_P95_LATENCY_MS` |
| 6 | `positive_expected_value` | expected value per 1 000 accounts at the chosen threshold > 0 | — |

Checks 2 and 3 are skipped, explicitly and visibly, when there is no champion:
the first model becomes the baseline. A skipped check is printed as `skip`, not
as `pass`, because a gate that reports six passes when it ran four is a gate
nobody should trust.

---

## 2. Why "no segment regresses" is the important one

An aggregate metric hides the thing a business cares about. A challenger can
gain 0.03 PR-AUC overall by getting better at SMB accounts — of which there are
thousands — while getting worse at enterprise accounts, which are worth roughly
ten times as much each. The average says ship it. The revenue says do not.

This is not hypothetical here. Measured on the current data, gradient boosting
beats logistic regression overall (test PR-AUC 0.3155 against 0.2856) and is
**worse on mid-market** (0.2392 against 0.2629). Under the default tolerance of
0.03 the regression of 0.0237 is inside budget and the promotion is allowed.
Tighten `POLARIS_MAX_SEGMENT_REGRESSION` to 0.01 and the same promotion is
refused with:

```
BLOCKED no_segment_regression: 2 segment(s) judged;
regressions: mid_market 0.2392 vs 0.2629 (-0.0237);
not evaluable (under 20 positives): enterprise
```

That is the gate doing its job: the number that decides is a business
tolerance, written in configuration where someone can argue about it, not a
constant buried in a scoring script.

### The minimum-evidence rule

The enterprise segment in the test split has 4 churns. The difference between
two PR-AUCs computed on 4 positives is noise, and a gate that blocks on noise
gets switched off within a month — which costs far more than the check was ever
worth.

So a segment is only **judged** when it has at least 20 positives
(`MIN_SEGMENT_POSITIVES`). Segments below that are neither judged nor ignored:
they are reported as *not evaluable*, by name, in the check's detail:

```
2 segment(s) judged; none regressed;
not evaluable (under 20 positives): enterprise
```

The absence of evidence is stated rather than silently read as a pass.

---

## 3. Latency is measured, not assumed

The gate loads the candidate artefact and times 200 single-row
`predict_proba` calls, taking the p95. Single-row rather than batched on
purpose: the serving path answers one account at a time when a CSM opens an
account page, and a batch measurement hides the per-call overhead that dominates
there. Measured on the current models: p95 15.8 ms for the logistic pipeline and
25.7 ms for gradient boosting, against a 50 ms budget.

---

## 4. `--force`

A controlled exception is sometimes right — an incident, a rollback, a
regression on a segment the business has decided it accepts this quarter. What
it must never be is accidental. So `--force`:

- is a named flag, never a default or an environment variable;
- logs a warning naming every check that failed;
- writes the override and the failed check names onto the model version row, so
  the registry says *why* this model is serving.

Exit codes make the distinction machine-readable: a blocked promotion exits `3`,
not `1`. A refused promotion is not a crash, and a scheduler must be able to
tell the difference without parsing text.

---

## 5. Exactly one model serves

`ml.model_version` has a partial unique index:

```sql
CREATE UNIQUE INDEX ux_model_one_production
    ON ml.model_version (model_name)
 WHERE stage = 'production';
```

The invariant is enforced by the database, not by application code that could be
bypassed by a second process, an interrupted promotion or a manual `UPDATE`. The
promotion transaction archives the incumbent and installs the challenger
together; if it fails halfway, nothing changed. The CI lifecycle job asserts the
invariant still holds after a full promotion cycle, and that a refused promotion
left the incumbent serving.

Stages: `candidate` → `production` → `archived`, or `candidate` → `rejected`
when the gate refuses. A rejected version keeps its artefact and its metrics —
what did not ship is part of the record.

---

## 6. What the gate does not check

Honesty about the gaps, because an interview will ask:

- **Fairness across anything but `segment`.** Country and industry are features;
  no check compares performance across them. For a churn model aimed at a
  retention budget this is defensible; for anything touching a person it would
  not be.
- **Drift at promotion time.** Drift is monitored after the fact
  (`docs/monitoring.md`), not made a precondition of promotion.
- **A live shadow comparison.** The challenger is judged on a held-out test
  split, not by scoring alongside the champion for two weeks. That is the right
  next step and it is not built.
- **Anything about the training data's provenance.** There is one source and it
  is simulated.
