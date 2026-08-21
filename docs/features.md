# The feature contract

> Synthetic data. Every number below was computed from the simulated business
> produced by `polaris simulate`.

`src/polaris/features/definitions.py` is the single source of truth for what the
model may see. It drives the training pipeline's column selection, the serving
payload validation, the drift monitor's per-feature loop, and this document.
Keeping one list rather than four is what stops a feature from being added to
training and forgotten in serving.

**Current version: `1.2.0` — 33 features over 8 groups.**

---

## 1. Why a version, and why it blocks

`FEATURE_VERSION` changes whenever the list or the SQL that computes it changes.
It is stored on every feature row, on every registered model, and compared by
the promotion gate: a model trained on `1.1.0` cannot be promoted while the
store holds `1.2.0`, because the column it learned from no longer means the same
thing. This is the first of the six gate checks and the cheapest one to pass —
which is the point.

---

## 2. What the model may not see

Nine columns live in the feature store and are excluded from the matrix by
`EXCLUDED_COLUMNS`:

| Column | Why it is withheld |
| --- | --- |
| `reference_date` | teaches the model the calendar, not the customer |
| `account_id` | an identifier is a memorisation shortcut |
| `churned_in_horizon` | the label |
| `label_known_at` | derived from the label's existence |
| `feature_version` | provenance |
| `computed_at` | when the row was built, not what it describes |
| `mrr_change_90d_pct` | computed by the SQL, left `NULL` — see below |
| `seats_change_90d` | same |

`mrr_change_90d_pct` and `seats_change_90d` are the honest ones. The simulated
billing system does not version price changes, so there is no way to know what
an account's MRR was 90 days ago. Two options were available: fabricate the
history, or ship the columns as `NULL` and exclude them. They are `NULL`, and
the CI contract job asserts that every stored column is either a declared
feature or a declared exclusion — so neither can be quietly promoted into the
matrix.

The same job asserts the reverse direction: nothing in the preprocessing
transformers names a column outside the contract. `remainder="drop"` in the
`ColumnTransformer` is the mechanism; the test is what stops it being changed by
accident.

---

## 3. Groups, and why they exist

The eight groups are not documentation. Features inside a group are strongly
correlated, and the serving layer's explanation perturbs **a whole group at a
time** for a reason: an account with 0.14 active users and zero sessions is
coherent, and the same account with the median eight active users and still zero
sessions is not. The model scores that contradiction as *more* risky, which
flips the sign of the explanation and tells a customer success manager the
opposite of the truth. That bug was in the first implementation; group
perturbation is the fix. See `docs/decisions.md`, ADR-009.

---

## 4. The contract

### product_usage — how much the product is being used

| Feature | Kind | Meaning | A missing value means |
| --- | --- | --- | --- |
| `active_users_28d_avg` | numeric | Mean daily active users over 28 days. | no usage records at all in the window |
| `seat_utilisation_28d` | numeric | Active users over licensed seats. | no usage records, or zero seats |
| `sessions_28d` | numeric | Sessions in the last 28 days. | — |
| `api_calls_28d` | numeric | API calls in the last 28 days. | — |
| `features_used_28d` | numeric | Mean distinct features used per day. | — |
| `usage_trend_28d_vs_90d` | numeric | 28-day average over 90-day average. Below 1 means the account is cooling. | not enough history for a trend |
| `days_since_last_use` | numeric | Days since anyone last logged in. | — |
| `zero_usage_days_28d` | numeric | Days with no active user in the last 28. | — |
| `error_rate_28d` | numeric | Error events per session. | no sessions to divide by |

### support — support history

| Feature | Kind | Meaning | A missing value means |
| --- | --- | --- | --- |
| `tickets_90d` | numeric | Tickets opened in the last 90 days. | — |
| `tickets_p1_90d` | numeric | P1 tickets opened in the last 90 days. | — |
| `open_tickets` | numeric | Tickets still open as of the reference date. | — |
| `avg_resolution_hours_90d` | numeric | Mean resolution time, tickets closed in window. | nothing was resolved in the window |
| `avg_satisfaction_180d` | numeric | Mean CSAT, 1-5. | the customer never answered a survey |

### billing — payment behaviour

| Feature | Kind | Meaning | A missing value means |
| --- | --- | --- | --- |
| `late_payments_180d` | numeric | Invoices paid late or overdue in 180 days. | — |
| `failed_payments_90d` | numeric | Failed payments in 90 days. | — |
| `max_days_overdue_180d` | numeric | Worst overdue stretch in 180 days. | — |

### sentiment — survey responses

| Feature | Kind | Meaning | A missing value means |
| --- | --- | --- | --- |
| `last_nps_score` | numeric | Most recent NPS score, 0-10. | never responded to a survey |
| `days_since_nps` | numeric | Days since the last NPS response. | never responded to a survey |

### relationship — the account relationship

| Feature | Kind | Meaning | A missing value means |
| --- | --- | --- | --- |
| `champion_left_180d` | boolean | The main contact left in the last 180 days. | — |
| `qbr_held_180d` | boolean | A quarterly business review took place. | — |
| `has_csm` | boolean | Whether a customer success manager is assigned. | — |

### commercial — the commercial shape of the account

| Feature | Kind | Meaning | A missing value means |
| --- | --- | --- | --- |
| `mrr_eur` | numeric | Monthly recurring revenue. | — |
| `seats` | numeric | Licensed seats. | — |
| `had_downgrade_180d` | boolean | A plan downgrade in the last 180 days. | — |
| `had_upgrade_180d` | boolean | A plan upgrade in the last 180 days. | — |
| `plan` | categorical | Subscription tier. | — |

### contract — contract and tenure

| Feature | Kind | Meaning | A missing value means |
| --- | --- | --- | --- |
| `contract_term_months` | numeric | 1 for monthly, 12 or 24 for committed terms. | — |
| `tenure_days` | numeric | Days since signup. | — |
| `days_to_renewal` | numeric | Days until the contract renews. | no scheduled renewal (monthly contract) |

### firmographics — who the customer is

| Feature | Kind | Meaning | A missing value means |
| --- | --- | --- | --- |
| `segment` | categorical | SMB, mid-market or enterprise. | — |
| `industry` | categorical | Customer's industry. | — |
| `country` | categorical | Billing country. | — |

---

## 5. Missingness

Nine features have a `missing_means` note above, and it is not decoration: those
gaps are information. A customer who never answered an NPS survey is not a
customer with a median NPS, and an account with no usage rows at all in the
28-day window is not an account with typical usage.

The numeric branch therefore imputes the median **and keeps an indicator
column** for every numeric feature that had a gap at fit time. On the current
data twelve indicators are produced: the nine declared ones plus
`sessions_28d`, `api_calls_28d` and `features_used_28d`, which are absent for
the same reason the usage averages are. A unit test asserts that each declared
column gets its indicator, rather than trusting that it happened.

Gradient boosting handles `NaN` natively and does not need the imputer at all.
It gets it anyway, because the preprocessing is identical across algorithms —
so a comparison between them is a comparison of the algorithms and not of two
different preprocessings.

---

## 6. Windows, and the one that is unbounded

Most features look back over a fixed window — 28, 90, 180 days — and the SQL
bounds the scan at 400 days so a two-year history does not get re-read for every
reference date.

`open_tickets` is the exception and has its own CTE with no lower bound. It
counts tickets still open as of the reference date, and a ticket opened 14 months
ago and never closed is still open. Bounding that scan produced a feature that
silently undercounted the worst accounts — the ones with ancient unresolved
problems. It cost a scan to fix and was worth it.

---

## 7. Adding a feature

1. Add the column to `sql/schema/003_features.sql`.
2. Compute it in `sql/features/build_features.sql`, with `< :reference_date` on
   every date comparison.
3. Declare it in `definitions.py`: name, kind, description, and
   `missing_means` if a gap carries meaning.
4. Put it in a group in `FEATURE_GROUPS` — the contract test fails otherwise.
5. Bump `FEATURE_VERSION`.
6. `polaris build-features` (a full recompute), then `polaris screen` before
   `polaris train`.

Step 6 is not optional. A new feature is the most likely place for a leak to
enter, and the screens exist because a leak shows up as a *better* number.
