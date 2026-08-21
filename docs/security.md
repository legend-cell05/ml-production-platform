# Security

> Synthetic data throughout. No real company, customer, invoice or support
> ticket exists in this repository, and nothing here has been deployed to a
> production environment.

This is a portfolio project, so the honest framing is: here are the controls
that **are** implemented and verifiable by reading the code, followed by the
ones that would be required before this scored anybody's real customers.

An ML platform has one risk profile that ordinary software does not: it reads
the whole customer base, writes a judgement about each account, and that
judgement gets acted on. The interesting failures are not only breaches.

---

## 1. Secrets

**No credential is in the repository.**

| Control | Where |
| --- | --- |
| `.env` is git-ignored; `.env.example` holds placeholders only | `.gitignore`, `.env.example` |
| The ignore rule is `.env` **and** `.env.*`, with a `!.env.example` exception | `.gitignore` |
| The database password is a Pydantic `SecretStr` | `config.py` |
| Logs, reports and `polaris doctor` print `safe_dsn`, never `dsn` | `config.py`, `db/engine.py`, `cli.py` |
| Compose defaults are obviously-local placeholders | `docker-compose.yml` |
| CI's PostgreSQL password is a test-only literal on an ephemeral container | `.github/workflows/ci.yml` |

`SecretStr` is the control that does real work. A plain `str` leaks through any
`repr()` of the settings object — which is what an unhandled exception prints in
a traceback, and what a logging framework prints when someone logs the config to
debug a connection problem. With `SecretStr` the same traceback shows
`SecretStr('**********')`.

`safe_dsn` exists for the same reason: the DSN is the one string that
legitimately needs to be shown to a human diagnosing a connection failure, and
it is the one string with the password in it.

---

## 2. SQL

Every value is a bound parameter. Nothing is formatted into SQL except schema
names, which cannot be bound.

Schema names come from configuration and are validated in `config.py` against
`^[a-z_][a-z0-9_]*$` before they are interpolated. The allow-list is the control:
a schema name that would change the meaning of a statement cannot pass the
validator.

The feature SQL lives in `.sql` files rather than in Python strings, which makes
this auditable — a reviewer can read what runs without reading the code that
runs it.

---

## 3. The container

| Control | Detail |
| --- | --- |
| Multi-stage build | the runtime image carries no compiler and no build cache |
| Non-root | a dedicated user at UID 10001, created in the image |
| No secrets baked in | configuration arrives as environment at run time |
| Explicit thread limits | `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2` |
| Healthcheck | `polaris doctor` |
| `.dockerignore` | excludes `.env`, `data/`, `mlruns/`, `mlflow.db*`, notebooks, tests |

CI asserts the image does not run as root:

```bash
test "$(docker run --rm --entrypoint id ml-production-platform:ci -u)" != "0"
```

**The images have never been built in the environment this project was developed
in** — no Docker daemon was available. They are built and smoke-tested by CI on
every push, which is the only claim being made about them.

---

## 4. The model is an attack surface too

Four risks specific to this kind of system, and what is done about each:

**The artefact is a pickle.** `joblib.load` on an untrusted file is arbitrary
code execution. Artefacts here are written by this application, into a directory
it controls, and loaded by path from a registry row — never from user input.
That is adequate for a single-tenant internal tool and inadequate the moment
artefacts cross a trust boundary; the fix there is signing, and it is not
implemented.

**The prediction is personal data about a customer relationship.** `ml.prediction`
holds a probability that a named account will leave, plus the explanation shown
to a CSM. That is commercially sensitive and, under GDPR, a record about
identifiable business contacts. There is no retention policy in this project and
there would need to be one.

**Serving trusts the feature store, not the caller.** The API scores accounts
that already have features; it does not accept a feature vector from the
network. A caller therefore cannot invent an account, probe the model with
crafted inputs, or extract the decision boundary one request at a time. That was
a deliberate design choice and it removes most of the model-inversion surface.

**The explanation leaks model behaviour by design.** A group contribution tells
the caller which part of the feature space moves the score. That is the point —
a CSM cannot act on a number alone — but it is information an adversary with API
access would also get.

---

## 5. What is not implemented

Named, because pretending otherwise is the failure mode this document exists to
avoid:

- **No authentication on the API.** It binds to localhost by default and there is
  no token, no mTLS, no rate limiting. It is a demonstration of a scoring
  surface, not a service.
- **No secret manager.** Configuration comes from `.env`. Real deployment means
  Azure Key Vault or equivalent, with rotation.
- **No artefact signing** (above).
- **No audit log of who promoted what.** The registry records the model, the
  gate result and the override reason — but not an identity, because there is no
  identity in this system.
- **No row-level or column-level access control.** The application connects with
  one role that can read and write all three schemas.
- **No retention or deletion path** for predictions or features.
- **No dependency signing or SBOM.** Dependencies are pinned by range and
  installed from PyPI.

---

## 6. Data provenance

Every number in this repository comes from `polaris simulate`, a seeded
generator in `generation/simulator.py`. The simulated business is called Vertex
Systems; the accounts are `ACC-000001` upward; the churn process is a latent
health variable with configurable hazards. It is not anonymised real data, not
scraped, and not derived from any dataset with a licence to respect.

The README, the model card and every document in `docs/` say so at the top,
because a portfolio project that is ambiguous about where its data came from is
worse than one with no data at all.
