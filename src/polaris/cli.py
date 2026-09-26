"""Command-line interface.

One command per operation in :mod:`polaris.pipeline`. Nothing here contains
logic; if a command needs a decision made, the decision belongs upstream and
the command exists only to print it.

Exit codes: 0 success, 1 handled failure, 2 misuse, 3 blocked by a gate. The
last is separate because a refused promotion is not a crash, and a scheduler
should be able to tell the difference without parsing text.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from polaris import __version__, pipeline
from polaris.config import get_settings
from polaris.db.engine import check_connection
from polaris.db.schema import drop_schemas, table_counts
from polaris.exceptions import PolarisError, PromotionBlocked
from polaris.features.definitions import FEATURE_VERSION, FEATURES
from polaris.logging_config import configure_logging
from polaris.registry.store import get_production, get_version, list_versions
from polaris.training.pipelines import ALGORITHMS
from polaris.training.train import training_runs

app = typer.Typer(
    add_completion=False,
    help="polaris -- churn prediction platform for Vertex Systems (synthetic data).",
)
console = Console()

EXIT_FAILURE, EXIT_MISUSE, EXIT_BLOCKED = 1, 2, 3


def _fail(message: str, code: int = EXIT_FAILURE) -> None:
    console.print(f"[bold red]FAILED[/bold red] {message}")
    raise typer.Exit(code=code)


def _require_database() -> None:
    if not check_connection(get_settings(), retries=5):
        _fail("database unreachable -- check .env and that PostgreSQL is running")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[bool, typer.Option("--version", help="Print the version and exit.")] = False,
) -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    if version:
        console.print(f"polaris {__version__} (features {FEATURE_VERSION})")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


@app.command()
def doctor() -> None:
    """Check configuration, database, feature store and registry."""
    settings = get_settings()
    table = Table(title="polaris doctor", header_style="bold")
    table.add_column("Check")
    table.add_column("Value")
    table.add_row("version", f"{__version__} (features {FEATURE_VERSION})")
    table.add_row("database", settings.safe_dsn)
    table.add_row("mlflow", settings.mlflow_uri)
    table.add_row("horizon", f"{settings.horizon_days} days")
    table.add_row("scoring cadence", f"every {settings.reference_interval_days} days")
    table.add_row(
        "economics",
        f"save {settings.value_of_saved_account:,.0f} EUR at {settings.cost_of_intervention:,.0f} "
        f"EUR, {settings.intervention_success_rate:.0%} success",
    )

    reachable = check_connection(settings, retries=2)
    table.add_row("database reachable", "[green]yes[/green]" if reachable else "[red]no[/red]")
    if reachable:
        counts = table_counts(settings)
        # A reachable database is not an initialised one. Asking the registry
        # for the serving model before `polaris simulate` has created the
        # schemas raises "relation does not exist" -- and a diagnostic command
        # that crashes on the state it exists to diagnose is worse than useless.
        initialised = f"{settings.ml_schema}.model_version" in counts
        table.add_row(
            "schemas",
            "ready"
            if initialised
            else "[yellow]not initialised -- run `polaris simulate`[/yellow]",
        )
        table.add_row(
            "feature rows", f"{counts.get(f'{settings.feature_schema}.churn_features', 0):,}"
        )
        if initialised:
            production = get_production("churn-60d", settings)
            table.add_row(
                "production model",
                f"v{production.version} ({production.feature_version})"
                if production
                else "[yellow]none[/yellow]",
            )
    console.print(table)
    if not reachable:
        raise typer.Exit(code=EXIT_FAILURE)


@app.command()
def features(
    show: Annotated[bool, typer.Option(help="List the feature contract.")] = False,
) -> None:
    """Show the feature contract."""
    if not show:
        _require_database()
        counts = table_counts(get_settings())
        settings = get_settings()
        console.print(
            f"feature store: {counts.get(f'{settings.feature_schema}.churn_features', 0):,} rows, "
            f"version {FEATURE_VERSION}"
        )
        return
    table = Table(title=f"Feature contract {FEATURE_VERSION}", header_style="bold")
    for column in ("feature", "kind", "missing means", "description"):
        table.add_column(column)
    for feature in FEATURES:
        table.add_row(
            feature.name, feature.kind, feature.missing_means or "-", feature.description[:60]
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@app.command()
def simulate() -> None:
    """Generate the simulated Vertex Systems business."""
    _require_database()
    try:
        result = pipeline.prepare(get_settings())
    except PolarisError as exc:
        _fail(str(exc))

    table = Table(title="Simulated source system", header_style="bold")
    table.add_column("table")
    table.add_column("rows", justify="right")
    for name, count in result["rows"].items():
        table.add_row(name, f"{count:,}")
    console.print(table)
    stats = result["stats"]
    console.print(
        f"[bold green]OK[/bold green] {stats['accounts']:,} accounts, "
        f"{stats['churned']:,} churned ({stats['churn_rate_pct']}%) over {stats['months']} months"
    )


@app.command(name="build-features")
def build_features_command(
    rebuild: Annotated[bool, typer.Option(help="Truncate before rebuilding.")] = True,
) -> None:
    """Compute point-in-time features for every reference date."""
    _require_database()
    try:
        report = pipeline.build_features(get_settings(), rebuild=rebuild)
    except PolarisError as exc:
        _fail(str(exc))
    console.print(
        f"[bold green]OK[/bold green] {report.rows:,} rows over "
        f"{len(report.reference_dates)} reference dates; {report.labelled_rows:,} labelled, "
        f"base rate {report.base_rate_pct:.2f}%"
    )


@app.command(name="screen")
def screen_command() -> None:
    """Run the leakage screens over the training data."""
    _require_database()
    try:
        findings = pipeline.screen(get_settings())
    except PolarisError as exc:
        _fail(str(exc))

    if not findings:
        console.print("[bold green]CLEAN[/bold green] no feature looks like a leak")
        return

    table = Table(title="Leakage screen", header_style="bold")
    for column in ("verdict", "screen", "feature", "detail"):
        table.add_column(column)
    for finding in findings:
        colour = {"LEAK": "red", "REVIEW": "yellow"}.get(finding.verdict, "white")
        table.add_row(
            f"[{colour}]{finding.verdict}[/{colour}]",
            finding.screen,
            finding.feature,
            finding.detail[:70],
        )
    console.print(table)
    if any(f.blocking for f in findings):
        raise typer.Exit(code=EXIT_FAILURE)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@app.command(name="train")
def train_command(
    algorithm: Annotated[
        str, typer.Option(help=f"One of {', '.join(ALGORITHMS)}.")
    ] = "gradient_boosting",
    promote: Annotated[bool, typer.Option(help="Try to promote it afterwards.")] = False,
    force: Annotated[bool, typer.Option(help="Promote even if the gate refuses.")] = False,
) -> None:
    """Train a model, register it, and optionally try to promote it."""
    _require_database()
    if algorithm not in ALGORITHMS:
        _fail(
            f"unknown algorithm {algorithm!r}; expected one of {', '.join(ALGORITHMS)}", EXIT_MISUSE
        )
    settings = get_settings()
    try:
        result, record = pipeline.train_and_register(algorithm, settings)  # type: ignore[arg-type]
    except PolarisError as exc:
        _fail(str(exc))

    table = Table(title=f"{record.model_name} v{record.version} ({algorithm})", header_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for name, value in result.headline.items():
        table.add_row(name, f"{value:,.4f}")
    console.print(table)

    segments = Table(title="Test PR-AUC by segment", header_style="bold")
    for column in ("segment", "rows", "churns", "pr_auc", "lift@100"):
        segments.add_column(column, justify="right" if column != "segment" else "left")
    for metrics in result.test.segments:
        segments.add_row(
            metrics.segment,
            f"{metrics.n_rows:,}",
            f"{metrics.n_positives:,}",
            f"{metrics.pr_auc:.4f}",
            f"{metrics.lift_at_100:.1f}",
        )
    console.print(segments)

    if not promote:
        console.print(
            f"[dim]registered as a candidate; `polaris promote {record.version}` to ship[/dim]"
        )
        return
    _try_promote(record.version, force=force)


@app.command(name="promote")
def promote_command(
    version: Annotated[int, typer.Argument(help="Version to promote.")],
    force: Annotated[bool, typer.Option(help="Promote even if the gate refuses.")] = False,
    model_name: Annotated[str, typer.Option(help="Model name.")] = "churn-60d",
) -> None:
    """Put a registered version into production, if the gate allows."""
    _require_database()
    _try_promote(version, force=force, model_name=model_name)


def _try_promote(version: int, *, force: bool, model_name: str = "churn-60d") -> None:
    settings = get_settings()
    try:
        record = get_version(model_name, version, settings)
        from polaris.data.dataset import build_dataset

        sample = build_dataset(settings).test.X.head(200)
        record, decision = pipeline.promote_candidate(
            record, settings, latency_sample=sample, force=force
        )
    except PromotionBlocked as exc:
        console.print(f"[bold red]BLOCKED[/bold red] {exc}")
        console.print("[dim]Pass --force to promote anyway; the reason is recorded.[/dim]")
        raise typer.Exit(code=EXIT_BLOCKED) from exc
    except PolarisError as exc:
        _fail(str(exc))

    table = Table(title=f"Promotion gate: {model_name} v{version}", header_style="bold")
    for column in ("check", "result", "detail"):
        table.add_column(column)
    for check in decision.checks:
        mark = (
            "[dim]skip[/dim]"
            if check.skipped
            else ("[green]pass[/green]" if check.passed else "[red]FAIL[/red]")
        )
        table.add_row(check.name, mark, check.detail[:80])
    console.print(table)
    console.print(
        f"[bold green]PROMOTED[/bold green] {model_name} v{record.version} is now serving"
    )


@app.command(name="models")
def models_command(
    model_name: Annotated[str | None, typer.Option(help="Restrict to one model.")] = None,
) -> None:
    """List registered versions."""
    _require_database()
    table = Table(title="Model registry", header_style="bold")
    for column in ("model", "version", "stage", "algorithm", "test PR-AUC", "threshold", "created"):
        table.add_column(column)
    for record in list_versions(model_name, get_settings()):
        colour = {"production": "green", "rejected": "red", "archived": "dim"}.get(
            record.stage, "white"
        )
        table.add_row(
            record.model_name,
            str(record.version),
            f"[{colour}]{record.stage}[/{colour}]",
            str(record.metrics.get("algorithm", "?")),
            f"{record.pr_auc:.4f}" if record.pr_auc else "-",
            f"{record.decision_threshold:.4f}",
            record.created_at.strftime("%Y-%m-%d %H:%M"),
        )
    console.print(table)


@app.command(name="runs")
def runs_command(limit: Annotated[int, typer.Option(help="How many.")] = 10) -> None:
    """Show the training run history."""
    _require_database()
    table = Table(title="Training runs", header_style="bold")
    for column in ("started", "algorithm", "features", "rows", "test PR-AUC", "status"):
        table.add_column(column)
    for run in training_runs(limit, get_settings()):
        metrics = run["metrics"] or {}
        table.add_row(
            run["started_at"].strftime("%Y-%m-%d %H:%M"),
            run["algorithm"],
            run["feature_version"],
            f"{run['train_rows']:,}",
            f"{float(metrics.get('test_pr_auc', 0)):.4f}",
            run["status"],
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Scoring and monitoring
# ---------------------------------------------------------------------------


@app.command(name="score")
def score_command(
    limit: Annotated[int, typer.Option(help="How many accounts to score.")] = 2000,
    top: Annotated[int, typer.Option(help="How many to print.")] = 15,
    explain: Annotated[bool, typer.Option(help="Explain each score (slower).")] = False,
    reference_date: Annotated[str | None, typer.Option(help="YYYY-MM-DD.")] = None,
) -> None:
    """Score accounts and record the predictions."""
    _require_database()
    settings = get_settings()
    target = None
    if reference_date:
        try:
            target = dt.date.fromisoformat(reference_date)
        except ValueError:
            _fail(f"invalid date {reference_date!r}", EXIT_MISUSE)
    try:
        predictions = pipeline.score_batch(
            settings, reference_date=target, limit=limit, explain=explain
        )
    except PolarisError as exc:
        _fail(str(exc))

    flagged = [p for p in predictions if p.decision]
    table = Table(
        title=f"Highest risk as of {predictions[0].reference_date}" if predictions else "No rows",
        header_style="bold",
    )
    for column in ("account", "probability", "act?", "expected value", "top driver"):
        table.add_column(column, justify="right" if column != "account" else "left")
    for prediction in predictions[:top]:
        driver = prediction.contributions[0].label if prediction.contributions else "-"
        table.add_row(
            prediction.account_id,
            f"{prediction.probability:.3f}",
            "[bold red]yes[/bold red]" if prediction.decision else "no",
            f"{prediction.expected_value_eur:,.0f} EUR",
            driver,
        )
    console.print(table)
    total = sum(p.expected_value_eur for p in flagged)
    console.print(
        f"[bold green]OK[/bold green] scored {len(predictions):,}, flagged {len(flagged):,} "
        f"for a combined expected value of {total:,.0f} EUR"
    )


@app.command(name="monitor")
def monitor_command(
    top: Annotated[int, typer.Option(help="How many drifting features to show.")] = 10,
) -> None:
    """Drift while the labels are pending, performance once they arrive."""
    _require_database()
    try:
        result = pipeline.monitor(get_settings())
    except PolarisError as exc:
        _fail(str(exc))
    if "error" in result:
        _fail(str(result["error"]))

    drift = result["drift"]
    table = Table(title=f"Feature drift -- {result['model']}", header_style="bold")
    for column in ("feature", "PSI", "status", "bins"):
        table.add_column(column, justify="right" if column == "PSI" else "left")
    for entry in drift[:top]:
        colour = {"ALERT": "red", "WARN": "yellow"}.get(entry.status, "green")
        table.add_row(
            entry.feature, f"{entry.psi:.4f}", f"[{colour}]{entry.status}[/{colour}]", entry.detail
        )
    console.print(table)

    console.print(
        Panel(str(result["prediction_drift"]), title="prediction drift", border_style="cyan")
    )

    performance = result["performance"]
    if performance is None:
        console.print(
            "[yellow]No predictions have matured yet[/yellow] -- labels arrive after the horizon."
        )
        return
    live = Table(title="Live performance (labels that have arrived)", header_style="bold")
    live.add_column("metric")
    live.add_column("value", justify="right")
    for name, value in performance.as_dict().items():
        live.add_row(name, str(value))
    console.print(live)


@app.command(name="cycle")
def cycle_command(
    algorithm: Annotated[str, typer.Option()] = "gradient_boosting",
    force: Annotated[bool, typer.Option(help="Promote even if the gate refuses.")] = False,
) -> None:
    """Train, register, promote if allowed, and score."""
    _require_database()
    try:
        report = pipeline.full_cycle(algorithm, get_settings(), force=force)  # type: ignore[arg-type]
    except PolarisError as exc:
        _fail(str(exc))
    table = Table(title="Cycle", header_style="bold")
    table.add_column("field")
    table.add_column("value", justify="right")
    for name, value in report.as_dict().items():
        table.add_row(name, str(value))
    console.print(table)
    if not report.promoted:
        raise typer.Exit(code=EXIT_BLOCKED)


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    reload: Annotated[bool, typer.Option(help="Reload on code changes.")] = False,
) -> None:
    """Run the scoring API."""
    import uvicorn

    settings = get_settings()
    console.print(f"[bold]polaris scoring[/bold] on http://{host}:{port}")
    uvicorn.run(
        "polaris.serving.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )


@app.command(name="reset")
def reset_command(
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask.")] = False,
    keep_source: Annotated[bool, typer.Option(help="Keep the simulated business.")] = False,
) -> None:
    """Drop the schemas."""
    settings = get_settings()
    dropped = (
        f"{settings.feature_schema} and {settings.ml_schema}"
        if keep_source
        else f"{settings.source_schema}, {settings.feature_schema} and {settings.ml_schema}"
    )
    if not yes and not typer.confirm(f"Drop {dropped} on {settings.safe_dsn}?", default=False):
        console.print("Cancelled.")
        raise typer.Exit()
    try:
        drop_schemas(settings, include_source=not keep_source)
    except PolarisError as exc:
        _fail(str(exc))
    console.print("[bold green]OK[/bold green] schemas dropped")


if __name__ == "__main__":  # pragma: no cover
    app()
