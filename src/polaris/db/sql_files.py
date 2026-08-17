"""Loading and rendering the .sql files.

Two jobs that look small and are not.

**Splitting.** psycopg refuses more than one statement in a prepared
statement, so a schema file has to be executed statement by statement. Naive
splitting on ``;`` breaks on semicolons inside string literals, comments and
dollar-quoted function bodies -- all of which appear in these files.

**Rendering.** Schema names cannot be bound parameters. The SQL files use
``${SOURCE}``, ``${FEATURES}`` and ``${ML}`` placeholders, substituted here from
settings whose values have already been validated against
``[a-z_][a-z0-9_]*``. The substitution never touches a value, only an
identifier.
"""

from __future__ import annotations

from pathlib import Path

from polaris.config import Settings, get_settings
from polaris.exceptions import DatabaseError


def split_statements(sql: str) -> list[str]:
    """Split a script into executable statements."""
    statements: list[str] = []
    current: list[str] = []
    in_single = in_double = in_line_comment = in_block_comment = False
    dollar_tag: str | None = None
    index = 0
    length = len(sql)

    while index < length:
        char = sql[index]
        pair = sql[index : index + 2]

        if in_line_comment:
            current.append(char)
            if char == "\n":
                in_line_comment = False
            index += 1
            continue

        if in_block_comment:
            current.append(char)
            if pair == "*/":
                current.append(sql[index + 1])
                in_block_comment = False
                index += 2
                continue
            index += 1
            continue

        if dollar_tag is not None:
            current.append(char)
            if sql.startswith(dollar_tag, index):
                current.append(sql[index + 1 : index + len(dollar_tag)])
                index += len(dollar_tag)
                dollar_tag = None
                continue
            index += 1
            continue

        if in_single:
            current.append(char)
            if char == "'":
                if sql[index + 1 : index + 2] == "'":  # escaped quote
                    current.append("'")
                    index += 2
                    continue
                in_single = False
            index += 1
            continue

        if in_double:
            current.append(char)
            if char == '"':
                in_double = False
            index += 1
            continue

        if pair == "--":
            in_line_comment = True
            current.append(pair)
            index += 2
            continue
        if pair == "/*":
            in_block_comment = True
            current.append(pair)
            index += 2
            continue
        if char == "'":
            in_single = True
            current.append(char)
            index += 1
            continue
        if char == '"':
            in_double = True
            current.append(char)
            index += 1
            continue
        if char == "$":
            end = sql.find("$", index + 1)
            if (end != -1 and sql[index + 1 : end].isidentifier()) or (
                end != -1 and end == index + 1
            ):
                dollar_tag = sql[index : end + 1]
                current.append(dollar_tag)
                index = end + 1
                continue

        if char == ";":
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            index += 1
            continue

        current.append(char)
        index += 1

    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


def render_sql(sql: str, settings: Settings | None = None) -> str:
    """Substitute the schema placeholders."""
    settings = settings or get_settings()
    return (
        sql.replace("${SOURCE}", settings.source_schema)
        .replace("${FEATURES}", settings.feature_schema)
        .replace("${ML}", settings.ml_schema)
    )


def sql_dir(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.project_root / "sql"


def read_sql(relative_path: str, settings: Settings | None = None) -> str:
    """Read one .sql file and render its placeholders."""
    path = sql_dir(settings) / relative_path
    if not path.is_file():
        raise DatabaseError(f"SQL file not found: {path}")
    return render_sql(path.read_text(encoding="utf-8"), settings)
