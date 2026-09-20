from __future__ import annotations

import argparse
import hashlib
import re
import sqlite3
from pathlib import Path


VERSION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = VERSION_DIR / "database" / "pipeline_v6.sqlite"
MIGRATIONS_DIR = VERSION_DIR / "database" / "migrations"
MIGRATION_PATTERN = re.compile(r"^(\d{3})_[a-z0-9_]+\.sql$")


class MigrationError(RuntimeError):
    """Raised when a database migration is missing, changed, or invalid."""


def connect_database(path: Path = DEFAULT_DATABASE) -> sqlite3.Connection:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _migration_files(directory: Path = MIGRATIONS_DIR) -> list[Path]:
    files = sorted(directory.glob("*.sql"))
    if not files:
        raise MigrationError(f"No SQL migrations found in {directory}")

    previous_number = 0
    for path in files:
        match = MIGRATION_PATTERN.fullmatch(path.name)
        if match is None:
            raise MigrationError(f"Invalid migration filename: {path.name}")
        number = int(match.group(1))
        if number != previous_number + 1:
            raise MigrationError(
                f"Migration sequence must be contiguous; expected {previous_number + 1:03d}"
            )
        previous_number = number
    return files


def initialize_database(
    path: Path = DEFAULT_DATABASE,
    migrations_dir: Path = MIGRATIONS_DIR,
) -> sqlite3.Connection:
    connection = connect_database(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            filename TEXT NOT NULL UNIQUE,
            sha256 TEXT NOT NULL CHECK (length(sha256) = 64),
            applied_at TEXT NOT NULL DEFAULT (
                strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            )
        ) STRICT
        """
    )
    connection.commit()

    applied = {
        row["version"]: row
        for row in connection.execute(
            "SELECT version, filename, sha256 FROM schema_migrations"
        )
    }

    for migration_path in _migration_files(migrations_dir):
        version = int(migration_path.name[:3])
        sql = migration_path.read_text(encoding="utf-8")
        checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
        existing = applied.get(version)
        if existing is not None:
            if existing["filename"] != migration_path.name:
                connection.close()
                raise MigrationError(
                    f"Migration {version:03d} filename changed after application"
                )
            if existing["sha256"] != checksum:
                connection.close()
                raise MigrationError(
                    f"Migration {migration_path.name} changed after application"
                )
            continue

        record_sql = (
            "INSERT INTO schema_migrations(version, filename, sha256) VALUES ("
            f"{version}, {_sql_literal(migration_path.name)}, {_sql_literal(checksum)}"
            ");"
        )
        try:
            connection.executescript(
                "BEGIN IMMEDIATE;\n" + sql + "\n" + record_sql + "\nCOMMIT;"
            )
        except sqlite3.Error as exc:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            connection.close()
            raise MigrationError(
                f"Failed to apply {migration_path.name}: {exc}"
            ) from exc

    return connection


def schema_summary(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT name, type
        FROM sqlite_master
        WHERE type IN ('table', 'index')
          AND name NOT LIKE 'sqlite_%'
        ORDER BY type DESC, name
        """
    ).fetchall()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or upgrade the v6 SQLite database."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("init", "summary"),
        default="init",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    connection = initialize_database(args.db)
    try:
        if args.command == "summary":
            for row in schema_summary(connection):
                print(f"{row['type']}: {row['name']}")
        else:
            migration_count = connection.execute(
                "SELECT COUNT(*) FROM schema_migrations"
            ).fetchone()[0]
            print(f"Database ready: {args.db.resolve()}")
            print(f"Applied migrations: {migration_count}")
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

