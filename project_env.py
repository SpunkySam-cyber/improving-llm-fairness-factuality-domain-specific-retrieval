"""Small, dependency-free helpers for loading project secrets from .env."""

from __future__ import annotations

import os
import re
from pathlib import Path


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_env_file(path: Path, *, override: bool = False) -> None:
    """Load KEY=VALUE pairs from *path* into os.environ.

    Existing environment variables win unless ``override`` is true. This supports
    the simple quoted or unquoted values used by this project without adding a
    python-dotenv dependency.
    """

    if not path.is_file():
        return

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"Invalid .env entry at {path}:{line_number}")

        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not _ENV_NAME.fullmatch(name):
            raise ValueError(f"Invalid environment variable at {path}:{line_number}")

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()

        if override or name not in os.environ:
            os.environ[name] = value


def require_env(name: str, *, env_path: Path) -> str:
    """Return a non-empty environment variable or raise a useful error."""

    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to {env_path} (see .env.example)."
        )
    return value
