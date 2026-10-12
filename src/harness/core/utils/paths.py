"""The harness home directory layout."""

import os
from dataclasses import dataclass
from pathlib import Path

from harness.core.consts import (
    BACKUPS_DIRNAME,
    BLOBS_DIRNAME,
    CONFIG_FILENAME,
    CORE_LOCK_FILENAME,
    DB_FILENAME,
    HARNESS_HOME_DIRNAME,
    HARNESS_HOME_ENV,
    HOOKS_DIRNAME,
    HOOKS_FILENAME,
    LOGS_DIRNAME,
    SEMANTIC_DIRNAME,
    TEMPLATES_DIRNAME,
    THEMES_DIRNAME,
    TOOLS_DIRNAME,
    USER_PROFILE_FILENAME,
    WORKSPACES_DIRNAME,
)


@dataclass(frozen=True, slots=True)
class HarnessHome:
    """Resolved paths under the harness home directory."""

    root: Path

    @property
    def config_file(self) -> Path:
        """Global ``config.yaml``."""
        return self.root / CONFIG_FILENAME

    @property
    def hooks_file(self) -> Path:
        """Global ``hooks.yaml``."""
        return self.root / HOOKS_FILENAME

    @property
    def user_profile_file(self) -> Path:
        """``user_profile.md`` (memory layer 5a)."""
        return self.root / USER_PROFILE_FILENAME

    @property
    def db_file(self) -> Path:
        """The SQLite database."""
        return self.root / DB_FILENAME

    @property
    def lock_file(self) -> Path:
        """The single-core lock file."""
        return self.root / CORE_LOCK_FILENAME

    @property
    def templates_dir(self) -> Path:
        """Agent templates."""
        return self.root / TEMPLATES_DIRNAME

    @property
    def hooks_dir(self) -> Path:
        """Python hooks."""
        return self.root / HOOKS_DIRNAME

    @property
    def tools_dir(self) -> Path:
        """Custom Python tools."""
        return self.root / TOOLS_DIRNAME

    @property
    def semantic_dir(self) -> Path:
        """LanceDB data."""
        return self.root / SEMANTIC_DIRNAME

    @property
    def blobs_dir(self) -> Path:
        """Blob store."""
        return self.root / BLOBS_DIRNAME

    @property
    def workspaces_dir(self) -> Path:
        """Default agent working directories."""
        return self.root / WORKSPACES_DIRNAME

    @property
    def logs_dir(self) -> Path:
        """Log files."""
        return self.root / LOGS_DIRNAME

    @property
    def backups_dir(self) -> Path:
        """Database backups taken before migrations."""
        return self.root / BACKUPS_DIRNAME

    @property
    def themes_dir(self) -> Path:
        """User GUI themes."""
        return self.root / THEMES_DIRNAME

    def workspace_for(self, agent_id: str) -> Path:
        """Default working directory of one agent."""
        return self.workspaces_dir / agent_id

    def ensure(self) -> None:
        """Create the directories that must exist before the core starts."""
        for directory in (
            self.root,
            self.templates_dir,
            self.hooks_dir,
            self.tools_dir,
            self.semantic_dir,
            self.blobs_dir,
            self.workspaces_dir,
            self.logs_dir,
            self.backups_dir,
            self.themes_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def resolve_home(override: Path | None = None) -> HarnessHome:
    """Return the harness home: explicit override, then ``HARNESS_HOME``, then ``~/.harness``.

    Reading this one environment variable here (instead of through the config manager)
    is deliberate: it decides where the config itself lives.
    """
    if override is not None:
        return HarnessHome(override.resolve())
    env = os.environ.get(HARNESS_HOME_ENV)
    if env:
        return HarnessHome(Path(env).resolve())
    return HarnessHome((Path.home() / HARNESS_HOME_DIRNAME).resolve())


def is_within(path: Path, parent: Path) -> bool:
    """Return whether resolved ``path`` is ``parent`` or inside it."""
    resolved, base = path.resolve(), parent.resolve()
    return resolved == base or base in resolved.parents
