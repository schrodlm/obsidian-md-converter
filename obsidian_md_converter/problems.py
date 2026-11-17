"""What validation reports: one problem per finding, with the check that found it."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from obsidian_md_converter.paths import ObsidianPath

if TYPE_CHECKING:
    from obsidian_md_converter.checks import Check


class Severity(Enum):
    """An error stops the run; a warning leaves a page that can still be written."""
    ERROR = 'error'
    WARNING = 'warning'


@dataclass(frozen=True)
class Problem:
    """One thing a check found: which check, what, and where in the vault."""
    check: 'Check'
    message: str
    note: Path | None = None
    line: int | None = None

    @property
    def severity(self) -> Severity:
        return self.check.severity

    @property
    def is_error(self) -> bool:
        return self.severity is Severity.ERROR

    def __str__(self) -> str:
        """Like a compiler: `notes/Hello.md:3: warning: ... [missing-heading]`."""
        where = ''
        if self.note is not None:
            where = _vault_relative(self.note)
            if self.line is not None:
                where += f":{self.line}"
            where += ': '
        return f"{where}{self.severity.value}: {self.message} [{self.check.name}]"


def _vault_relative(path: Path) -> str:
    root = getattr(ObsidianPath, '_root', None)
    path = Path(path)
    if root is not None and path.is_relative_to(root):
        return path.relative_to(root).as_posix()
    return path.name
