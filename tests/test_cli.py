"""What a run prints at the end, and its exit code: every problem in vault order, then the counts."""

from pathlib import Path

import pytest

from obsidian_md_converter.checks import InvalidFrontMatter, MissingFolder, MissingHeading, MissingImage
from obsidian_md_converter.cli import report
from obsidian_md_converter.paths import ObsidianPath
from obsidian_md_converter.problems import Problem

VAULT = Path("/vault")


@pytest.fixture(autouse=True)
def configure_root():
    ObsidianPath.configure_root(VAULT)


HELLO_IMAGE = Problem(MissingImage(), "Missing image: gone.png", VAULT / "notes/Hello.md", 3)
HELLO_HEADING = Problem(MissingHeading(), "names a missing heading", VAULT / "notes/Hello.md", 1)
BROKEN = Problem(InvalidFrontMatter(), "Invalid front matter YAML", VAULT / "notes/Broken.md")
GONE = Problem(MissingFolder(), "Source directory not found: /vault/gone")


class TestReport:
    def test_a_clean_run_succeeds(self):
        assert report([], 12, validate_only=False) == (["Transfer completed: 0 error(s), 0 warning(s), 12 files processed"], 0)

    def test_warnings_are_listed_and_the_run_succeeds(self):
        lines, exit_code = report([HELLO_IMAGE], 12, validate_only=False)
        assert lines == [
            "notes/Hello.md:3: warning: Missing image: gone.png [missing-image]",
            "Transfer completed: 0 error(s), 1 warning(s), 12 files processed",
        ]
        assert exit_code == 0

    def test_an_error_fails_the_run_and_every_warning_is_still_listed(self):
        lines, exit_code = report([HELLO_IMAGE, BROKEN], 12, validate_only=False)
        assert lines == [
            "notes/Broken.md: error: Invalid front matter YAML [invalid-front-matter]",
            "notes/Hello.md:3: warning: Missing image: gone.png [missing-image]",
            "Transfer FAILED: 1 error(s), 1 warning(s)",
            "Nothing was written to the site.",
        ]
        assert exit_code == 1

    def test_validation_fails_without_saying_nothing_was_written(self):
        lines, exit_code = report([BROKEN], 12, validate_only=True)
        assert lines[-1] == "Validation FAILED: 1 error(s), 0 warning(s)"
        assert exit_code == 1

    def test_problems_are_in_vault_order_with_run_wide_ones_first(self):
        lines, _ = report([HELLO_IMAGE, BROKEN, HELLO_HEADING, GONE], 12, validate_only=False)
        assert [line.split(":")[0] for line in lines[:4]] == ["warning", "notes/Broken.md", "notes/Hello.md", "notes/Hello.md"]
        assert lines[2].startswith("notes/Hello.md:1:")
        assert lines[3].startswith("notes/Hello.md:3:")
