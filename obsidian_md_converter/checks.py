"""Validation: one check per kind of problem, each finding its problems in the index before any page is written.

A check only reads the index, so finding problems never touches the disk. A problem that `--fix`
can resolve is fixed in a separate step, after every problem has been found.
"""

import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from obsidian_md_converter.config import Config
from obsidian_md_converter.front_matter import FrontMatter, front_matter_class
from obsidian_md_converter.index import IndexedNote, NoteIndex
from obsidian_md_converter.mapping import Mapping
from obsidian_md_converter.problems import Problem, Severity
from obsidian_md_converter.scanning import WikiLink


class Check(ABC):
    """
    One kind of problem, and how to find it in the index.

    Its name, shown with each problem it finds, is its class name in lowercase words joined by
    hyphens: MissingHeading -> 'missing-heading'. Its severity is the class's unless one is given.
    """
    name: str
    severity: Severity

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if 'name' not in vars(cls):
            cls.name = re.sub(r'(?<!^)(?=[A-Z])', '-', cls.__name__).lower()

    def __init__(self, severity: Severity | None = None):
        if severity is not None:
            self.severity = severity

    @abstractmethod
    def find(self, index: NoteIndex) -> Iterator[Problem]:
        """Every problem of this kind in the index."""

    def fix(self, problem: Problem) -> Problem:
        """
        With --fix: resolve one of this check's problems in the vault, and return what to report instead.

        Most problems need the author, so by default the problem is reported as it is.
        """
        return problem


class NoteCheck(Check):
    """A check that looks at one published note at a time."""

    def find(self, index: NoteIndex) -> Iterator[Problem]:
        for note in index.notes:
            yield from self.find_in(note, index)

    @abstractmethod
    def find_in(self, note: IndexedNote, index: NoteIndex) -> Iterator[Problem]:
        """The problems of this kind in one note."""


def checks_for(config: Config) -> list[Check]:
    """The checks a run makes, in the order their problems are reported."""
    checks = [MissingFolder()]
    if not config.validate_only and not config.force_create:
        checks.append(MissingOutputFolder())
    return checks + [
        UnreadableNote(),
        UnusableName(),
        Clash(),
        TabsInFrontMatter(),
        InvalidFrontMatter(),
        UnnamablePage(front_matter_class(config.front_matter_config)),
        MissingImage(),
        MissingHeading(),
    ]


def validate(index: NoteIndex, checks: list[Check]) -> list[Problem]:
    """Every problem the checks find in the index, check by check."""
    return [problem for check in checks for problem in check.find(index)]


def fix(problems: list[Problem]) -> list[Problem]:
    """Resolve in the vault every problem that can be (--fix), reporting the fix in its place."""
    return [problem.check.fix(problem) for problem in problems]


class MissingFolder(Check):
    """A mapping whose vault folder does not exist; nothing is published from it, so its site folder is emptied."""
    severity = Severity.WARNING

    def find(self, index: NoteIndex) -> Iterator[Problem]:
        for folders in index.folders:
            if not folders.source_exists:
                yield Problem(self, f"Source directory not found: {folders.mapping.source}; nothing is published from it")


class MissingOutputFolder(Check):
    """A mapping whose site folder does not exist, in a run that would write into it without creating it."""
    severity = Severity.ERROR

    def find(self, index: NoteIndex) -> Iterator[Problem]:
        for folders in index.folders:
            # A mapping without its vault folder is already reported, and writes nothing.
            if folders.source_exists and not folders.destination_exists:
                yield Problem(self, f"Output directory not found: {folders.mapping.destination} (use --force to create it)")


class UnreadableNote(Check):
    """A note in a mapped folder that cannot be read as UTF-8 text."""
    severity = Severity.ERROR

    def find(self, index: NoteIndex) -> Iterator[Problem]:
        for source, reason in index.unreadable.items():
            yield Problem(self, f"File not readable: {reason}", source)


class Clash(Check):
    """Two published notes of one mapping that would be written to the same page."""
    severity = Severity.ERROR

    def find(self, index: NoteIndex) -> Iterator[Problem]:
        sources_by_page: dict[tuple[Mapping, Path], list[Path]] = {}
        for note in index.notes:
            if note.page is not None:
                sources_by_page.setdefault((note.mapping, note.page.neutral_path), []).append(note.source)
        for (_, path), sources in sources_by_page.items():
            if len(sources) > 1:
                names = " and ".join(f"'{source.name}'" for source in sources)
                yield Problem(self, f"{names} would both be written to {path}; rename one of them")


class UnusableName(NoteCheck):
    """A note whose name has no letters or digits to build its slug from."""
    severity = Severity.ERROR

    def find_in(self, note: IndexedNote, index: NoteIndex) -> Iterator[Problem]:
        if note.page is None:
            yield Problem(self, "its name has no letters or digits to build a URL from; rename it", note.source)


class TabsInFrontMatter(NoteCheck):
    """Front matter indented with tabs, which YAML does not allow; the page is written with spaces instead."""
    severity = Severity.WARNING

    def find_in(self, note: IndexedNote, index: NoteIndex) -> Iterator[Problem]:
        if note.front_matter.yaml is not None and '\t' in note.front_matter.yaml:
            yield Problem(self, "Tabs in front matter YAML (use --fix to resolve)", note.source)

    def fix(self, problem: Problem) -> Problem:
        """Replace the tabs with spaces in the note itself."""
        fm = FrontMatter(problem.note.read_text(encoding='utf-8'))
        fm.fix_tabs()
        problem.note.write_text(fm.serialize(), encoding='utf-8')
        return Problem(self, "Fixed tabs in front matter YAML", problem.note)


class InvalidFrontMatter(NoteCheck):
    """Front matter that is not valid YAML, even with its tabs read as spaces."""
    severity = Severity.ERROR

    def find_in(self, note: IndexedNote, index: NoteIndex) -> Iterator[Problem]:
        if note.front_matter.error is not None:
            yield Problem(self, f"Invalid front matter YAML: {note.front_matter.error}", note.source)


class UnnamablePage(NoteCheck):
    """A page the site generator cannot name from the note's front matter, such as a Jekyll post without a date."""
    severity = Severity.ERROR

    def __init__(self, front_matter_class: type[FrontMatter], severity: Severity | None = None):
        super().__init__(severity)
        self.front_matter_class = front_matter_class

    def find_in(self, note: IndexedNote, index: NoteIndex) -> Iterator[Problem]:
        if note.page is None:
            return
        try:
            note.page_path(self.front_matter_class)
        except ValueError as e:
            yield Problem(self, str(e), note.source)


class MissingImage(NoteCheck):
    """A vault image a note shows that is not in the vault's image folder; the page has its alt text instead."""
    severity = Severity.WARNING

    def find_in(self, note: IndexedNote, index: NoteIndex) -> Iterator[Problem]:
        for image in note.images:
            if not image.exists:
                yield Problem(self, f"Missing image: {image.name}; writing its alt text instead", note.source, image.line)


class MissingHeading(NoteCheck):
    """A heading link naming a heading its note does not have; it points at the page instead. Links to unwritten notes are plain text and not checked."""
    severity = Severity.WARNING

    def find_in(self, note: IndexedNote, index: NoteIndex) -> Iterator[Problem]:
        for link in note.links:
            message = self._missing(link, note, index)
            if message is not None:
                yield Problem(self, message, note.source, link.span.line)

    @staticmethod
    def _missing(link: WikiLink, note: IndexedNote, index: NoteIndex) -> str | None:
        if not link.note:
            if note.anchors.anchor_for(link.heading) is None:
                return f"[[{link.raw}]] names heading '{link.heading}', which this note does not have"
            return None
        target = index.find(link.note)
        if target is None or not link.heading or target.anchors.anchor_for(link.heading) is not None:
            return None
        return f"[[{link.raw}]] names heading '{link.heading}', which '{link.note}' does not have; linking to the page"
