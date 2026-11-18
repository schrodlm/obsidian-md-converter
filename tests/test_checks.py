"""Each check on a small index built in memory: no vault on disk, except where --fix rewrites a note."""

from pathlib import Path

import pytest

from obsidian_md_converter.checks import (
    Clash, InvalidFrontMatter, MissingFolder, MissingHeading, MissingImage, MissingOutputFolder, TabsInFrontMatter,
    UnnamablePage, UnreadableNote, UnusableName, fix, validate,
)
from obsidian_md_converter.processors import JekyllProcessor
from obsidian_md_converter.headings import HeadingAnchors
from obsidian_md_converter.index import (
    ImageReference, IndexedFrontMatter, IndexedNote, MappingFolders, NoteIndex, Page, read_front_matter,
)
from obsidian_md_converter.mapping import Mapping
from obsidian_md_converter.naming import slugify_text
from obsidian_md_converter.paths import ObsidianPath, OutputPath
from obsidian_md_converter.problems import Problem, Severity
from obsidian_md_converter.scanning import scan_body

VAULT = Path("/vault")
SITE = Path("/site")


@pytest.fixture(autouse=True)
def configure_roots():
    ObsidianPath.configure_root(VAULT)
    OutputPath.configure_root(SITE)


def mapping(folder: str = "notes") -> Mapping:
    return Mapping(ObsidianPath(VAULT / folder), OutputPath(SITE / f"_{folder}"))


def indexed_note(name: str, body: str = "", front_matter: str | None = None, folder: str = "notes",
                 images: list[ImageReference] | None = None) -> IndexedNote:
    """A published note as the index records it, built from its name, body and front matter."""
    slug = slugify_text(name)
    fm = read_front_matter("" if front_matter is None else f"---\n{front_matter}\n---\n")
    return IndexedNote(
        source=VAULT / folder / f"{name}.md",
        mapping=mapping(folder),
        page=Page(Path(f"{slug}.md"), f"/{folder}/{slug}.html") if slug else None,
        front_matter=IndexedFrontMatter(yaml=front_matter, values=fm.metadata, error=fm.yaml_error),
        anchors=HeadingAnchors(body, JekyllProcessor("/{collection}/{slug}.html").anchor_ids),
        links=scan_body(body).links,
        images=images or [],
    )


def find(check, *notes: IndexedNote) -> list[Problem]:
    return list(check.find(NoteIndex(list(notes))))


class TestCheck:
    """A check is named after its class, and its severity is the class's unless the instance is given one."""

    def test_name_is_the_class_name_in_lowercase_words_joined_by_hyphens(self):
        assert MissingHeading.name == "missing-heading"
        assert TabsInFrontMatter().name == "tabs-in-front-matter"

    def test_problem_takes_the_severity_of_its_check(self):
        [problem] = find(MissingHeading(), indexed_note("Hello", "See [[#Absent]].\n"))
        assert problem.severity is Severity.WARNING
        assert not problem.is_error

    def test_severity_can_be_set_per_check(self):
        [problem] = find(MissingHeading(Severity.ERROR), indexed_note("Hello", "See [[#Absent]].\n"))
        assert problem.is_error
        assert MissingHeading.severity is Severity.WARNING


class TestProblem:
    """A problem prints like a compiler message: where, severity, what, and the check's name."""

    def test_names_the_note_by_its_vault_path_and_line(self):
        problem = Problem(MissingHeading(), "names a missing heading", VAULT / "notes" / "Hello.md", 3)
        assert str(problem) == "notes/Hello.md:3: warning: names a missing heading [missing-heading]"

    def test_without_a_line(self):
        problem = Problem(InvalidFrontMatter(), "Invalid front matter YAML: bad", VAULT / "notes" / "Hello.md")
        assert str(problem) == "notes/Hello.md: error: Invalid front matter YAML: bad [invalid-front-matter]"

    def test_without_a_note(self):
        assert str(Problem(Clash(), "two pages")) == "error: two pages [clash]"


class TestFolders:
    def test_missing_vault_folder_is_reported(self):
        index = NoteIndex([], folders=[MappingFolders(mapping("gone"), source_exists=False, destination_exists=False)])
        [problem] = MissingFolder().find(index)
        assert problem.message == "Source directory not found: /vault/gone; nothing is published from it"

    def test_missing_site_folder_is_reported_only_when_the_vault_folder_exists(self):
        index = NoteIndex([], folders=[
            MappingFolders(mapping("notes"), source_exists=True, destination_exists=False),
            MappingFolders(mapping("gone"), source_exists=False, destination_exists=False),
        ])
        [problem] = MissingOutputFolder().find(index)
        assert problem.message == "Output directory not found: /site/_notes (use --force to create it)"


class TestUnreadableNote:
    def test_each_unreadable_note_is_reported_with_its_reason(self):
        index = NoteIndex([], unreadable={VAULT / "notes" / "Binary.md": "invalid start byte"})
        [problem] = UnreadableNote().find(index)
        assert problem.note == VAULT / "notes" / "Binary.md"
        assert problem.message == "File not readable: invalid start byte"


class TestUnusableName:
    def test_name_without_letters_or_digits_is_reported(self):
        [problem] = find(UnusableName(), indexed_note("???"), indexed_note("Hello"))
        assert problem.note == VAULT / "notes" / "???.md"

    def test_names_in_other_scripts_are_usable(self):
        assert find(UnusableName(), indexed_note("Привет мир")) == []


class TestFrontMatter:
    TABS = "tags:\n\t- a"

    def test_tabs_are_a_warning(self):
        [problem] = find(TabsInFrontMatter(), indexed_note("Hello", "Body\n", front_matter=self.TABS))
        assert problem.message == "Tabs in front matter YAML (use --fix to resolve)"
        assert not problem.is_error

    def test_tabs_alone_do_not_make_the_yaml_invalid(self):
        assert find(InvalidFrontMatter(), indexed_note("Hello", "Body\n", front_matter=self.TABS)) == []

    def test_invalid_yaml_is_an_error_with_its_line(self):
        [problem] = find(InvalidFrontMatter(), indexed_note("Hello", "Body\n", front_matter="tags:\n      - a\n  - b"))
        assert problem.message.startswith("Invalid front matter YAML: ")
        assert "(line 4)" in problem.message


class TestUnnamablePage:
    def test_post_without_a_date_is_an_error(self):
        [problem] = find(UnnamablePage(JekyllProcessor("/{collection}/{slug}.html")), indexed_note("Turing Machine", "Body\n", front_matter="layout: post"))
        assert problem.message == "Missing 'date' field for post layout"

    def test_dated_post_and_other_layouts_pass(self):
        dated = indexed_note("Turing Machine", "Body\n", front_matter="layout: post\ndate: 2024-12-20")
        page = indexed_note("Hello", "Body\n", front_matter="layout: note")
        assert find(UnnamablePage(JekyllProcessor("/{collection}/{slug}.html")), dated, page) == []

    def test_invalid_yaml_is_not_also_reported_as_a_missing_date(self):
        note = indexed_note("Hello", "Body\n", front_matter="layout: post\ntags:\n      - a\n  - b")
        assert find(UnnamablePage(JekyllProcessor("/{collection}/{slug}.html")), note) == []


class TestFixingTabs:
    """--fix resolves tabs in the vault after validation, and reports the fix instead of the problem."""

    def test_fix_rewrites_the_note_with_spaces(self, tmp_path):
        ObsidianPath.configure_root(tmp_path)
        source = tmp_path / "Hello.md"
        source.write_text("---\ntags:\n\t- a\n---\nBody\n", encoding="utf-8")
        problem = Problem(TabsInFrontMatter(), "Tabs in front matter YAML (use --fix to resolve)", source)

        [fixed] = fix([problem])

        assert source.read_text(encoding="utf-8") == "---\ntags:\n  - a\n---\nBody\n"
        assert fixed.message == "Fixed tabs in front matter YAML"
        assert fixed.check is problem.check

    def test_problems_without_a_fix_are_reported_unchanged(self):
        problem = Problem(MissingHeading(), "names a missing heading", VAULT / "notes" / "Hello.md", 3)
        assert fix([problem]) == [problem]


class TestMissingImage:
    def test_each_missing_image_is_reported_on_its_line(self):
        note = indexed_note("Hello", images=[ImageReference("there.png", 2, exists=True),
                                             ImageReference("splay results.png", 5, exists=False)])
        [problem] = find(MissingImage(), note)
        assert problem.message == "Missing image: splay results.png; writing its alt text instead"
        assert problem.line == 5


class TestMissingHeading:
    TARGET = ("Second Note", "## Present\n")

    def test_link_to_a_heading_another_note_lacks_is_reported_on_its_line(self):
        note = indexed_note("Hello", "Intro\n\nSee [[Second Note#Renamed|it]].\n")
        [problem] = find(MissingHeading(), indexed_note(*self.TARGET), note)
        assert problem.message == ("[[Second Note#Renamed|it]] names heading 'Renamed', "
                                   "which 'Second Note' does not have; linking to the page")
        assert problem.line == 3

    def test_link_to_a_heading_the_note_itself_lacks_is_reported(self):
        [problem] = find(MissingHeading(), indexed_note("Hello", "## Present\nSee [[#Absent]].\n"))
        assert problem.message == "[[#Absent]] names heading 'Absent', which this note does not have"

    def test_links_to_present_headings_and_whole_notes_pass(self):
        note = indexed_note("Hello", "## Own\nSee [[#own]], [[second note#present]] and [[Second Note]].\n")
        assert find(MissingHeading(), indexed_note(*self.TARGET), note) == []

    def test_link_to_an_unwritten_note_is_not_checked(self):
        assert find(MissingHeading(), indexed_note("Hello", "See [[Not Written Yet#Part]].\n")) == []


class TestClash:
    def test_notes_written_to_the_same_page_are_reported_once(self):
        [problem] = find(Clash(), indexed_note("Délivery (protocols)"), indexed_note("delivery protocols"),
                         indexed_note("Hello"))
        assert problem.message == ("'Délivery (protocols).md' and 'delivery protocols.md' would both be written to "
                                   "delivery-protocols.md; rename one of them")

    def test_same_page_name_in_different_mappings_does_not_clash(self):
        assert find(Clash(), indexed_note("Log", folder="notes"), indexed_note("Log", folder="logs")) == []


class TestValidate:
    def test_problems_are_reported_check_by_check(self):
        index = NoteIndex([
            indexed_note("Hello", "See [[#Absent]].\n", images=[ImageReference("gone.png", 1, exists=False)]),
            indexed_note("World", images=[ImageReference("gone.png", 1, exists=False)]),
        ])
        problems = validate(index, [MissingHeading(), MissingImage()])
        assert [(problem.check.name, problem.note.stem) for problem in problems] == [
            ("missing-heading", "Hello"), ("missing-image", "Hello"), ("missing-image", "World"),
        ]
