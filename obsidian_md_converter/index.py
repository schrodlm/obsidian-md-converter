"""The index: what a run knows about every published note before it writes any page."""

from dataclasses import dataclass
from pathlib import Path

from obsidian_md_converter.config import Config
from obsidian_md_converter.errors import ConversionError
from obsidian_md_converter.front_matter import FrontMatter, create_front_matter
from obsidian_md_converter.headings import HeadingAnchors
from obsidian_md_converter.images import local_image_name
from obsidian_md_converter.mapping import Mapping
from obsidian_md_converter.naming import slugify
from obsidian_md_converter.paths import ObsidianPath
from obsidian_md_converter.processors import JekyllProcessor
from obsidian_md_converter.publishing import is_published
from obsidian_md_converter import scanning
from obsidian_md_converter.scanning import Embed, MarkdownImage, WikiLink


@dataclass(frozen=True)
class Page:
    """Where a note is published: its slugified path below its mapping's site folder, and its address."""
    # Before the site generator's own naming, such as Jekyll's date in front of a post: 'moje-projekty/first-entry.md'.
    neutral_path: Path
    url: str


@dataclass(frozen=True)
class IndexedFrontMatter:
    """A note's front matter as its page is written, with tabs in the YAML read as spaces."""
    # The YAML exactly as the note has it, or None when the note has no front matter.
    yaml: str | None
    # The parsed fields; empty when there are none or the YAML does not parse.
    values: dict
    # Why the YAML does not parse, or None when it does.
    error: str | None


@dataclass(frozen=True)
class ImageReference:
    """A vault image a note shows: its name in the vault's image folder, the line that shows it, and whether the file is there."""
    name: str
    line: int
    exists: bool


@dataclass(frozen=True)
class IndexedNote:
    """A published note as the checks, other notes and its writer see it: its names, front matter, headings and references, but not its text."""
    source: Path
    mapping: Mapping
    # None when the note's name has no letters or digits to build a slug from.
    page: Page | None
    front_matter: IndexedFrontMatter
    anchors: HeadingAnchors
    links: list[WikiLink]
    # Images on the web are not the vault's, and are left out.
    images: list[ImageReference]

    def page_path(self, processor: type[JekyllProcessor]) -> Path:
        """
        The page's path below its mapping's site folder, named by the site generator's convention.

        Example: 'publish/posts/Turing Machine.md' with 'layout: post' and 'date: 2024-12-20' -> '2024-12-20-turing-machine.md'

        Raises:
            ValueError: if the front matter lacks what the convention needs.
        """
        name = processor.apply_slug_transform(self.page.neutral_path.name, self.front_matter.values)
        return self.page.neutral_path.with_name(name)


@dataclass(frozen=True)
class MappingFolders:
    """Whether a mapping's vault folder and site folder exist."""
    mapping: Mapping
    source_exists: bool
    destination_exists: bool


class NoteIndex:
    """Every published note, found by the name a link gives it, every note that could not be read, and the mapped folders."""

    def __init__(self, notes: list[IndexedNote], unreadable: dict[Path, str] | None = None,
                 folders: list[MappingFolders] | None = None):
        self.notes = notes
        # Why each unreadable note could not be read.
        self.unreadable = unreadable or {}
        self.folders = folders or []
        # Exact names and lowercased names share one table; a later note replaces an earlier one under the same key.
        self._by_name: dict[str, IndexedNote] = {}
        for note in notes:
            if note.page is None:
                continue
            self._by_name[note.source.stem] = note
            self._by_name[note.source.stem.lower()] = note

    def __len__(self) -> int:
        return len(self.notes)

    def find(self, name: str) -> IndexedNote | None:
        """The note a link names, matched by its exact name first and then case-insensitively, or None."""
        return self._by_name.get(name) or self._by_name.get(name.lower())

    def notes_in(self, mapping: Mapping) -> list[IndexedNote]:
        return [note for note in self.notes if note.mapping == mapping]


def read_front_matter(content: str, fm_config: dict) -> FrontMatter:
    """A note's front matter as its page is written: tabs in the YAML are replaced by spaces."""
    fm = create_front_matter(content, {'front_matter': fm_config})
    fm.fix_tabs()
    return fm


def build_index(config: Config) -> NoteIndex:
    """
    Read every published note of every mapping into the index, each note once.

    Example: 'notes/2PC.md' is indexed with url '/notes/2pc.html', found by '2PC' or '2pc'.
    """
    notes = []
    unreadable = {}
    folders = []

    for mapping in config.source_mappings:
        mapping : Mapping
        folders.append(MappingFolders(mapping, mapping.source.exists(), mapping.destination.exists()))
        if not mapping.source.exists():
            continue

        for md_file in mapping.source.rglob("*.md"):
            try:
                content = md_file.read_text(encoding='utf-8')
            except Exception as e:
                unreadable[md_file] = str(e)
                continue
            if is_published(content):
                notes.append(_index_note(md_file, content, mapping, config))

    return NoteIndex(notes, unreadable, folders)


def _index_note(md_file: Path, content: str, mapping: Mapping, config: Config) -> IndexedNote:
    scanned = scanning.scan(content)
    fm = read_front_matter(content, config.front_matter_config)
    return IndexedNote(
        source=md_file,
        mapping=mapping,
        page=_page(md_file, mapping, config),
        front_matter=IndexedFrontMatter(yaml=scanned.front_matter, values=fm.metadata, error=fm.yaml_error),
        anchors=HeadingAnchors(scanned.body),
        links=scanned.links,
        images=_images([*scanned.embeds, *scanned.markdown_images], config),
    )


def _images(references: list[Embed | MarkdownImage], config: Config) -> list[ImageReference]:
    images = []
    for reference in sorted(references, key=lambda reference: reference.span.start):
        name = local_image_name(reference)
        if name is not None:
            images.append(ImageReference(name, reference.span.line, (config.obsidian_image_dir / name).exists()))
    return images


def _page(md_file: Path, mapping: Mapping, config: Config) -> Page | None:
    try:
        output_slug = slugify(ObsidianPath(md_file))
    except ConversionError:
        return None
    neutral_path = mapping.subdirectory_for(md_file) / output_slug
    # Remove leading underscore from collection name for URL (Jekyll convention)
    collection_name = mapping.destination.name.lstrip('_')
    url = config.url_format.format(collection=collection_name, slug=neutral_path.with_suffix('').as_posix())
    return Page(neutral_path=neutral_path, url=url)
