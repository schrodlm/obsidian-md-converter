"""The index: what a run knows about every published note before it writes any page."""

from dataclasses import dataclass
from pathlib import Path

from obsidian_md_converter.config import Config
from obsidian_md_converter.errors import ConversionError
from obsidian_md_converter.front_matter import FrontMatter
from obsidian_md_converter.headings import HeadingAnchors
from obsidian_md_converter.mapping import Mapping
from obsidian_md_converter.naming import slugify
from obsidian_md_converter.paths import ObsidianPath
from obsidian_md_converter.publishing import get_directory_md_files


@dataclass(frozen=True)
class IndexedNote:
    """A published note as other notes see it: where its page is, and the anchors its headings get there."""
    source: Path
    url: str
    anchors: HeadingAnchors


class NoteIndex:
    """Every published note, found by the name a link gives it."""

    def __init__(self, notes: list[IndexedNote]):
        self.notes = notes
        # Exact names and lowercased names share one table; a later note replaces an earlier one under the same key.
        self._by_name: dict[str, IndexedNote] = {}
        for note in notes:
            self._by_name[note.source.stem] = note
            self._by_name[note.source.stem.lower()] = note

    def __len__(self) -> int:
        return len(self.notes)

    def find(self, name: str) -> IndexedNote | None:
        """The note a link names, matched by its exact name first and then case-insensitively, or None."""
        return self._by_name.get(name) or self._by_name.get(name.lower())


def build_index(config: Config) -> NoteIndex:
    """
    Read every published note of every mapping into the index.

    Example: 'notes/2PC.md' is indexed as IndexedNote(url='/notes/2pc.html', anchors=...), found by '2PC' or '2pc'.
    """
    notes = []

    for mapping in config.source_mappings:
        mapping : Mapping
        if not mapping.source.exists():
            continue

        for md_file in get_directory_md_files(mapping.source):
            content = md_file.read_text(encoding='utf-8')
            fm = FrontMatter(content)
            try:
                output_slug = slugify(ObsidianPath(md_file))
            except ConversionError:
                # Its conversion reports the unusable name; nothing can link to it.
                continue
            # Remove .md extension for URL; a file in a subdirectory keeps it in its URL
            output_slug_no_ext = (mapping.subdirectory_for(md_file) / Path(output_slug).stem).as_posix()

            # Build output URL based on config format
            # Remove leading underscore from collection name for URL (Jekyll convention)
            collection_name = mapping.destination.name.lstrip('_')
            output_url = config.url_format.format(collection=collection_name, slug=output_slug_no_ext)

            notes.append(IndexedNote(source=md_file, url=output_url, anchors=HeadingAnchors(fm.body)))

    return NoteIndex(notes)
