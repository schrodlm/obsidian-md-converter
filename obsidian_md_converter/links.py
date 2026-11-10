"""Where every published note lives, and how one wikilink is rewritten into a link to it."""

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
from obsidian_md_converter.scanning import WikiLink


@dataclass
class LinkTarget:
    """Where a note is published, and the anchors its headings get there."""
    url: str
    anchors: HeadingAnchors


@dataclass
class NoteLinks:
    """What rewriting one note's links needs: every published note, the note's own headings, and where to report problems."""
    link_mapping: dict[str, LinkTarget]
    own_anchors: HeadingAnchors
    source_name: str
    warnings: list[str]


def rewrite_wikilink(link: WikiLink, links: NoteLinks) -> str:
    """
    The Markdown that replaces a scanned wikilink: a link to the published note, or its display text.

    A link to a heading the note does not have points at the page and is reported.

    Examples:
        [[2PC]] -> [2PC](/notes/2pc.html)
        [[2PC|Two Phase Commit]] -> [Two Phase Commit](/notes/2pc.html)
        [[tmux#Prefix key]] -> [tmux > Prefix key](/notes/tmux.html#prefix-key)
        [[#Prefix key]] -> [Prefix key](#prefix-key)
        [[Unknown Note]] -> Unknown Note
    """
    if link.display is not None:
        display_text = link.display
    elif link.note and link.heading:
        display_text = f"{link.note} > {link.heading}"
    else:
        display_text = link.note or link.heading

    if not link.note:
        anchor = links.own_anchors.anchor_for(link.heading) if link.heading else None
        if anchor is None:
            links.warnings.append(f"{links.source_name}: [[{link.raw}]] names heading '{link.heading}', which this note does not have")
            return display_text
        return f"[{display_text}](#{anchor})"

    target = links.link_mapping.get(link.note) or links.link_mapping.get(link.note.lower())
    if target is None:
        return display_text
    if not link.heading:
        return f"[{display_text}]({target.url})"

    anchor = target.anchors.anchor_for(link.heading)
    if anchor is None:
        links.warnings.append(
            f"{links.source_name}: [[{link.raw}]] names heading '{link.heading}', which '{link.note}' does not have; linking to the page"
        )
        return f"[{display_text}]({target.url})"
    return f"[{display_text}]({target.url}#{anchor})"


def build_link_mapping(config: Config) -> dict[str, LinkTarget]:
    """
    Build a mapping of note titles/filenames to where they are published.
    This enables conversion of [[wikilinks]] to proper markdown links.

    Args:
        config: Configuration dictionary from YAML

    Returns:
        Dictionary mapping note title (without extension) to its output URL and heading anchors
        Example: {'2PC': LinkTarget(url='/notes/2pc.html', anchors=...)}
    """
    link_mapping = {}

    for mapping in config.source_mappings:
        mapping : Mapping
        if not mapping.source.exists():
            continue

        # Get all markdown files
        md_files = get_directory_md_files(mapping.source)

        for md_file in md_files:
            # Get the note title (filename without extension)
            note_title = md_file.stem

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

            # Map both the original title and lowercase version
            target = LinkTarget(url=output_url, anchors=HeadingAnchors(fm.body))
            link_mapping[note_title] = target
            link_mapping[note_title.lower()] = target

    return link_mapping
