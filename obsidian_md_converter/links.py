"""How one wikilink is rewritten into a link to the published note it names."""

from dataclasses import dataclass

from obsidian_md_converter.headings import HeadingAnchors
from obsidian_md_converter.index import NoteIndex
from obsidian_md_converter.processors.base import PageLink, Processor
from obsidian_md_converter.scanning import WikiLink


@dataclass
class NoteLinks:
    """What rewriting one note's links needs: every published note and the note's own headings."""
    index: NoteIndex
    own_anchors: HeadingAnchors


def rewrite_wikilink(link: WikiLink, links: NoteLinks, processor: Processor) -> str:
    """
    The Markdown that replaces a scanned wikilink: the processor's link to the published note, or its display text.

    A link to a heading the note does not have points at the page; validation reports it.
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
            return display_text
        return processor.link_markdown(PageLink(display_text, f"#{anchor}"))

    target = links.index.find(link.note)
    if target is None:
        return display_text
    anchor = target.anchors.anchor_for(link.heading) if link.heading else None
    url = target.page.url if anchor is None else f"{target.page.url}#{anchor}"
    return processor.link_markdown(PageLink(display_text, url))
