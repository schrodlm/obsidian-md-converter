"""How one wikilink is rewritten into a link to the published note it names."""

from dataclasses import dataclass

from obsidian_md_converter.headings import HeadingAnchors
from obsidian_md_converter.index import NoteIndex
from obsidian_md_converter.scanning import WikiLink


@dataclass
class NoteLinks:
    """What rewriting one note's links needs: every published note, the note's own headings, and where to report problems."""
    index: NoteIndex
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

    target = links.index.find(link.note)
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
