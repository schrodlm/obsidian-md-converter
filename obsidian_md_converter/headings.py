import re
from collections.abc import Callable

from obsidian_md_converter import scanning

_EXPLICIT_ID_RE = re.compile(r'[ \t]*\{#(?P<id>[^}\s]+)\}$')

# Characters Obsidian replaces with a space when matching a link to a heading.
_OBSIDIAN_HEADING_PUNCTUATION_RE = re.compile(r'[!"#$%&()*+,.:;<=>?@^`{|}~\\/\[\]\r\n]')


def strip_heading(text: str) -> str:
    """Normalize heading text the way Obsidian does before comparing it with a link's heading."""
    stripped = _OBSIDIAN_HEADING_PUNCTUATION_RE.sub(' ', text)
    return re.sub(r'\s+', ' ', stripped).strip()


def heading_key(text: str) -> str:
    """The key under which a heading and a link to it meet: Obsidian matches headings case-insensitively."""
    return strip_heading(text).lower()


def rendered_heading_text(source: str) -> str:
    """The heading's visible text once its links are rendered: a link shows only its display text."""
    scanned = scanning.scan_body(source)
    pieces = []
    position = 0
    for structure in sorted([*scanned.links, *scanned.embeds, *scanned.markdown_links, *scanned.markdown_images],
                            key=lambda structure: structure.span.start):
        pieces.append(source[position:structure.span.start])
        pieces.append(_visible_text(structure))
        position = structure.span.end
    pieces.append(source[position:])
    return "".join(pieces)


def _visible_text(structure) -> str:
    if isinstance(structure, (scanning.WikiLink, scanning.Embed)):
        target, separator, display = structure.raw.partition('|')
        return (display if separator else target).strip()
    if isinstance(structure, scanning.MarkdownLink):
        return structure.text
    return structure.alt


class HeadingAnchors:
    """The anchors a page gives its headings, looked up by the heading text an Obsidian link names.

    `anchor_ids` is the site's rule for naming headings, given their rendered text in order;
    a heading with an explicit `{#id}` keeps that id instead.
    """

    def __init__(self, body: str, anchor_ids: Callable[[list[str]], list[str]]):
        # Each heading's key, with its explicit id or None when the site names it.
        headings: list[tuple[str, str | None]] = []
        named_by_site: list[str] = []
        for source in (heading.text for heading in scanning.scan_body(body).headings):
            explicit = _EXPLICIT_ID_RE.search(source)
            if explicit:
                headings.append((heading_key(source[:explicit.start()]), explicit.group('id')))
            else:
                headings.append((heading_key(source), None))
                named_by_site.append(rendered_heading_text(source))

        site_ids = iter(anchor_ids(named_by_site))
        self._anchors: dict[str, str] = {}
        for key, explicit_id in headings:
            anchor = explicit_id if explicit_id is not None else next(site_ids)
            self._anchors.setdefault(key, anchor)

    def anchor_for(self, heading: str) -> str | None:
        """The page anchor for the heading a link names, or None when the note has no such heading."""
        return self._anchors.get(heading_key(heading))
